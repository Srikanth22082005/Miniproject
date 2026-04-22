"""
Face Registration Module — YOLOv8n-face + ArcFace (InsightFace).

Scans the dataset/ directory, detects faces with YOLOv8n-face,
extracts 512-D ArcFace embeddings (w600k_r50), and builds a FAISS
IndexFlatIP for fast cosine-similarity matching at runtime.

Usage:
    python register.py
"""

import os
import sys
import pickle
import numpy as np

# Fix M1 Mac Segmentation Fault (Import PyTorch/YOLO BEFORE cv2 to prevent BLAS library conflicts)
import torch
from ultralytics import YOLO
import cv2

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR    = os.path.join(BASE_DIR, "dataset")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "face_index.faiss")
LABELS_PATH      = os.path.join(EMBEDDINGS_DIR, "labels.pkl")

# ── Config (must match recognize.py) ─────────────────────────────────────────
YOLO_IMGSZ       = 320
YOLO_CONF        = 0.35    # slightly lower threshold for registration photos
FACE_SIZE        = 112     # ArcFace input size
MODEL_DETECTOR   = "yolov8n-face.pt"
MODEL_RECOGNIZER = "w600k_mbf"   # InsightFace ArcFace MobileNet (buffalo_sc)
YOLO_WEIGHTS_URL = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
)


def _ensure_yolo_weights() -> str:
    """Download yolov8n-face.pt from GitHub if not present. Returns local path."""
    model_path = os.path.join(BASE_DIR, MODEL_DETECTOR)
    if not os.path.exists(model_path):
        import urllib.request
        print(f"[REGISTER] Downloading {MODEL_DETECTOR} from GitHub (~6 MB) ...")
        try:
            urllib.request.urlretrieve(YOLO_WEIGHTS_URL, model_path)
            print(f"[REGISTER] {MODEL_DETECTOR} saved to {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download {MODEL_DETECTOR}: {e}")
            print(f"  Manual download: {YOLO_WEIGHTS_URL}")
            sys.exit(1)
    return model_path


def _load_arcface(providers=None):
    """
    Load InsightFace ArcFace recognizer via direct ONNX path.
    Falls back to downloading buffalo_sc if not cached.
    """
    if providers is None:
        providers = ["CPUExecutionProvider"]
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model as _gm
    import glob as _glob

    home     = os.path.expanduser("~")
    pattern  = os.path.join(home, ".insightface", "models", "*", "w600k_mbf.onnx")
    candidates = _glob.glob(pattern)

    if candidates:
        onnx_path = candidates[0]
        print(f"[REGISTER] Using ArcFace ONNX: {onnx_path}")
        rec = _gm(onnx_path, providers=providers)
        rec.prepare(ctx_id=-1)
        return rec

    print("[REGISTER] Downloading buffalo_sc (ArcFace) ...")
    app = FaceAnalysis(name="buffalo_sc", providers=providers)
    app.prepare(ctx_id=-1, det_size=(320, 320))
    candidates = _glob.glob(pattern)
    if candidates:
        onnx_path = candidates[0]
        rec = _gm(onnx_path, providers=providers)
        rec.prepare(ctx_id=-1)
        return rec

    raise RuntimeError("Could not locate w600k_mbf.onnx after downloading buffalo_sc.")


def align_face(frame: np.ndarray, bbox: list, target_size: int = FACE_SIZE) -> np.ndarray | None:
    """
    Crop and resize a face region to target_size × target_size with 10% padding.
    Consistent with recognize.py alignment logic.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]

    pad = int((x2 - x1) * 0.10)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (target_size, target_size))


def register_faces():
    """Scan dataset/, detect faces with YOLO, embed with ArcFace, build FAISS index."""

    # ── Import dependencies ──
    try:
        import faiss
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Run: pip install ultralytics insightface faiss-cpu onnxruntime")
        sys.exit(1)

    # ── Load YOLOv8n-face detector ──
    print(f"[REGISTER] Loading YOLOv8n-face detector ({MODEL_DETECTOR}) ...")
    model_path = _ensure_yolo_weights()
    yolo = YOLO(model_path)
    dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
    yolo(dummy, imgsz=YOLO_IMGSZ, verbose=False)   # warm-up
    print("[REGISTER] YOLOv8n-face detector ready.")

    # ── Load ArcFace recognizer ──
    print(f"[REGISTER] Loading ArcFace recognizer ({MODEL_RECOGNIZER}) ...")
    recognizer = _load_arcface(providers=["CPUExecutionProvider"])
    print("[REGISTER] ArcFace recognizer ready.")

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # ── Validate dataset directory ──
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        print("Create: dataset/<person_name>/<image>.jpg")
        sys.exit(1)

    person_dirs = [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith(".")
    ]

    if not person_dirs:
        print("[ERROR] No person directories found in dataset/")
        print("Create folders like: dataset/alice/1.jpg")
        sys.exit(1)

    print(f"[REGISTER] Found {len(person_dirs)} person(s): {person_dirs}")

    # ── Process each person ──
    embeddings: list[np.ndarray] = []
    labels:     list[str]        = []

    for person_name in sorted(person_dirs):
        person_path = os.path.join(DATASET_DIR, person_name)
        image_files = [
            f for f in os.listdir(person_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        if not image_files:
            print(f"  [WARN] No images found for '{person_name}', skipping.")
            continue

        print(f"  [REGISTER] '{person_name}' — {len(image_files)} image(s)")

        for img_file in sorted(image_files):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"    [WARN] Could not read: {img_file}")
                continue

            # ── Detect faces with YOLOv8n-face ──
            results = yolo(img, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
            boxes   = results[0].boxes

            if boxes is None or len(boxes) == 0:
                print(f"    [WARN] No face detected in: {img_file}")
                continue

            # Pick the largest detected face (most prominent)
            best_box  = None
            best_area = 0.0
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if area > best_area:
                    best_area = area
                    best_box  = xyxy

            if best_box is None:
                print(f"    [WARN] Could not extract best box from: {img_file}")
                continue

            # ── Align and embed ──
            aligned = align_face(img, best_box)
            if aligned is None:
                print(f"    [WARN] Face alignment failed for: {img_file}")
                continue

            try:
                embedding = recognizer.get_feat(aligned)
                if embedding is None:
                    print(f"    [WARN] No embedding from: {img_file}")
                    continue

                embedding = embedding.flatten().astype(np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding /= norm

                embeddings.append(embedding)
                labels.append(person_name)
                print(f"    [OK] {img_file} — 512-D ArcFace embedding registered")

            except Exception as e:
                print(f"    [WARN] Embedding error for {img_file}: {e}")

    if not embeddings:
        print("[ERROR] No faces were registered. Add clear face photos to dataset/.")
        sys.exit(1)

    # ── Build FAISS IndexFlatIP (cosine similarity on normalised vectors) ──
    print(f"\n[REGISTER] Building FAISS index with {len(embeddings)} embedding(s) ...")
    matrix    = np.array(embeddings, dtype=np.float32)
    dimension = matrix.shape[1]   # 512
    index     = faiss.IndexFlatIP(dimension)
    index.add(matrix)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"[REGISTER] FAISS index saved → {FAISS_INDEX_PATH}")

    with open(LABELS_PATH, "wb") as f:
        pickle.dump(labels, f)
    print(f"[REGISTER] Labels saved    → {LABELS_PATH}")

    print(f"\n{'='*52}")
    print(f"  ✅ Registration complete!")
    print(f"  Stack     : YOLOv8n-face + ArcFace (w600k_r50)")
    print(f"  Persons   : {len(set(labels))}")
    print(f"  Embeddings: {len(embeddings)}")
    print(f"  Dimension : {dimension}-D")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    register_faces()

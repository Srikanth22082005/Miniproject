"""
Face Registration Module.
Scans the dataset/ directory, detects faces using SCRFD,
extracts ArcFace 512-D embeddings, and builds a FAISS index for fast matching.
"""

import os
import sys
import pickle
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "face_index.faiss")
LABELS_PATH = os.path.join(EMBEDDINGS_DIR, "labels.pkl")


def register_faces():
    """Scan dataset folder, extract face embeddings, and build FAISS index."""
    try:
        import faiss
        from insightface.app import FaceAnalysis
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Run: pip install insightface faiss-cpu onnxruntime")
        sys.exit(1)

    # Initialize InsightFace (SCRFD + ArcFace)
    print("[REGISTER] Initializing InsightFace (SCRFD + ArcFace)...")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    embeddings = []
    labels = []
    import cv2

    # Scan dataset directories
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        print("Create dataset/<person_name>/ folders with face images.")
        sys.exit(1)

    person_dirs = [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith(".")
    ]

    if not person_dirs:
        print("[ERROR] No person directories found in dataset/")
        print("Create folders like: dataset/srikanth/1.jpg")
        sys.exit(1)

    print(f"[REGISTER] Found {len(person_dirs)} person(s): {person_dirs}")

    for person_name in sorted(person_dirs):
        person_path = os.path.join(DATASET_DIR, person_name)
        image_files = [
            f for f in os.listdir(person_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        if not image_files:
            print(f"  [WARN] No images found for '{person_name}', skipping.")
            continue

        print(f"  [REGISTER] Processing '{person_name}' — {len(image_files)} image(s)")

        for img_file in sorted(image_files):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"    [WARN] Could not read: {img_file}")
                continue

            # Detect and extract face embeddings
            faces = app.get(img)

            if not faces:
                print(f"    [WARN] No face detected in: {img_file}")
                continue

            # Use the largest face (most prominent)
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embedding = face.embedding  # 512-D ArcFace vector

            if embedding is not None:
                # Normalize embedding for cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                embeddings.append(embedding)
                labels.append(person_name)
                print(f"    [OK] {img_file} — face registered (512-D embedding)")
            else:
                print(f"    [WARN] No embedding extracted from: {img_file}")

    if not embeddings:
        print("[ERROR] No faces were registered. Add clear face photos to dataset/.")
        sys.exit(1)

    # Build FAISS index
    print(f"\n[REGISTER] Building FAISS index with {len(embeddings)} embedding(s)...")
    embeddings_matrix = np.array(embeddings, dtype=np.float32)

    # Use IndexFlatIP for cosine similarity (inner product on normalized vectors)
    dimension = embeddings_matrix.shape[1]  # 512
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)

    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"[REGISTER] FAISS index saved: {FAISS_INDEX_PATH}")

    # Save labels
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(labels, f)
    print(f"[REGISTER] Labels saved: {LABELS_PATH}")

    print(f"\n{'='*50}")
    print(f"  Registration complete!")
    print(f"  Persons: {len(set(labels))}")
    print(f"  Total embeddings: {len(embeddings)}")
    print(f"  Index dimension: {dimension}")
    print(f"{'='*50}")


if __name__ == "__main__":
    register_faces()

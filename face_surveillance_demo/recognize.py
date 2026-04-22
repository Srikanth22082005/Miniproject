"""
Real-time Face Recognition Engine — YOLOv8n-face + ArcFace (InsightFace).
OPTIMIZED FOR CPU — targets 10-20+ FPS.

Detection  : YOLOv8n-face  (ultralytics) — ultra-fast CPU face detector
Recognition: ArcFace w600k_mbf (InsightFace) — 512-D embeddings
Tracking   : ByteTrack (custom) — stable multi-face track IDs
Matching   : FAISS IndexFlatIP — cosine similarity search

Performance optimizations:
  - Cap max faces per frame (MAX_FACES = 5)
  - Cache embeddings per track_id — skip ArcFace if track already identified
  - Threaded inference — YOLO+ArcFace run in background, don't block frame read
  - Skip frames aggressively (PROCESS_EVERY_N = 5)
  - Minimum face size filter (skip tiny/distant faces)
  - Reuse numpy arrays, avoid unnecessary copies
"""

import os
import sys
import time
import pickle
import threading
from collections import deque
import numpy as np

# Fix M1 Mac Segmentation Fault (Import PyTorch/YOLO BEFORE cv2 to prevent BLAS library conflicts)
import torch
from ultralytics import YOLO
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "embeddings", "face_index.faiss")
LABELS_PATH      = os.path.join(BASE_DIR, "embeddings", "labels.pkl")

# ─── Configuration ──────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD  = 0.45   # cosine similarity threshold for recognition
LOG_COOLDOWN          = 10     # seconds between logging the same track_id
UNKNOWN_SAVE_COOLDOWN = 60     # seconds before saving same unknown track again
PROCESS_EVERY_N       = 3      # run detection every Nth frame (higher → faster FPS)
CAMERA_WIDTH          = 640
CAMERA_HEIGHT         = 480
YOLO_IMGSZ            = 320    # YOLO inference image size
YOLO_CONF             = 0.50   # minimum face detection confidence (higher = fewer false positives)
FACE_SIZE             = 112    # ArcFace input size (112×112)
MAX_FACES             = 4      # max faces to process per frame (largest faces first)
MIN_FACE_SIZE         = 40     # minimum face width/height in pixels (skip tiny faces)
EMBEDDING_CACHE_TTL   = 8.0    # seconds before re-running ArcFace on same track
MODEL_DETECTOR        = "yolov8n-face.pt"
MODEL_RECOGNIZER      = "w600k_mbf"
YOLO_WEIGHTS_URL      = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
)


def _ensure_yolo_weights():
    """Download yolov8n-face.pt from GitHub if not present."""
    model_path = os.path.join(BASE_DIR, MODEL_DETECTOR)
    if not os.path.exists(model_path):
        import urllib.request
        print(f"[ENGINE] Downloading {MODEL_DETECTOR} from GitHub (~6 MB) ...")
        try:
            urllib.request.urlretrieve(YOLO_WEIGHTS_URL, model_path)
            print(f"[ENGINE] {MODEL_DETECTOR} saved to {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download {MODEL_DETECTOR}: {e}")
            print(f"  Manual download: {YOLO_WEIGHTS_URL}")
            sys.exit(1)
    return model_path


def _load_arcface(providers=None):
    """
    Load InsightFace ArcFace recognizer via buffalo_sc ONNX path.
    Falls back to downloading buffalo_sc if not cached.
    """
    if providers is None:
        providers = ["CPUExecutionProvider"]
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model as _gm
    import glob as _glob

    home = os.path.expanduser("~")
    pattern = os.path.join(home, ".insightface", "models", "*", "w600k_mbf.onnx")
    candidates = _glob.glob(pattern)

    if candidates:
        onnx_path = candidates[0]
        print(f"[ENGINE] Using ArcFace ONNX: {onnx_path}")
        rec = _gm(onnx_path, providers=providers)
        rec.prepare(ctx_id=-1)
        return rec

    print("[ENGINE] Downloading buffalo_sc (ArcFace) ...")
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
    """Crop and resize a face region to target_size × target_size."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h, w = frame.shape[:2]

    # Small padding for context
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


class FaceRecognizer:
    """Real-time face recognition engine: YOLOv8n-face + ArcFace — CPU optimized."""

    def __init__(self):
        self.yolo          = None
        self.recognizer    = None
        self.faiss_index   = None
        self.labels        = None
        self.tracker       = None
        self.cap           = None
        self.running       = False
        self.frame         = None
        self.frame_lock    = threading.Lock()
        self.log_timestamps      = {}
        self.unknown_saved_tracks = set()
        self.start_time    = None
        self.fps           = 0
        self.frame_counter = 0
        self.last_detections = []

        # ── Smooth-streaming additions ──
        self._raw_frame      = None          # latest frame from camera reader thread
        self._raw_lock       = threading.Lock()
        self._jpeg_buf       = None          # pre-encoded JPEG bytes
        self._jpeg_lock      = threading.Lock()
        self._jpeg_event     = threading.Event()  # signals new JPEG available

        # ── Embedding cache: track_id → {label, similarity, timestamp} ──
        self.track_cache: dict[int, dict] = {}

    # ── Initialisation ──────────────────────────────────────────────────────

    def initialize(self):
        """Load all models and data."""
        try:
            import faiss
            from ultralytics import YOLO
            from tracker import ByteTracker
            from database import init_db
        except ImportError as e:
            print(f"[ERROR] Missing dependency: {e}")
            print("Run: pip install ultralytics insightface faiss-cpu onnxruntime")
            sys.exit(1)

        init_db()

        # ── YOLOv8n-face detector ──
        print(f"[ENGINE] Loading YOLOv8n-face detector ({MODEL_DETECTOR}) ...")
        model_path = _ensure_yolo_weights()
        self.yolo = YOLO(model_path)
        # Warm-up
        dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
        self.yolo(dummy, imgsz=YOLO_IMGSZ, verbose=False)
        print("[ENGINE] YOLOv8n-face detector ready.")

        # ── ArcFace recognizer ──
        print(f"[ENGINE] Loading ArcFace recognizer ({MODEL_RECOGNIZER}) ...")
        self.recognizer = _load_arcface(providers=["CPUExecutionProvider"])
        print("[ENGINE] ArcFace recognizer ready.")

        # ── FAISS index ──
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(LABELS_PATH):
            print("[WARN] No FAISS index found. Run register.py first.")
            self.faiss_index = None
            self.labels      = None
        else:
            self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(LABELS_PATH, "rb") as f:
                self.labels = pickle.load(f)
            print(
                f"[ENGINE] FAISS index loaded: {self.faiss_index.ntotal} embeddings, "
                f"{len(set(self.labels))} person(s)"
            )

        # ── ByteTracker ──
        self.tracker = ByteTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            high_thresh=0.5,
            low_thresh=0.1,
        )
        print("[ENGINE] ByteTrack tracker ready.")
        print(f"[ENGINE] ✅ Stack: YOLOv8n-face → ArcFace → FAISS → ByteTrack")
        print(f"[ENGINE] ⚡ Optimizations: MAX_FACES={MAX_FACES}, SKIP={PROCESS_EVERY_N}, "
              f"MIN_SIZE={MIN_FACE_SIZE}, CACHE_TTL={EMBEDDING_CACHE_TTL}s")

    # ── Detection (YOLO only — fast) ─────────────────────────────────────────

    def detect_faces_fast(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLOv8n-face on frame. Returns bbox+confidence only.
        ArcFace embedding is deferred (only run on unidentified tracks).
        """
        results = self.yolo(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
        detections = []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # Collect all valid detections
        raw = []
        for box in boxes:
            xyxy  = box.xyxy[0].cpu().numpy().tolist()
            conf  = float(box.conf[0].cpu().numpy())
            face_w = xyxy[2] - xyxy[0]
            face_h = xyxy[3] - xyxy[1]

            # Skip tiny faces (distant/noisy detections)
            if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
                continue

            area = face_w * face_h
            raw.append((xyxy, conf, area))

        # Sort by area descending → process largest faces first
        raw.sort(key=lambda x: x[2], reverse=True)

        # Cap at MAX_FACES
        for xyxy, conf, _ in raw[:MAX_FACES]:
            detections.append({
                "bbox":       xyxy,
                "confidence": conf,
                "label":      "Unknown",
                "embedding":  None,
                "similarity": 0.0,
            })

        return detections

    # ── Recognition (ArcFace — expensive, cached) ────────────────────────────

    def identify_tracks(self, frame: np.ndarray, active_tracks: list):
        """
        Run ArcFace ONLY on tracks that haven't been identified yet or
        whose cache has expired. This is the key optimization — we skip
        ArcFace entirely for already-known faces.
        """
        now = time.time()
        from tracker import iou as compute_iou

        for track in active_tracks:
            tid = track.track_id

            # Check cache: if track was recently identified, skip ArcFace
            cached = self.track_cache.get(tid)
            if cached and (now - cached["timestamp"]) < EMBEDDING_CACHE_TTL:
                track.label      = cached["label"]
                track.confidence = cached["similarity"]
                continue

            # Need to run ArcFace on this track
            aligned = align_face(frame, track.bbox.tolist())
            if aligned is None:
                continue

            embedding = self._get_embedding(aligned)
            label, similarity = self.match_face(embedding)

            track.label      = label
            track.confidence = similarity

            # Cache the result
            self.track_cache[tid] = {
                "label":      label,
                "similarity": similarity,
                "timestamp":  now,
            }

        # Clean up stale cache entries (tracks that no longer exist)
        active_ids = {t.track_id for t in active_tracks}
        stale_keys = [k for k in self.track_cache if k not in active_ids]
        for k in stale_keys:
            del self.track_cache[k]

    def _get_embedding(self, aligned_face: np.ndarray) -> np.ndarray | None:
        """Extract 512-D ArcFace embedding from a 112×112 BGR face crop."""
        try:
            embedding = self.recognizer.get_feat(aligned_face)
            if embedding is None:
                return None
            embedding = embedding.flatten().astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm
            return embedding
        except Exception as e:
            print(f"[WARN] Embedding extraction failed: {e}")
            return None

    # ── Matching ─────────────────────────────────────────────────────────────

    def match_face(self, embedding: np.ndarray | None) -> tuple[str, float]:
        """Match a 512-D embedding against the FAISS index."""
        if embedding is None or self.faiss_index is None or self.labels is None:
            return "Unknown", 0.0

        query = np.array([embedding], dtype=np.float32)
        scores, indices = self.faiss_index.search(query, k=1)

        score = float(scores[0][0])
        idx   = int(indices[0][0])

        if score >= SIMILARITY_THRESHOLD and 0 <= idx < len(self.labels):
            return self.labels[idx], score
        return "Unknown", score

    # ── Logging helpers ──────────────────────────────────────────────────────

    def should_log(self, track_id: int) -> bool:
        now  = time.time()
        last = self.log_timestamps.get(track_id, 0)
        if now - last >= LOG_COOLDOWN:
            self.log_timestamps[track_id] = now
            return True
        return False

    # ── Frame processing ─────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Optimized pipeline:
          1. YOLO detection every Nth frame (fast — no ArcFace here)
          2. ByteTracker every frame (smooth bounding boxes)
          3. ArcFace only on UN-identified tracks (cached for known faces)
          4. Draw + log
        """
        from database import log_visit_and_attendance, log_unknown_face

        self.frame_counter += 1

        # ── Step 1: YOLO detection (every Nth frame) ──
        if self.frame_counter % PROCESS_EVERY_N == 0:
            self.last_detections = self.detect_faces_fast(frame)

        detections = self.last_detections

        # ── Step 2: ByteTracker (every frame for smooth tracking) ──
        active_tracks = self.tracker.update(detections)

        # ── Step 3: ArcFace recognition (only on un-cached tracks) ──
        # This is the KEY optimization: we only call ArcFace on tracks
        # whose identity is unknown or whose cache expired.
        if self.frame_counter % PROCESS_EVERY_N == 0:
            self.identify_tracks(frame, active_tracks)
        else:
            # On skipped frames, just apply cached labels
            for track in active_tracks:
                cached = self.track_cache.get(track.track_id)
                if cached:
                    track.label      = cached["label"]
                    track.confidence = cached["similarity"]

        # ── Step 4: Draw + log ──
        for track in active_tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            name       = track.label
            similarity = track.confidence
            track_id   = track.track_id

            color  = (0, 220, 100) if name != "Unknown" else (0, 80, 255)
            status = "Recognized"  if name != "Unknown" else "Unknown"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label badge
            label_text = f"{name} ({similarity:.2f}) ID:{track_id}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # DB log (with cooldown)
            if self.should_log(track_id):
                log_visit_and_attendance(name, similarity, track_id, status)

            # Save unknown face crop once per track
            if name == "Unknown" and track_id not in self.unknown_saved_tracks:
                self.unknown_saved_tracks.add(track_id)
                unknown_dir = os.path.join(BASE_DIR, "static", "unknown_faces")
                os.makedirs(unknown_dir, exist_ok=True)
                try:
                    h, w = frame.shape[:2]
                    sY = max(0, int(y1) - 20)
                    eY = min(h, int(y2) + 20)
                    sX = max(0, int(x1) - 20)
                    eX = min(w, int(x2) + 20)
                    if sY < eY and sX < eX:
                        face_crop = frame[sY:eY, sX:eX]
                        if face_crop.size > 0:
                            filename = f"unknown_{int(time.time())}_{track_id}.jpg"
                            filepath = os.path.join(unknown_dir, filename)
                            cv2.imwrite(filepath, face_crop)
                            log_unknown_face(f"/static/unknown_faces/{filename}")
                except Exception as e:
                    print(f"[WARN] Error saving unknown face: {e}")

        # HUD
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Tracks: {len(active_tracks)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return frame

    # ── Camera & loop ────────────────────────────────────────────────────────

    def start_camera(self, camera_index: int = 0):
        """Open and configure the webcam."""
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            print(f"[ENGINE] Opening camera {camera_index}... (attempt {attempt}/{max_retries})")
            self.cap = (
                cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                if sys.platform == "win32"
                else cv2.VideoCapture(camera_index)
            )
            if self.cap.isOpened():
                ret, _ = self.cap.read()
                if ret:
                    break
                self.cap.release()
            if attempt < max_retries:
                print("[ENGINE] Camera not ready, retrying in 2 s...")
                time.sleep(2)
            else:
                print("[ERROR] Cannot open camera after all retries!")
                sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Minimize camera buffering — always read the latest frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[ENGINE] Camera opened: {actual_w}×{actual_h}")

    # ── Dedicated camera reader thread ──────────────────────────────────────

    def _camera_reader(self):
        """
        Continuously grab frames from the camera in a tight loop.
        This drains the OS/driver buffer so the processing thread
        always gets the *latest* frame instead of a stale queued one.
        """
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                continue
            with self._raw_lock:
                self._raw_frame = frame

    def run_loop(self):
        """
        Background recognition loop (for Flask thread).
        Architecture:
          Thread 1 (_camera_reader) — reads camera as fast as possible
          Thread 2 (this method)    — processes + encodes JPEG
        The reader thread ensures we never read a stale buffered frame.
        """
        self.running    = True
        self.start_time = time.time()
        frame_count     = 0
        fps_timer       = time.time()

        # Start the dedicated camera reader
        reader = threading.Thread(target=self._camera_reader, daemon=True)
        reader.start()
        print("[ENGINE] Camera reader thread started.")

        # JPEG encode params (allocated once)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]

        while self.running:
            # Grab the latest frame from the reader thread
            with self._raw_lock:
                raw_frame = self._raw_frame

            if raw_frame is None:
                time.sleep(0.005)
                continue

            try:
                processed = self.process_frame(raw_frame)
            except Exception as e:
                import traceback
                print(f"[ERROR] Frame processing error: {e}")
                traceback.print_exc()
                processed = raw_frame

            # Pre-encode JPEG once (avoid re-encoding per HTTP request)
            _, buf = cv2.imencode(".jpg", processed, encode_params)
            jpeg_bytes = buf.tobytes()

            with self._jpeg_lock:
                self._jpeg_buf = jpeg_bytes
            self._jpeg_event.set()

            # Also keep raw numpy frame for legacy access
            with self.frame_lock:
                self.frame = processed

            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self.fps     = frame_count / elapsed
                frame_count  = 0
                fps_timer    = time.time()

        if self.cap:
            self.cap.release()

    def get_frame_jpeg(self) -> bytes | None:
        """Return the latest pre-encoded JPEG bytes (zero-copy, no re-encode)."""
        with self._jpeg_lock:
            return self._jpeg_buf

    def get_uptime(self) -> float:
        return time.time() - self.start_time if self.start_time else 0

    def stop(self):
        self.running = False


# ── Standalone mode ──────────────────────────────────────────────────────────

def run_standalone():
    """Run with an OpenCV window (no Flask)."""
    engine = FaceRecognizer()
    engine.initialize()
    engine.start_camera(0)

    print("\n[ENGINE] Starting recognition. Press 'q' to quit.\n")
    engine.start_time = time.time()
    frame_count = 0
    fps_timer   = time.time()

    while True:
        ret, raw_frame = engine.cap.read()
        if not ret:
            continue

        processed = engine.process_frame(raw_frame)

        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            engine.fps  = frame_count / elapsed
            frame_count = 0
            fps_timer   = time.time()

        cv2.imshow("Face Surveillance — YOLOv8n-face + ArcFace", processed)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    engine.cap.release()
    cv2.destroyAllWindows()
    print("[ENGINE] Stopped.")


if __name__ == "__main__":
    run_standalone()

"""
Real-time Face Recognition Engine.
Uses SCRFD detection, ArcFace embeddings, ByteTrack tracking,
FAISS matching, and SQLite logging. Provides both standalone
OpenCV window and MJPEG streaming for Flask.
"""

import os
import sys
import time
import pickle
import threading
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "embeddings", "face_index.faiss")
LABELS_PATH = os.path.join(BASE_DIR, "embeddings", "labels.pkl")

# Recognition settings
SIMILARITY_THRESHOLD = 0.45
LOG_COOLDOWN = 3  # seconds between logging the same track_id


class FaceRecognizer:
    """Real-time face recognition engine."""

    def __init__(self):
        self.face_app = None
        self.faiss_index = None
        self.labels = None
        self.tracker = None
        self.cap = None
        self.running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.log_timestamps = {}  # track_id -> last_log_time
        self.start_time = None
        self.fps = 0

    def initialize(self):
        """Load all models and data."""
        try:
            import faiss
            from insightface.app import FaceAnalysis
            from tracker import ByteTracker
            from database import init_db
        except ImportError as e:
            print(f"[ERROR] Missing dependency: {e}")
            sys.exit(1)

        # Initialize database
        init_db()

        # Initialize InsightFace
        print("[ENGINE] Loading InsightFace (SCRFD + ArcFace)...")
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self.face_app.prepare(ctx_id=-1, det_size=(640, 640))
        print("[ENGINE] InsightFace ready.")

        # Load FAISS index
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(LABELS_PATH):
            print("[WARN] No FAISS index found. Run register.py first.")
            print("[WARN] Running in detection-only mode (all faces = Unknown).")
            self.faiss_index = None
            self.labels = None
        else:
            self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(LABELS_PATH, "rb") as f:
                self.labels = pickle.load(f)
            print(f"[ENGINE] FAISS index loaded: {self.faiss_index.ntotal} embeddings, {len(set(self.labels))} person(s)")

        # Initialize tracker
        self.tracker = ByteTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            high_thresh=0.5,
            low_thresh=0.1,
        )
        print("[ENGINE] ByteTrack tracker ready.")

    def match_face(self, embedding):
        """Match a face embedding against the FAISS index."""
        if self.faiss_index is None or self.labels is None:
            return "Unknown", 0.0

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        query = np.array([embedding], dtype=np.float32)
        scores, indices = self.faiss_index.search(query, k=1)

        score = float(scores[0][0])
        idx = int(indices[0][0])

        if score >= SIMILARITY_THRESHOLD and 0 <= idx < len(self.labels):
            return self.labels[idx], score
        else:
            return "Unknown", score

    def should_log(self, track_id):
        """Check if we should log this track (cooldown-based)."""
        now = time.time()
        last = self.log_timestamps.get(track_id, 0)
        if now - last >= LOG_COOLDOWN:
            self.log_timestamps[track_id] = now
            return True
        return False

    def process_frame(self, frame):
        """Process a single video frame through the full pipeline."""
        from database import insert_log

        detections = []

        # 1. Face detection + embedding extraction
        faces = self.face_app.get(frame)

        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            det_score = float(face.det_score)

            # 2. Match against FAISS
            label = "Unknown"
            similarity = 0.0
            embedding = None

            if face.embedding is not None:
                embedding = face.embedding
                label, similarity = self.match_face(embedding)

            detections.append({
                "bbox": bbox,
                "confidence": det_score,
                "label": label,
                "embedding": embedding,
                "similarity": similarity,
            })

        # 3. Update ByteTrack tracker
        active_tracks = self.tracker.update(detections)

        # Map detections to tracks for labeling
        # Associate closest detection to each track
        det_labels = {}
        for det in detections:
            for track in active_tracks:
                from tracker import iou as compute_iou
                if compute_iou(det["bbox"], track.bbox) > 0.3:
                    if track.label == "Unknown" and det["label"] != "Unknown":
                        track.label = det["label"]
                        track.confidence = det["similarity"]
                    det_labels[track.track_id] = {
                        "label": track.label,
                        "similarity": det.get("similarity", track.confidence),
                    }

        # 4. Draw results and log
        for track in active_tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            label = track.label
            info = det_labels.get(track.track_id, {"label": label, "similarity": track.confidence})
            name = info["label"]
            similarity = info["similarity"]
            track_id = track.track_id

            # Color: green for known, red for unknown
            if name != "Unknown":
                color = (0, 220, 100)
                status = "Recognized"
            else:
                color = (0, 80, 255)
                status = "Unknown"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_text = f"{name} ({similarity:.2f}) ID:{track_id}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 5. Log to SQLite (with cooldown)
            if self.should_log(track_id):
                insert_log(name, similarity, track_id, status)

        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Draw track count
        cv2.putText(frame, f"Tracks: {len(active_tracks)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        return frame

    def start_camera(self, camera_index=0):
        """Start the webcam capture with retry for macOS permission dialog."""
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            print(f"[ENGINE] Opening camera {camera_index}... (attempt {attempt}/{max_retries})")
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                # Try reading a test frame to confirm it actually works
                ret, _ = self.cap.read()
                if ret:
                    break
                else:
                    self.cap.release()
            if attempt < max_retries:
                print("[ENGINE] Camera not ready. If you see a macOS permission dialog, click 'Allow'.")
                print(f"[ENGINE] Retrying in 3 seconds...")
                time.sleep(3)
            else:
                print("[ERROR] Cannot open camera after all retries!")
                print("[TIP] Go to System Settings → Privacy & Security → Camera")
                print("      and enable access for Terminal / your IDE.")
                sys.exit(1)

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[ENGINE] Camera opened: {actual_w}x{actual_h}")

    def run_loop(self):
        """Main recognition loop (for thread)."""
        self.running = True
        self.start_time = time.time()
        frame_count = 0
        fps_timer = time.time()

        while self.running:
            ret, raw_frame = self.cap.read()
            if not ret:
                print("[WARN] Frame capture failed, retrying...")
                time.sleep(0.1)
                continue

            # Process frame
            processed = self.process_frame(raw_frame)

            # Update FPS counter
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            # Store processed frame for streaming
            with self.frame_lock:
                self.frame = processed.copy()

        # Cleanup
        if self.cap:
            self.cap.release()

    def get_frame_jpeg(self):
        """Get the current processed frame as JPEG bytes."""
        with self.frame_lock:
            if self.frame is None:
                return None
            _, buffer = cv2.imencode(".jpg", self.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buffer.tobytes()

    def get_uptime(self):
        """Get uptime in seconds."""
        if self.start_time:
            return time.time() - self.start_time
        return 0

    def stop(self):
        """Stop the recognition engine."""
        self.running = False


def run_standalone():
    """Run the recognizer with an OpenCV window (standalone mode)."""
    engine = FaceRecognizer()
    engine.initialize()
    engine.start_camera(0)

    print("\n[ENGINE] Starting recognition. Press 'q' to quit.\n")
    engine.start_time = time.time()
    frame_count = 0
    fps_timer = time.time()

    while True:
        ret, raw_frame = engine.cap.read()
        if not ret:
            continue

        processed = engine.process_frame(raw_frame)

        # FPS calculation
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            engine.fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        cv2.imshow("Face Surveillance Demo", processed)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    engine.cap.release()
    cv2.destroyAllWindows()
    print("[ENGINE] Stopped.")


if __name__ == "__main__":
    run_standalone()

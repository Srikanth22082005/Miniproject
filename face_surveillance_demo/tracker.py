"""
ByteTrack-style face tracker.
Lightweight IoU-based multi-object tracker with Kalman filter prediction.
Assigns stable Track IDs across video frames to reduce redundant recognition.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanFilter2D:
    """Simple 2D Kalman filter for bounding box center tracking."""

    def __init__(self, bbox):
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 10
        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = 1  # cx += vx
        self.F[1, 5] = 1  # cy += vy
        self.F[2, 6] = 1  # w += vw
        self.F[3, 7] = 1  # h += vh
        self.Q = np.eye(8, dtype=np.float32) * 1.0
        self.H = np.eye(4, 8, dtype=np.float32)
        self.R = np.eye(4, dtype=np.float32) * 5.0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_bbox()

    def update(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w, h], dtype=np.float32)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P
        return self.get_bbox()

    def get_bbox(self):
        cx, cy, w, h = self.state[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class Track:
    """Single tracked object."""
    _id_counter = 0

    def __init__(self, bbox, confidence, label=None, embedding=None):
        Track._id_counter += 1
        self.track_id = Track._id_counter
        self.kalman = KalmanFilter2D(bbox)
        self.bbox = np.array(bbox, dtype=np.float32)
        self.confidence = confidence
        self.label = label or "Unknown"
        self.embedding = embedding
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.is_confirmed = False

    def predict(self):
        self.bbox = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox, confidence, label=None, embedding=None):
        self.bbox = self.kalman.update(bbox)
        self.confidence = confidence
        if label and label != "Unknown":
            self.label = label
        if embedding is not None:
            self.embedding = embedding
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= 3:
            self.is_confirmed = True

    @staticmethod
    def reset_id_counter():
        Track._id_counter = 0


def iou(bbox1, bbox2):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


class ByteTracker:
    """
    ByteTrack-style multi-object tracker.
    
    Features:
    - Two-stage association (high-confidence + low-confidence detections)
    - Kalman filter prediction for motion estimation
    - IoU-based matching with Hungarian algorithm
    - Stable Track ID assignment
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, high_thresh=0.5, low_thresh=0.1):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.tracks = []

    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: list of dicts with keys:
                - 'bbox': [x1, y1, x2, y2]
                - 'confidence': float
                - 'label': str (optional)
                - 'embedding': ndarray (optional)
        
        Returns:
            list of active Track objects
        """
        # Predict existing tracks
        for track in self.tracks:
            track.predict()

        if not detections:
            # Remove dead tracks
            self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
            return [t for t in self.tracks if t.is_confirmed]

        # Split detections into high and low confidence
        high_dets = [d for d in detections if d["confidence"] >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d["confidence"] < self.high_thresh]

        # --- First association: high confidence detections ---
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed]
        unmatched_tracks_idx, unmatched_dets_idx = self._associate(
            confirmed_tracks, high_dets
        )

        # --- Second association: low confidence detections with unmatched tracks ---
        remaining_tracks = [confirmed_tracks[i] for i in unmatched_tracks_idx]
        unmatched_tracks_idx_2, _ = self._associate(remaining_tracks, low_dets)

        # --- Third association: unconfirmed tracks with remaining high detections ---
        unconfirmed_tracks = [t for t in self.tracks if not t.is_confirmed]
        remaining_high_dets = [high_dets[i] for i in unmatched_dets_idx]
        _, unmatched_new_dets_idx = self._associate(unconfirmed_tracks, remaining_high_dets)

        # Create new tracks for unmatched high-confidence detections
        for i in unmatched_new_dets_idx:
            det = remaining_high_dets[i]
            new_track = Track(
                det["bbox"],
                det["confidence"],
                det.get("label"),
                det.get("embedding"),
            )
            self.tracks.append(new_track)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        return [t for t in self.tracks if t.is_confirmed]

    def _associate(self, tracks, detections):
        """Associate tracks with detections using IoU + Hungarian algorithm."""
        if not tracks or not detections:
            return list(range(len(tracks))), list(range(len(detections)))

        # Build cost matrix (negative IoU for minimization)
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1.0 - iou(track.bbox, det["bbox"])

        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < (1.0 - self.iou_threshold):
                tracks[row].update(
                    detections[col]["bbox"],
                    detections[col]["confidence"],
                    detections[col].get("label"),
                    detections[col].get("embedding"),
                )
                matched_tracks.add(row)
                matched_dets.add(col)

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]

        return unmatched_tracks, unmatched_dets

    def reset(self):
        """Reset all tracks."""
        self.tracks = []
        Track.reset_id_counter()

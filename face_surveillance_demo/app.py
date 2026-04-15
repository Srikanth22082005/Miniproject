"""
Flask Web Dashboard for Face Surveillance System.
Provides live video feed, recognition logs, and statistics.
"""

import os
import sys
import time
import threading
from flask import Flask, render_template, Response, jsonify

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from database import init_db, get_recent_logs, get_stats, clear_logs
from recognize import FaceRecognizer

app = Flask(__name__)
engine = FaceRecognizer()


def start_engine():
    """Initialize and start the recognition engine in a background thread."""
    engine.initialize()
    engine.start_camera(0)
    engine_thread = threading.Thread(target=engine.run_loop, daemon=True)
    engine_thread.start()
    print("[APP] Recognition engine started in background thread.")


def generate_frames():
    """Generator for MJPEG video stream."""
    while True:
        frame_bytes = engine.get_frame_jpeg()
        if frame_bytes is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.033)  # ~30 FPS cap


# ─── Routes ───────────────────────────────────────────────────

@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/logs")
def api_logs():
    """Get recent recognition logs as JSON."""
    logs = get_recent_logs(limit=100)
    return jsonify(logs)


@app.route("/api/stats")
def api_stats():
    """Get recognition statistics as JSON."""
    stats = get_stats()
    stats["uptime"] = round(engine.get_uptime(), 1)
    stats["fps"] = round(engine.fps, 1)
    return jsonify(stats)


@app.route("/api/clear_logs", methods=["POST"])
def api_clear_logs():
    """Clear all recognition logs."""
    clear_logs()
    return jsonify({"status": "ok", "message": "Logs cleared."})


# ─── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    start_engine()
    print("\n" + "=" * 55)
    print("  🎯 Face Surveillance Dashboard")
    print("  Open: http://localhost:5000")
    print("  Press Ctrl+C to stop.")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

"""
Flask API Server for Face Attendance System.
Provides face streaming, analytics endpoints, and student CRUD operations for React frontend.
"""

import os
import sys
import time
import threading
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from database import init_db, get_analytics, get_all_students, get_student_insights, get_entry_exit_logs
from recognize import FaceRecognizer

app = Flask(__name__)
# Enable CORS for React dev server (e.g., port 5173)
CORS(app)

engine = FaceRecognizer()


def start_engine():
    """Initialize and start the recognition engine in a background thread."""
    engine.initialize()
    engine.start_camera(0)
    engine_thread = threading.Thread(target=engine.run_loop, daemon=True)
    engine_thread.start()
    print("[APP] Recognition engine started in background thread.")


def generate_frames():
    """Generator for MJPEG video stream — event-driven, zero artificial delay."""
    while True:
        # Wait up to 100ms for a new frame (event-driven, no busy-loop)
        engine._jpeg_event.wait(timeout=0.1)
        engine._jpeg_event.clear()

        frame_bytes = engine.get_frame_jpeg()
        if frame_bytes is None:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# ─── API Routes ───────────────────────────────────────────────────

@app.route("/api/status")
def status():
    return jsonify({
        "status": "running",
        "uptime": round(engine.get_uptime(), 1) if engine.running else 0,
        "fps": round(engine.fps, 1)
    })

@app.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint."""
    resp = Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    # Disable all buffering for real-time streaming
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

@app.route("/api/analytics")
def api_analytics():
    """Get analytics data."""
    try:
        data = get_analytics()
        # Add entry/exit logs 
        data["entryExitLogs"] = get_entry_exit_logs()
        # Add insights
        data["insights"] = get_student_insights()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/students", methods=["GET"])
def api_students():
    """Get all students."""
    try:
        students = get_all_students()
        return jsonify(students)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/unknown_faces", methods=["GET"])
def api_unknown_faces():
    """Get unknown faces logs from db."""
    from database import get_connection
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM unknown_faces ORDER BY id DESC")
    faces = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify(faces)

@app.route("/api/attendance", methods=["GET"])
def api_attendance():
    """Get attendance logs."""
    from database import get_connection
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT a.id, a.date, a.time, a.status, s.name, s.student_id, s.department
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
        ORDER BY a.date DESC, a.time DESC
    """)
    logs = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify(logs)

# ─── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    start_engine()
    print("\n" + "=" * 55)
    print("  🎯 Face Attendance System API")
    print("  Running on: http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

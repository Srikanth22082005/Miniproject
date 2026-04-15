"""
Database module — SQLite logging for face surveillance events.
Stores recognition events with person name, confidence, track ID, and timestamp.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs.db")


def get_connection():
    """Get a thread-safe SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database and create the logs table if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            track_id INTEGER,
            timestamp TEXT NOT NULL,
            status TEXT DEFAULT 'Recognized'
        )
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Database initialized at {DB_PATH}")


def insert_log(person_name, confidence, track_id=None, status="Recognized"):
    """Insert a new recognition event into the log."""
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO logs (person_name, confidence, track_id, timestamp, status) VALUES (?, ?, ?, ?, ?)",
        (person_name, round(confidence, 4), track_id, timestamp, status),
    )
    conn.commit()
    conn.close()


def get_recent_logs(limit=50):
    """Retrieve the most recent recognition logs."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_stats():
    """Get aggregated statistics from the logs."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM logs")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(DISTINCT person_name) as unique_persons FROM logs WHERE person_name != 'Unknown'")
    unique_persons = cursor.fetchone()["unique_persons"]

    cursor.execute("SELECT COUNT(*) as unknowns FROM logs WHERE person_name = 'Unknown'")
    unknowns = cursor.fetchone()["unknowns"]

    cursor.execute("SELECT person_name, COUNT(*) as count FROM logs GROUP BY person_name ORDER BY count DESC LIMIT 10")
    top_persons = [dict(row) for row in cursor.fetchall()]

    # Recent activity — detections per minute for the last 30 minutes
    cursor.execute("""
        SELECT strftime('%H:%M', timestamp) as minute, COUNT(*) as count
        FROM logs
        WHERE timestamp >= datetime('now', '-30 minutes', 'localtime')
        GROUP BY minute
        ORDER BY minute
    """)
    activity = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return {
        "total_detections": total,
        "unique_persons": unique_persons,
        "unknown_detections": unknowns,
        "top_persons": top_persons,
        "recent_activity": activity,
    }


def clear_logs():
    """Clear all logs from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM logs")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("[DB] Database ready.")

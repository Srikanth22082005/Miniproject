"""
Database module — SQLite logging for face surveillance and attendance system.
Handles students, attendance semantics, visits, and unknown faces.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")


def get_connection():
    """Get a thread-safe SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database and create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Students table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT,
            year TEXT,
            image_path TEXT,
            registered_at TEXT
        )
    """)
    
    # Attendance table (Once per day per student)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'Present',
            FOREIGN KEY (student_id) REFERENCES students (student_id),
            UNIQUE(student_id, date)
        )
    """)
    
    # Visits table (Every time they are seen - Entry/Exit tracking)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT NOT NULL,
            student_id TEXT,
            confidence REAL,
            track_id INTEGER,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT DEFAULT 'Recognized'
        )
    """)
    
    # Unknown faces table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS unknown_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"[DB] Database initialized at {DB_PATH}")


def log_visit_and_attendance(person_name, confidence, track_id, status="Recognized", student_id=None, time_window_start="08:30", time_window_end="10:00"):
    """
    Log a visit. If the person is a known student, also attempt to log attendance
    based on whether they've been marked today.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    current_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Log visit
    cursor.execute("""
        INSERT INTO visits (person_name, student_id, confidence, track_id, date, time, timestamp, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (person_name, student_id, round(confidence, 4), track_id, current_date, current_time, current_timestamp, status))
    
    # 2. Log attendance if known student
    if status == "Recognized" and person_name != "Unknown":
        # Look up student_id if not provided
        if not student_id:
            cursor.execute("SELECT student_id FROM students WHERE name = ?", (person_name,))
            row = cursor.fetchone()
            if row:
                student_id = row['student_id']
                
        if student_id:
            # Check if attendance already marked today
            cursor.execute("SELECT id FROM attendance WHERE student_id = ? AND date = ?", (student_id, current_date))
            if not cursor.fetchone():
                # Determine status based on time window
                # Parse times
                curr_t = datetime.strptime(current_time, "%H:%M:%S").time()
                start_t = datetime.strptime(time_window_start, "%H:%M").time()
                end_t = datetime.strptime(time_window_end, "%H:%M").time()
                
                attendance_status = "Present"
                if curr_t > end_t:
                    attendance_status = "Late"
                
                try:
                    cursor.execute("""
                        INSERT INTO attendance (student_id, date, time, status)
                        VALUES (?, ?, ?, ?)
                    """, (student_id, current_date, current_time, attendance_status))
                except sqlite3.IntegrityError:
                    # Already exists (race condition)
                    pass
    
    conn.commit()
    conn.close()


def log_unknown_face(image_path):
    """Save record of an unknown face."""
    conn = get_connection()
    cursor = conn.cursor()
    
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    current_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO unknown_faces (image_path, date, time, timestamp)
        VALUES (?, ?, ?, ?)
    """, (image_path, current_date, current_time, current_timestamp))
    
    conn.commit()
    conn.close()


def get_analytics():
    """Get analytics data for dashboard."""
    conn = get_connection()
    cursor = conn.cursor()
    
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    
    # 1. Total visits today
    cursor.execute("SELECT COUNT(*) as total FROM visits WHERE date = ?", (current_date,))
    total_visits_today = cursor.fetchone()["total"]
    
    # 2. Present / Late / Absent counts today
    cursor.execute("SELECT COUNT(*) as total FROM students")
    total_students = cursor.fetchone()["total"]
    
    cursor.execute("SELECT COUNT(*) as present FROM attendance WHERE date = ? AND status='Present'", (current_date,))
    present_today = cursor.fetchone()["present"]
    
    cursor.execute("SELECT COUNT(*) as late FROM attendance WHERE date = ? AND status='Late'", (current_date,))
    late_today = cursor.fetchone()["late"]
    
    absent_today = total_students - present_today - late_today
    
    # 3. Faces per hour (today)
    cursor.execute("""
        SELECT strftime('%H', timestamp) as hour, COUNT(*) as count 
        FROM visits 
        WHERE date = ?
        GROUP BY hour ORDER BY hour
    """, (current_date,))
    faces_per_hour = [dict(row) for row in cursor.fetchall()]
    
    # 4. Unknown vs Known ratio (today)
    cursor.execute("SELECT COUNT(*) as count FROM visits WHERE date = ? AND status = 'Recognized'", (current_date,))
    known_count = cursor.fetchone()["count"]
    cursor.execute("SELECT COUNT(*) as count FROM visits WHERE date = ? AND status = 'Unknown'", (current_date,))
    unknown_count = cursor.fetchone()["count"]
    
    # 5. Most frequent person
    cursor.execute("""
        SELECT person_name, COUNT(*) as count 
        FROM visits 
        WHERE status = 'Recognized' 
        GROUP BY person_name 
        ORDER BY count DESC LIMIT 5
    """)
    most_frequent = [dict(row) for row in cursor.fetchall()]

    conn.close()
    
    return {
        "stats": {
            "totalVisitsToday": total_visits_today,
            "presentToday": present_today,
            "lateToday": late_today,
            "absentToday": absent_today,
            "totalStudents": total_students
        },
        "charts": {
            "facesPerHour": faces_per_hour,
            "knownVsUnknown": {"known": known_count, "unknown": unknown_count},
            "mostFrequent": most_frequent
        }
    }


def get_student_insights():
    """Detect low attendance (<75%) and frequent absentees."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get total distinct days with at least one attendance recorded
    cursor.execute("SELECT COUNT(DISTINCT date) as total_days FROM attendance")
    total_days = cursor.fetchone()["total_days"]
    
    insights = []
    
    if total_days > 0:
        cursor.execute("SELECT student_id, name FROM students")
        students = cursor.fetchall()
        
        for std in students:
            sid = std["student_id"]
            name = std["name"]
            
            cursor.execute("SELECT COUNT(*) as present_days FROM attendance WHERE student_id = ?", (sid,))
            present_days = cursor.fetchone()["present_days"]
            
            percentage = (present_days / total_days) * 100
            
            if percentage < 75.0:
                insights.append({
                    "student_id": sid,
                    "name": name,
                    "attendance_percentage": round(percentage, 1),
                    "insight": "Low Attendance"
                })
                
    conn.close()
    return insights


def get_all_students():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students")
    students = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return students

def get_entry_exit_logs():
    """Fetch recent visit logs with first/last seen aggregation."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT person_name, 
               MIN(time) as first_seen, 
               MAX(time) as last_seen, 
               COUNT(*) as total_visits
        FROM visits 
        WHERE date = date('now', 'localtime') AND status = 'Recognized'
        GROUP BY person_name
    """)
    logs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return logs

if __name__ == "__main__":
    init_db()

"""
Generate 2 months of realistic synthetic attendance & visit data.
Populates: attendance, visits tables for all 10 students.
Run: python generate_synthetic_data.py
"""

import sqlite3
import os
import random
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")

# 10 Students (must match seed_db.py)
STUDENTS = [
    ("STU001", "John Doe"),
    ("STU002", "Jane Smith"),
    ("STU003", "Alice Johnson"),
    ("STU004", "Bob Brown"),
    ("STU005", "Charlie Davis"),
    ("STU006", "Diana Miller"),
    ("STU007", "Ethan Wilson"),
    ("STU008", "Fiona Moore"),
    ("STU009", "George Taylor"),
    ("STU010", "Hannah Anderson"),
]

def is_working_day(dt):
    """Monday=0 ... Sunday=6. Skip weekends."""
    return dt.weekday() < 5

def random_time(hour_min=8, hour_max=10):
    """Generate a random time between hour_min and hour_max."""
    h = random.randint(hour_min, hour_max)
    m = random.randint(0, 59)
    s = random.randint(0, 59)
    return f"{h:02d}:{m:02d}:{s:02d}"

def generate():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Clear old synthetic data (keep today's real data)
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("DELETE FROM attendance WHERE date != ?", (today,))
    c.execute("DELETE FROM visits WHERE date != ?", (today,))
    conn.commit()

    # Generate working days for the last 2 months
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=60)

    working_days = []
    current = start_date
    while current <= end_date:
        if is_working_day(current):
            working_days.append(current)
        current += timedelta(days=1)

    print(f"[SYNTH] Generating data for {len(working_days)} working days ({start_date} to {end_date})")
    print(f"[SYNTH] Students: {len(STUDENTS)}")

    attendance_rows = []
    visit_rows = []

    for day in working_days:
        date_str = day.strftime("%Y-%m-%d")

        for sid, name in STUDENTS:
            # Each student has a different attendance probability
            # Some students are reliable, some are not (for insights to show)
            if sid == "STU004":       # Bob Brown - chronic absentee (~55%)
                show_prob = 0.55
            elif sid == "STU008":     # Fiona Moore - low attendance (~65%)
                show_prob = 0.65
            elif sid == "STU010":     # Hannah Anderson - borderline (~72%)
                show_prob = 0.72
            elif sid == "STU001":     # John Doe - perfect (~98%)
                show_prob = 0.98
            else:                     # Everyone else ~85-92%
                show_prob = random.uniform(0.85, 0.92)

            if random.random() > show_prob:
                continue  # Student absent this day

            # Determine arrival time
            is_late = random.random() < 0.12  # 12% chance of being late
            if is_late:
                arrival_time = random_time(10, 11)  # After 10:00 AM
                status = "Late"
            else:
                arrival_time = random_time(8, 9)    # 8:00-9:59 AM
                status = "Present"

            timestamp = f"{date_str} {arrival_time}"

            # Insert attendance (once per day)
            attendance_rows.append((sid, date_str, arrival_time, status))

            # Insert visits (1-4 visits per day for this student)
            num_visits = random.randint(1, 4)
            for v in range(num_visits):
                if v == 0:
                    visit_time = arrival_time
                else:
                    visit_time = random_time(9, 16)  # Later visits during the day

                visit_ts = f"{date_str} {visit_time}"
                confidence = round(random.uniform(0.55, 0.95), 4)
                track_id = random.randint(1, 9999)

                visit_rows.append((
                    name, sid, confidence, track_id,
                    date_str, visit_time, visit_ts, "Recognized"
                ))

        # Also add some unknown visits per day (2-6)
        num_unknowns = random.randint(2, 6)
        for _ in range(num_unknowns):
            unk_time = random_time(8, 16)
            unk_ts = f"{date_str} {unk_time}"
            confidence = round(random.uniform(0.15, 0.40), 4)
            track_id = random.randint(10000, 99999)

            visit_rows.append((
                "Unknown", None, confidence, track_id,
                date_str, unk_time, unk_ts, "Unknown"
            ))

    # Bulk insert attendance
    c.executemany("""
        INSERT OR IGNORE INTO attendance (student_id, date, time, status)
        VALUES (?, ?, ?, ?)
    """, attendance_rows)

    # Bulk insert visits
    c.executemany("""
        INSERT INTO visits (person_name, student_id, confidence, track_id, date, time, timestamp, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, visit_rows)

    conn.commit()

    # Print summary
    c.execute("SELECT COUNT(*) FROM attendance")
    att_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM visits")
    vis_count = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT date) FROM attendance")
    day_count = c.fetchone()[0]

    conn.close()

    print(f"\n{'='*50}")
    print(f"  [OK] Synthetic Data Generated")
    print(f"  Working days:      {day_count}")
    print(f"  Attendance records: {att_count}")
    print(f"  Visit records:     {vis_count}")
    print(f"{'='*50}")

if __name__ == "__main__":
    generate()

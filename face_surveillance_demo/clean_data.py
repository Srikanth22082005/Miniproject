import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Clear all data
cursor.execute("DELETE FROM attendance")
cursor.execute("DELETE FROM visits")
cursor.execute("DELETE FROM students")
cursor.execute("DELETE FROM unknown_faces")

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Insert real students
cursor.execute("INSERT INTO students (student_id, name, department, year, image_path, registered_at) VALUES (?, ?, ?, ?, ?, ?)", 
    ("STU001", "srikanth", "CS", "Year 1", "dataset/srikanth", now))
cursor.execute("INSERT INTO students (student_id, name, department, year, image_path, registered_at) VALUES (?, ?, ?, ?, ?, ?)", 
    ("STU002", "tamil", "CS", "Year 1", "dataset/tamil", now))

conn.commit()
conn.close()
print("Synthetic data removed and real users registered in DB.")

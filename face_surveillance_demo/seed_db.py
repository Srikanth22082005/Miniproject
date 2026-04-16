import os
import sqlite3
from datetime import datetime
import database

# 10 Synthetic Students
STUDENTS = [
    ("STU001", "John Doe", "Computer Science", "Year 1"),
    ("STU002", "Jane Smith", "Electrical Eng", "Year 2"),
    ("STU003", "Alice Johnson", "Mechanical Eng", "Year 3"),
    ("STU004", "Bob Brown", "Civil Eng", "Year 4"),
    ("STU005", "Charlie Davis", "Computer Science", "Year 1"),
    ("STU006", "Diana Miller", "Electrical Eng", "Year 2"),
    ("STU007", "Ethan Wilson", "Mechanical Eng", "Year 3"),
    ("STU008", "Fiona Moore", "Civil Eng", "Year 4"),
    ("STU009", "George Taylor", "Computer Science", "Year 1"),
    ("STU010", "Hannah Anderson", "Electrical Eng", "Year 2"),
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

def seed_database():
    database.init_db()
    conn = database.get_connection()
    cursor = conn.cursor()
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for s_id, name, dept, year in STUDENTS:
        # Create directory
        student_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(student_dir, exist_ok=True)
        
        # Create a placeholder file so git/user knows where to put photos
        with open(os.path.join(student_dir, "placeholder.txt"), "w") as f:
            f.write("Please upload face images (.jpg, .png) of this student here.")
            
        img_path = f"dataset/{name}"
        
        try:
            cursor.execute("""
                INSERT INTO students (student_id, name, department, year, image_path, registered_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (s_id, name, dept, year, img_path, now))
            print(f"Inserted {name} into database.")
        except sqlite3.IntegrityError:
            print(f"{name} already exists.")
            
    conn.commit()
    conn.close()
    print("Database seeding completed.")

if __name__ == "__main__":
    seed_database()

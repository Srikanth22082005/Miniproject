# 🎯 SmartAttend — AI-Powered Face Recognition Attendance System

## Complete Project Documentation

---

## 📌 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Project Structure](#5-project-structure)
6. [Backend — Python Modules](#6-backend--python-modules)
   - 6.1 [app.py — Flask API Server](#61-apppy--flask-api-server)
   - 6.2 [recognize.py — Face Recognition Engine](#62-recognizepy--face-recognition-engine)
   - 6.3 [register.py — Face Registration Module](#63-registerpy--face-registration-module)
   - 6.4 [tracker.py — ByteTrack Face Tracker](#64-trackerpy--bytetrack-face-tracker)
   - 6.5 [database.py — SQLite Database Module](#65-databasepy--sqlite-database-module)
   - 6.6 [seed_db.py — Database Seeding Script](#66-seed_dbpy--database-seeding-script)
   - 6.7 [generate_synthetic_data.py — Synthetic Data Generator](#67-generate_synthetic_datapy--synthetic-data-generator)
7. [Database Schema](#7-database-schema)
8. [REST API Endpoints](#8-rest-api-endpoints)
9. [Frontend — React Dashboard](#9-frontend--react-dashboard)
   - 9.1 [App.jsx — Root Layout & Routing](#91-appjsx--root-layout--routing)
   - 9.2 [DashboardPage.jsx — Main Dashboard](#92-dashboardpagejsx--main-dashboard)
   - 9.3 [StudentsPage.jsx — Student Management](#93-studentspagejsx--student-management)
   - 9.4 [ReportsPage.jsx — Attendance Reports](#94-reportspagejsx--attendance-reports)
   - 9.5 [AlertsPage.jsx — Unknown Faces & Alerts](#95-alertspagejsx--unknown-faces--alerts)
   - 9.6 [Frontend Styling & Design System](#96-frontend-styling--design-system)
10. [AI / ML Pipeline](#10-ai--ml-pipeline)
    - 10.1 [Face Detection — SCRFD](#101-face-detection--scrfd)
    - 10.2 [Face Recognition — ArcFace](#102-face-recognition--arcface)
    - 10.3 [Face Tracking — ByteTrack](#103-face-tracking--bytetrack)
    - 10.4 [Face Matching — FAISS](#104-face-matching--faiss)
11. [Configuration & Parameters](#11-configuration--parameters)
12. [Data Flow & Processing Pipeline](#12-data-flow--processing-pipeline)
13. [Setup & Installation Guide](#13-setup--installation-guide)
14. [Running the Application](#14-running-the-application)
15. [Deployment Notes](#15-deployment-notes)
16. [Known Limitations & Future Enhancements](#16-known-limitations--future-enhancements)

---

## 1. Project Overview

**SmartAttend** is a real-time AI-powered Face Recognition Attendance System that uses computer vision and deep learning to automatically detect, recognize, and track individuals via a webcam feed. The system logs attendance data to a local SQLite database and provides a modern, interactive React-based dashboard for monitoring, analytics, and reporting.

**Key Capabilities:**
- **Real-time face detection** from webcam video using SCRFD (InsightFace)
- **Face recognition** using 512-D ArcFace embeddings matched via FAISS
- **Multi-object face tracking** with a custom ByteTrack implementation using Kalman filters
- **Automated attendance logging** — marks students as Present/Late/Absent based on arrival time
- **Unknown face capture** — saves snapshots of unrecognized persons for security review
- **MJPEG live video streaming** to a web-based dashboard
- **Rich analytics dashboard** with charts, KPIs, insights, and exportable CSV reports

---

## 2. Objectives

| # | Objective |
|---|-----------|
| 1 | Automate attendance tracking using AI-based face recognition |
| 2 | Eliminate manual roll-call and paper-based attendance systems |
| 3 | Provide real-time monitoring through a live camera feed |
| 4 | Track multiple faces simultaneously using multi-object tracking |
| 5 | Detect and log unknown/unregistered individuals for security |
| 6 | Generate attendance analytics, insights, and exportable reports |
| 7 | Support both standalone (OpenCV window) and web-based (Flask + React) modes |
| 8 | Optimize for CPU performance (no GPU/CUDA required) |

---

## 3. System Architecture

### High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                               │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              React Frontend (Vite + React 19)             │    │
│  │  ┌────────────┬────────────┬──────────┬────────────────┐  │    │
│  │  │ Dashboard  │  Students  │ Reports  │     Alerts     │  │    │
│  │  └────────────┴────────────┴──────────┴────────────────┘  │    │
│  │       Chart.js  │  Axios  │  Framer Motion  │  Lucide    │    │
│  └──────────────────────────────────────────────────────────┘    │
│                              │ HTTP (port 5173 → 5000)           │
└──────────────────────────────┼───────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────┐
│                        API LAYER                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                 Flask REST API (port 5000)                 │    │
│  │  /api/analytics  /api/students  /api/attendance           │    │
│  │  /api/status     /api/unknown_faces  /video_feed (MJPEG)  │    │
│  └──────────────────────────────────────────────────────────┘    │
│                              │                                    │
└──────────────────────────────┼───────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────┐
│                     PROCESSING LAYER                              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────────┐   │
│  │  SCRFD   │→│ ArcFace  │→│ ByteTrack │→│ FAISS Matcher │   │
│  │Detection │  │Embedding │  │ Tracker   │  │ (Cosine Sim)  │   │
│  │(320×320) │  │ (512-D)  │  │(Kalman+IoU│  │ (IndexFlatIP) │   │
│  └──────────┘  └──────────┘  └───────────┘  └───────────────┘   │
│                              │                                    │
└──────────────────────────────┼───────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────┐
│                       DATA LAYER                                  │
│  ┌────────────────┐  ┌────────────────────────────────────────┐  │
│  │ OpenCV Camera  │  │   SQLite Database (attendance.db)      │  │
│  │  (640×480)     │  │   Tables: students, attendance,        │  │
│  │  VideoCapture  │  │   visits, unknown_faces                │  │
│  └────────────────┘  └────────────────────────────────────────┘  │
│                                                                    │
│  ┌────────────────┐  ┌────────────────────────────────────────┐  │
│  │ FAISS Index    │  │ Face Dataset (dataset/<name>/*.jpg)    │  │
│  │ (face_index    │  │ 11 registered persons with face        │  │
│  │  .faiss)       │  │ images for embedding extraction        │  │
│  └────────────────┘  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
Webcam → Frame Capture → SCRFD Detection → ArcFace Embedding →
FAISS Matching → ByteTrack Tracking → SQLite Logging →
Flask API → React Dashboard
```

---

## 4. Technology Stack

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Core programming language |
| **Flask** | ≥ 3.0.0 | REST API web server |
| **Flask-CORS** | — | Cross-Origin Resource Sharing for React frontend |
| **OpenCV** (cv2) | ≥ 4.8.0 | Video capture, frame processing, MJPEG encoding |
| **InsightFace** | ≥ 0.7.3 | SCRFD face detection + ArcFace face recognition |
| **ONNX Runtime** | ≥ 1.16.0 | AI model inference engine (CPU) |
| **FAISS** (faiss-cpu) | ≥ 1.7.4 | Vector similarity search for face matching |
| **NumPy** | ≥ 1.24.0 | Numerical computation for embeddings |
| **SciPy** | ≥ 1.10.0 | Hungarian algorithm for track association |
| **scikit-learn** | ≥ 1.3.0 | Supporting ML utilities |
| **Pillow** | ≥ 10.0.0 | Image processing |
| **SQLite3** | Built-in | Local database for attendance/visit logs |

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 19.2.4 | UI component framework |
| **Vite** | 8.0.4 | Build tool & dev server |
| **React Router DOM** | 7.14.1 | Client-side routing |
| **Axios** | 1.15.0 | HTTP client for API communication |
| **Chart.js** | 4.5.1 | Data visualization library |
| **react-chartjs-2** | 5.3.1 | React wrapper for Chart.js |
| **Framer Motion** | 12.38.0 | Animation library |
| **Lucide React** | 1.8.0 | Icon library |
| **Inter** (Google Font) | — | Typography |

### AI Models

| Model | Architecture | Purpose |
|-------|-------------|---------|
| **buffalo_s** (InsightFace) | SCRFD + ArcFace | Small, fast model optimized for CPU. Includes both face detection and recognition |
| **SCRFD** | Single-stage anchor-free detector | Face bounding box detection + 5-point facial landmarks |
| **ArcFace** | Deep CNN with Additive Angular Margin Loss | 512-dimensional face feature embedding extraction |

### Infrastructure

| Component | Technology |
|-----------|-----------|
| **Database** | SQLite (file-based, thread-safe) |
| **Vector Index** | FAISS IndexFlatIP (flat inner product for cosine similarity) |
| **Video Streaming** | MJPEG over HTTP multipart |
| **Deployment** | Local (webcam-based), no cloud dependency |
| **OS Compatibility** | Windows, macOS (M1/M2), Linux |

---

## 5. Project Structure

```
Miniproject/
└── face_surveillance_demo/
    ├── app.py                        # Flask API server (main entry point)
    ├── recognize.py                  # Real-time face recognition engine
    ├── register.py                   # Face registration & FAISS index builder
    ├── tracker.py                    # ByteTrack multi-object face tracker
    ├── database.py                   # SQLite database operations
    ├── seed_db.py                    # Database seeding script (10 students)
    ├── generate_synthetic_data.py    # 2-month synthetic attendance data generator
    ├── requirements.txt              # Python dependencies
    ├── README.md                     # Quick-start README
    │
    ├── attendance.db                 # SQLite database file (auto-generated)
    ├── logs.db                       # Legacy log database
    │
    ├── dataset/                      # Face image dataset
    │   ├── srikanth/                 # Real user face images
    │   │   ├── 1.jpg
    │   │   └── 2.jpg
    │   ├── John Doe/                 # Synthetic placeholder directories
    │   ├── Jane Smith/
    │   ├── Alice Johnson/
    │   ├── Bob Brown/
    │   ├── Charlie Davis/
    │   ├── Diana Miller/
    │   ├── Ethan Wilson/
    │   ├── Fiona Moore/
    │   ├── George Taylor/
    │   └── Hannah Anderson/
    │
    ├── embeddings/                   # FAISS index & labels
    │   ├── face_index.faiss          # Pre-built FAISS index (~12KB)
    │   └── labels.pkl                # Pickle file mapping index → names
    │
    ├── static/                       # Static assets for Flask dashboard
    │   ├── style.css                 # Premium dark theme CSS (750 lines)
    │   └── unknown_faces/            # Auto-saved unknown face images
    │       └── unknown_<timestamp>_<trackid>.jpg
    │
    ├── templates/                    # Flask HTML template
    │   └── index.html                # Standalone surveillance dashboard (322 lines)
    │
    └── frontend/                     # React frontend (Vite)
        ├── index.html                # HTML entry point
        ├── package.json              # NPM dependencies
        ├── vite.config.js            # Vite configuration
        ├── eslint.config.js          # ESLint configuration
        ├── dist/                     # Production build output
        ├── node_modules/             # NPM packages
        ├── public/                   # Public static assets
        └── src/
            ├── main.jsx              # React entry point
            ├── App.jsx               # Root component with sidebar & routing
            ├── index.css             # Global CSS design system (633 lines)
            ├── assets/               # Static assets (images, etc.)
            └── pages/
                ├── DashboardPage.jsx  # Main dashboard with KPIs, charts, live feed
                ├── StudentsPage.jsx   # Student management with search
                ├── ReportsPage.jsx    # Attendance reports with CSV export
                └── AlertsPage.jsx     # Unknown face alerts & gallery
```

---

## 6. Backend — Python Modules

### 6.1 `app.py` — Flask API Server

**File Size:** 3,874 bytes | **Lines:** 127

**Purpose:** Main entry point. Creates the Flask web server, initializes the face recognition engine in a background thread, and exposes REST API endpoints for the React frontend.

**Key Components:**

| Component | Description |
|-----------|-------------|
| `start_engine()` | Initializes FaceRecognizer, starts webcam, runs recognition loop in daemon thread |
| `generate_frames()` | Python generator yielding MJPEG frames for video streaming |
| `CORS(app)` | Enables cross-origin requests from React dev server (port 5173) |

**API Routes Defined:**
- `GET /api/status` — Engine status, uptime, FPS
- `GET /video_feed` — MJPEG live video stream
- `GET /api/analytics` — Dashboard analytics (stats + charts + insights)
- `GET /api/students` — All enrolled students
- `GET /api/unknown_faces` — Unknown face detection records
- `GET /api/attendance` — Full attendance log with student details

**Startup Flow:**
1. `init_db()` — Creates database tables
2. `start_engine()` — Loads models, starts camera, begins recognition loop
3. `app.run()` — Flask server on `0.0.0.0:5000` (threaded mode)

---

### 6.2 `recognize.py` — Face Recognition Engine

**File Size:** 13,556 bytes | **Lines:** 374

**Purpose:** Core real-time face recognition engine. Handles video capture, face detection, embedding extraction, FAISS matching, ByteTrack tracking, annotation rendering, and SQLite logging.

**Class: `FaceRecognizer`**

| Method | Description |
|--------|-------------|
| `__init__()` | Initialize all instance variables, frame lock, caches |
| `initialize()` | Load InsightFace model (buffalo_s), FAISS index, ByteTracker |
| `match_face(embedding)` | Normalize embedding, query FAISS index, return (name, score) |
| `should_log(track_id)` | Cooldown-based check (10s) to prevent redundant visit logging |
| `process_frame(frame)` | Full pipeline: detect → embed → match → track → annotate → log |
| `start_camera(index)` | Open webcam with retry logic (5 attempts), set resolution |
| `run_loop()` | Main loop: capture frame → process → update FPS → store for streaming |
| `get_frame_jpeg()` | Thread-safe JPEG encoding for MJPEG streaming (quality=70) |
| `get_uptime()` | Calculate engine uptime in seconds |
| `stop()` | Gracefully stop the recognition loop |

**Function: `run_standalone()`**
- Runs the recognizer with an OpenCV window (no Flask, no React)
- Press `q` to quit

**CPU Optimizations Applied:**

| Optimization | Detail |
|-------------|--------|
| Model | `buffalo_s` (small/fast) instead of `buffalo_l` (large) |
| Detection Size | 320×320 pixels (4× fewer pixels than 640×640) |
| Frame Skipping | Process face detection every 4th frame (`PROCESS_EVERY_N = 4`) |
| Camera Resolution | 640×480 instead of 1280×720 |
| Unknown Face Saving | Once per track_id (not every frame) |
| Visit Logging | 10-second cooldown per track_id |
| JPEG Quality | 70% (reduced for faster streaming) |

**Frame Processing Pipeline (`process_frame`):**

```
Frame → (every Nth frame) SCRFD Detection → ArcFace Embedding → FAISS Match
    ↓ (every frame)
ByteTrack Update → Label Mapping → Draw Bounding Boxes → Log to SQLite
    ↓
Return annotated frame
```

---

### 6.3 `register.py` — Face Registration Module

**File Size:** 4,754 bytes | **Lines:** 137

**Purpose:** Scans the `dataset/` directory, detects faces in each image, extracts 512-D ArcFace embeddings, normalizes them for cosine similarity, and builds a FAISS index for fast matching.

**Function: `register_faces()`**

**Registration Process:**
1. Initialize InsightFace with `buffalo_s` model (same as recognition engine)
2. Scan `dataset/` for person subdirectories
3. For each person's directory:
   - Read each image file (JPG, PNG, BMP, WebP)
   - Detect faces using SCRFD
   - Select the largest face (most prominent)
   - Extract 512-D ArcFace embedding
   - L2-normalize the embedding
   - Store embedding + label (person name)
4. Build FAISS `IndexFlatIP` index (inner product on normalized vectors = cosine similarity)
5. Save index to `embeddings/face_index.faiss`
6. Save labels to `embeddings/labels.pkl` (Python pickle)

**Supported Image Formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

**Output Files:**
- `embeddings/face_index.faiss` — Binary FAISS index file
- `embeddings/labels.pkl` — Python pickle file mapping index positions to person names

---

### 6.4 `tracker.py` — ByteTrack Face Tracker

**File Size:** 8,125 bytes | **Lines:** 221

**Purpose:** Implements a ByteTrack-style multi-object tracker for stable face tracking across video frames. Assigns persistent Track IDs to reduce redundant recognition calls.

**Classes:**

#### `KalmanFilter2D`
- **State Vector:** `[cx, cy, w, h, vx, vy, vw, vh]` (center x/y, width, height, velocities)
- **Covariance Matrix:** 8×8 identity × 10
- **Process Noise (Q):** Identity × 1.0
- **Measurement Noise (R):** Identity × 5.0
- **Methods:** `predict()`, `update(bbox)`, `get_bbox()`

#### `Track`
- Represents a single tracked face
- Properties: `track_id`, `kalman`, `bbox`, `confidence`, `label`, `embedding`, `age`, `hits`, `time_since_update`, `is_confirmed`
- A track is **confirmed** after 3 consecutive hits (`min_hits`)
- Track IDs are globally auto-incremented

#### `ByteTracker`
- **Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_age` | 30 | Maximum frames a track can survive without update |
| `min_hits` | 3 | Minimum consecutive hits to confirm a track |
| `iou_threshold` | 0.3 | Minimum IoU for successful match |
| `high_thresh` | 0.5 | Confidence threshold for high-confidence detections |
| `low_thresh` | 0.1 | Confidence threshold for low-confidence detections |

- **Three-Stage Association Algorithm:**
  1. **Stage 1:** Match high-confidence detections to confirmed tracks (IoU + Hungarian)
  2. **Stage 2:** Match low-confidence detections to remaining unmatched confirmed tracks
  3. **Stage 3:** Match remaining high-confidence detections to unconfirmed tracks

- **Matching:** Uses IoU-based cost matrix solved by the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)

#### `iou(bbox1, bbox2)`
- Utility function to compute Intersection over Union between two bounding boxes (`[x1, y1, x2, y2]` format)

---

### 6.5 `database.py` — SQLite Database Module

**File Size:** 9,571 bytes | **Lines:** 285

**Purpose:** Handles all database operations — table creation, visit logging, attendance marking, analytics queries, and student/insight data retrieval.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `get_connection()` | Thread-safe SQLite connection with `Row` factory |
| `init_db()` | Create all 4 tables if they don't exist |
| `log_visit_and_attendance(...)` | Log a visit record; auto-mark attendance for known students |
| `log_unknown_face(image_path)` | Save unknown face record |
| `get_analytics()` | Comprehensive dashboard data (stats, charts, frequencies) |
| `get_student_insights()` | Detect students with <75% attendance |
| `get_all_students()` | Fetch complete student list |
| `get_entry_exit_logs()` | Aggregated first/last seen + total visits per person today |

**Attendance Logic in `log_visit_and_attendance()`:**
1. Always log a visit record
2. If the person is known (not "Unknown"):
   - Look up `student_id` from `students` table
   - Check if attendance already marked today (`UNIQUE(student_id, date)`)
   - If not marked:
     - If arrival time ≤ `time_window_end` (10:00 AM) → `Present`
     - If arrival time > `time_window_end` → `Late`
   - Insert attendance record

**Configurable Time Window:**
- `time_window_start`: `08:30` (start of attendance window)
- `time_window_end`: `10:00` (cutoff for "Present" status; after this → "Late")

---

### 6.6 `seed_db.py` — Database Seeding Script

**File Size:** 2,022 bytes | **Lines:** 58

**Purpose:** Populates the `students` table with 10 synthetic student records, creates dataset directories with placeholder files for face image uploads.

**Seeded Students:**

| Student ID | Name | Department | Year |
|-----------|------|------------|------|
| STU001 | John Doe | Computer Science | Year 1 |
| STU002 | Jane Smith | Electrical Eng | Year 2 |
| STU003 | Alice Johnson | Mechanical Eng | Year 3 |
| STU004 | Bob Brown | Civil Eng | Year 4 |
| STU005 | Charlie Davis | Computer Science | Year 1 |
| STU006 | Diana Miller | Electrical Eng | Year 2 |
| STU007 | Ethan Wilson | Mechanical Eng | Year 3 |
| STU008 | Fiona Moore | Civil Eng | Year 4 |
| STU009 | George Taylor | Computer Science | Year 1 |
| STU010 | Hannah Anderson | Electrical Eng | Year 2 |

---

### 6.7 `generate_synthetic_data.py` — Synthetic Data Generator

**File Size:** 5,491 bytes | **Lines:** 163

**Purpose:** Generates 2 months (~60 days) of realistic synthetic attendance and visit data for all 10 students. Skips weekends, varies attendance probability per student, and includes unknown visitor entries.

**Student Attendance Probabilities:**

| Student | Attendance Rate | Profile |
|---------|----------------|---------|
| John Doe (STU001) | ~98% | Perfect attendance |
| Bob Brown (STU004) | ~55% | Chronic absentee |
| Fiona Moore (STU008) | ~65% | Low attendance |
| Hannah Anderson (STU010) | ~72% | Borderline |
| Others | ~85-92% | Regular attendees |

**Data Generation Logic:**
- **Working days only:** Monday–Friday
- **Late probability:** 12% chance per student per day
  - Late: arrival 10:00–11:59 AM
  - On time: arrival 8:00–9:59 AM
- **Visits per day per student:** 1–4 (random)
- **Unknown visits per day:** 2–6 (random IDs >10000, low confidence 0.15–0.40)
- **Preserves today's real data:** Deletes only historical synthetic data

---

## 7. Database Schema

**Database File:** `attendance.db` (SQLite)

### Table: `students`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `student_id` | TEXT | PRIMARY KEY | Unique student identifier (e.g., STU001) |
| `name` | TEXT | NOT NULL | Full name |
| `department` | TEXT | — | Academic department |
| `year` | TEXT | — | Academic year |
| `image_path` | TEXT | — | Path to face image dataset directory |
| `registered_at` | TEXT | — | Registration timestamp |

### Table: `attendance`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Auto-generated ID |
| `student_id` | TEXT | FOREIGN KEY → students | Student reference |
| `date` | TEXT | NOT NULL, UNIQUE(student_id, date) | Date (YYYY-MM-DD) |
| `time` | TEXT | NOT NULL | Arrival time (HH:MM:SS) |
| `status` | TEXT | DEFAULT 'Present' | Present / Late |

### Table: `visits`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Auto-generated ID |
| `person_name` | TEXT | NOT NULL | Detected person name or "Unknown" |
| `student_id` | TEXT | — | Student ID if matched |
| `confidence` | REAL | — | FAISS similarity score |
| `track_id` | INTEGER | — | ByteTrack assigned ID |
| `date` | TEXT | NOT NULL | Date (YYYY-MM-DD) |
| `time` | TEXT | NOT NULL | Detection time (HH:MM:SS) |
| `timestamp` | TEXT | NOT NULL | Full timestamp (YYYY-MM-DD HH:MM:SS) |
| `status` | TEXT | DEFAULT 'Recognized' | Recognized / Unknown |

### Table: `unknown_faces`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Auto-generated ID |
| `image_path` | TEXT | NOT NULL | Relative path to saved face crop |
| `date` | TEXT | NOT NULL | Capture date (YYYY-MM-DD) |
| `time` | TEXT | NOT NULL | Capture time (HH:MM:SS) |
| `timestamp` | TEXT | NOT NULL | Full timestamp |

### Entity Relationship

```
students (1) ←── (N) attendance
    │
    └───── student_id ─────→ visits.student_id (optional)

unknown_faces (standalone — no foreign keys)
```

---

## 8. REST API Endpoints

**Base URL:** `http://localhost:5000`

### `GET /api/status`
Returns engine runtime status.

**Response:**
```json
{
  "status": "running",
  "uptime": 1234.5,
  "fps": 15.2
}
```

### `GET /video_feed`
Live MJPEG video stream of the annotated camera feed.

**Content-Type:** `multipart/x-mixed-replace; boundary=frame`

### `GET /api/analytics`
Comprehensive analytics data for the dashboard.

**Response:**
```json
{
  "stats": {
    "totalVisitsToday": 45,
    "presentToday": 8,
    "lateToday": 1,
    "absentToday": 1,
    "totalStudents": 10
  },
  "charts": {
    "facesPerHour": [
      { "hour": "08", "count": 12 },
      { "hour": "09", "count": 18 }
    ],
    "knownVsUnknown": { "known": 38, "unknown": 7 },
    "mostFrequent": [
      { "person_name": "John Doe", "count": 15 }
    ]
  },
  "entryExitLogs": [
    {
      "person_name": "John Doe",
      "first_seen": "08:15:30",
      "last_seen": "16:45:12",
      "total_visits": 8
    }
  ],
  "insights": [
    {
      "student_id": "STU004",
      "name": "Bob Brown",
      "attendance_percentage": 55.0,
      "insight": "Low Attendance"
    }
  ]
}
```

### `GET /api/students`
Returns all enrolled students.

**Response:**
```json
[
  {
    "student_id": "STU001",
    "name": "John Doe",
    "department": "Computer Science",
    "year": "Year 1",
    "image_path": "dataset/John Doe",
    "registered_at": "2026-04-15 10:00:00"
  }
]
```

### `GET /api/attendance`
Returns full attendance log with student details (JOINed with students table).

**Response:**
```json
[
  {
    "id": 1,
    "date": "2026-04-16",
    "time": "08:30:15",
    "status": "Present",
    "name": "John Doe",
    "student_id": "STU001",
    "department": "Computer Science"
  }
]
```

### `GET /api/unknown_faces`
Returns all unknown face detection records.

**Response:**
```json
[
  {
    "id": 1,
    "image_path": "/static/unknown_faces/unknown_1713000000_42.jpg",
    "date": "2026-04-16",
    "time": "14:30:00",
    "timestamp": "2026-04-16 14:30:00"
  }
]
```

---

## 9. Frontend — React Dashboard

### 9.1 `App.jsx` — Root Layout & Routing

**Lines:** 67

**Layout Structure:**
- **Sidebar** (280px fixed width) with brand icon, navigation links, system status
- **Main Content Area** with route-based page rendering

**Navigation Routes:**

| Path | Component | Description |
|------|-----------|-------------|
| `/` | `DashboardPage` | Main dashboard with live feed, KPIs, and charts |
| `/students` | `StudentsPage` | Student enrolled list with search |
| `/reports` | `ReportsPage` | Attendance reports with filtering & export |
| `/alerts` | `AlertsPage` | Unknown face alerts gallery |

**Sidebar Sections:**
- **Main:** Dashboard, Students
- **Data:** Reports, Alerts
- **Footer:** System Online status badge with pulse animation

---

### 9.2 `DashboardPage.jsx` — Main Dashboard

**Lines:** 305

**Features:**
1. **Header Bar:** Page title, uptime badge, FPS indicator (Wifi/WifiOff icon)
2. **KPI Cards (4 stat cards):**
   - 👁 Total Visits Today (purple)
   - ✓ Present Count / Total Students (green)
   - 🕐 Late Count (amber)
   - ✗ Absent Count (red)
3. **Attendance Rate Bar:** Progress bar showing today's attendance percentage (green ≥75%, red <75%)
4. **Live Camera Feed:** MJPEG stream from `/video_feed` with LIVE/OFFLINE badge
5. **Entry/Exit Log Table:** Person name, first seen, last seen, total visits
6. **3-Column Chart Grid:**
   - Line Chart: Detections per Hour (Chart.js)
   - Bar Chart: Most Frequent Visitors (horizontal)
   - Doughnut Chart: Known vs Unknown ratio
7. **Low Attendance Insights:** Alerts for students with <75% attendance

**Data Polling:** Every 3 seconds via `setInterval` calling `/api/analytics` and `/api/status`

---

### 9.3 `StudentsPage.jsx` — Student Management

**Lines:** 89

**Features:**
- **Search Bar:** Filter students by name, ID, or department
- **Student Table:** Avatar (color-coded initials), Name, Student ID, Department, Year, Status badge
- **Info Box:** Instructions for adding face images and running registration
- **Avatar Colors:** 10-color palette cycled per student

---

### 9.4 `ReportsPage.jsx` — Attendance Reports

**Lines:** 304

**Features:**
1. **Header:** Record counts, CSV export button
2. **Period Tabs:** Daily / Weekly / Monthly / Semester views
3. **Status Filter:** All / Present / Late buttons with counts
4. **Per-Student Summary Table:**
   - Name, Present days, Late days, Absent days
   - Attendance percentage with progress bar
   - Status badges: Good (≥85%), Warning (≧75%), Critical (<75%)
5. **Stacked Bar Chart:** Per-student attendance breakdown (Present/Late/Absent)
6. **Grouped Records:** Expandable sections grouped by selected period

**CSV Export:** Downloads filtered attendance data as a `.csv` file with columns: Date, Time, Student ID, Name, Department, Status

**Helper Functions:**
- `groupBy(arr, fn)` — Groups array items by a key function
- `getWeek(dateStr)` — Calculates ISO week number
- `getMonthLabel(dateStr)` — Returns "Month Year" label

---

### 9.5 `AlertsPage.jsx` — Unknown Faces & Alerts

**Lines:** 65

**Features:**
- **Header:** Total captured count
- **Face Gallery:** Grid of unknown face snapshots with hover overlay showing date/time
- **Auto-refresh:** Polls `/api/unknown_faces` every 5 seconds
- **Empty State:** "All clear" shield icon when no unknown faces detected
- **Image Source:** Loaded from `http://localhost:5000/static/unknown_faces/`

---

### 9.6 Frontend Styling & Design System

**File:** `src/index.css` | **Lines:** 633

**Design System — Light Professional Theme**

**Color Tokens:**

| Token | Value | Usage |
|-------|-------|-------|
| `--accent` | `#4f46e5` (Indigo) | Primary brand color |
| `--success` | `#059669` (Emerald) | Present, good status |
| `--warning` | `#d97706` (Amber) | Late, warning status |
| `--danger` | `#dc2626` (Red) | Absent, critical alerts |
| `--info` | `#2563eb` (Blue) | Informational elements |
| `--bg-body` | `#f0f2f5` | Page background |
| `--bg-card` | `#ffffff` | Card backgrounds |
| `--text-primary` | `#111827` | Main text |
| `--text-muted` | `#9ca3af` | Secondary/muted text |

**Design Features:**
- **Typography:** Inter font (300–800 weights)
- **Border Radius:** sm(8px), md(12px), lg(16px), xl(20px), full(9999px)
- **Shadows:** 5 levels (xs, sm, md, lg, accent)
- **Animations:** Pulse, spin, log slide-in
- **Responsive Breakpoints:** 1100px, 768px, 480px

**Component Styles Included:**
- Sidebar (fixed, sticky, with brand icon)
- Cards (with hover shadow lift)
- Stat KPI cards (with decorative circle accent)
- Data tables (with hover rows, rounded headers)
- Badges (present, late, absent, active, unknown)
- Progress bars (with gradient fills)
- Video feed wrapper (16:9 aspect ratio)
- Unknown faces gallery (grid with hover overlay)
- Buttons (primary gradient, outline)
- Search input (with icon, focus ring)
- Empty states & error screens
- Info box (blue tint)

---

### Legacy Flask Dashboard

**File:** `templates/index.html` | **Lines:** 322

A standalone HTML dashboard (no React) served by Flask at `http://localhost:5000`. Features a **premium dark theme** with glassmorphism effects, including:
- Live MJPEG camera feed
- Stats grid (Total Detections, Unique Persons, Unknown Faces, Uptime)
- Recognition logs with real-time updates
- Activity bar chart (last 30 minutes)
- Auto-refresh every 2 seconds

**File:** `static/style.css` | **Lines:** 750

Dark theme CSS with:
- Animated background grid
- Ambient glow effects
- Glassmorphism cards
- JetBrains Mono monospace font for technical data
- Custom tooltips
- Scrollbar styling

---

## 10. AI / ML Pipeline

### 10.1 Face Detection — SCRFD

**Model:** SCRFD (Sample and Computation Redistribution for Efficient Face Detection)

| Parameter | Value |
|-----------|-------|
| Input Resolution | 320×320 pixels |
| Architecture | Single-stage, anchor-free |
| Model Pack | `buffalo_s` (InsightFace) |
| Execution Provider | CPUExecutionProvider (ONNX Runtime) |
| Output | Bounding boxes + 5-point landmarks + detection scores |

**Key Characteristics:**
- Part of the InsightFace `buffalo_s` model package
- Provides facial bounding boxes `[x1, y1, x2, y2]`
- Returns detection confidence scores
- Extracts 5-point facial landmarks (eyes, nose, mouth corners)
- Optimized for CPU with small input resolution (320×320)

---

### 10.2 Face Recognition — ArcFace

**Model:** ArcFace (Additive Angular Margin Loss for Deep Face Recognition)

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 512 |
| Model Pack | `buffalo_s` (InsightFace) |
| Loss Function | Additive Angular Margin (ArcFace) |
| Normalization | L2-normalized for cosine similarity |
| Output | 512-D float32 feature vector |

**Pipeline:**
1. Crop face region from input frame using detector output
2. Align face using 5-point landmarks
3. Resize to model input size
4. Forward pass through ArcFace CNN
5. Output 512-dimensional embedding vector
6. L2-normalize the embedding

---

### 10.3 Face Tracking — ByteTrack

**Algorithm:** ByteTrack (custom implementation)

| Feature | Detail |
|---------|--------|
| Motion Model | Kalman Filter (8-state: cx, cy, w, h + velocities) |
| Association | IoU-based cost matrix + Hungarian algorithm |
| Multi-stage | High + Low confidence detection matching |
| Track Management | Confirmation (≥3 hits), aging (max 30 frames) |
| ID Assignment | Globally auto-incremented integers |

**Why ByteTrack?**
- Handles low-confidence detections (recovers occluded faces)
- Stable Track IDs across frames → reduces redundant recognition
- Kalman filter smooths bounding box jitter
- Lightweight enough for real-time CPU processing

---

### 10.4 Face Matching — FAISS

**Library:** FAISS (Facebook AI Similarity Search)

| Parameter | Value |
|-----------|-------|
| Index Type | `IndexFlatIP` (Flat Inner Product) |
| Similarity Metric | Cosine Similarity (inner product on normalized vectors) |
| Dimension | 512 |
| k (top-k) | 1 |
| Threshold | 0.45 |

**Matching Process:**
1. Normalize query embedding (L2 norm)
2. Search FAISS index for top-1 nearest neighbor
3. Return match if score ≥ 0.45, else "Unknown"

---

## 11. Configuration & Parameters

### Recognition Engine (`recognize.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SIMILARITY_THRESHOLD` | 0.45 | Minimum cosine similarity for a match |
| `LOG_COOLDOWN` | 10 seconds | Time between logging the same track_id |
| `UNKNOWN_SAVE_COOLDOWN` | 60 seconds | Cooldown for saving unknown face crops |
| `PROCESS_EVERY_N` | 4 | Frame skipping (detect every 4th frame) |
| `CAMERA_WIDTH` | 640 | Webcam width resolution |
| `CAMERA_HEIGHT` | 480 | Webcam height resolution |
| `DET_SIZE` | (320, 320) | SCRFD detection input resolution |
| `MODEL_NAME` | `buffalo_s` | InsightFace model package name |

### ByteTracker (`tracker.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_age` | 30 | Max frames a track survives without updates |
| `min_hits` | 3 | Consecutive hits needed to confirm a track |
| `iou_threshold` | 0.3 | Minimum IoU for data association match |
| `high_thresh` | 0.5 | High-confidence detection threshold |
| `low_thresh` | 0.1 | Low-confidence detection threshold |

### Attendance Rules (`database.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `time_window_start` | 08:30 | Start of attendance acceptance window |
| `time_window_end` | 10:00 | Cutoff for "Present"; after this → "Late" |
| Attendance per day | 1 per student | `UNIQUE(student_id, date)` constraint |
| Insight threshold | 75% | Students below this are flagged |

### Flask Server (`app.py`)

| Parameter | Value |
|-----------|-------|
| Host | `0.0.0.0` |
| Port | `5000` |
| Debug | `False` |
| Threaded | `True` |
| MJPEG frame rate | ~30 FPS (33ms delay) |

### React Frontend

| Parameter | Value |
|-----------|-------|
| Dev Server Port | 5173 (Vite default) |
| API Base URL | `http://localhost:5000` |
| Dashboard poll interval | 3000ms |
| Alerts poll interval | 5000ms |

---

## 12. Data Flow & Processing Pipeline

### Complete Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Camera Capture                                          │
│ • OpenCV VideoCapture (index 0)                                 │
│ • Resolution: 640×480 @ 30 FPS                                  │
│ • Platform-aware: CAP_DSHOW on Windows                          │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Frame Processing (process_frame)                        │
│ • Frame counter incremented                                     │
│ • If frame_counter % 4 == 0 → run detection (expensive)         │
│ • Else → reuse cached detections                                │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Face Detection (SCRFD via InsightFace)                  │
│ • Input: raw BGR frame                                          │
│ • Output: list of Face objects with:                            │
│   - bbox [x1, y1, x2, y2]                                      │
│   - det_score (detection confidence)                            │
│   - embedding (512-D ArcFace vector) or None                    │
│   - landmarks (5-point facial landmarks)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Face Matching (FAISS)                                   │
│ • For each face with embedding:                                 │
│   - L2-normalize the embedding                                  │
│   - Search FAISS IndexFlatIP for top-1 match                    │
│   - If similarity ≥ 0.45 → return (name, score)                │
│   - Else → return ("Unknown", score)                            │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Multi-Object Tracking (ByteTrack)                       │
│ • Predict all existing track positions                           │
│ • Three-stage data association:                                  │
│   1. High-confidence → confirmed tracks                         │
│   2. Low-confidence → unmatched confirmed tracks                │
│   3. Remaining high → unconfirmed tracks                        │
│ • Create new tracks for unmatched detections                    │
│ • Remove tracks exceeding max_age                               │
│ • Return list of confirmed active tracks                        │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Annotation & Logging                                    │
│ • Draw bounding boxes (green=known, orange=unknown)             │
│ • Draw label text with confidence and track ID                  │
│ • If should_log(track_id) (10s cooldown):                       │
│   → log_visit_and_attendance() to SQLite                        │
│ • If unknown & first time for this track_id:                    │
│   → Save face crop to static/unknown_faces/                    │
│   → log_unknown_face() to SQLite                                │
│ • Overlay FPS and active track count                            │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: Frame Delivery                                          │
│ • Store processed frame in thread-safe buffer                   │
│ • MJPEG generator encodes frame as JPEG (quality=70)            │
│ • Yield to Flask Response stream → video_feed endpoint          │
│ • React <img> tag auto-renders the MJPEG stream                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. Setup & Installation Guide

### Prerequisites

- **Python** 3.10 or higher
- **Node.js** 18+ and **npm** (for React frontend)
- **Webcam** (built-in or USB)
- **OS**: Windows 10/11, macOS (M1/M2 supported), or Linux

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Miniproject/face_surveillance_demo
```

### Step 2: Create Python Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note:** First run downloads ~300MB of InsightFace SCRFD/ArcFace models automatically.

### Step 4: Seed the Database

```bash
python seed_db.py
```

This creates `attendance.db` with 10 student records and dataset directories.

### Step 5: Add Face Photos

Place 2-5 clear face photos per person in:
```
dataset/<person_name>/1.jpg
dataset/<person_name>/2.jpg
```

**Tips for best results:**
- Use well-lit, front-facing photos
- Avoid sunglasses or heavy occlusion
- Different angles and expressions improve recognition
- Minimum face size: ~100×100 pixels

### Step 6: Register Faces (Build FAISS Index)

```bash
python register.py
```

This scans all images in `dataset/`, extracts 512-D embeddings, and creates `embeddings/face_index.faiss`.

### Step 7: (Optional) Generate Synthetic Data

```bash
python generate_synthetic_data.py
```

Generates 2 months of attendance/visit data for analytics and reports.

### Step 8: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## 14. Running the Application

### Mode 1: Full System (Flask API + React Dashboard)

**Terminal 1 — Start Backend:**
```bash
python app.py
```
Output:
```
  🎯 Face Attendance System API
  Running on: http://localhost:5000
```

**Terminal 2 — Start Frontend:**
```bash
cd frontend
npm run dev
```
Output:
```
  VITE v8.0.4  ready in 300 ms
  ➜  Local:   http://localhost:5173/
```

Open `http://localhost:5173` in your browser.

### Mode 2: Standalone (OpenCV Window Only)

```bash
python recognize.py
```
- Opens an OpenCV window with annotated video feed
- Press `q` to quit
- No web server, no dashboard

### Mode 3: Legacy Flask Dashboard

```bash
python app.py
```
Open `http://localhost:5000` for the dark-themed HTML dashboard (served by Flask).

---

## 15. Deployment Notes

### Mac M1/M2 Notes
- Uses `CPUExecutionProvider` for ONNX Runtime (no CUDA)
- `ctx_id=-1` ensures CPU mode in InsightFace
- Native ARM performance — no Rosetta required

### Windows Notes
- Uses `cv2.CAP_DSHOW` (DirectShow) for camera access
- Camera retry logic handles driver initialization delays

### Performance Benchmarks (Approximate)

| Config | FPS (CPU) |
|--------|-----------|
| buffalo_s, 320×320, skip=4 | ~15-20 FPS |
| buffalo_s, 320×320, skip=2 | ~10-15 FPS |
| buffalo_l, 640×640, skip=1 | ~3-5 FPS |

### Production Considerations
- Replace SQLite with PostgreSQL for concurrent access
- Add authentication (JWT) to API endpoints
- Configure CORS origins (currently allows all)
- Use Gunicorn/uWSGI instead of Flask dev server
- Consider GPU acceleration for higher FPS (CUDAExecutionProvider)

---

## 16. Known Limitations & Future Enhancements

### Current Limitations

| # | Limitation |
|---|-----------|
| 1 | Single camera input only (no multi-camera support) |
| 2 | SQLite is not ideal for high-concurrency production use |
| 3 | No user authentication on API endpoints |
| 4 | FAISS index must be rebuilt when new students are added |
| 5 | Frame skipping can miss fast-moving faces |
| 6 | Similarity threshold (0.45) is fixed — no adaptive thresholding |
| 7 | No GPU acceleration — CPU only |
| 8 | No face anti-spoofing (vulnerable to photo/video attacks) |

### Potential Enhancements

| # | Enhancement |
|---|------------|
| 1 | Multi-camera support with camera selection UI |
| 2 | PostgreSQL or MongoDB for scalable data storage |
| 3 | JWT/OAuth2 authentication on API endpoints |
| 4 | Incremental FAISS index updates (add faces without full rebuild) |
| 5 | Face anti-spoofing / liveness detection |
| 6 | Email/SMS notifications for unknown face alerts |
| 7 | GPU acceleration with CUDA for real-time 30+ FPS |
| 8 | Mobile app for attendance check-in |
| 9 | Cloud deployment (AWS/GCP/Azure) with S3 for images |
| 10 | Active learning — auto-suggest registration for frequently seen unknowns |
| 11 | Integration with institutional ERP/LMS systems |
| 12 | Dark/Light theme toggle on the React dashboard |

---

## 📝 File Summary Table

| File | Size | Lines | Purpose |
|------|------|-------|---------|
| `app.py` | 3.8 KB | 127 | Flask REST API server |
| `recognize.py` | 13.6 KB | 374 | Face recognition engine |
| `register.py` | 4.8 KB | 137 | FAISS index builder |
| `tracker.py` | 8.1 KB | 221 | ByteTrack face tracker |
| `database.py` | 9.6 KB | 285 | SQLite operations |
| `seed_db.py` | 2.0 KB | 58 | Database seeder |
| `generate_synthetic_data.py` | 5.5 KB | 163 | Synthetic data generator |
| `requirements.txt` | 162 B | 10 | Python dependencies |
| `templates/index.html` | 14.6 KB | 322 | Flask HTML dashboard |
| `static/style.css` | 19.5 KB | 750 | Dark theme CSS |
| `frontend/src/App.jsx` | 2.4 KB | 67 | React root + routing |
| `frontend/src/main.jsx` | 235 B | 11 | React entry point |
| `frontend/src/index.css` | 22.1 KB | 633 | Light theme CSS |
| `frontend/src/pages/DashboardPage.jsx` | 11.8 KB | 305 | Dashboard page |
| `frontend/src/pages/StudentsPage.jsx` | 3.3 KB | 89 | Students page |
| `frontend/src/pages/ReportsPage.jsx` | 12.2 KB | 304 | Reports page |
| `frontend/src/pages/AlertsPage.jsx` | 2.0 KB | 65 | Alerts page |
| `frontend/package.json` | 785 B | 34 | NPM dependencies |

---

## 🔗 Dependencies (requirements.txt)

```
opencv-python>=4.8.0
flask>=3.0.0
insightface>=0.7.3
onnxruntime>=1.16.0
faiss-cpu>=1.7.4
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
Pillow>=10.0.0
```

**Additional (auto-installed by Flask):** `flask-cors`

**Frontend (package.json):**
- `react`, `react-dom` (v19.2.4)
- `react-router-dom` (v7.14.1)
- `axios` (v1.15.0)
- `chart.js` (v4.5.1), `react-chartjs-2` (v5.3.1)
- `framer-motion` (v12.38.0)
- `lucide-react` (v1.8.0)
- `vite` (v8.0.4), `@vitejs/plugin-react` (v6.0.1)

---

*Last Updated: April 16, 2026*
*Project: SmartAttend — AI-Powered Face Recognition Attendance System*

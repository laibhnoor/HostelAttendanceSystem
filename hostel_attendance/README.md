# Hostel Attendance System

## Overview
This project is a real-time AI-powered hostel attendance system built with Python, OpenCV, and DeepFace. It enrolls each student by storing face embeddings rather than raw images for privacy and performance. A web interface captures camera snapshots, performs matching, and displays live attendance alerts.

## Architecture
The system operates in two phases: enrollment and recognition. In enrollment, multiple face embeddings are captured and saved per student to improve match reliability. In recognition, Haar Cascade detects faces, DeepFace generates embeddings, and cosine similarity identifies the best match before marking attendance with live alerts.

## Setup
1. Clone the repository.
2. Create and activate a Python 3.10+ virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Copy `.env.example` to `.env` and adjust paths if needed.

## Usage
1. Start the web server with `uvicorn main:app --host 0.0.0.0 --port 8000`.
2. Open `http://localhost:8000/enroll` to register a student.
3. Open `http://localhost:8000/recognize` to run recognition with live alerts.
4. Open `http://localhost:8000/` for the manager dashboard.

## Configuration
| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| DB_PATH | string | attendance.db | SQLite database file path |
| EMBEDDINGS_DIR | string | embeddings/ | Directory where embeddings are stored |
| SIMILARITY_THRESHOLD | float | 0.68 | Minimum cosine similarity to accept a match |
| CAMERA_INDEX | int | 0 | OpenCV camera index |
| MODEL_NAME | string | Facenet512 | DeepFace model name |
| DETECTOR_BACKEND | string | opencv | DeepFace detector backend label |
| HAAR_CASCADE_PATH | string | haarcascade_frontalface_default.xml | Haar Cascade path for face detection |
| ENROLLMENT_SAMPLES | int | 5 | Number of samples to capture per student |
| LOG_LEVEL | string | INFO | Logging level |
| HOSTEL_OPTIONS | string | Amna,Khadija | Comma-separated hostel list |
| DEFAULT_HOSTEL | string | Amna | Default hostel option |
| SNAPSHOT_INTERVAL_MS | int | 700 | Webcam snapshot interval in ms |
| VIDEO_WIDTH | int | 640 | Webcam capture width |
| VIDEO_HEIGHT | int | 480 | Webcam capture height |
| JPEG_QUALITY | float | 0.9 | JPEG quality for snapshots |
| EMBEDDINGS_REFRESH_SECONDS | int | 10 | Reload embeddings interval |
| DASHBOARD_REFRESH_SECONDS | int | 10 | Dashboard auto-refresh interval |

## Database Schema
The database contains two tables:
- `students`: stores `student_id`, `name`, `hostel`, and enrollment time.
- `attendance`: stores `student_id`, `timestamp`, and match confidence.

## How Matching Works
Each face image is converted into an embedding vector using DeepFace. Cosine similarity is computed between the live embedding and all stored embeddings per student. The highest score per student is selected, and a match is accepted when the best score is above the configured threshold.

## Performance Tips
Use the Facenet512 model for a strong speed/accuracy balance. If available, run DeepFace with GPU acceleration (TensorFlow GPU) to improve frame rate. You can also increase `ENROLLMENT_SAMPLES` to improve robustness.

## Known Limitations
Lighting changes can reduce accuracy without additional preprocessing. Enrollment quality depends on capturing diverse angles with only one face in frame. Threshold tuning may be required for different camera setups.

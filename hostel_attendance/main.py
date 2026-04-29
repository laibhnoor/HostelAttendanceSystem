from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from config import (
    DASHBOARD_REFRESH_SECONDS,
    DEFAULT_HOSTEL,
    EMBEDDINGS_DIR,
    EMBEDDINGS_REFRESH_SECONDS,
    ENROLLMENT_SAMPLES,
    FACE_PADDING,
    HAAR_CASCADE_PATH,
    HAAR_MIN_NEIGHBORS,
    HAAR_MIN_SIZE,
    HAAR_SCALE_FACTOR,
    HOSTEL_OPTIONS,
    JPEG_QUALITY,
    LOG_LEVEL,
    MODEL_NAME,
    SIMILARITY_THRESHOLD,
    SNAPSHOT_INTERVAL_MS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from database import (
    add_student,
    get_all_students,
    get_attendance_today,
    get_student_record,
    init_db,
    mark_attendance,
    student_exists,
)
from utils.embedding import get_embedding, load_all_embeddings, save_embedding
from utils.similarity import match_face

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Hostel Attendance Web")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_alert_lock = asyncio.Lock()
_alert_queues: List[asyncio.Queue] = []

_embeddings_lock = asyncio.Lock()
_embeddings_db: Dict[str, List[np.ndarray]] = {}
_embeddings_last_load = 0.0


async def _broadcast_alert(message: str, kind: str) -> None:
    """Broadcast a live alert to all connected clients.

    Args:
        message: Alert message text.
        kind: Alert type (success or error).

    Returns:
        None.
    """
    payload = {"message": message, "kind": kind}
    async with _alert_lock:
        queues = list(_alert_queues)
    for queue in queues:
        await queue.put(payload)


def _load_cascade() -> Optional[cv2.CascadeClassifier]:
    """Load the Haar Cascade classifier.

    Args:
        None.

    Returns:
        Cascade classifier or None if loading failed.
    """
    cascade = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
    if cascade.empty():
        logger.error("Failed to load Haar Cascade from %s", HAAR_CASCADE_PATH)
        return None
    return cascade


CASCADE = _load_cascade()


def _decode_image(file_bytes: bytes) -> Optional[np.ndarray]:
    """Decode an image from raw bytes.

    Args:
        file_bytes: Uploaded image bytes.

    Returns:
        Decoded BGR image or None on failure.
    """
    try:
        buffer = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Failed to decode image bytes")
            return None
        return image
    except (ValueError, cv2.error) as exc:
        logger.error("Image decode failed: %s", exc)
        return None


def _detect_faces(frame: np.ndarray) -> List[tuple[int, int, int, int]]:
    """Detect faces using Haar Cascade.

    Args:
        frame: BGR image frame.

    Returns:
        List of face rectangles.
    """
    if CASCADE is None:
        return []
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=(HAAR_MIN_SIZE, HAAR_MIN_SIZE),
        )
        return list(faces)
    except cv2.error as exc:
        logger.error("Face detection failed: %s", exc)
        return []


def _select_primary_face(faces: List[tuple[int, int, int, int]]) -> Optional[tuple[int, int, int, int]]:
    """Select the largest face from detected faces.

    Args:
        faces: List of face rectangles.

    Returns:
        The largest face rectangle or None.
    """
    if not faces:
        return None
    return max(faces, key=lambda rect: rect[2] * rect[3])


def _crop_with_padding(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a face region with padding.

    Args:
        frame: BGR image frame.
        rect: Face rectangle.

    Returns:
        Cropped face image.
    """
    x, y, w, h = rect
    height, width = frame.shape[:2]
    x1 = max(x - FACE_PADDING, 0)
    y1 = max(y - FACE_PADDING, 0)
    x2 = min(x + w + FACE_PADDING, width)
    y2 = min(y + h + FACE_PADDING, height)
    return frame[y1:y2, x1:x2]


def _count_embeddings(student_id: str) -> int:
    """Count saved embeddings for a student.

    Args:
        student_id: Unique student identifier.

    Returns:
        Number of embeddings found.
    """
    student_dir = EMBEDDINGS_DIR / student_id
    try:
        if not student_dir.exists():
            return 0
        files = [
            f
            for f in student_dir.iterdir()
            if f.is_file() and f.name.startswith(f"{student_id}_") and f.suffix == ".npy"
        ]
        return len(files)
    except OSError as exc:
        logger.error("Failed to count embeddings for %s: %s", student_id, exc)
        return 0


async def _get_embeddings_db() -> Dict[str, List[np.ndarray]]:
    """Get embeddings database, reloading periodically.

    Args:
        None.

    Returns:
        Embeddings database dictionary.
    """
    global _embeddings_db, _embeddings_last_load
    now = time.time()
    async with _embeddings_lock:
        if now - _embeddings_last_load >= EMBEDDINGS_REFRESH_SECONDS:
            _embeddings_db = load_all_embeddings(str(EMBEDDINGS_DIR))
            _embeddings_last_load = now
    return _embeddings_db


@app.on_event("startup")
async def _startup() -> None:
    """Initialize app resources on startup.

    Args:
        None.

    Returns:
        None.
    """
    init_db()
    await _get_embeddings_db()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Render the manager dashboard.

    Args:
        request: FastAPI request.

    Returns:
        HTML response.
    """
    attendance = get_attendance_today()
    students = get_all_students()
    context = {
        "request": request,
        "title": "Hostel Dashboard",
        "heading": "Manager Dashboard",
        "subtitle": "Live attendance and student registry.",
        "page": "dashboard",
        "attendance": attendance,
        "students": students,
        "hostels": HOSTEL_OPTIONS,
        "default_hostel": DEFAULT_HOSTEL,
        "snapshot_interval_ms": SNAPSHOT_INTERVAL_MS,
        "video_width": VIDEO_WIDTH,
        "video_height": VIDEO_HEIGHT,
        "jpeg_quality": JPEG_QUALITY,
        "dashboard_refresh_seconds": DASHBOARD_REFRESH_SECONDS,
    }
    return templates.TemplateResponse("dashboard.html", context)


@app.get("/enroll", response_class=HTMLResponse)
async def enroll_page(request: Request) -> HTMLResponse:
    """Render the enrollment page.

    Args:
        request: FastAPI request.

    Returns:
        HTML response.
    """
    context = {
        "request": request,
        "title": "Enroll Student",
        "heading": "Student Enrollment",
        "subtitle": "Capture multiple samples with a clean camera feed.",
        "page": "enroll",
        "hostels": HOSTEL_OPTIONS,
        "default_hostel": DEFAULT_HOSTEL,
        "enrollment_samples": ENROLLMENT_SAMPLES,
        "snapshot_interval_ms": SNAPSHOT_INTERVAL_MS,
        "video_width": VIDEO_WIDTH,
        "video_height": VIDEO_HEIGHT,
        "jpeg_quality": JPEG_QUALITY,
    }
    return templates.TemplateResponse("enroll.html", context)


@app.get("/recognize", response_class=HTMLResponse)
async def recognize_page(request: Request) -> HTMLResponse:
    """Render the recognition page.

    Args:
        request: FastAPI request.

    Returns:
        HTML response.
    """
    context = {
        "request": request,
        "title": "Recognize Students",
        "heading": "Live Recognition",
        "subtitle": "Alerts appear instantly when a match is found.",
        "page": "recognize",
        "hostels": HOSTEL_OPTIONS,
        "default_hostel": DEFAULT_HOSTEL,
        "snapshot_interval_ms": SNAPSHOT_INTERVAL_MS,
        "video_width": VIDEO_WIDTH,
        "video_height": VIDEO_HEIGHT,
        "jpeg_quality": JPEG_QUALITY,
    }
    return templates.TemplateResponse("recognize.html", context)


@app.post("/api/enroll")
async def api_enroll(
    student_id: str = Form(...),
    name: str = Form(...),
    hostel: str = Form(...),
    frame: UploadFile = File(...),
) -> JSONResponse:
    """Handle enrollment frame uploads.

    Args:
        student_id: Unique student identifier.
        name: Full student name.
        hostel: Hostel name.
        frame: Uploaded image frame.

    Returns:
        JSON response payload.
    """
    if not student_id.isalnum() or not name.strip() or hostel not in HOSTEL_OPTIONS:
        return JSONResponse({"ok": False, "message": "Invalid enrollment data."})

    record = get_student_record(student_id)
    if record is None:
        if not add_student(student_id, name.strip(), hostel):
            return JSONResponse({"ok": False, "message": "Student already exists."})
    else:
        if record["hostel"] != hostel:
            return JSONResponse({"ok": False, "message": "Hostel mismatch for student."})
        if record["name"].strip().lower() != name.strip().lower():
            return JSONResponse({"ok": False, "message": "Name does not match record."})

    try:
        image_bytes = await frame.read()
    except (RuntimeError, OSError) as exc:
        logger.error("Failed to read enrollment frame: %s", exc)
        return JSONResponse({"ok": False, "message": "Failed to read frame."})

    image = _decode_image(image_bytes)
    if image is None:
        return JSONResponse({"ok": False, "message": "Invalid image data."})

    faces = _detect_faces(image)
    if len(faces) != 1:
        return JSONResponse({"ok": False, "message": "Ensure exactly one face is visible."})

    face_roi = _crop_with_padding(image, faces[0])
    embedding = get_embedding(face_roi, MODEL_NAME)
    if embedding is None:
        return JSONResponse({"ok": False, "message": "Embedding extraction failed."})

    saved = save_embedding(student_id, embedding, str(EMBEDDINGS_DIR))
    if saved is None:
        return JSONResponse({"ok": False, "message": "Failed to save embedding."})

    captured = _count_embeddings(student_id)
    return JSONResponse({"ok": True, "captured": captured, "required": ENROLLMENT_SAMPLES})


@app.post("/api/recognize")
async def api_recognize(
    hostel: str = Form(...),
    frame: UploadFile = File(...),
) -> JSONResponse:
    """Handle recognition frame uploads.

    Args:
        hostel: Selected hostel name.
        frame: Uploaded image frame.

    Returns:
        JSON response payload.
    """
    if hostel not in HOSTEL_OPTIONS:
        return JSONResponse({"ok": False, "message": "Invalid hostel selection."})

    try:
        image_bytes = await frame.read()
    except (RuntimeError, OSError) as exc:
        logger.error("Failed to read recognition frame: %s", exc)
        return JSONResponse({"ok": False, "message": "Failed to read frame."})

    image = _decode_image(image_bytes)
    if image is None:
        return JSONResponse({"ok": False, "message": "Invalid image data."})

    faces = _detect_faces(image)
    face = _select_primary_face(faces)
    if face is None:
        return JSONResponse({"ok": True, "message": "No face detected."})

    face_roi = _crop_with_padding(image, face)
    embedding = get_embedding(face_roi, MODEL_NAME)
    if embedding is None:
        return JSONResponse({"ok": False, "message": "Embedding extraction failed."})

    embeddings_db = await _get_embeddings_db()
    if not embeddings_db:
        return JSONResponse({"ok": False, "message": "No enrolled students found."})

    student_id, score = match_face(embedding, embeddings_db, SIMILARITY_THRESHOLD)
    if student_id is None:
        return JSONResponse({"ok": True, "message": "Unknown face", "score": score})

    record = get_student_record(student_id)
    if record is None:
        return JSONResponse({"ok": False, "message": "Student record missing."})

    name = record["name"]
    student_hostel = record["hostel"]

    if student_hostel != hostel:
        await _broadcast_alert(
            f"Wrong hostel: {name} (registered {student_hostel})", "error"
        )
        return JSONResponse(
            {
                "ok": True,
                "status": "wrong_hostel",
                "student_id": student_id,
                "name": name,
                "score": score,
            }
        )

    marked = mark_attendance(student_id, score)
    if marked:
        await _broadcast_alert(f"Marked present: {name}", "success")
        status = "marked"
    else:
        await _broadcast_alert(f"Already marked: {name}", "success")
        status = "duplicate"

    return JSONResponse(
        {
            "ok": True,
            "status": status,
            "student_id": student_id,
            "name": name,
            "score": score,
        }
    )


@app.get("/api/alerts")
async def api_alerts() -> EventSourceResponse:
    """Stream live alerts to the client.

    Args:
        None.

    Returns:
        SSE response.
    """
    queue: asyncio.Queue = asyncio.Queue()
    async with _alert_lock:
        _alert_queues.append(queue)

    async def event_generator() -> AsyncIterator[dict]:
        try:
            while True:
                payload = await queue.get()
                yield {"event": "message", "data": json.dumps(payload)}
        finally:
            async with _alert_lock:
                if queue in _alert_queues:
                    _alert_queues.remove(queue)

    return EventSourceResponse(event_generator())

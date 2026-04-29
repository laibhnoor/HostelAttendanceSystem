from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Tuple

import cv2

from config import (
    CAMERA_INDEX,
    COLOR_GREEN,
    DEFAULT_HOSTEL,
    ENROLLMENT_SAMPLES,
    EMBEDDINGS_DIR,
    FACE_PADDING,
    FONT_SCALE_LARGE,
    HAAR_CASCADE_PATH,
    HAAR_MIN_NEIGHBORS,
    HAAR_MIN_SIZE,
    HAAR_SCALE_FACTOR,
    HOSTEL_OPTIONS,
    RECT_THICKNESS,
    TEXT_POS_X,
    TEXT_POS_Y,
    TEXT_THICKNESS,
    WAIT_KEY_DELAY_MS,
    LOG_LEVEL,
    MODEL_NAME,
)
from database import add_student, init_db
from utils.embedding import get_embedding, save_embedding

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


@dataclass
class VideoCaptureContext:
    """Context manager for OpenCV video capture.

    Args:
        index: Camera index.

    Returns:
        OpenCV VideoCapture instance.
    """

    index: int

    def __enter__(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
        self._cap = cap
        return cap

    def __exit__(self, exc_type, exc, tb) -> None:
        if hasattr(self, "_cap"):
            self._cap.release()
        cv2.destroyAllWindows()


def _validate_inputs(student_id: str, name: str, hostel: str) -> bool:
    """Validate enrollment inputs.

    Args:
        student_id: Unique student identifier.
        name: Student full name.

    Returns:
        True if valid, False otherwise.
    """
    return student_id.isalnum() and bool(name.strip()) and hostel in HOSTEL_OPTIONS


def _select_hostel() -> str:
    """Prompt the user to select a hostel.

    Args:
        None.

    Returns:
        Selected hostel name.
    """
    options = ", ".join(HOSTEL_OPTIONS)
    prompt = f"Enter hostel ({options}) [default: {DEFAULT_HOSTEL}]: "
    hostel = input(prompt).strip()
    if not hostel:
        return DEFAULT_HOSTEL
    return hostel


def _crop_with_padding(frame, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    """Compute padded crop coordinates within frame bounds.

    Args:
        frame: Frame image.
        x: Face x.
        y: Face y.
        w: Face width.
        h: Face height.

    Returns:
        Tuple of (x1, y1, x2, y2) bounds.
    """
    height, width = frame.shape[:2]
    x1 = max(x - FACE_PADDING, 0)
    y1 = max(y - FACE_PADDING, 0)
    x2 = min(x + w + FACE_PADDING, width)
    y2 = min(y + h + FACE_PADDING, height)
    return x1, y1, x2, y2


def main() -> None:
    """Run the enrollment workflow.

    Args:
        None.

    Returns:
        None.
    """
    init_db()

    student_id = input("Enter student ID (alphanumeric): ").strip()
    name = input("Enter full name: ").strip()

    hostel = _select_hostel()

    if not _validate_inputs(student_id, name, hostel):
        logger.error("Invalid student ID or name")
        sys.exit(1)

    if not add_student(student_id, name, hostel):
        logger.error("Student already exists or could not be added")
        sys.exit(1)

    cascade = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
    if cascade.empty():
        logger.error("Failed to load Haar Cascade from %s", HAAR_CASCADE_PATH)
        sys.exit(1)

    captured = 0

    try:
        with VideoCaptureContext(CAMERA_INDEX) as cap:
            while captured < ENROLLMENT_SAMPLES:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=HAAR_SCALE_FACTOR,
                    minNeighbors=HAAR_MIN_NEIGHBORS,
                    minSize=(HAAR_MIN_SIZE, HAAR_MIN_SIZE),
                )

                for (x, y, w, h) in faces:
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        COLOR_GREEN,
                        RECT_THICKNESS,
                    )

                cv2.putText(
                    frame,
                    f"Captured: {captured} / {ENROLLMENT_SAMPLES}",
                    (TEXT_POS_X, TEXT_POS_Y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE_LARGE,
                    COLOR_GREEN,
                    TEXT_THICKNESS,
                )

                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    x1, y1, x2, y2 = _crop_with_padding(frame, x, y, w, h)
                    face_roi = frame[y1:y2, x1:x2]
                    embedding = get_embedding(face_roi, MODEL_NAME)
                    if embedding is not None:
                        saved = save_embedding(student_id, embedding, str(EMBEDDINGS_DIR))
                        if saved:
                            captured += 1

                cv2.imshow("Enrollment", frame)
                cv2.waitKey(WAIT_KEY_DELAY_MS)
    except KeyboardInterrupt:
        logger.info("Enrollment interrupted by user")
        return
    except RuntimeError as exc:
        logger.error("Enrollment failed: %s", exc)
        return

    logger.info("Enrollment complete for %s", student_id)


if __name__ == "__main__":
    main()

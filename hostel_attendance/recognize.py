from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import cv2

from config import (
    CAMERA_INDEX,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_RED,
    DETECTOR_BACKEND,
    EMBEDDINGS_DIR,
    FACE_PADDING,
    FONT_SCALE_LARGE,
    FONT_SCALE_SMALL,
    FPS_AVG_WINDOW,
    HAAR_CASCADE_PATH,
    HAAR_MIN_NEIGHBORS,
    HAAR_MIN_SIZE,
    HAAR_SCALE_FACTOR,
    LABEL_OFFSET_Y,
    LOG_LEVEL,
    MODEL_NAME,
    RECT_THICKNESS,
    RELOAD_EMBEDDINGS_EVERY_FRAMES,
    SIMILARITY_THRESHOLD,
    TEXT_POS_X,
    TEXT_POS_Y,
    TEXT_THICKNESS,
    WAIT_KEY_DELAY_MS,
)
from database import get_student_map, init_db, mark_attendance
from utils.embedding import get_embedding, load_all_embeddings
from utils.similarity import match_face

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


def _crop_with_padding(frame, x: int, y: int, w: int, h: int):
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
    """Run the real-time recognition workflow.

    Args:
        None.

    Returns:
        None.
    """
    if DETECTOR_BACKEND.lower() != "opencv":
        logger.warning("Only Haar Cascade (opencv) detector is supported in this script")

    init_db()
    embeddings_db = load_all_embeddings(str(EMBEDDINGS_DIR))
    if not embeddings_db:
        logger.warning("No embeddings found. Enroll students first.")
        return

    student_map = get_student_map()

    cascade = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
    if cascade.empty():
        logger.error("Failed to load Haar Cascade from %s", HAAR_CASCADE_PATH)
        return

    frame_times = deque(maxlen=FPS_AVG_WINDOW)
    frame_count = 0

    try:
        with VideoCaptureContext(CAMERA_INDEX) as cap:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue

                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(
                        gray,
                        scaleFactor=HAAR_SCALE_FACTOR,
                        minNeighbors=HAAR_MIN_NEIGHBORS,
                        minSize=(HAAR_MIN_SIZE, HAAR_MIN_SIZE),
                    )

                    for (x, y, w, h) in faces:
                        x1, y1, x2, y2 = _crop_with_padding(frame, x, y, w, h)
                        face_roi = frame[y1:y2, x1:x2]
                        embedding = get_embedding(face_roi, MODEL_NAME)
                        if embedding is None:
                            continue

                        student_id, score = match_face(
                            embedding, embeddings_db, SIMILARITY_THRESHOLD
                        )
                        if student_id is not None:
                            marked = mark_attendance(student_id, score)
                            name = student_map.get(student_id, student_id)
                            label = f"{name} | Score: {score:.2f}"
                            color = COLOR_GREEN
                            if marked:
                                logger.info("Marked present: %s (%.2f)", student_id, score)
                            else:
                                logger.info("Duplicate skipped: %s (%.2f)", student_id, score)
                        else:
                            label = f"Unknown | Score: {score:.2f}"
                            color = COLOR_RED

                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + w, y + h),
                            color,
                            RECT_THICKNESS,
                        )
                        cv2.putText(
                            frame,
                            label,
                            (x, y - LABEL_OFFSET_Y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SCALE_SMALL,
                            color,
                            TEXT_THICKNESS,
                        )

                    frame_count += 1
                    if frame_count % RELOAD_EMBEDDINGS_EVERY_FRAMES == 0:
                        embeddings_db = load_all_embeddings(str(EMBEDDINGS_DIR))
                        student_map = get_student_map()
                        logger.info("Reloaded embeddings and student map")

                    frame_times.append(time.time() - start_time)
                    if frame_times:
                        fps = len(frame_times) / sum(frame_times)
                        cv2.putText(
                            frame,
                            f"FPS: {fps:.1f}",
                            (TEXT_POS_X, TEXT_POS_Y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SCALE_LARGE,
                            COLOR_CYAN,
                            TEXT_THICKNESS,
                        )

                    cv2.imshow("Recognition", frame)
                except (ValueError, RuntimeError, cv2.error) as exc:
                    logger.error("Frame processing error: %s", exc)

                if cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        logger.info("Recognition interrupted by user")
    except RuntimeError as exc:
        logger.error("Recognition failed: %s", exc)


if __name__ == "__main__":
    main()

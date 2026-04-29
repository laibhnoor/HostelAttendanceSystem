from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    """Get a required environment variable.

    Args:
        name: Environment variable name.

    Returns:
        The environment variable value.

    Raises:
        ValueError: If the variable is missing or empty.
    """
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


def _parse_int(name: str) -> int:
    """Parse an integer environment variable.

    Args:
        name: Environment variable name.

    Returns:
        Parsed integer value.

    Raises:
        ValueError: If the variable cannot be parsed as int.
    """
    raw = _require_env(name)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _parse_float(name: str) -> float:
    """Parse a float environment variable.

    Args:
        name: Environment variable name.

    Returns:
        Parsed float value.

    Raises:
        ValueError: If the variable cannot be parsed as float.
    """
    raw = _require_env(name)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name}: {raw}") from exc


DB_PATH: Path = Path(_require_env("DB_PATH"))
EMBEDDINGS_DIR: Path = Path(_require_env("EMBEDDINGS_DIR"))
SIMILARITY_THRESHOLD: float = _parse_float("SIMILARITY_THRESHOLD")
CAMERA_INDEX: int = _parse_int("CAMERA_INDEX")
MODEL_NAME: str = _require_env("MODEL_NAME")
DETECTOR_BACKEND: str = _require_env("DETECTOR_BACKEND")
HAAR_CASCADE_PATH: Path = Path(_require_env("HAAR_CASCADE_PATH"))
ENROLLMENT_SAMPLES: int = _parse_int("ENROLLMENT_SAMPLES")
LOG_LEVEL_NAME: str = _require_env("LOG_LEVEL").upper()
HOSTEL_OPTIONS_RAW: str = _require_env("HOSTEL_OPTIONS")
DEFAULT_HOSTEL: str = _require_env("DEFAULT_HOSTEL")
SNAPSHOT_INTERVAL_MS: int = _parse_int("SNAPSHOT_INTERVAL_MS")
VIDEO_WIDTH: int = _parse_int("VIDEO_WIDTH")
VIDEO_HEIGHT: int = _parse_int("VIDEO_HEIGHT")
JPEG_QUALITY: float = _parse_float("JPEG_QUALITY")
EMBEDDINGS_REFRESH_SECONDS: int = _parse_int("EMBEDDINGS_REFRESH_SECONDS")
DASHBOARD_REFRESH_SECONDS: int = _parse_int("DASHBOARD_REFRESH_SECONDS")

if not (0.0 <= SIMILARITY_THRESHOLD <= 1.0):
    raise ValueError("SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
if ENROLLMENT_SAMPLES < 3:
    raise ValueError("ENROLLMENT_SAMPLES must be at least 3")
if CAMERA_INDEX < 0:
    raise ValueError("CAMERA_INDEX must be >= 0")
if LOG_LEVEL_NAME not in logging._nameToLevel:
    raise ValueError(
        "LOG_LEVEL must be a valid logging level name (e.g., DEBUG, INFO)"
    )
HOSTEL_OPTIONS: list[str] = [opt.strip() for opt in HOSTEL_OPTIONS_RAW.split(",") if opt.strip()]
if len(HOSTEL_OPTIONS) < 2:
    raise ValueError("HOSTEL_OPTIONS must contain at least two entries")
if DEFAULT_HOSTEL not in HOSTEL_OPTIONS:
    raise ValueError("DEFAULT_HOSTEL must be one of HOSTEL_OPTIONS")
if SNAPSHOT_INTERVAL_MS < 200:
    raise ValueError("SNAPSHOT_INTERVAL_MS must be >= 200")
if VIDEO_WIDTH <= 0 or VIDEO_HEIGHT <= 0:
    raise ValueError("VIDEO_WIDTH and VIDEO_HEIGHT must be > 0")
if not (0.5 <= JPEG_QUALITY <= 1.0):
    raise ValueError("JPEG_QUALITY must be between 0.5 and 1.0")
if EMBEDDINGS_REFRESH_SECONDS < 2:
    raise ValueError("EMBEDDINGS_REFRESH_SECONDS must be >= 2")
if DASHBOARD_REFRESH_SECONDS < 5:
    raise ValueError("DASHBOARD_REFRESH_SECONDS must be >= 5")

LOG_LEVEL: int = logging._nameToLevel[LOG_LEVEL_NAME]

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

ATTENDANCE_COOLDOWN_SECONDS: int = 60
FACE_PADDING: int = 20
FPS_AVG_WINDOW: int = 30
RELOAD_EMBEDDINGS_EVERY_FRAMES: int = 300

HAAR_SCALE_FACTOR: float = 1.1
HAAR_MIN_NEIGHBORS: int = 5
HAAR_MIN_SIZE: int = 60

WAIT_KEY_DELAY_MS: int = 1
FONT_SCALE_LARGE: float = 0.8
FONT_SCALE_SMALL: float = 0.6
TEXT_THICKNESS: int = 2
RECT_THICKNESS: int = 2
LABEL_OFFSET_Y: int = 10
TEXT_POS_X: int = 10
TEXT_POS_Y: int = 30

COLOR_GREEN: tuple[int, int, int] = (0, 255, 0)
COLOR_RED: tuple[int, int, int] = (0, 0, 255)
COLOR_CYAN: tuple[int, int, int] = (255, 255, 0)

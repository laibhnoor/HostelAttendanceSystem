from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
from deepface import DeepFace

logger = logging.getLogger(__name__)


def get_embedding(face_img_bgr: np.ndarray, model_name: str) -> Optional[np.ndarray]:
    """Generate an L2-normalized embedding for a face image.

    Args:
        face_img_bgr: Cropped face image in BGR format.
        model_name: DeepFace model name.

    Returns:
        Normalized embedding vector or None on failure.
    """
    try:
        result = DeepFace.represent(
            img_path=face_img_bgr,
            model_name=model_name,
            enforce_detection=False,
        )
        if not result:
            logger.warning("DeepFace returned no embedding")
            return None
        embedding = result[0].get("embedding")
        if embedding is None:
            logger.warning("DeepFace embedding missing in response")
            return None
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            logger.warning("Embedding norm is zero")
            return None
        return vec / norm
    except (ValueError, RuntimeError, TypeError) as exc:
        logger.error("Failed to generate embedding: %s", exc)
        return None


def save_embedding(student_id: str, embedding: np.ndarray, embeddings_dir: str) -> Optional[str]:
    """Save a face embedding to disk.

    Args:
        student_id: Unique student identifier.
        embedding: Embedding vector to save.
        embeddings_dir: Base directory to store embeddings.

    Returns:
        Path to the saved embedding or None on failure.
    """
    student_dir = os.path.join(embeddings_dir, student_id)
    try:
        os.makedirs(student_dir, exist_ok=True)
    except OSError as exc:
        logger.error("Failed to create embeddings directory %s: %s", student_dir, exc)
        return None

    try:
        existing = [
            name
            for name in os.listdir(student_dir)
            if name.startswith(f"{student_id}_") and name.endswith(".npy")
        ]
    except OSError as exc:
        logger.error("Failed to list embeddings directory %s: %s", student_dir, exc)
        return None

    indexes: List[int] = []
    for name in existing:
        try:
            index = int(name.split("_")[-1].split(".")[0])
            indexes.append(index)
        except ValueError:
            continue

    next_index = max(indexes) + 1 if indexes else 1
    file_path = os.path.join(student_dir, f"{student_id}_{next_index}.npy")
    try:
        np.save(file_path, embedding)
        return file_path
    except (OSError, ValueError) as exc:
        logger.error("Failed to save embedding %s: %s", file_path, exc)
        return None


def load_all_embeddings(embeddings_dir: str) -> Dict[str, List[np.ndarray]]:
    """Load all embeddings from disk.

    Args:
        embeddings_dir: Base directory that stores embeddings.

    Returns:
        Dictionary mapping student_id to list of embeddings.
    """
    db: Dict[str, List[np.ndarray]] = {}
    try:
        for root, _, files in os.walk(embeddings_dir):
            for filename in files:
                if not filename.endswith(".npy"):
                    continue
                student_id = os.path.basename(root)
                file_path = os.path.join(root, filename)
                try:
                    embedding = np.load(file_path)
                except (OSError, ValueError) as exc:
                    logger.warning("Skipping corrupted embedding %s: %s", file_path, exc)
                    continue
                db.setdefault(student_id, []).append(embedding)
        return db
    except OSError as exc:
        logger.error("Failed to walk embeddings directory %s: %s", embeddings_dir, exc)
        return {}

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity in range [-1, 1].
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def match_face(
    query_embedding: np.ndarray,
    embeddings_db: Dict[str, List[np.ndarray]],
    threshold: float,
) -> Tuple[Optional[str], float]:
    """Match a query embedding against stored embeddings.

    Args:
        query_embedding: Embedding vector to match.
        embeddings_db: Mapping of student_id to list of embeddings.
        threshold: Minimum similarity score required.

    Returns:
        Tuple of best student_id (or None) and best score.
    """
    best_student: Optional[str] = None
    best_score = -1.0

    for student_id, embeddings in embeddings_db.items():
        if not embeddings:
            continue
        scores = [cosine_similarity(query_embedding, emb) for emb in embeddings]
        max_score = max(scores)
        if max_score > best_score:
            best_score = max_score
            best_student = student_id

    if best_score >= threshold:
        return best_student, best_score
    return None, best_score

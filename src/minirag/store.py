"""In-memory vector store: a list of chunks plus a numpy matrix.

``search`` returns the ``k`` chunks most similar to a query vector, ranked by
cosine similarity. Vectors are L2-normalized on the way in (zero vectors stay
zero), so cosine similarity is a single matrix-vector product and the store
does not depend on the embedder having normalized its output.

Brute-force search is exact and fine for a few thousand chunks; a real vector
index (HNSW/IVF) only pays off beyond that.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Chunk = Dict[str, str]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class VectorStore:
    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._matrix: Optional[np.ndarray] = None  # shape (n, dim), unit rows

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dim(self) -> Optional[int]:
        """Vector dimension, or ``None`` while the store is empty."""
        return None if self._matrix is None else self._matrix.shape[1]

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Append ``chunks`` with their embedding ``vectors`` (one row each).

        Raises ``ValueError`` if the row count does not match the number of
        chunks, if vectors are not 2-D, or if the dimension differs from the
        vectors already stored. Nothing is added when validation fails.
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2-D, got shape %s" % (vectors.shape,))
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                "got %d chunks but %d vectors" % (len(chunks), vectors.shape[0])
            )
        if len(chunks) == 0:
            return
        if self._matrix is not None and vectors.shape[1] != self._matrix.shape[1]:
            raise ValueError(
                "vector dimension %d does not match store dimension %d"
                % (vectors.shape[1], self._matrix.shape[1])
            )
        vectors = _normalize_rows(vectors)
        self._matrix = (
            vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        )
        self._chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return up to ``k`` ``(chunk, cosine_score)`` pairs, best first.

        Returns ``[]`` for an empty store. Ties keep insertion order. Raises
        ``ValueError`` if ``k <= 0`` or the query dimension does not match.
        """
        if k <= 0:
            raise ValueError("k must be positive, got %d" % k)
        query = np.asarray(query_vec, dtype=np.float64)
        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 1:
            raise ValueError("query_vec must be a single vector, got shape %s" % (query.shape,))
        if self._matrix is None:
            return []
        if query.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                "query dimension %d does not match store dimension %d"
                % (query.shape[0], self._matrix.shape[1])
            )
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        scores = self._matrix @ query.astype(np.float32)
        # Stable sort on the negated scores: best first, ties in insertion order.
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._chunks[i], float(scores[i])) for i in order]

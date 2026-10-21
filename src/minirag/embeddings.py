"""Text embedders.

Every embedder implements ``embed(texts) -> np.ndarray`` returning a float32
matrix of shape ``(len(texts), dim)`` whose rows are L2-normalized, so cosine
similarity between two embeddings is just their dot product.

* ``FakeEmbedder``: deterministic hashed bag-of-words. No network, no AWS
  credentials. Used for tests and offline runs. It only captures word overlap,
  not meaning, so it is NOT a stand-in for real semantic quality.
* ``BedrockEmbedder``: Amazon Titan text embeddings through ``bedrock-runtime``.
  Uses the default AWS credential chain; no keys in code.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, List, Optional, Protocol, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

DEFAULT_BEDROCK_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_BEDROCK_REGION = "us-east-1"


class Embedder(Protocol):
    """Anything that turns texts into L2-normalized vectors.

    Contract shared by every implementation:

    * returns a float32 matrix of shape ``(len(texts), dim)``;
    * an empty batch gives shape ``(0, dim)``;
    * a ``str`` instead of a sequence raises ``TypeError``;
    * empty or whitespace-only text gives the zero vector (it never raises),
      which scores 0 against everything in the vector store.
    """

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows. All-zero rows are left as zeros (no division by 0)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder (the "hashing trick").

    Each lowercase word is hashed (MD5, so results are stable across processes,
    unlike the built-in ``hash``) to one of ``dim`` buckets, with a hash-derived
    sign to reduce collision bias. Texts that share words get a high cosine
    score; texts with no shared words score near 0. A text with no word
    characters embeds to the zero vector.
    """

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive, got %d" % dim)
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            raise TypeError("embed() expects a sequence of strings, not a single string")
        matrix = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:8], "big") % self.dim
                sign = 1.0 if digest[8] % 2 == 0 else -1.0
                matrix[row, bucket] += sign
        return _normalize(matrix)


class BedrockEmbedder:
    """Amazon Titan text embeddings via the Bedrock ``InvokeModel`` API.

    Configuration (constructor args win over env vars):

    * model id: ``MINIRAG_EMBED_MODEL_ID`` (default Titan Text Embeddings V2)
    * region: ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` (default ``us-east-1``)

    Credentials come from the default AWS chain. The boto3 client is created
    lazily on the first ``embed`` call, so constructing this class needs no
    credentials. Pass ``client`` to inject a client (useful for tests).

    Titan takes one text per request, so texts are embedded one by one.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        dimensions: int = 1024,
        client: Any = None,
    ) -> None:
        self.model_id = (
            model_id
            or os.environ.get("MINIRAG_EMBED_MODEL_ID")
            or DEFAULT_BEDROCK_EMBED_MODEL_ID
        )
        self.region_name = (
            region_name
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or DEFAULT_BEDROCK_REGION
        )
        self.dimensions = dimensions
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3  # imported lazily so tests/offline use don't need it

            self._client = boto3.client("bedrock-runtime", region_name=self.region_name)
        return self._client

    def _embed_one(self, text: str) -> List[float]:
        body = json.dumps(
            {"inputText": text, "dimensions": self.dimensions, "normalize": True}
        )
        response = self._get_client().invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        return payload["embedding"]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            raise TypeError("embed() expects a sequence of strings, not a single string")
        if len(texts) == 0:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        # Blank texts follow the Embedder contract (zero vector, like
        # FakeEmbedder) and are not sent to Bedrock, which rejects empty input.
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float64)
        for row, text in enumerate(texts):
            if text.strip():
                matrix[row] = self._embed_one(text)
        # Titan already normalizes; do it again so the contract never depends on it.
        return _normalize(matrix)

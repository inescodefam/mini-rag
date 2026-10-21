"""Runtime wiring shared by the API and the CLI.

Everything is configured through environment variables (no keys in code; AWS
credentials come from the default credential chain):

* ``MINIRAG_OFFLINE``: ``1``/``true`` uses ``FakeEmbedder`` + ``StubLLM`` (no
  AWS). Default: Bedrock embeddings and Bedrock LLM.
* ``MINIRAG_CORPUS_DIR``: folder of ``*.md`` files (default: ``corpus/`` next
  to ``src/``).
* ``MINIRAG_TOP_K``: chunks to retrieve (default ``DEFAULT_K``).
* ``MINIRAG_THRESHOLD``: minimum cosine score (default ``DEFAULT_THRESHOLD``).
  It depends on the embedder, so re-tune it when switching to Bedrock.
* ``MINIRAG_EMBED_MODEL_ID`` / ``MINIRAG_LLM_MODEL_ID`` / ``AWS_REGION``: see
  ``BedrockEmbedder`` and ``BedrockLLM``.
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Optional

from .embeddings import BedrockEmbedder, Embedder, FakeEmbedder
from .llm import LLM, BedrockLLM, StubLLM
from .rag import DEFAULT_K, DEFAULT_THRESHOLD, ingest
from .store import VectorStore

DEFAULT_CORPUS_DIR = pathlib.Path(__file__).resolve().parents[2] / "corpus"

_TRUE = {"1", "true", "yes", "on"}


@dataclass
class Pipeline:
    """Everything ``answer`` needs, ready to use."""

    embedder: Embedder
    store: VectorStore
    llm: LLM
    k: int = DEFAULT_K
    threshold: float = DEFAULT_THRESHOLD


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def build_pipeline(
    offline: Optional[bool] = None,
    corpus_dir: Optional[str] = None,
    k: Optional[int] = None,
    threshold: Optional[float] = None,
) -> Pipeline:
    """Build embedder, LLM and an ingested store from args, falling back to env vars.

    Raises ``FileNotFoundError`` if the corpus folder is missing and
    ``ValueError`` if it contains no chunks (an empty store would refuse every
    question, which hides a misconfiguration).
    """
    if offline is None:
        offline = env_flag("MINIRAG_OFFLINE")
    path = pathlib.Path(corpus_dir or os.environ.get("MINIRAG_CORPUS_DIR") or DEFAULT_CORPUS_DIR)
    if not path.is_dir():
        raise FileNotFoundError("corpus directory not found: %s" % path)
    if k is None:
        k = int(os.environ.get("MINIRAG_TOP_K", DEFAULT_K))
    if threshold is None:
        threshold = float(os.environ.get("MINIRAG_THRESHOLD", DEFAULT_THRESHOLD))

    embedder: Embedder = FakeEmbedder() if offline else BedrockEmbedder()
    llm: LLM = StubLLM() if offline else BedrockLLM()
    store = VectorStore()
    if ingest(path, embedder, store) == 0:
        raise ValueError("no .md chunks found in corpus directory: %s" % path)
    return Pipeline(embedder=embedder, store=store, llm=llm, k=k, threshold=threshold)

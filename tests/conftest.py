import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minirag.embeddings import FakeEmbedder  # noqa: E402
from minirag.rag import ingest  # noqa: E402
from minirag.store import VectorStore  # noqa: E402

CORPUS_DIR = ROOT / "corpus"


@pytest.fixture(scope="session")
def corpus_index():
    """The real bundled corpus, embedded with FakeEmbedder (no AWS needed)."""
    embedder = FakeEmbedder()
    store = VectorStore()
    assert ingest(CORPUS_DIR, embedder, store) > 0
    return embedder, store

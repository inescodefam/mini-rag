import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from minirag.embeddings import FakeEmbedder  # noqa: E402
from minirag.store import VectorStore  # noqa: E402


def _chunk(name):
    return {"source": name + ".md", "heading": "", "text": name}


def test_empty_store_returns_nothing():
    store = VectorStore()
    assert len(store) == 0
    assert store.dim is None
    assert store.search(np.array([1.0, 0.0]), k=3) == []


def test_results_sorted_by_cosine_best_first():
    store = VectorStore()
    store.add(
        [_chunk("far"), _chunk("exact"), _chunk("near")],
        np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
    )
    results = store.search(np.array([1.0, 0.0]), k=3)
    assert [c["text"] for c, _ in results] == ["exact", "near", "far"]
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    assert scores[1] == pytest.approx(2 ** -0.5, abs=1e-6)
    assert scores[2] == pytest.approx(0.0, abs=1e-6)


def test_k_limits_results_and_larger_k_returns_all():
    store = VectorStore()
    store.add([_chunk("a"), _chunk("b")], np.eye(2))
    assert len(store.search(np.array([1.0, 0.0]), k=1)) == 1
    assert len(store.search(np.array([1.0, 0.0]), k=10)) == 2


def test_cosine_ignores_vector_magnitude():
    store = VectorStore()
    store.add([_chunk("big"), _chunk("small")], np.array([[100.0, 0.0], [0.0, 0.01]]))
    results = store.search(np.array([0.5, 0.0]), k=2)
    assert results[0][0]["text"] == "big"
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)


def test_ties_keep_insertion_order():
    store = VectorStore()
    store.add([_chunk("first"), _chunk("second")], np.array([[1.0, 0.0], [1.0, 0.0]]))
    results = store.search(np.array([1.0, 0.0]), k=2)
    assert [c["text"] for c, _ in results] == ["first", "second"]


def test_add_is_incremental():
    store = VectorStore()
    store.add([_chunk("a")], np.array([[1.0, 0.0]]))
    store.add([_chunk("b")], np.array([[0.0, 1.0]]))
    assert len(store) == 2
    assert store.dim == 2
    assert store.search(np.array([0.0, 1.0]), k=1)[0][0]["text"] == "b"


def test_zero_vectors_do_not_produce_nan():
    store = VectorStore()
    store.add([_chunk("zero"), _chunk("one")], np.array([[0.0, 0.0], [1.0, 0.0]]))
    results = store.search(np.array([1.0, 0.0]), k=2)
    assert results[0][0]["text"] == "one"
    assert results[1][1] == 0.0
    zero_query = store.search(np.array([0.0, 0.0]), k=2)
    assert all(score == 0.0 for _, score in zero_query)


def test_add_validation_leaves_store_unchanged():
    store = VectorStore()
    store.add([_chunk("a")], np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        store.add([_chunk("b"), _chunk("c")], np.array([[1.0, 0.0]]))  # count mismatch
    with pytest.raises(ValueError):
        store.add([_chunk("b")], np.array([[1.0, 0.0, 0.0]]))  # dim mismatch
    with pytest.raises(ValueError):
        store.add([_chunk("b")], np.array([1.0, 0.0]))  # not 2-D
    assert len(store) == 1


def test_add_empty_is_noop():
    store = VectorStore()
    store.add([], np.zeros((0, 4)))
    assert len(store) == 0
    assert store.dim is None


def test_search_validation():
    store = VectorStore()
    store.add([_chunk("a")], np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        store.search(np.array([1.0, 0.0]), k=0)
    with pytest.raises(ValueError):
        store.search(np.array([1.0, 0.0, 0.0]), k=1)
    with pytest.raises(ValueError):
        store.search(np.zeros((2, 2)), k=1)


def test_accepts_single_row_matrix_query_from_embedder():
    emb = FakeEmbedder()
    store = VectorStore()
    texts = ["reduce hallucinations by grounding answers", "terraform state locking"]
    store.add([_chunk("rag"), _chunk("tf")], emb.embed(texts))
    results = store.search(emb.embed(["how to reduce hallucinations"]), k=1)
    assert results[0][0]["text"] == "rag"

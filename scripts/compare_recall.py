"""Compare recall@3 and score ranges: FakeEmbedder vs BedrockEmbedder.

Reuses the eval set from ``tests/test_retrieval.py`` and the out-of-scope
questions from ``tests/test_refusal.py``. Needs AWS credentials for the Bedrock
half (e.g. ``AWS_PROFILE=mini-rag``).

    AWS_PROFILE=mini-rag python scripts/compare_recall.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from minirag.embeddings import BedrockEmbedder, FakeEmbedder  # noqa: E402
from minirag.rag import ingest, retrieve  # noqa: E402
from minirag.store import VectorStore  # noqa: E402
from test_refusal import OUT_OF_SCOPE  # noqa: E402
from test_retrieval import EVAL_SET, K  # noqa: E402


def evaluate(name, embedder):
    store = VectorStore()
    n = ingest(ROOT / "corpus", embedder, store)
    print("== %s (%d chunks) ==" % (name, n))
    found = 0
    in_scope_best = []
    for question, expected in EVAL_SET:
        hits = retrieve(question, embedder, store, k=K)
        got = [(c["source"], c["heading"]) for c, _ in hits]
        rank = got.index(expected) + 1 if expected in got else None
        found += rank is not None
        in_scope_best.append(hits[0][1])
        print("  [%s] rank=%s top1=%.3f  %s" % ("ok" if rank else "MISS", rank, hits[0][1], question))
        if not rank:
            print("        got: %r" % got)
    print("  recall@%d = %d/%d = %.2f" % (K, found, len(EVAL_SET), found / len(EVAL_SET)))
    out_best = []
    for question in OUT_OF_SCOPE:
        top = retrieve(question, embedder, store, k=1)[0][1]
        out_best.append(top)
        print("  [out-of-scope] top1=%.3f  %s" % (top, question))
    print(
        "  in-scope top1: min %.3f | out-of-scope top1: max %.3f"
        % (min(in_scope_best), max(out_best))
    )
    print()


if __name__ == "__main__":
    evaluate("FakeEmbedder", FakeEmbedder())
    evaluate("BedrockEmbedder (%s)" % BedrockEmbedder().model_id, BedrockEmbedder())

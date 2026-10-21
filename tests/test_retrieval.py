"""Retrieval quality on the real corpus: recall@3 on a tiny hand-written eval set.

Uses FakeEmbedder (word overlap only), so this measures lexical retrieval, not
semantic quality. Re-run the same eval against BedrockEmbedder for real numbers.
"""
from minirag.rag import retrieve

K = 3

# (question, (expected source file, expected section heading))
EVAL_SET = [
    (
        "Why is an IAM role better than access keys?",
        ("aws-infrastructure.md", "What's an IAM role and why is it better than access keys?"),
    ),
    (
        "Where should database passwords and other secrets be stored?",
        ("delivery-operations.md", "How do you handle secrets and config?"),
    ),
    (
        "What is indirect prompt injection in a RAG app?",
        (
            "llm-safety.md",
            "What is prompt injection? Give an indirect example and mitigations.",
        ),
    ),
]


def recall_at_k(index, k=K):
    """Fraction of eval questions whose expected section is in the top-k chunks."""
    embedder, store = index
    found = 0
    for question, expected in EVAL_SET:
        hits = retrieve(question, embedder, store, k=k)
        if expected in {(c["source"], c["heading"]) for c, _ in hits}:
            found += 1
    return found / len(EVAL_SET)


def test_recall_at_3_is_perfect_on_eval_set(corpus_index):
    recall = recall_at_k(corpus_index)
    print("recall@%d = %.2f" % (K, recall))
    assert recall == 1.0


def test_each_question_finds_its_expected_chunk(corpus_index):
    # Per-question check so a failure names the question that broke.
    embedder, store = corpus_index
    for question, expected in EVAL_SET:
        hits = retrieve(question, embedder, store, k=K)
        got = [(c["source"], c["heading"]) for c, _ in hits]
        assert expected in got, "%r: expected %r in top-%d, got %r" % (question, expected, K, got)

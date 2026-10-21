"""Out-of-scope questions must be refused without calling the LLM."""
import pytest

from minirag.llm import StubLLM
from minirag.rag import DEFAULT_THRESHOLD, REFUSAL_MESSAGE, answer

OUT_OF_SCOPE = [
    "What is the capital of Peru?",
    "What is the recipe for chocolate cake?",
    "How tall is Mount Everest?",
]


@pytest.mark.parametrize("question", OUT_OF_SCOPE)
def test_out_of_scope_question_is_refused_without_llm(corpus_index, question):
    embedder, store = corpus_index
    llm = StubLLM()
    result = answer(question, embedder, store, llm)
    assert result == {"answer": REFUSAL_MESSAGE, "sources": [], "refused": True}
    assert llm.calls == []


def test_in_scope_question_is_answered_with_sources(corpus_index):
    embedder, store = corpus_index
    llm = StubLLM()
    result = answer("Why is an IAM role better than access keys?", embedder, store, llm)
    assert result["refused"] is False
    assert len(llm.calls) == 1
    assert result["sources"][0]["source"] == "aws-infrastructure.md"
    assert all(s["score"] >= DEFAULT_THRESHOLD for s in result["sources"])

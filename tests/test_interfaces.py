import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from minirag.api import create_app  # noqa: E402
from minirag.cli import main  # noqa: E402
from minirag.config import Pipeline, build_pipeline  # noqa: E402
from minirag.embeddings import FakeEmbedder  # noqa: E402
from minirag.llm import StubLLM  # noqa: E402
from minirag.rag import REFUSAL_MESSAGE, ingest  # noqa: E402
from minirag.store import VectorStore  # noqa: E402


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "pets.md").write_text(
        "# Cats\nCats sleep a lot and purr when happy.\n", encoding="utf-8"
    )
    (tmp_path / "money.md").write_text(
        "# Tax\nIncome tax is due every April.\n", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def pipeline(corpus):
    embedder, store, llm = FakeEmbedder(), VectorStore(), StubLLM()
    ingest(corpus, embedder, store)
    return Pipeline(embedder=embedder, store=store, llm=llm, k=2, threshold=0.2)


def test_ask_returns_answer_and_sources(pipeline):
    with TestClient(create_app(pipeline)) as client:
        resp = client.post("/ask", json={"question": "Do cats purr when happy?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert body["sources"][0]["citation"] == "pets.md#Cats"
    assert "Cats sleep a lot" in body["answer"]  # StubLLM echoes the prompt


def test_ask_refuses_out_of_scope_without_calling_llm(pipeline):
    with TestClient(create_app(pipeline)) as client:
        resp = client.post("/ask", json={"question": "xylophone zebra quartz"})
    assert resp.status_code == 200
    assert resp.json() == {"answer": REFUSAL_MESSAGE, "sources": [], "refused": True}
    assert pipeline.llm.calls == []


@pytest.mark.parametrize(
    "payload",
    [{}, {"question": ""}, {"question": "   "}, {"question": 5}, {"question": "x" * 2001}],
)
def test_ask_rejects_bad_requests(pipeline, payload):
    with TestClient(create_app(pipeline)) as client:
        assert client.post("/ask", json=payload).status_code == 422


def test_ask_maps_upstream_failure_to_502(pipeline):
    class Boom:
        def complete(self, system, user):
            raise RuntimeError("secret provider detail")

    pipeline.llm = Boom()
    with TestClient(create_app(pipeline)) as client:
        resp = client.post("/ask", json={"question": "Do cats purr?"})
    assert resp.status_code == 502
    assert "secret" not in resp.text


def test_build_pipeline_offline_uses_fakes(corpus):
    p = build_pipeline(offline=True, corpus_dir=str(corpus))
    assert isinstance(p.embedder, FakeEmbedder)
    assert isinstance(p.llm, StubLLM)
    assert len(p.store) == 2


def test_build_pipeline_reads_env(corpus, monkeypatch):
    monkeypatch.setenv("MINIRAG_OFFLINE", "1")
    monkeypatch.setenv("MINIRAG_CORPUS_DIR", str(corpus))
    monkeypatch.setenv("MINIRAG_TOP_K", "1")
    monkeypatch.setenv("MINIRAG_THRESHOLD", "0.5")
    p = build_pipeline()
    assert (p.k, p.threshold) == (1, 0.5)


def test_build_pipeline_missing_or_empty_corpus(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_pipeline(offline=True, corpus_dir=str(tmp_path / "nope"))
    with pytest.raises(ValueError):
        build_pipeline(offline=True, corpus_dir=str(tmp_path))


def test_cli_prints_answer_and_sources(corpus, capsys):
    code = main(["--offline", "--corpus", str(corpus), "Do cats purr when happy?"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Sources:" in out and "pets.md#Cats" in out


def test_cli_refusal(corpus, capsys):
    code = main(["--offline", "--corpus", str(corpus), "xylophone zebra quartz"])
    out = capsys.readouterr().out
    assert code == 0
    assert REFUSAL_MESSAGE in out and "Sources:" not in out


def test_cli_errors_exit_nonzero(tmp_path, capsys):
    assert main(["--offline", "--corpus", str(tmp_path / "nope"), "hi"]) == 1
    assert "error:" in capsys.readouterr().err
    assert main(["--offline", "--corpus", str(tmp_path), "   "]) == 2

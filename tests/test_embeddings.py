import io
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from minirag.embeddings import BedrockEmbedder, FakeEmbedder  # noqa: E402


def test_shape_and_dtype():
    out = FakeEmbedder(dim=64).embed(["hello world", "another text", "third"])
    assert out.shape == (3, 64)
    assert out.dtype == np.float32


def test_rows_are_unit_length():
    out = FakeEmbedder().embed(["retrieval augmented generation", "vector store"])
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_deterministic():
    a = FakeEmbedder().embed(["same text here"])
    b = FakeEmbedder().embed(["same text here"])
    assert np.array_equal(a, b)


def test_case_and_punctuation_insensitive():
    out = FakeEmbedder().embed(["Hello, World!", "hello world"])
    assert np.allclose(out[0], out[1])


def test_shared_words_score_higher_than_unrelated():
    emb = FakeEmbedder()
    q, related, unrelated = emb.embed(
        [
            "how do I reduce hallucinations in RAG",
            "reduce hallucinations by grounding RAG answers",
            "terraform state locking with dynamodb",
        ]
    )
    assert q @ related > q @ unrelated
    assert q @ related > 0.3
    assert abs(q @ unrelated) < 0.2


def test_identical_texts_have_cosine_one():
    out = FakeEmbedder().embed(["foo bar baz", "foo bar baz"])
    assert out[0] @ out[1] == pytest.approx(1.0, abs=1e-6)


def test_empty_and_wordless_text_is_zero_vector_without_nan():
    out = FakeEmbedder().embed(["", "   ", "!!!"])
    assert np.all(out == 0)
    assert not np.isnan(out).any()


def test_empty_batch():
    assert FakeEmbedder(dim=16).embed([]).shape == (0, 16)


def test_single_string_is_rejected():
    with pytest.raises(TypeError):
        FakeEmbedder().embed("not a list")


def test_invalid_dim():
    with pytest.raises(ValueError):
        FakeEmbedder(dim=0)


# --- BedrockEmbedder, with a fake boto3 client (no AWS needed) ---------------


class _FakeBedrockClient:
    def __init__(self, vector):
        self.vector = vector
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        body = json.dumps({"embedding": self.vector}).encode()
        return {"body": io.BytesIO(body)}


def test_bedrock_embedder_request_and_normalization(monkeypatch):
    monkeypatch.delenv("MINIRAG_EMBED_MODEL_ID", raising=False)
    client = _FakeBedrockClient([3.0, 4.0])
    emb = BedrockEmbedder(dimensions=2, client=client)

    out = emb.embed(["first", "second"])

    assert out.shape == (2, 2)
    assert np.allclose(out, [[0.6, 0.8], [0.6, 0.8]])
    assert len(client.calls) == 2
    call = client.calls[0]
    assert call["modelId"] == "amazon.titan-embed-text-v2:0"
    assert json.loads(call["body"]) == {
        "inputText": "first",
        "dimensions": 2,
        "normalize": True,
    }


def test_bedrock_embedder_model_id_from_env(monkeypatch):
    monkeypatch.setenv("MINIRAG_EMBED_MODEL_ID", "custom.model-v1")
    assert BedrockEmbedder(client=_FakeBedrockClient([1.0])).model_id == "custom.model-v1"


def test_bedrock_embedder_blank_text_is_zero_vector_without_api_call():
    client = _FakeBedrockClient([3.0, 4.0])
    emb = BedrockEmbedder(dimensions=2, client=client)

    out = emb.embed(["", "  \n"])

    assert out.shape == (2, 2)
    assert out.dtype == np.float32
    assert np.all(out == 0)
    assert client.calls == []


def test_bedrock_embedder_mixed_batch_keeps_row_order():
    client = _FakeBedrockClient([3.0, 4.0])
    emb = BedrockEmbedder(dimensions=2, client=client)

    out = emb.embed(["", "real text", " "])

    assert np.allclose(out, [[0, 0], [0.6, 0.8], [0, 0]])
    assert len(client.calls) == 1


@pytest.mark.parametrize("make", [lambda: FakeEmbedder(dim=2), lambda: BedrockEmbedder(dimensions=2, client=_FakeBedrockClient([3.0, 4.0]))])
def test_embedders_share_edge_case_contract(make):
    emb = make()
    assert np.all(emb.embed(["", "   "]) == 0)
    assert emb.embed([]).shape == (0, 2)
    with pytest.raises(TypeError):
        emb.embed("not a list")


def test_bedrock_embedder_needs_no_credentials_to_construct():
    BedrockEmbedder()  # lazy client: must not touch boto3/AWS

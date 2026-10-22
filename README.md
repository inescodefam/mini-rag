# mini-rag

A small retrieval-augmented generation (RAG) pipeline over five Markdown guides: heading-aware chunking, embeddings, an in-memory numpy vector store, a similarity threshold that makes it refuse out-of-scope questions, and a grounded prompt with citations. It ships with a FastAPI endpoint (`POST /ask`) and a CLI.
<img width="848" height="520" alt="image" src="https://github.com/user-attachments/assets/d58900a6-1a57-4f81-b185-b72e218d15bb" />


Design notes and the honest limits of the evaluation are in [explain.md](explain.md).

## Two modes

| Mode | Embeddings | LLM | Needs AWS | Use for |
| --- | --- | --- | --- | --- |
| Offline (`MINIRAG_OFFLINE=1` or `--offline`) | `FakeEmbedder` (hashed bag-of-words) | `StubLLM` (returns the prompt) | No | Tests, development, inspecting the prompt |
| Bedrock (default) | Titan Text Embeddings V2 | Claude through the Bedrock Converse API | Yes | Real answers |

In offline mode the "answer" is the full prompt sent to the LLM, not a real answer. Retrieval in offline mode matches words, not meaning.

## Setup

Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The corpus lives in `corpus/` (five flat `.md` files: `rag-basics.md`, `llm-safety.md`, `evaluation-performance.md`, `aws-infrastructure.md`, `delivery-operations.md`). That folder is listed in `.gitignore`, so a fresh clone will not have it. Copy the guides in, or point `MINIRAG_CORPUS_DIR` at any folder of `.md` files.

## Add your own documents

Put your files in `corpus/` (or in any folder and point `MINIRAG_CORPUS_DIR` at it). The format is plain Markdown:

- **Extension and location:** `.md` files directly inside the folder. Subfolders and other extensions (`.txt`, `.pdf`) are ignored.
- **Encoding:** UTF-8.
- **Headings define sections.** Split your document with `#` headings (`# Title`, `## Section`). A chunk never crosses a heading, and each answer cites `file.md#Heading`, so use clear, specific headings. Text before the first heading is cited as just `file.md`. A `#` line inside a fenced code block is not treated as a heading.
- **File names are citations.** Give each file a unique, descriptive name (`refund-policy.md`, not `notes.md`).
- **Length:** long sections are cut into 800-character chunks with 100 characters of overlap, so write one idea per section.

Example `corpus/pets.md`:

```markdown
# Pets

## Feeding
Cats eat twice a day. Dogs need walks.

## Vet
Vaccination is yearly.
```

Test it (no AWS needed). The corpus is read once at startup, so restart the API after adding files.

```bash
# In scope: the answer should cite pets.md#Feeding
PYTHONPATH=src python -m minirag.cli --offline --corpus corpus "How often do cats eat?"

# Out of scope: should print "I don't know based on the provided documents."
PYTHONPATH=src python -m minirag.cli --offline "What is the capital of Peru?"
```

In offline mode the printed "answer" is the prompt, and retrieval matches shared words, not meaning. Check that the `Sources:` list shows the section you expected. If good questions are refused, lower `--threshold` (default 0.2). The default was tuned on the bundled guides, so re-check it for your own documents.

The bundled tests (`python -m pytest`) check retrieval against the five bundled guides. If you replace them with your own files, `test_retrieval.py` will fail. To keep it working, leave the bundled guides in `corpus/` and use `MINIRAG_CORPUS_DIR` for your own folder. To test your own documents, add `(question, (file, heading))` entries to `EVAL_SET` in `tests/test_retrieval.py`.

## Run the tests

No AWS credentials needed. Tests use `FakeEmbedder` and `StubLLM`.

```bash
python -m pytest
```

## Ask a question (CLI)

Offline:

```bash
PYTHONPATH=src python -m minirag.cli --offline "Why is an IAM role better than access keys?"
PYTHONPATH=src python -m minirag.cli --offline "What is the capital of Peru?"   # refused
```

With Bedrock (see [AWS setup](#aws-setup)):

```bash
PYTHONPATH=src python -m minirag.cli "Why is an IAM role better than access keys?"
```

The CLI prints the answer, then the sources with their scores. Options: `--corpus DIR`, `-k N`, `--threshold X`.

## Ask a question (API)

```bash
PYTHONPATH=src MINIRAG_OFFLINE=1 uvicorn minirag.api:app --port 8000
# or, with Bedrock: PYTHONPATH=src uvicorn minirag.api:app --port 8000

curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Where should database passwords be stored?"}'
```

Response (the answer text and score are illustrative):

```json
{
  "answer": "...",
  "sources": [
    {"source": "delivery-operations.md", "heading": "How do you handle secrets and config?",
     "citation": "delivery-operations.md#How do you handle secrets and config?", "score": 0.41}
  ],
  "refused": false
}
```

For a refused question, `answer` is `"I don't know based on the provided documents."`, `sources` is empty, `refused` is `true`, and the LLM is never called. `GET /health` returns `{"status": "ok"}`. The corpus is ingested once at startup. A model or embedding failure returns HTTP 502 with a generic message (no provider details). Questions are limited to 2000 characters.

## Configuration

All settings are environment variables. No keys appear in code.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MINIRAG_OFFLINE` | off | `1`/`true`/`yes`/`on` uses `FakeEmbedder` + `StubLLM` |
| `MINIRAG_CORPUS_DIR` | `corpus/` next to `src/` | Folder of `*.md` files |
| `MINIRAG_TOP_K` | `3` | Chunks to retrieve |
| `MINIRAG_THRESHOLD` | `0.2` | Minimum cosine score for a chunk to be used. Depends on the embedder |
| `MINIRAG_EMBED_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Bedrock embedding model |
| `MINIRAG_LLM_MODEL_ID` | `eu.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock model (an inference profile ID) |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | embeddings: from `~/.aws/config`; LLM: `eu-north-1` | Bedrock region |

## AWS setup

1. Configure credentials through the default AWS chain (for example `aws configure`, `AWS_PROFILE=...`, SSO, or an IAM role) and check them with `aws sts get-caller-identity`.
2. In the Bedrock console, enable model access in your region for the Titan embedding model and the Claude model.
3. Set the region (`AWS_REGION`). The embedder has no built-in region default, so without it (or a region in `~/.aws/config`) boto3 raises `NoRegionError`.
4. If your region needs a different Claude inference profile, set `MINIRAG_LLM_MODEL_ID`.

## Compare fake and Bedrock retrieval

```bash
AWS_PROFILE=<your-profile> AWS_REGION=<region> python scripts/compare_recall.py
```

This runs the 3-question eval set (recall@3) and the 3 out-of-scope questions with both embedders, and prints the best score for each question so you can pick a threshold. The eval set is tiny, so treat the numbers as a smoke test, not a benchmark.

## Layout

```
corpus/                 # the 5 guides (gitignored)
src/minirag/
  chunking.py           # chunk(text, size, overlap), heading-aware
  embeddings.py         # FakeEmbedder, BedrockEmbedder
  store.py              # VectorStore: list + numpy, cosine top-k
  llm.py                # StubLLM, BedrockLLM
  rag.py                # ingest(), retrieve(), answer()
  config.py             # env-var wiring shared by API and CLI
  api.py                # FastAPI POST /ask
  cli.py                # python -m minirag.cli "question"
scripts/compare_recall.py
tests/
explain.md              # design notes and production changes
```

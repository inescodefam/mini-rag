# Design notes

This is a teaching-sized RAG pipeline. This file says what it does, what was and was not measured, and what you would change for production.

## What it does

1. **Chunk.** Each Markdown file is split on `#` headings (ignoring `#` inside code fences), then a sliding character window (800 characters, 100 overlap) runs over each section. A chunk never crosses a heading, so every chunk can be cited as `file#heading`. The bundled corpus (about 38 KB) gives 75 chunks.
2. **Embed.** An `Embedder` turns texts into L2-normalized vectors, so cosine similarity is a dot product.
3. **Store.** A Python list of chunks plus a numpy matrix. Search is brute force: one matrix-vector product, exact top-k.
4. **Retrieve and refuse.** Take the top 3 chunks and keep those scoring at least the threshold (0.2). If none do, return "I don't know based on the provided documents." and do not call the LLM.
5. **Prompt.** The system prompt says to answer only from the context, cite sources as `[file.md#Heading]`, say "I don't know" otherwise, and treat the context as untrusted data. Chunks are wrapped in `<context>` / `<document>` tags, and any such tags inside chunk text are neutralized so a document cannot close the block.
6. **Generate.** The LLM is called through a small interface. `BedrockLLM` uses the Converse API at temperature 0.

The pipeline depends on the `Embedder` and `LLM` interfaces, not on Bedrock. That is what lets the whole test suite run without AWS credentials.

## Which embeddings were used where

This is the part most easily misread, so plainly:

- **Tests and offline mode use `FakeEmbedder`.** It hashes each non-stop-word into one of 512 buckets (a bag of words). It knows nothing about meaning: "car" and "automobile" are unrelated to it, and a question only retrieves a chunk if they share words. All 84 tests, including the recall@3 and refusal tests, run on it. A passing test therefore says the plumbing works and that lexical retrieval finds the right chunk for these questions. It says nothing about semantic retrieval quality.
- **Real runs use `BedrockEmbedder`**, which calls Amazon Titan Text Embeddings V2 (1024 dimensions, normalized) and `BedrockLLM`, which calls Claude (default: Haiku 4.5 through an `eu.` inference profile).
- **`StubLLM` never answers.** It returns the prompt. In offline mode, the CLI and API print the prompt, not a model answer.

### What was measured

On the 3-question eval set (hand-written, one expected section each) and 3 out-of-scope questions:

| | Fake embedder | Titan v2 |
| --- | --- | --- |
| recall@3 | 3/3, all at rank 1 (re-run for this file) | Not recorded in the repo. Re-run `scripts/compare_recall.py` |
| Lowest best-chunk score, in-scope | 0.259 (re-run for this file) | about 0.41 (from the note in `rag.py`) |
| Highest best-chunk score, out-of-scope | 0.149 (re-run for this file) | about 0.11 (from the note in `rag.py`) |

Caveats:

- The Titan score ranges come from the comment in `src/minirag/rag.py`, written after an earlier run of `scripts/compare_recall.py`. I did not re-run Bedrock while writing this file, so treat those two numbers as recorded, not freshly verified.
- Six questions is far too few to estimate recall or pick a threshold with confidence. The eval set was also written by looking at the corpus, and the fake embedder's stop-word list was tuned until the out-of-scope questions fell below the in-scope ones. That is fitting to the test, so 3/3 is expected and weak evidence.
- Answer quality (faithfulness of the Claude output, citation correctness) was not evaluated systematically. The prompt is designed to ground and cite, but nothing here measures how often the model obeys it.

### The threshold depends on the embedder

The score scale differs per embedder, so the threshold of 0.2 is not portable. It sits in the gap for the fake embedder (0.149 to 0.259, a narrow gap). The Titan gap is wider (about 0.11 to 0.41), so 0.2 works there too and something near 0.25 would sit more centrally. If you change the embedding model, chunk size, or corpus, re-measure and set `MINIRAG_THRESHOLD`. A fixed global threshold is also a blunt tool: an on-topic question the corpus does not answer can still score above it. Refusal here means "nothing similar enough", not "the answer is absent".

## Known limitations

- **Character-based chunks.** Size is in characters, not tokens. A character window can cut a sentence in half and has no relation to the embedding model's token limit. Chunks are small (800 characters), so one answer can span several chunks.
- **One-text-per-request embedding.** Titan takes one text per call, so ingesting is 75 sequential calls on every startup. Embeddings are not cached or persisted.
- **Everything in memory.** The store is rebuilt at startup and lost on exit. Nothing is incremental.
- **No conversation memory.** Each question is independent.
- **Python version range.** Developed on Python 3.12 and tested in CI on 3.10, 3.11 and 3.12. Python 3.9 is not supported (boto3 dropped it), and 3.13 or newer needs a newer `numpy` than the pinned 2.0.2.
- **Prompt-injection defenses are partial.** Delimiting, tag neutralization, and instructions reduce the risk. They do not prevent a model from following injected text. Treat them as one layer.
- **The bundled tests are tied to the bundled corpus.** `tests/test_retrieval.py` expects the five guides in `corpus/`. If you replace them with your own files, update `EVAL_SET`.

## What I would change for production

**Retrieval quality**

- **Hybrid search.** Combine BM25 (keyword) with vector search and fuse the rankings (for example reciprocal rank fusion). Vectors handle paraphrase, BM25 handles exact terms such as error codes, names and acronyms. The fake embedder is, in effect, a weak keyword search, so this would also catch what the embeddings miss.
- **Reranking.** Retrieve 20 to 50 candidates cheaply, then rerank with a cross-encoder or a rerank model and keep the top few. Apply the refusal threshold to the reranker score, which is usually better calibrated than raw cosine.
- **Token-based, structure-aware chunking.** Count tokens with the embedding model's tokenizer, split on sentence or paragraph boundaries, keep tables and code blocks whole, and prepend the document title and heading to each chunk before embedding. Consider returning the parent section for context while matching on small chunks.
- **Query handling.** Query rewriting for follow-up questions, and optionally multiple query variants.

**Data and access**

- **A real vector database** (pgvector, OpenSearch, or a managed service) with an ANN index (HNSW/IVF), persistence, and incremental upserts and deletes. Brute force is fine up to a few thousand chunks and not beyond.
- **Metadata filters and ACLs.** Store source, owner, tenant, sensitivity and date with each chunk. Apply access-control filters inside the search query, before ranking, never after generation. Otherwise the model can quote documents the user may not read.
- **Ingestion pipeline.** Batch and parallel embedding with retries and backoff, content hashing to skip unchanged files, and an index version tied to the embedding model so vectors from two models are never mixed.
- **Caching.** Cache query embeddings and, where acceptable, full answers keyed by normalized question and index version. Use prompt caching for the long fixed system prompt.

**Safety**

- **Guardrails and prompt-injection defenses.** Keep the delimiting and instruction hierarchy, and add: input and output filtering (for example Bedrock Guardrails or a classifier), PII redaction, stripping of hidden or active content from ingested documents, least-privilege tools (this app has none, which is a real protection), and output checks that every citation refers to a chunk that was actually retrieved. Red-team with an injection test set.
- **Abuse controls on the API.** Authentication, rate limits, request logging with redaction, and timeouts on model calls. The API already hides provider errors behind a 502 and caps question length.

**Evaluation and operations**

- **Evaluation in CI.** Grow the eval set to dozens or hundreds of real questions, including unanswerable ones and near-misses. Track recall@k, MRR, refusal precision and recall, and answer faithfulness (an LLM judge spot-checked by humans). Run the retrieval part on every change to chunking, embeddings or the threshold and fail the build on regressions. Keep the offline fake-embedder tests for plumbing, but do not count them as quality evidence.
- **Threshold calibration.** Pick it from a labeled set at a chosen precision/recall trade-off, per embedder and per index, and re-run it whenever either changes.
- **Observability.** Log the question, retrieved chunk IDs and scores, whether it refused, latency per stage, and token use and cost. Add user feedback, and review refusals and low-score answers regularly.
- **Infrastructure.** Run behind an IAM role (no long-lived keys), with regional or cross-region inference configured deliberately, and quotas and budget alarms on Bedrock.

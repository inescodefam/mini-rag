# mini-rag

A small Retrieval-Augmented Generation (RAG) app: ingest documents, retrieve relevant chunks, and have an LLM answer using only that context. This repo's docs explain the concepts and the AWS deployment choices behind it.

## Guide

| # | Guide | Covers |
|---|-------|--------|
| 1 | [README.md](README.md) (this file) | Tokens, RAG for PMs and engineers, chunking, embeddings, RAG vs fine-tuning |
| 2 | [docs/llm-behavior-and-safety](docs/llm-behavior-and-safety/README.md) | Hallucinations, function calling, agents, prompt injection |
| 3 | [docs/evaluation-and-performance](docs/evaluation-and-performance/README.md) | Evaluating RAG, latency and cost, streaming to React |
| 4 | [docs/aws-infrastructure](docs/aws-infrastructure/README.md) | Bedrock vs OpenAI, ECS/EKS/Lambda, IAM, networking |
| 5 | [docs/delivery-and-operations](docs/delivery-and-operations/README.md) | CI/CD, secrets and config, multi-AZ, observability |

---

## 1. What's a token? Why does it matter for cost and design?

A **token** is the unit of text an LLM actually reads and writes. It is a chunk produced by a tokenizer (for example BPE): a whole short word, part of a long word, punctuation, or whitespace. Rough rule of thumb for English: **1 token is about 4 characters, or about 0.75 words**. Other languages and code usually need more tokens per word.

Why it matters:

- **Cost.** Providers bill per token, with separate (usually higher) prices for output tokens than input tokens. In RAG, the retrieved context is input, so every extra chunk you stuff into the prompt costs money on every request.
- **Context window.** Every model has a maximum number of tokens (prompt + answer combined). System prompt, chat history, retrieved chunks, and the answer must all fit.
- **Latency.** Input tokens add to time-to-first-token. Output tokens are generated one at a time, so a long answer is slow.
- **Quality.** More context is not always better. Models can miss information buried in the middle of a long prompt, and irrelevant chunks add noise.

Design consequences: set a token budget (for example 500 system + 3,000 context + 1,000 answer), count tokens with the model's tokenizer rather than characters, cap `max_tokens`, and trim chat history.

## 2. Explain RAG

**To a product manager.**
An LLM only knows what it was trained on. It doesn't know your company's documents, and it sometimes makes things up. RAG fixes this by giving the model an "open-book exam": when a user asks a question, we first search our own documents for the most relevant passages, then hand those passages to the model and tell it to answer from them. The benefits are:

- Answers reflect up-to-date, private company knowledge.
- We can show citations, which builds user trust.
- Updating knowledge means updating documents, not retraining a model.
- Access control is possible, because we only retrieve what the user is allowed to see.

The trade-offs: answer quality is only as good as the search and the documents, and each question costs a search plus an LLM call.

**To an engineer.**
RAG has two pipelines.

*Ingestion (offline):*
1. Load documents (PDF, HTML, Markdown, and so on) and extract text.
2. Split into chunks (with overlap) and attach metadata (source, page, ACL, timestamp).
3. Embed each chunk with an embedding model into a vector.
4. Store vectors and metadata in a vector index (pgvector, OpenSearch, and so on).

*Query (online):*
1. Optionally rewrite the user query (use chat history to resolve "what about the second one?").
2. Embed the query with the same embedding model.
3. Retrieve top-k nearest chunks, optionally combining with keyword search (BM25) and filtering by metadata.
4. Optionally rerank with a cross-encoder to keep the best few.
5. Build a prompt: instructions + retrieved chunks (clearly delimited) + question.
6. Call the LLM, stream the answer, and return citations.

```
 Docs -> chunk -> embed -> [vector DB]
                               ^
 Question -> embed -> top-k search -> (rerank) -> prompt -> LLM -> answer + citations
```

## 3. How do you choose chunk size? What goes wrong if it's too large or small?

Chunks are the unit of retrieval, so the chunk size sets the trade-off between **precision** and **context**.

| | Too small (< ~100 tokens) | Too large (> ~1,000 tokens) |
|---|---|---|
| Retrieval | Precise matches, but fragments lose meaning ("It was approved in 2021." has no subject) | One vector blurs several topics, so similarity gets diluted and retrieval gets worse |
| Prompt | Need many chunks to cover an answer | A few chunks eat the token budget and add noise |
| Cost and latency | More chunks to embed, store and search | More input tokens per request |
| Answers | Missing context, so wrong or partial answers | Model distracted by irrelevant text, or the relevant bit is lost in the middle |

How to choose:

- **Start around 300-500 tokens with 10-20% overlap**, then tune using your evaluation set (see the [evaluation guide](docs/evaluation-and-performance/README.md)). Measure retrieval hit rate and answer quality at a few sizes rather than guessing.
- **Match the document's structure.** Split on headings, paragraphs, and sentences (recursive splitting) before falling back to fixed length. Keep tables and code blocks intact.
- **Check the embedding model's input limit.** Text beyond it is silently truncated by some models.
- **Consider small-to-big retrieval.** Embed small chunks for precision, but send the surrounding parent section to the LLM.
- Add context to each chunk (document title, section heading) so it stands alone.

## 4. What are embeddings? How is "similarity" computed?

An **embedding** is a list of numbers (a vector, commonly 384 to 3,072 dimensions) produced by a model so that texts with similar *meaning* land close together in vector space. "How do I reset my password?" and "I forgot my login credentials" end up near each other even without shared words.

**Similarity** is a distance measure between two vectors:

- **Cosine similarity** (most common): the cosine of the angle between the vectors. Range -1 to 1; higher is more similar. It ignores vector length.

  `cos(a, b) = (a · b) / (‖a‖ ‖b‖)`

- **Dot product**: `a · b`. Equivalent to cosine if vectors are normalized to length 1 (many embedding APIs return normalized vectors), and slightly faster.
- **Euclidean (L2) distance**: straight-line distance; lower is more similar.

At scale, exact search over millions of vectors is slow, so vector databases use **approximate nearest neighbor** indexes (HNSW, IVF) which trade a little recall for much more speed.

Gotchas:

- Query and documents must use the **same embedding model** (and version). Changing models means re-embedding everything.
- Similarity means "semantically related", not "correct". Exact IDs, names, and error codes are often better served by keyword search, which is why hybrid search is common.

## 5. RAG vs fine-tuning. When each?

| | RAG | Fine-tuning |
|---|---|---|
| Teaches the model | Facts at query time (via context) | Behavior, style, format, and skills (via weights) |
| Knowledge freshness | Update documents, effective immediately | Needs retraining to change |
| Citations and traceability | Yes, you know which chunk was used | No |
| Access control | Filter at retrieval time | Hard; knowledge is baked into the weights |
| Up-front cost | Pipeline and index | Data curation and training runs |
| Per-request cost | Longer prompts | Can shorten prompts; may use a smaller model |
| Hallucination on facts | Reduced when grounded | Not reliably reduced |

**Use RAG when** the answer depends on private, large, or frequently changing knowledge, or you need citations and per-user permissions.

**Use fine-tuning when** you need a consistent tone, a strict output format, a specialized task (classification, extraction), or lower latency and cost by distilling into a smaller model.

**Use both** when you want, for example, a fine-tuned model that reliably follows your answer format while RAG supplies the facts. Try prompt engineering first, then RAG, then fine-tuning, since each step costs more.

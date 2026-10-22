# Evaluation and Performance

Back to the [main README](../../README.md).

## How do you evaluate a RAG system before shipping?

Evaluate **retrieval** and **generation** separately, so you can tell which half is failing.

### 1. Build a golden dataset
- 50-200 realistic questions (from support tickets, search logs, domain experts), each with the expected answer and the source document(s)/chunk(s) that contain it.
- Include **edge cases**: multi-document questions, ambiguous questions, questions with **no answer** in the corpus, adversarial and prompt-injection attempts, and different phrasings.
- Version it in the repo and grow it with every production bug.

### 2. Measure retrieval
| Metric | Question it answers |
|--------|--------------------|
| Recall@k / hit rate | Is a relevant chunk in the top k? |
| MRR / nDCG | How high is it ranked? |
| Precision@k | How much of the top k is relevant? |

If recall@k is low, no prompt will save you. Fix chunking, embeddings, hybrid search, or reranking first.

### 3. Measure generation
- **Faithfulness / groundedness**: is every claim supported by the retrieved context?
- **Answer relevance**: does it answer the question?
- **Correctness** against the reference answer.
- **Citation accuracy**: do the cited sources really support the statement?
- **Correct refusals**: does it say "I don't know" for unanswerable questions?
- Use **LLM-as-judge** (with a clear rubric and a stronger model) for scale, **calibrated against human-labeled samples**. Frameworks such as Ragas, TruLens, or DeepEval can help, but spot-check by hand.

### 4. Non-functional and safety checks
- Latency (p50/p95 time-to-first-token and total), cost per request, token usage.
- Security: injection tests, PII leakage, access-control tests (user A must never retrieve user B's documents).
- Robustness: empty input, very long input, other languages, timeouts and provider errors.

### 5. Make it a release gate
- Run the evaluation suite in CI on every change to prompts, models, chunking, or retrieval settings, and fail the build on regressions beyond a threshold.
- Compare variants (A/B offline) before changing anything.
- After launch: shadow or canary rollout, thumbs up/down feedback, sampled human review, and drift monitoring.

## How do you reduce LLM latency and cost in production?

Latency and cost are mostly driven by **tokens**, so reduce tokens first.

### Reduce tokens
- Retrieve fewer, better chunks (rerank, then keep the top 3-5 instead of 20).
- Compress or trim chat history (summarize older turns).
- Keep the system prompt short; cap `max_tokens` and ask for concise answers.

### Choose the right model
- **Route by difficulty**: a small, fast model for simple questions, classification, and query rewriting; a larger one only when needed.
- Consider a distilled or fine-tuned small model for narrow tasks.

### Cache
- **Prompt caching** (supported by Anthropic, OpenAI, and Bedrock for some models) makes a long, stable prefix (system prompt, tool definitions) cheaper and faster.
- **Response cache** for repeated identical questions; **semantic cache** for near-duplicates (check carefully for correctness and per-user data).
- Cache embeddings for repeated queries and for unchanged documents during ingestion.

### Parallelize and shorten the pipeline
- Run independent steps concurrently (vector search and keyword search; multiple tool calls).
- Skip unneeded steps (don't rewrite queries for a first-turn question).
- Use a fast vector index (HNSW) and keep it close to the app (same region and AZ where possible).

### Improve perceived latency
- **Stream the response** so the user sees text within hundreds of milliseconds (see below).
- Show progress states and sources early.

### Operational controls
- Batch APIs for offline jobs (ingestion, evaluation) at lower prices.
- Provisioned throughput or higher rate-limit tiers if you hit throttling; retry with exponential backoff and jitter.
- Set per-user and per-tenant budgets, rate limits, and alarms on token spend.
- Measure it: track tokens in and out, cost per request, and p95 latency per pipeline stage.

## How do you stream an LLM answer to a React UI?

**Overall path:** LLM provider (streaming API) -> your backend (reads chunks as they arrive) -> HTTP response (Server-Sent Events or chunked stream) -> React (reads the stream and appends to state).

### Backend (Node/Express example)

```ts
app.post("/api/chat", async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  const controller = new AbortController();
  req.on("close", () => controller.abort()); // user left: stop paying for tokens

  try {
    for await (const token of streamLlmAnswer(req.body.question, controller.signal)) {
      res.write(`data: ${JSON.stringify({ token })}\n\n`);
    }
    res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
  } catch (err) {
    res.write(`data: ${JSON.stringify({ error: "generation_failed" })}\n\n`);
  } finally {
    res.end();
  }
});
```

`streamLlmAnswer` wraps the provider's streaming call (for example Bedrock `ConverseStream` or OpenAI `stream: true`).

### React client
`EventSource` only supports GET, so for a POST with a body use `fetch` and read the stream:

```tsx
async function ask(question: string, onToken: (t: string) => void, signal: AbortSignal) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error("Request failed");

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? ""; // keep the incomplete tail
    for (const evt of events) {
      if (!evt.startsWith("data: ")) continue;
      const msg = JSON.parse(evt.slice(6));
      if (msg.token) onToken(msg.token);
    }
  }
}

function Chat() {
  const [answer, setAnswer] = useState("");
  const abortRef = useRef<AbortController>();

  const send = (q: string) => {
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    setAnswer("");
    ask(q, (t) => setAnswer((a) => a + t), abortRef.current.signal).catch(console.error);
  };
  const stop = () => abortRef.current?.abort();
  // render answer, a "Stop" button, and error state
}
```

Or use a library such as the Vercel AI SDK (`useChat`), which handles this protocol for you.

### Gotchas
- **Proxy buffering**: nginx, CDNs, and some gateways buffer responses and break streaming. Disable buffering (`X-Accel-Buffering: no`) and check your ALB/CloudFront/API Gateway setup. ALB has an idle timeout (60s by default), so send keep-alive comments or events for slow generations.
- **Cancellation**: wire the Stop button and browser disconnects to the provider call so you stop paying for tokens.
- **Errors mid-stream**: the HTTP status is already 200, so send an error event in the stream and show it in the UI.
- **Rendering**: batch state updates (for example with `requestAnimationFrame`) if streaming markdown causes re-render jank, and sanitize rendered markdown.
- **Citations and metadata**: send sources as separate event types (`{"type":"sources", ...}`) before or after the tokens.
- **Guardrails**: output moderation on a stream is harder; decide whether to buffer in chunks or run checks after the fact.

"""Command line interface: ``python -m minirag.cli "question"``.

Prints the answer and its sources. Use ``--offline`` (or ``MINIRAG_OFFLINE=1``)
to run with the fake embedder and stub LLM, without AWS. In offline mode the
"answer" is the prompt itself, which is useful for inspecting it.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .config import build_pipeline
from .rag import answer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minirag", description="Ask the mini-RAG corpus a question.")
    parser.add_argument("question", help="the question to ask")
    parser.add_argument("--offline", action="store_true", help="use FakeEmbedder + StubLLM (no AWS)")
    parser.add_argument("--corpus", help="corpus directory (default: MINIRAG_CORPUS_DIR or ./corpus)")
    parser.add_argument("-k", type=int, help="number of chunks to retrieve")
    parser.add_argument("--threshold", type=float, help="minimum cosine score to keep a chunk")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.question.strip():
        print("error: question must not be blank", file=sys.stderr)
        return 2
    try:
        pipeline = build_pipeline(
            offline=True if args.offline else None,
            corpus_dir=args.corpus,
            k=args.k,
            threshold=args.threshold,
        )
        result = answer(
            args.question,
            pipeline.embedder,
            pipeline.store,
            pipeline.llm,
            k=pipeline.k,
            threshold=pipeline.threshold,
        )
    except Exception as exc:
        print("error: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1

    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for s in result["sources"]:
            print("  - %s (score %.3f)" % (s["citation"], s["score"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

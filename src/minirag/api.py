"""FastAPI app: ``POST /ask``.

Run with ``uvicorn minirag.api:app`` (from ``src/``, or with ``PYTHONPATH=src``).
The corpus is ingested once at startup. Set ``MINIRAG_OFFLINE=1`` to run with
the fake embedder and stub LLM (no AWS).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .config import Pipeline, build_pipeline
from .rag import answer

MAX_QUESTION_CHARS = 2000


class AskRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class Source(BaseModel):
    source: str
    heading: str
    citation: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    refused: bool


def create_app(pipeline: Optional[Pipeline] = None) -> FastAPI:
    """Create the app. Pass a ``pipeline`` to inject components (tests);
    otherwise one is built from the environment at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline if pipeline is not None else build_pipeline()
        yield

    app = FastAPI(title="mini-rag", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # Plain ``def`` so FastAPI runs the blocking Bedrock calls in a worker thread.
    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        p: Pipeline = app.state.pipeline
        try:
            result = answer(
                request.question, p.embedder, p.store, p.llm, k=p.k, threshold=p.threshold
            )
        except Exception as exc:  # embedding / model call failed
            # Do not leak provider error details to the client.
            raise HTTPException(status_code=502, detail="upstream model error") from exc
        return AskResponse(**result)

    return app


app = create_app()

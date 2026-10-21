"""Heading-aware text chunking.

``chunk`` first splits Markdown text into sections on ``#`` heading lines, then
slides a fixed-size character window (with overlap) over each section. Chunks
never cross a heading boundary, so every chunk can be cited as
``source#heading``.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

Chunk = Dict[str, str]

_HEADING_RE = re.compile(r"^ {0,3}#{1,6}[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(```|~~~)")


def _split_sections(text: str) -> List[Tuple[str, str]]:
    """Return ``(heading, body)`` pairs. Text before any heading has heading ``""``.

    Lines inside fenced code blocks are never treated as headings, because
    shell comments such as ``# install deps`` look like Markdown headings.
    """
    sections: List[Tuple[str, str]] = []
    heading = ""
    body: List[str] = []
    fence = None  # the fence marker that opened the current code block

    for line in text.splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            body.append(line)
            continue

        heading_match = None if fence else _HEADING_RE.match(line)
        if heading_match:
            sections.append((heading, "\n".join(body)))
            heading = heading_match.group(1).strip()
            body = []
        else:
            body.append(line)

    sections.append((heading, "\n".join(body)))
    return sections


def _windows(text: str, size: int, overlap: int) -> List[str]:
    """Slide a window of ``size`` characters, advancing by ``size - overlap``."""
    step = size - overlap
    pieces: List[str] = []
    start = 0
    while True:
        end = min(start + size, len(text))
        pieces.append(text[start:end])
        if end == len(text):
            return pieces
        start += step


def chunk(text: str, size: int, overlap: int, source: str = "") -> List[Chunk]:
    """Split ``text`` into chunks of at most ``size`` characters.

    Consecutive chunks within a section share ``overlap`` characters. Returns a
    list of ``{"source", "heading", "text"}`` dicts. Empty or whitespace-only
    sections (e.g. a heading directly followed by another heading) produce no
    chunks.

    Raises ``ValueError`` if ``size <= 0``, ``overlap < 0`` or
    ``overlap >= size`` (the window would never advance).
    """
    if size <= 0:
        raise ValueError("size must be positive, got %d" % size)
    if overlap < 0:
        raise ValueError("overlap must be non-negative, got %d" % overlap)
    if overlap >= size:
        raise ValueError(
            "overlap (%d) must be smaller than size (%d)" % (overlap, size)
        )

    chunks: List[Chunk] = []
    for heading, body in _split_sections(text):
        body = body.strip()
        if not body:
            continue
        for piece in _windows(body, size, overlap):
            if piece.strip():
                chunks.append({"source": source, "heading": heading, "text": piece})
    return chunks

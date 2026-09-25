"""Exact, server-owned citation passages from this question's retrieved chunks."""

import re
from dataclasses import dataclass

from app.retrieval import Chunk

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    chunk: Chunk


def build_passages(chunks: list[Chunk]) -> list[Passage]:
    passages = []
    for chunk in chunks:
        start, number = 0, 0
        while start < len(chunk.text):
            end = min(start + 600, len(chunk.text))
            if end < len(chunk.text):
                # Prefer sentence ends, then word boundaries. The 80-character
                # minimum bounds the ID count even for adversarial tiny sentences.
                breaks = list(_SENTENCE_BREAK.finditer(chunk.text, start + 80, end))
                boundary = breaks[-1].start() if breaks else chunk.text.rfind(" ", start + 80, end)
                if boundary != -1:
                    end = boundary
            text = chunk.text[start:end].strip()
            if text:
                passages.append(Passage(f"{chunk.id}:p{number}", text, chunk))
                number += 1
            start = end
    return passages

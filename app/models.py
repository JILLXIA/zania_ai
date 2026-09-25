import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class AppError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class Source(BaseModel):
    text: str
    source_type: Literal["text", "image"] = "text"
    page: int | None = None
    source_path: str | None = None
    context: str = ""


class Visual(BaseModel):
    page: int
    image_path: str
    width: int
    height: int
    context: str
    digest: str


class ParsedDocument(BaseModel):
    sources: list[Source]
    visuals: list[Visual] = Field(default_factory=list)


class Citation(BaseModel):
    source_type: Literal["text", "image"]
    page: int | None = None
    source_path: str | None = None
    excerpt: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class Result(BaseModel):
    question: str
    answer: str
    status: Literal["answered", "partial", "not_found", "error"]
    citations: list[Citation] = Field(default_factory=list)
    error: ErrorDetail | None = None


class QAResponse(BaseModel):
    request_id: str
    results: list[Result]
    warnings: list[str] = Field(default_factory=list)


# Provider schemas have no optional/default fields: strict structured output requires every key.
class AnswerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "partial", "not_found"]
    answer: str
    missing_details: list[str]
    evidence_ids: list[str]


class VisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["content", "decorative", "unreadable"]
    observations: list[str]

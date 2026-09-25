"""Local embeddings, chunking, and request-local FAISS + BM25 hybrid retrieval."""

import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from app.config import EMBEDDING_MODEL, Settings
from app.logging import event
from app.models import AppError, Source

MAX_CONTEXT_BYTES = 16_000
MAX_EVIDENCE_CHUNKS = 8


class LocalEmbeddings:
    def __init__(self, model_directory: str):
        import onnxruntime
        from fastembed import TextEmbedding
        from tokenizers import Tokenizer

        directory = Path(model_directory)
        if not (directory / "tokenizer.json").exists():
            raise RuntimeError("Run python scripts/prepare_model.py first.")
        onnxruntime.disable_telemetry_events()
        self.model = TextEmbedding(
            EMBEDDING_MODEL,
            specific_model_path=str(directory),
            local_files_only=True,
            threads=2,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self.model.embed(texts, batch_size=32)), dtype="float32")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        # Long compound questions keep every part in retrieval instead of silent truncation.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=0,
            length_function=self.token_count,
        )
        parts = [splitter.split_text(text) for text in texts]
        vectors = list(self.model.query_embed([p for group in parts for p in group]))
        result, offset = [], 0
        for group in parts:
            result.append(np.mean(vectors[offset : offset + len(group)], axis=0))
            offset += len(group)
        return np.asarray(result, dtype="float32")


@dataclass
class Chunk:
    id: str
    text: str
    source: Source
    context: str = ""


def lexical_tokens(text: str) -> list[str]:
    # Keep identifiers such as CC6.1 and TLS-1.2 intact; retain negation words.
    return re.findall(r"\w+(?:[.-]\w+)*", text.casefold())


def reciprocal_rank_fusion(*rankings: list[int]) -> list[int]:
    """Equal-weight RRF (k=60); raw cosine and BM25 scores are not comparable."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (60 + rank)
    # Stable sorting preserves dense rank order when fused scores tie.
    return sorted(scores, key=lambda chunk_id: -scores[chunk_id])


def build_chunks(sources: list[Source], embeddings, settings: Settings) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=40,
        length_function=lambda text: max(embeddings.token_count(text), len(text.encode()) // 4),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for source in sources:
        context = source.context.encode()[:320].decode("utf-8", errors="ignore")
        while context and embeddings.token_count(context) > 80:
            context = context[: len(context) // 2]
        for text in splitter.split_text(source.text):
            chunks.append(Chunk(id=f"c{len(chunks)}", text=text, source=source, context=context))
            if len(chunks) > settings.max_chunks:
                raise AppError(413, "too_many_chunks", "The document exceeds the chunk limit.")
    return chunks


def retrieve_all(sources: list[Source], questions: list[str], embeddings, settings: Settings):
    chunks = build_chunks(sources, embeddings, settings)
    if not chunks:
        return [[] for _ in questions]
    texts = []
    for chunk in chunks:
        context = chunk.context
        # Added context helps retrieval but is never accepted as a quote from this chunk.
        texts.append(f"{context}\n{chunk.text}" if context else chunk.text)
    vectors = embeddings.embed_documents(texts)
    queries = embeddings.embed_queries(questions)
    faiss.normalize_L2(vectors)
    faiss.normalize_L2(queries)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    event("vector_index", index_type="IndexFlatIP", vectors=index.ntotal, dimensions=index.d)
    _, neighbors = index.search(queries, min(12, len(chunks)))
    tokens = [lexical_tokens(text) for text in texts]
    # An all-punctuation document has no vocabulary and cannot build a BM25 index.
    bm25 = BM25Okapi(tokens) if any(tokens) else None
    result = []
    for question_index, (question, ids) in enumerate(zip(questions, neighbors, strict=True)):
        selected = maximal_marginal_relevance(
            queries[question_index], vectors[ids], lambda_mult=0.7, k=6
        )
        dense_ids = [int(ids[i]) for i in selected]
        lexical_ids = []
        if bm25 is not None:
            scores = bm25.get_scores(lexical_tokens(question))
            # Do not give RRF credit to zero-score ties or nonpositive BM25 matches.
            lexical_ids = [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] > 0][:12]
        # Small sources can be returned whole. Merge their chunk hits before RRF,
        # retaining the best dense hit (or lexical hit for a lexical-only source).
        source_rows: dict[int, int] = {}
        rankings = [
            [
                source_rows.setdefault(id(chunks[i].source), i)
                if len(chunks[i].source.text.encode()) <= 5000
                else i
                for i in candidates
            ]
            for candidates in (dense_ids, lexical_ids)
        ]
        ranked_ids = reciprocal_rank_fusion(*rankings)
        context, seen, size = [], set(), 0
        for chunk_id in ranked_ids:
            chunk = chunks[chunk_id]
            # A nearby sentence can qualify an answer (e.g. an incident-notification time).
            # Expand only if it fits; otherwise keep the actual matching chunk.
            available = MAX_CONTEXT_BYTES - size - len(chunk.context.encode("utf-8")) - 120
            if len(chunk.source.text.encode()) <= min(5000, available):
                chunk = Chunk(chunk.id, chunk.source.text, chunk.source, chunk.context)
            if chunk.text in seen:
                continue
            # Bound context without downloading another tokenizer (~4K English tokens).
            cost = len(chunk.text.encode("utf-8")) + len(chunk.context.encode("utf-8")) + 120
            if size + cost > MAX_CONTEXT_BYTES:
                continue
            context.append(chunk)
            seen.add(chunk.text)
            size += cost
            if len(context) == MAX_EVIDENCE_CHUNKS:
                break
        event(
            "hybrid_retrieval",
            question_index=question_index,
            dense_candidates=len(ids),
            dense_selected=len(dense_ids),
            lexical_candidates=len(lexical_ids),
            fused_candidates=len(ranked_ids),
            selected_chunks=len(context),
        )
        result.append(context)
    return result

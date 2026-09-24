"""Local embeddings, LangChain splitting, and one FAISS index per uploaded document."""

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import EMBEDDING_MODEL, Settings
from app.models import AppError, Source


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
    _, neighbors = index.search(queries, min(12, len(chunks)))
    result = []
    for query, ids in zip(queries, neighbors, strict=True):
        selected = maximal_marginal_relevance(query, vectors[ids], lambda_mult=0.7, k=6)
        context, seen, size = [], set(), 0
        for rank in selected:
            chunk = chunks[int(ids[rank])]
            # A nearby sentence can qualify an answer (e.g. an incident-notification time).
            # Include the full same-page/record source when small enough, never another source.
            if len(chunk.source.text.encode()) <= 5000:
                chunk = Chunk(chunk.id, chunk.source.text, chunk.source, chunk.context)
            if chunk.text in seen:
                continue
            # Bound context without downloading another tokenizer (~3K English tokens).
            cost = len(chunk.text.encode("utf-8")) + len(chunk.context.encode("utf-8")) + 120
            if size + cost > 12000:
                continue
            context.append(chunk)
            seen.add(chunk.text)
            size += cost
        result.append(context)
    return result

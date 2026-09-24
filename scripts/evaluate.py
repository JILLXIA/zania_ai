"""Opt-in sample evaluation. Retrieval is local; --live enables billable model calls."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from app.config import Settings
from app.ingestion import parse_document, questions_from_bytes
from app.logging import configure_logging
from app.provider import OpenAIProvider
from app.retrieval import LocalEmbeddings, retrieve_all
from app.service import QAService


async def evaluate(args):
    settings = Settings()
    questions = questions_from_bytes(args.questions.read_bytes(), settings)
    if args.document.suffix.lower() not in {".pdf", ".json"}:
        raise ValueError("Use one PDF or JSON document, not the CSV export.")
    if args.document.stat().st_size > settings.max_document_bytes:
        raise ValueError("Document exceeds the configured size limit.")
    if args.live and not settings.openai_api_key.get_secret_value():
        raise ValueError("--live requires OPENAI_API_KEY; no provider call was made.")
    model = await asyncio.to_thread(LocalEmbeddings, settings.model_cache)
    start = time.monotonic()
    with TemporaryDirectory(prefix="zania-eval-") as directory:
        path = Path(directory) / f"document{args.document.suffix.lower()}"
        path.write_bytes(args.document.read_bytes())
        if args.live:
            configure_logging()
            provider = OpenAIProvider(settings)
            try:
                service = QAService(settings, model, provider)
                async with asyncio.timeout(settings.processing_timeout):
                    response, status = await service.run(questions, path, path.suffix[1:], "eval")
                output = {"http_status": status, **response.model_dump(exclude_none=True)}
            finally:
                await provider.close()
        else:
            document = await parse_document(path, path.suffix[1:], settings)
            contexts = await asyncio.to_thread(
                retrieve_all, document.sources, questions, model, settings
            )
            output = {
                "mode": "local text retrieval only; images are NOT analyzed without --live",
                "visual_candidate_pages": [v.page for v in document.visuals],
                "results": [
                    {
                        "question": q,
                        "evidence": [
                            {
                                "chunk_id": c.id,
                                "page": c.source.page,
                                "source_path": c.source.source_path,
                                "text": c.text,
                            }
                            for c in chunks
                        ],
                    }
                    for q, chunks in zip(questions, contexts, strict=True)
                ],
            }
    output["elapsed_seconds"] = round(time.monotonic() - start, 2)
    # Explicit evaluation output contains source text; never send it to shared production logs.
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("--questions", type=Path, default=Path("examples/questions.json"))
    parser.add_argument("--live", action="store_true", help="Enable paid gpt-4o-mini vision and QA")
    asyncio.run(evaluate(parser.parse_args()))

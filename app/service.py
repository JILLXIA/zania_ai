import asyncio
from functools import partial
from pathlib import Path

from app.config import Settings
from app.ingestion import parse_document, vision_tokens
from app.logging import event
from app.models import (
    AnswerOutput,
    AppError,
    Citation,
    ErrorDetail,
    QAResponse,
    Result,
    Source,
    normalize,
)
from app.retrieval import retrieve_all


def grounded_result(question, output: AnswerOutput, chunks) -> Result:
    if output.status == "not_found":
        if output.evidence or output.missing_details:
            raise AppError(502, "invalid_answer", "Inconsistent missing-evidence response.")
        return Result(question=question, answer="Not found in document", status="not_found")
    if not output.answer.strip() or not output.evidence:
        raise AppError(502, "invalid_answer", "A factual answer must include evidence.")
    if (output.status == "partial") != bool(output.missing_details):
        raise AppError(502, "invalid_answer", "Partial-answer details are inconsistent.")
    lookup, citations = {c.id: c for c in chunks}, []
    for evidence in output.evidence:
        chunk = lookup.get(evidence.chunk_id)
        quote = normalize(evidence.excerpt)
        if chunk is None or not quote or quote not in normalize(chunk.text):
            raise AppError(502, "invalid_citation", "The answer contains an unverified citation.")
        citation = Citation(
            source_type=chunk.source.source_type,
            page=chunk.source.page,
            source_path=chunk.source.source_path,
            excerpt=quote,
        )
        if citation not in citations:
            citations.append(citation)
    answer = output.answer.strip()
    if output.missing_details:
        missing = "; ".join(s.strip() for s in output.missing_details if s.strip())
        if not missing:
            raise AppError(502, "invalid_answer", "Missing details must be named.")
        answer += f"\n\nNot found in document: {missing}."
    return Result(question=question, answer=answer, status=output.status, citations=citations)


def failed_result(question: str, error: AppError) -> Result:
    return Result(
        question=question,
        answer=error.message,
        status="error",
        error=ErrorDetail(code=error.code, message=error.message),
    )


class QAService:
    def __init__(self, settings: Settings, embeddings, provider, parser=parse_document):
        self.settings, self.embeddings, self.provider, self.parser = (
            settings,
            embeddings,
            provider,
            parser,
        )
        self.active_requests = 0
        self.embedding_lock = asyncio.Lock()

    async def run_cpu(self, function, *args):
        # Cancellation must not release the model while its worker thread is still using it.
        async with self.embedding_lock:
            task = asyncio.create_task(asyncio.to_thread(partial(function, *args)))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise

    async def run(self, questions: list[str], path: Path, kind: str, request_id: str):
        document = await self.parser(path, kind, self.settings)
        warnings = []
        if document.visuals:
            unique = {}
            for visual in document.visuals:
                unique.setdefault((visual.digest, visual.context), visual)
            estimate = sum(
                vision_tokens(v.width, v.height) + len(v.context.encode()) + 1000
                for v in unique.values()
            )
            if estimate > self.settings.vision_token_budget:
                raise AppError(
                    413, "vision_budget_exceeded", "Visual content exceeds the token budget."
                )
            budget = [self.settings.vision_token_budget]
            semaphore = asyncio.Semaphore(2)
            analyzed = {}

            async def analyze(key, visual):
                async with semaphore:
                    output = await self.provider.analyze_image(visual, budget)
                if output.kind == "unreadable":
                    raise AppError(
                        422,
                        "unreadable_image",
                        f"Visual content on page {visual.page} is illegible.",
                    )
                if (output.kind == "content") != bool(output.observations):
                    raise AppError(
                        502, "invalid_visual_output", "Inconsistent visual observations."
                    )
                analyzed[key] = output

            tasks = [asyncio.create_task(analyze(key, v)) for key, v in unique.items()]
            try:
                async with asyncio.timeout(self.settings.vision_timeout):
                    await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            for visual in document.visuals:
                output = analyzed[(visual.digest, visual.context)]
                if output.kind == "content":
                    document.sources.append(
                        Source(
                            text=normalize(" ".join(output.observations)),
                            source_type="image",
                            page=visual.page,
                        )
                    )
            warnings.append(
                "Image citations contain model-extracted observations; check the original page."
            )
            event("visual_ingestion", candidates=len(document.visuals), calls=len(unique))
        if not any(source.text.strip() for source in document.sources):
            raise AppError(422, "empty_document", "No usable document evidence was found.")
        if sum(len(s.text) for s in document.sources) > self.settings.max_text_chars:
            raise AppError(413, "document_too_large", "Combined evidence exceeds the text limit.")
        unique_questions = list(dict.fromkeys(q.strip() for q in questions))
        contexts = await self.run_cpu(
            retrieve_all,
            document.sources,
            unique_questions,
            self.embeddings,
            self.settings,
        )
        event(
            "retrieval",
            sources=len(document.sources),
            questions=len(questions),
            unique_questions=len(unique_questions),
        )
        semaphore = asyncio.Semaphore(self.settings.calls_per_request)
        results, statuses = {}, {}

        async def answer(question, context):
            try:
                if not context:
                    result = Result(
                        question=question, answer="Not found in document", status="not_found"
                    )
                else:
                    async with semaphore:
                        output = await self.provider.answer(question, context)
                    result = grounded_result(question, output, context)
                results[question] = result
            except AppError as exc:
                results[question] = failed_result(question, exc)
                statuses[question] = exc.status

        tasks = [
            asyncio.create_task(answer(q, c))
            for q, c in zip(unique_questions, contexts, strict=True)
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            # Preserve completed answers; timed-out questions still get a result entry.
            if not results:
                raise
            for question in unique_questions:
                if question not in results:
                    results[question] = failed_result(
                        question,
                        AppError(504, "processing_timeout", "Question processing timed out."),
                    )
                    statuses[question] = 504
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        expanded = [results[q.strip()].model_copy(update={"question": q}) for q in questions]
        status = 200
        if all(result.status == "error" for result in expanded):
            status = 504 if 504 in statuses.values() else 503 if 503 in statuses.values() else 502
        return QAResponse(request_id=request_id, results=expanded, warnings=warnings), status

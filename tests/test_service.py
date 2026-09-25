import asyncio
import threading

import pytest

from app.evidence import build_passages
from app.models import AnswerOutput, AppError, ParsedDocument, Source, Visual
from app.retrieval import Chunk, build_chunks, retrieve_all
from app.service import grounded_result
from tests.conftest import FakeEmbeddings


@pytest.mark.parametrize("evidence_id", ["missing", "c0", " "])
def test_unavailable_citation_ids_are_errors(evidence_id):
    chunk = Chunk("c0", "GCP", Source(text="GCP", page=12))
    output = AnswerOutput(
        status="answered",
        answer="Claim",
        missing_details=[],
        evidence_ids=[evidence_id],
    )
    with pytest.raises(AppError) as error:
        grounded_result("Question?", output, build_passages([chunk]))
    assert error.value.code == "invalid_citation"


def test_image_citation_modality_and_location_are_server_owned():
    chunk = Chunk("c0", "Redis in GCP", Source(text="Redis in GCP", page=15, source_type="image"))
    output = AnswerOutput(
        status="answered",
        answer="Redis",
        missing_details=[],
        evidence_ids=["c0:p0"],
    )
    result = grounded_result("Components?", output, build_passages([chunk]))
    assert result.citations[0].source_type == "image"
    assert result.citations[0].page == 15


@pytest.mark.parametrize(
    "status,missing,evidence_ids",
    [
        ("answered", [], []),
        ("partial", [], []),
        ("answered", ["region"], ["c0:p0"]),
        ("not_found", [], ["c0:p0"]),
    ],
)
def test_inconsistent_answer_fields(status, missing, evidence_ids):
    output = AnswerOutput(
        status=status, answer="Claim", missing_details=missing, evidence_ids=evidence_ids
    )
    with pytest.raises(AppError):
        grounded_result(
            "Question?", output, build_passages([Chunk("c0", "GCP", Source(text="GCP", page=1))])
        )


def test_real_faiss_retrieval_and_bounded_chunks(settings):
    embeddings = FakeEmbeddings()
    sources = [Source(text="Cloud hosting GCP.", page=1), Source(text="Employee training.", page=2)]
    context = retrieve_all(sources, ["Cloud hosting"], embeddings, settings)[0]
    assert context[0].source.page == 1
    assert embeddings.document_calls == 1
    chunks = build_chunks([Source(text="Long content. " * 2000, page=8)], embeddings, settings)
    assert len(chunks) > 1
    assert all(embeddings.token_count(c.text) <= 400 and c.source.page == 8 for c in chunks)
    settings.max_chunks = 1
    with pytest.raises(AppError) as error:
        build_chunks([Source(text="Long content. " * 2000)], embeddings, settings)
    assert error.value.code == "too_many_chunks"


async def test_visual_observations_reused_and_budget_preflight(tmp_path, service, settings):
    visual = Visual(page=1, image_path="unused", width=512, height=512, context="", digest="abc")

    async def parser(*args):
        return ParsedDocument(sources=[], visuals=[visual, visual.model_copy(update={"page": 2})])

    service.parser = parser
    response, status = await service.run(
        ["diagram components?", "diagram hosting?"], tmp_path / "doc.pdf", "pdf", "test"
    )
    assert status == 200
    assert len(service.provider.images) == 1
    assert all(r.status == "answered" for r in response.results)
    settings.vision_token_budget = 100
    with pytest.raises(AppError) as error:
        await service.run(["Q"], tmp_path / "doc.pdf", "pdf", "test")
    assert error.value.code == "vision_budget_exceeded"
    assert len(service.provider.images) == 1


async def test_cpu_cancellation_keeps_lock_until_worker_finishes(service):
    started, release = threading.Event(), threading.Event()

    def work():
        started.set()
        release.wait(timeout=2)

    task = asyncio.create_task(service.run_cpu(work))
    while not started.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    await asyncio.sleep(0.01)
    try:
        assert service.embedding_lock.locked()
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not service.embedding_lock.locked()

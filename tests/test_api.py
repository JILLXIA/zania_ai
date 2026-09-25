import asyncio
import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.logging import logger
from app.main import RequestLimits, create_app, process_connected
from app.models import AppError, ParsedDocument, Source, VisionOutput
from tests.conftest import pdf_bytes, uploads


def test_json_flow_partial_missing_duplicate_and_isolation(client, service):
    questions = ["Cloud provider?", "unsupported topic?", "Notification SLA?", " Cloud provider? "]
    document = [
        {"question": "Source-only question", "answer": "Hosted on GCP."},
        {"answer": "We notify affected parties without undue delay."},
    ]
    response = client.post("/qa", files=uploads(questions, document))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["request_id"] == response.headers["x-request-id"]
    results = body["results"]
    assert [r["question"] for r in results] == questions
    assert [r["status"] for r in results] == ["answered", "not_found", "partial", "answered"]
    assert results[0]["citations"] == [
        {"source_type": "text", "source_path": "/0", "excerpt": json.dumps(document[0])}
    ]
    assert results[1]["answer"] == "Not found in document"
    assert results[1]["citations"] == []
    assert "without undue delay" in results[2]["answer"]
    assert "Not found in document: a numeric notification SLA" in results[2]["answer"]
    assert len(service.provider.answers) == 3
    assert service.embeddings.document_calls == 1
    assert not service.provider.images
    isolated = client.post("/qa", files=uploads(document={"text": "An unrelated document."}))
    assert isolated.json()["results"][0]["status"] == "not_found"


@pytest.mark.parametrize("image", [False, True])
def test_pdf_endpoint_with_real_parser(client, service, image):
    questions = ["What is in the diagram?"] if image else ["Cloud provider?"]
    document = pdf_bytes(text="See the diagram below." if image else "Hosted on GCP.", image=image)
    response = client.post("/qa", files=uploads(questions, document, kind="pdf"))
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["status"] == "answered"
    citation = result["citations"][0]
    assert citation["page"] == 1
    assert citation["source_type"] == ("image" if image else "text")
    assert "source_path" not in citation
    if image:
        assert response.json()["warnings"]
        assert not Path(service.provider.images[0].image_path).exists()


def test_unreadable_visual_is_not_missing_evidence(client, service):
    service.provider.visual_output = VisionOutput(kind="unreadable", observations=[])
    response = client.post("/qa", files=uploads(document=pdf_bytes(image=True), kind="pdf"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unreadable_image"
    assert not service.provider.answers
    assert service.active_requests == 0


def test_mixed_and_all_upstream_failures(client):
    response = client.post("/qa", files=uploads(["Cloud provider?", "failure?"]))
    assert response.status_code == 200
    assert response.json()["results"][1]["status"] == "error"
    response = client.post("/qa", files=uploads(["failure?"]))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "all_questions_failed"
    assert response.json()["results"][0]["status"] == "error"


def test_wrong_question_role_rejected_before_generation(client, service):
    response = client.post("/qa", files=uploads([{"question": "Q", "answer": "Yes"}]))
    assert response.status_code == 422
    assert not service.provider.answers


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("missing", 422),
        ("duplicate", 422),
        ("extra_file", 400),
        ("wrong_extension", 415),
        ("wrong_mime", 415),
        ("fake_pdf", 415),
        ("invalid_json", 400),
        ("empty", 422),
    ],
)
def test_file_validation(client, mutation, expected):
    files = list(uploads().items())
    if mutation == "missing":
        files.pop()
    elif mutation == "duplicate":
        files[1] = files[0]
    elif mutation == "extra_file":
        files.append(("extra", files[1][1]))
    else:
        contents = {
            "wrong_extension": ("doc.csv", b"data", "text/csv"),
            "wrong_mime": ("doc.json", b"{}", "application/pdf"),
            "fake_pdf": ("doc.pdf", b"not a PDF", "application/pdf"),
            "invalid_json": ("doc.json", b"{", "application/json"),
            "empty": ("doc.json", b"", "application/json"),
        }
        files[1] = ("document", contents[mutation])
    response = client.post("/qa", files=files)
    assert response.status_code == expected, response.text
    assert response.json()["error"]["code"]


def test_non_file_fields_and_non_multipart(client):
    assert client.post("/qa", json={}).status_code == 415
    response = client.post("/qa", files=uploads(), data={"unexpected": "field"})
    assert response.status_code == 400


def test_file_size_and_busy_limits(client, service, settings):
    settings.max_document_bytes = 5
    assert client.post("/qa", files=uploads()).status_code == 413
    service.active_requests = settings.max_requests
    assert client.post("/qa", files=uploads()).status_code == 503


@pytest.mark.parametrize("length", [None, b"1", b"999", b"bad", b"-1"])
async def test_receive_boundary_counts_actual_bytes(settings, length):
    settings.max_request_bytes = 10
    messages = []

    async def consume(scope, receive, send):
        await receive()

    async def receive():
        return {"type": "http.request", "body": b"x" * 11, "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "headers": [] if length is None else [(b"content-length", length)]}
    await RequestLimits(consume, settings)(scope, receive, send)
    assert messages[0]["status"] == (400 if length in (b"bad", b"-1") else 413)


async def test_deadline_preserves_completed_answers(settings, service, monkeypatch):
    original = service.provider.answer

    async def answer(question, chunks):
        if "slow" in question:
            await asyncio.sleep(10)
        return await original(question, chunks)

    async def parser(*args):
        from app.models import ParsedDocument, Source

        return ParsedDocument(sources=[Source(text="Hosted on GCP.", source_path="")])

    service.parser = parser
    settings.processing_timeout = 0.2
    monkeypatch.setattr(service.provider, "answer", answer)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(settings, service)), base_url="http://test"
    ) as client:
        response = await client.post("/qa", files=uploads(["Cloud provider?", "slow?"]))
    assert response.status_code == 200, response.text
    assert [r["status"] for r in response.json()["results"]] == ["answered", "error"]
    assert service.active_requests == 0


def test_health_readiness_openapi_and_static(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 503
        assert client.post("/qa", files=uploads()).status_code == 503
        home = client.get("/")
        assert home.status_code == 200
        assert "frame-ancestors 'none'" in home.headers["content-security-policy"]
        assert client.get("/static/app.js").status_code == 200
        schema = client.get("/openapi.json").json()
        assert "multipart/form-data" in schema["paths"]["/qa"]["post"]["requestBody"]["content"]


async def test_concurrent_requests_share_capacity(settings, service, monkeypatch):
    async def parser(*args):
        await asyncio.sleep(0.05)
        return ParsedDocument(sources=[Source(text="GCP", source_path="")])

    service.parser = parser
    settings.max_requests = 1
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(settings, service)), base_url="http://test"
    ) as client:
        responses = await asyncio.gather(*(client.post("/qa", files=uploads()) for _ in range(3)))
    assert sorted(r.status_code for r in responses) == [200, 503, 503]
    assert service.active_requests == 0


async def test_disconnect_cancels_and_joins_processing():
    started, cleaned = asyncio.Event(), asyncio.Event()

    async def receive():
        await started.wait()
        return {"type": "http.disconnect"}

    async def work():
        try:
            started.set()
            await asyncio.sleep(10)
        finally:
            cleaned.set()

    with pytest.raises(AppError) as error:
        await process_connected(SimpleNamespace(receive=receive), work(), 1)
    assert error.value.code == "client_disconnected"
    assert cleaned.is_set()


def test_json_logs_exclude_sensitive_content(client):
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    logger.addHandler(handler)
    try:
        response = client.post(
            "/qa", files=uploads(["private-question-marker"], {"secret": "private-document-marker"})
        )
        assert response.status_code == 200
    finally:
        logger.removeHandler(handler)
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records and all("request_id" in record and "stage" in record for record in records)
    assert "private-question-marker" not in output.getvalue()
    assert "private-document-marker" not in output.getvalue()

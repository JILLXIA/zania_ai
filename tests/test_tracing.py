import asyncio
import json

import httpx
import pytest
import requests
from langsmith import Client, tracing_context
from pydantic import SecretStr

from app.config import Settings
from app.ingestion import parse_document
from app.models import AppError, ParsedDocument, Source, Visual
from app.provider import OpenAIProvider
from app.service import QAService
from app.tracing import create_trace_client
from tests.conftest import FakeEmbeddings
from tests.test_provider import ANSWER_PASSAGES, completion, missing_response


class RecordingSession(requests.Session):
    """Capture the real LangSmith SDK's HTTP payloads without using sockets."""

    def __init__(self):
        super().__init__()
        self.payloads = []

    def request(self, method, url, **kwargs):
        if kwargs.get("data"):
            self.payloads.append(json.loads(kwargs["data"]))
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"version":"test"}'
        response.url = url
        return response

    def runs(self):
        runs = {}
        for payload in self.payloads:
            if "id" in payload:
                runs.setdefault(payload["id"], {}).update(payload)
        return list(runs.values())


@pytest.fixture
def tracing_session(monkeypatch):
    session = RecordingSession()

    def unexpected_default_client(**kwargs):
        pytest.fail("Tracing must keep the explicit recording client, never a default client.")

    monkeypatch.setattr("langsmith.run_trees.get_cached_client", unexpected_default_client)

    def client(**kwargs):
        return Client(**kwargs, session=session, auto_batch_tracing=False, info={"version": "test"})

    monkeypatch.setattr("app.tracing.Client", client)
    return session


def enable_tracing(settings):
    settings.langsmith_tracing = True
    settings.langsmith_api_key = SecretStr("test-langsmith-secret")
    settings.langsmith_endpoint = "https://tracing.invalid"
    settings.openai_api_key = SecretStr("test-openai-secret")


def test_standard_dotenv_settings(tmp_path, monkeypatch):
    for name in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_WORKSPACE_ID",
        "LANGSMITH_HIDE_INPUTS",
        "LANGSMITH_HIDE_OUTPUTS",
    ):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "LANGSMITH_TRACING=true\nLANGSMITH_API_KEY=test-langsmith-secret\n"
        "LANGSMITH_PROJECT=demo-test\nLANGSMITH_ENDPOINT=https://tracing.invalid\n"
        "LANGSMITH_HIDE_INPUTS=false\nLANGSMITH_HIDE_OUTPUTS=false\n"
    )
    settings = Settings(_env_file=path, OPENAI_API_KEY="")
    assert settings.langsmith_tracing
    assert settings.langsmith_project == "demo-test"
    assert settings.langsmith_endpoint == "https://tracing.invalid"
    assert not settings.langsmith_hide_inputs and not settings.langsmith_hide_outputs
    assert settings.langsmith_api_key.get_secret_value() == "test-langsmith-secret"
    assert "test-langsmith-secret" not in repr(settings)


@pytest.mark.parametrize("enabled", [False, True])
def test_disabled_or_missing_key_never_creates_client(settings, monkeypatch, enabled):
    settings.langsmith_tracing = enabled

    def unexpected_client(**kwargs):
        pytest.fail("No LangSmith client should be created.")

    monkeypatch.setattr("app.tracing.Client", unexpected_client)
    assert settings.langsmith_hide_inputs and settings.langsmith_hide_outputs
    assert create_trace_client(settings) is None


def test_client_setup_failure_is_nonfatal(settings, monkeypatch):
    enable_tracing(settings)

    def broken_client(**kwargs):
        raise ValueError("private-error")

    monkeypatch.setattr("app.tracing.Client", broken_client)
    assert create_trace_client(settings) is None


@pytest.mark.parametrize("hide_content", [True, False])
async def test_request_tree_and_content_controls(settings, tracing_session, tmp_path, hide_content):
    enable_tracing(settings)
    settings.langsmith_hide_inputs = hide_content
    settings.langsmith_hide_outputs = hide_content

    async def parser(*args):
        return ParsedDocument(
            sources=[Source(text="private-source-sentinel: Hosted on GCP.", page=1)]
        )

    answer = completion(
        {
            "status": "answered",
            "answer": "Hosted on GCP.",
            "missing_details": [],
            "evidence_ids": ["c0:p0"],
        }
    )
    provider = OpenAIProvider(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(lambda _: answer))
    )
    service = QAService(settings, FakeEmbeddings(), provider, parser=parser)
    try:
        response, status = await service.run(
            ["private-question-sentinel: Who hosts the service?"],
            tmp_path / "document.pdf",
            "pdf",
            "request-one",
        )
        assert status == 200 and response.results[0].status == "answered"
    finally:
        await provider.close()

    runs = tracing_session.runs()
    root = next(r for r in runs if r.get("name") == "document_qa")
    assert root["session_name"] == "zania-demo"
    assert root["extra"]["metadata"]["request_id"] == "request-one"
    assert root["extra"]["metadata"]["result_statuses"] == ["answered"]
    assert root["extra"]["metadata"]["http_status"] == 200
    children = [r for r in runs if r.get("parent_run_id") == root["id"]]
    assert {r["name"] for r in children} >= {
        "parse_document",
        "retrieve_evidence",
        "answer_attempt_1",
    }
    assert any(r.get("run_type") == "llm" for r in runs)
    assert all(r["trace_id"] == root["id"] for r in runs)
    retrieval = next(r for r in runs if r.get("name") == "retrieve_evidence")
    event = next(e for e in retrieval["events"] if e["name"] == "vector_index")
    assert event["kwargs"]["vectors"] == 1 and event["time"]
    hybrid = next(e for e in retrieval["events"] if e["name"] == "hybrid_retrieval")
    assert hybrid["kwargs"]["dense_candidates"] == 1
    assert hybrid["kwargs"]["selected_chunks"] == 1
    assert any(e["name"] == "answer" for e in root["events"])
    assert any(e["name"] == "citations_resolved" for e in root["events"])
    body = json.dumps(tracing_session.payloads)
    assert ("private-source-sentinel" in body) is not hide_content
    assert ("private-question-sentinel" in body) is not hide_content
    assert "test-openai-secret" not in body and "test-langsmith-secret" not in body


async def test_vision_payload_is_hidden(settings, tracing_session, tmp_path):
    enable_tracing(settings)
    path = tmp_path / "image.png"
    path.write_bytes(b"private-image")
    provider = OpenAIProvider(
        settings,
        httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: completion({"kind": "content", "observations": ["private-observation"]})
            )
        ),
    )
    try:
        await provider.analyze_image(
            Visual(
                page=1,
                image_path=str(path),
                width=512,
                height=512,
                context="private-caption",
                digest="test",
            ),
            [100000],
        )
    finally:
        await provider.close()
    assert any(r.get("name") == "vision_attempt_1" for r in tracing_session.runs())
    body = json.dumps(tracing_session.payloads)
    for private in ("data:image", "private-caption", "private-observation"):
        assert private not in body


async def test_disabled_tracing_overrides_ambient_context(settings, tracing_session, tmp_path):
    provider = OpenAIProvider(
        settings.model_copy(update={"openai_api_key": SecretStr("test")}),
        httpx.AsyncClient(transport=httpx.MockTransport(lambda _: missing_response())),
    )

    async def parser(*args):
        return ParsedDocument(sources=[Source(text="Hosted on GCP.")])

    ambient = Client(
        api_url="https://tracing.invalid",
        api_key="test",
        session=tracing_session,
        auto_batch_tracing=False,
        info={"version": "test"},
    )
    try:
        with tracing_context(enabled=True, client=ambient):
            await QAService(settings, FakeEmbeddings(), provider, parser=parser).run(
                ["Who hosts it?"], tmp_path / "document.json", "json", "untraced"
            )
    finally:
        await provider.close()
        ambient.close(timeout=1)
    assert not tracing_session.payloads


async def test_retry_attempts_are_visible(settings, tracing_session, monkeypatch):
    enable_tracing(settings)
    attempts = 0

    async def no_delay(_seconds):
        pass

    def handler(_request):
        nonlocal attempts
        attempts += 1
        return (
            httpx.Response(429, json={"error": {"message": "limited"}})
            if attempts == 1
            else missing_response()
        )

    monkeypatch.setattr("app.provider.asyncio.sleep", no_delay)
    provider = OpenAIProvider(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    try:
        assert (await provider.answer("Question", ANSWER_PASSAGES)).status == "not_found"
    finally:
        await provider.close()
    assert {r.get("name") for r in tracing_session.runs()} >= {
        "answer_attempt_1",
        "answer_attempt_2",
    }


async def test_parser_worker_does_not_receive_tracing_settings(settings, tmp_path, monkeypatch):
    enable_tracing(settings)
    monkeypatch.setenv("LANGSMITH_API_KEY", "private-langsmith-key")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "private-legacy-key")
    monkeypatch.setenv("OPENAI_API_KEY", "private-openai-key")
    path = tmp_path / "document.json"

    class Process:
        returncode = 0

        async def wait(self):
            (tmp_path / "parsed.json").write_text('{"sources": [], "visuals": []}')

    async def spawn(*args, **kwargs):
        config = json.loads(args[-1])
        assert not any(k.startswith("langsmith_") for k in config)
        assert "openai_api_key" not in config
        assert not any(k.startswith(("LANGSMITH_", "LANGCHAIN_")) for k in kwargs["env"])
        assert "OPENAI_API_KEY" not in kwargs["env"]
        return Process()

    monkeypatch.setattr("app.ingestion.asyncio.create_subprocess_exec", spawn)
    await parse_document(path, "json", settings)


async def test_failed_request_is_recorded(settings, tracing_session, tmp_path):
    enable_tracing(settings)

    async def parser(*args):
        raise AppError(422, "empty_document", "The document has no evidence.")

    provider = OpenAIProvider(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(lambda _: missing_response()))
    )
    try:
        with pytest.raises(AppError):
            await QAService(settings, FakeEmbeddings(), provider, parser=parser).run(
                ["Q"], tmp_path / "document.json", "json", "failed-request"
            )
    finally:
        await provider.close()
    root = next(r for r in tracing_session.runs() if r.get("name") == "document_qa")
    assert root["error"]


@pytest.mark.parametrize("failed_method", ["POST", "PATCH"])
async def test_tracing_auth_failure_does_not_fail_qa(
    settings, tracing_session, tmp_path, monkeypatch, failed_method
):
    enable_tracing(settings)
    original = tracing_session.request

    def unauthorized(method, url, **kwargs):
        response = original(method, url, **kwargs)
        if method.upper() == failed_method:
            response.status_code = 401
            response._content = b'{"detail":"Invalid tracing key"}'
            response.request = requests.Request(method, url).prepare()
        return response

    monkeypatch.setattr(tracing_session, "request", unauthorized)

    async def parser(*args):
        return ParsedDocument(sources=[Source(text="Hosted on GCP.")])

    provider = OpenAIProvider(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(lambda _: missing_response()))
    )
    try:
        response, status = await QAService(settings, FakeEmbeddings(), provider, parser=parser).run(
            ["Who hosts it?"], tmp_path / "document.json", "json", "trace-auth-failure"
        )
        assert status == 200 and response.results[0].status == "not_found"
    finally:
        await provider.close()


async def test_concurrent_requests_have_separate_trace_trees(settings, tracing_session, tmp_path):
    enable_tracing(settings)

    async def parser(*args):
        await asyncio.sleep(0)
        return ParsedDocument(sources=[Source(text="Hosted on GCP.")])

    provider = OpenAIProvider(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(lambda _: missing_response()))
    )
    service = QAService(settings, FakeEmbeddings(), provider, parser=parser)
    try:
        await asyncio.gather(
            *(
                service.run(["Who hosts it?"], tmp_path / "document.json", "json", request_id)
                for request_id in ("request-a", "request-b")
            )
        )
    finally:
        await provider.close()
    runs = tracing_session.runs()
    roots = {r["id"]: r for r in runs if r.get("name") == "document_qa"}
    assert len(roots) == 2
    assert {r["extra"]["metadata"]["request_id"] for r in roots.values()} == {
        "request-a",
        "request-b",
    }
    for run in runs:
        root = roots[run["trace_id"]]
        assert run["extra"]["metadata"]["request_id"] == root["extra"]["metadata"]["request_id"]

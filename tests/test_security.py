import asyncio

import httpx
import pytest
from pydantic import SecretStr

from app.models import AppError, Visual
from app.provider import OpenAIProvider


async def test_visual_retry_charged_to_same_budget(settings, tmp_path, monkeypatch):
    attempts = []

    async def no_delay(seconds):
        pass

    async def handler(request):
        attempts.append(request)
        return httpx.Response(429, json={"error": {"message": "limited"}})

    monkeypatch.setattr("app.provider.asyncio.sleep", no_delay)
    settings.openai_api_key = SecretStr("test-key")
    provider = OpenAIProvider(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    path = tmp_path / "image.png"
    path.write_bytes(b"test")
    visual = Visual(page=1, image_path=str(path), width=512, height=512, context="", digest="x")
    try:
        with pytest.raises(AppError) as error:
            await provider.analyze_image(visual, [10000])
        assert error.value.code == "vision_budget_exhausted"
        assert len(attempts) == 1
    finally:
        await provider.close()


async def test_vision_deadline_stops_pending_calls(service, settings, tmp_path, monkeypatch):
    from app.models import ParsedDocument

    cleaned = asyncio.Event()

    async def parser(*args):
        return ParsedDocument(
            sources=[],
            visuals=[
                Visual(page=1, image_path="unused", width=512, height=512, context="", digest="x")
            ],
        )

    async def slow_vision(*args):
        try:
            await asyncio.sleep(10)
        finally:
            cleaned.set()

    settings.vision_timeout = 0.01
    service.parser = parser
    monkeypatch.setattr(service.provider, "analyze_image", slow_vision)
    with pytest.raises(TimeoutError):
        await service.run(["Q"], tmp_path / "doc.pdf", "pdf", "test")
    assert cleaned.is_set()
    assert not service.provider.answers

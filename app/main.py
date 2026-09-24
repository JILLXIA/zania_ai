import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

from app.config import Settings
from app.ingestion import questions_from_bytes
from app.logging import configure_logging, event, request_id
from app.models import AppError, QAResponse
from app.provider import OpenAIProvider
from app.retrieval import LocalEmbeddings
from app.service import QAService


def error_response(error: AppError):
    return JSONResponse(
        {
            "request_id": request_id.get(),
            "error": {
                "code": error.code,
                "message": error.message,
            },
        },
        status_code=error.status,
    )


class RequestLimits:
    """Count bytes before the multipart parser can spool an unlimited upload."""

    def __init__(self, app, settings: Settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        token = request_id.set(str(uuid.uuid4()))
        start, count, response_status = time.monotonic(), 0, 500
        body_complete = False

        async def limited_receive():
            nonlocal count, body_complete
            if body_complete:
                return await receive()
            remaining = self.settings.upload_timeout - (time.monotonic() - start)
            if remaining <= 0:
                raise AppError(408, "upload_timeout", "File upload timed out.")
            try:
                message = await asyncio.wait_for(receive(), remaining)
            except TimeoutError as exc:
                raise AppError(408, "upload_timeout", "File upload timed out.") from exc
            count += len(message.get("body", b""))
            if message["type"] == "http.request" and not message.get("more_body", False):
                body_complete = True
            if count > self.settings.max_request_bytes:
                raise AppError(413, "request_too_large", "The multipart request exceeds the limit.")
            return message

        async def tracked_send(message):
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message["status"]
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", request_id.get().encode()),
                    (b"x-content-type-options", b"nosniff"),
                ]
            await send(message)

        try:
            try:
                length = int(dict(scope["headers"]).get(b"content-length", b"0"))
                if length < 0:
                    raise ValueError()
            except ValueError as exc:
                raise AppError(400, "invalid_content_length", "Invalid content length.") from exc
            if length > self.settings.max_request_bytes:
                raise AppError(413, "request_too_large", "The multipart request exceeds the limit.")
            await self.app(scope, limited_receive, tracked_send)
        except AppError as exc:
            await error_response(exc)(scope, receive, tracked_send)
        finally:
            event(
                "request",
                status=response_status,
                received_bytes=count,
                elapsed_ms=round((time.monotonic() - start) * 1000),
            )
            request_id.reset(token)


def validate_type(file: UploadFile, allowed: set[str]) -> str:
    kind = Path(file.filename or "").suffix.lower().lstrip(".")
    if kind not in allowed:
        raise AppError(415, "unsupported_file", f"Expected a {', '.join(sorted(allowed))} file.")
    mime = (file.content_type or "application/octet-stream").lower().split(";")[0].strip()
    expected = "application/pdf" if kind == "pdf" else "application/json"
    if mime not in (expected, "application/octet-stream"):
        raise AppError(415, "file_type_mismatch", "File extension and content type must agree.")
    return kind


async def read_bounded(file: UploadFile, limit: int) -> bytes:
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise AppError(413, "file_too_large", "An uploaded file exceeds its size limit.")
    if not data:
        raise AppError(422, "empty_file", "Uploaded files must not be empty.")
    return data


async def process_connected(request: Request, work, timeout: float):
    """Stop provider calls and reap workers when a client leaves after uploading."""

    async def disconnected():
        while (await request.receive())["type"] != "http.disconnect":
            pass

    async def bounded():
        async with asyncio.timeout(timeout):
            return await work

    processing = asyncio.create_task(bounded())
    watching = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait([processing, watching], return_when=asyncio.FIRST_COMPLETED)
        if processing in done:
            return await processing
        raise AppError(499, "client_disconnected", "The client disconnected.")
    finally:
        processing.cancel()
        watching.cancel()
        await asyncio.gather(processing, watching, return_exceptions=True)


def create_app(settings: Settings | None = None, service: QAService | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(application):
        owned_provider = None
        application.state.service = service
        if service is None and settings.openai_api_key.get_secret_value():
            try:
                embeddings = await asyncio.to_thread(LocalEmbeddings, settings.model_cache)
                owned_provider = OpenAIProvider(settings)
                application.state.service = QAService(settings, embeddings, owned_provider)
            except Exception:
                event("startup", code="dependencies_not_ready")
        try:
            yield
        finally:
            if owned_provider is not None:
                await owned_provider.close()

    application = FastAPI(title="Document QA", version="0.1.0", lifespan=lifespan)
    application.state.service = service
    application.add_middleware(RequestLimits, settings=settings)
    static = Path(__file__).parent / "static"
    application.mount("/static", StaticFiles(directory=static), name="static")

    @application.exception_handler(AppError)
    async def app_error(_request, exc):
        return error_response(exc)

    @application.exception_handler(HTTPException)
    async def http_error(_request, exc):
        return error_response(
            AppError(exc.status_code, "invalid_request", "Invalid request format.")
        )

    @application.get("/", include_in_schema=False)
    async def home():
        return FileResponse(
            static / "index.html",
            headers={
                "Content-Security-Policy": (
                    "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
                ),
            },
        )

    @application.get("/health")
    async def health():
        return {"status": "ok"}

    @application.get("/ready")
    async def ready():
        if application.state.service is None:
            raise AppError(
                503, "not_ready", "Set OPENAI_API_KEY and prepare the local embedding model."
            )
        return {"status": "ready"}

    @application.post(
        "/qa",
        response_model=QAResponse,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["questions", "document"],
                            "properties": {
                                "questions": {"type": "string", "format": "binary"},
                                "document": {"type": "string", "format": "binary"},
                            },
                        },
                    },
                },
            },
        },
    )
    async def qa(request: Request):
        current = application.state.service
        if current is None:
            raise AppError(
                503, "not_ready", "Set OPENAI_API_KEY and prepare the local embedding model."
            )
        if current.active_requests >= settings.max_requests:
            raise AppError(503, "busy", "The service is busy; retry shortly.")
        current.active_requests += 1
        try:
            if (
                not request.headers.get("content-type", "")
                .lower()
                .startswith("multipart/form-data")
            ):
                raise AppError(
                    415, "multipart_required", "Send questions and document as multipart files."
                )
            async with request.form(
                max_files=2, max_fields=0, max_part_size=settings.max_document_bytes
            ) as form:
                if len(form.multi_items()) != 2 or set(form) != {"questions", "document"}:
                    raise AppError(
                        422, "invalid_fields", "Upload exactly one questions file and one document."
                    )
                if any(not isinstance(value, UploadFile) for value in form.values()):
                    raise AppError(422, "files_required", "Both fields must contain files.")
                validate_type(form["questions"], {"json"})
                kind = validate_type(form["document"], {"pdf", "json"})
                questions = questions_from_bytes(
                    await read_bounded(form["questions"], settings.max_questions_bytes),
                    settings,
                )
                document = await read_bounded(form["document"], settings.max_document_bytes)
                if kind == "pdf" and not document.startswith(b"%PDF-"):
                    raise AppError(415, "invalid_pdf_signature", "The document is not a PDF.")
            with TemporaryDirectory(prefix="zania-") as directory:
                path = Path(directory) / f"document.{kind}"
                path.write_bytes(document)
                result, status = await process_connected(
                    request,
                    current.run(questions, path, kind, request_id.get()),
                    settings.processing_timeout,
                )
                payload = result.model_dump(exclude_none=True)
                if status != 200:
                    payload["error"] = {
                        "code": "all_questions_failed",
                        "message": "All questions failed to process.",
                    }
                return JSONResponse(payload, status_code=status)
        except TimeoutError as exc:
            raise AppError(
                504, "processing_timeout", "Document processing exceeded its deadline."
            ) from exc
        except (AppError, HTTPException):
            raise
        except Exception as exc:
            event("request_failure", code="internal_error", exception_type=type(exc).__name__)
            raise AppError(500, "internal_error", "Unable to process this request.") from exc
        finally:
            current.active_requests -= 1

    return application


app = create_app()

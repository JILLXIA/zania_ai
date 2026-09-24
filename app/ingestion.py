"""Bounded document parsing. The CLI runs untrusted PDF work in a disposable process."""

import asyncio
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from app.config import Settings
from app.models import AppError, ParsedDocument, Source, Visual, normalize


def parse_json(data: bytes):
    def reject_constant(value):
        raise ValueError(f"Non-JSON constant: {value}")

    try:
        return json.loads(
            data.decode("utf-8-sig"),
            parse_constant=reject_constant,
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AppError(400, "invalid_json", "Upload valid UTF-8 JSON.") from exc


def questions_from_bytes(data: bytes, settings: Settings) -> list[str]:
    value = parse_json(data)
    if not isinstance(value, list) or not 1 <= len(value) <= settings.max_questions:
        raise AppError(422, "invalid_questions", f"Supply 1-{settings.max_questions} questions.")
    if any(
        not isinstance(q, str) or not q.strip() or len(q) > settings.max_question_chars
        for q in value
    ):
        raise AppError(422, "invalid_questions", "Questions must be nonempty, bounded strings.")
    return value


def json_sources(data: bytes, settings: Settings) -> ParsedDocument:
    root = parse_json(data)
    if not isinstance(root, (dict, list)):
        raise AppError(422, "invalid_document", "The JSON document must be an object or array.")

    def check(value, depth=0):
        if depth > settings.max_json_depth:
            raise AppError(413, "json_too_deep", "JSON nesting exceeds the configured limit.")
        if isinstance(value, dict):
            return any([check(v, depth + 1) for v in value.values()])
        if isinstance(value, list):
            return any([check(v, depth + 1) for v in value])
        return isinstance(value, str) and bool(value.strip())

    if not check(root):
        raise AppError(422, "empty_document", "The document contains no usable text.")

    sources = []

    def emit(value, pointer: str, context=""):
        # Keep Q/A records together, especially when the answer is just Yes or No.
        is_record = isinstance(value, dict) and "question" in value
        if is_record:
            context = str(value["question"])
        text = normalize(json.dumps(value, ensure_ascii=False))
        if (isinstance(value, (dict, list)) and len(text) > 6000) or (
            isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value)
        ):
            items = value.items() if isinstance(value, dict) else enumerate(value)
            for key, child in items:
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                emit(child, f"{pointer}/{escaped}", context or str(key))
        elif text not in ("{}", "[]", "null", '""'):
            sources.append(Source(text=text, source_path=pointer, context=context))

    if isinstance(root, list):
        for i, record in enumerate(root):
            emit(record, f"/{i}")
    else:
        emit(root, "")  # The empty JSON Pointer identifies the root object.
    if sum(len(s.text) for s in sources) > settings.max_text_chars:
        raise AppError(413, "document_too_large", "Extracted document text exceeds the limit.")
    return ParsedDocument(sources=sources)


def pdf_document(path: Path, directory: Path, settings: Settings) -> ParsedDocument:
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw
    from pypdf import PdfReader

    reader = PdfReader(path)
    if reader.is_encrypted:
        raise AppError(422, "encrypted_pdf", "Password-protected PDFs are not supported.")
    if len(reader.pages) > settings.max_pages:
        raise AppError(413, "too_many_pages", "The PDF exceeds the configured page limit.")
    texts, total = [], 0
    for page in reader.pages:
        # A compressed content stream can be much larger than the uploaded PDF.
        text = page.extract_text() or ""
        total += len(text)
        if total > settings.max_text_chars:
            raise AppError(413, "document_too_large", "Extracted PDF text exceeds the limit.")
        texts.append(text)
    edges = Counter()
    for text in texts:
        lines = [normalize(line) for line in text.splitlines() if line.strip()]
        edges.update(set(lines[:2] + lines[-2:]))
    sources = []
    cleaned = []
    for i, text in enumerate(texts):
        lines = text.splitlines()
        text = normalize(
            " ".join(
                line
                for j, line in enumerate(lines)
                if not (j < 2 or j >= len(lines) - 2)
                or not re.search(
                    r"confidential|all rights reserved|^\s*(?:page )?\d+\s*$", line, re.IGNORECASE
                )
                or edges[normalize(line)] < max(3, len(texts) // 2)
            )
        )
        cleaned.append(text)
        if text:
            sources.append(Source(text=text, page=i + 1))

    candidates = []
    with pdfium.PdfDocument(path) as pdf:
        for i, page in enumerate(pdf):
            width, height = page.get_size()
            bounds, vector_bounds = [], []
            try:
                for n, obj in enumerate(page.get_objects()):
                    if n > 20_000:
                        raise AppError(413, "complex_pdf", "PDF page complexity exceeds the limit.")
                    left, bottom, right, top = obj.get_bounds()
                    box = (max(0, left), max(0, bottom), min(width, right), min(height, top))
                    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
                    in_body = top > height * 0.12 and bottom < height * 0.88
                    if obj.type == raw.FPDF_PAGEOBJ_IMAGE and area >= width * height * 0.02:
                        if in_body:
                            bounds.append(box)
                    elif (
                        obj.type == raw.FPDF_PAGEOBJ_PATH
                        and bottom > height * 0.12
                        and top < height * 0.88
                        and 100 < area < width * height * 0.8
                    ):
                        vector_bounds.append(box)
                # Captioned vector diagrams, not the grid on every control-table page.
                caption = re.search(
                    r"(?:following|below).{0,30}(?:diagram|chart)|"
                    r"(?:diagram|chart).{0,30}(?:following|below)",
                    cleaned[i],
                    re.IGNORECASE,
                )
                if caption and not bounds:
                    bounds.extend(vector_bounds)
                if bounds:
                    box = (
                        max(0, min(b[0] for b in bounds) - 8),
                        max(0, min(b[1] for b in bounds) - 8),
                        min(width, max(b[2] for b in bounds) + 8),
                        min(height, max(b[3] for b in bounds) + 8),
                    )
                    candidates.append((i, box))
            finally:
                page.close()
        if len(candidates) > settings.max_visual_pages:
            raise AppError(413, "too_many_visuals", "Too many visual pages; upload a smaller PDF.")
        visuals = []
        render_start = time.monotonic()
        for i, (left, bottom, right, top) in candidates:
            if time.monotonic() - render_start > settings.rendering_timeout:
                raise AppError(504, "render_timeout", "PDF rendering exceeded its deadline.")
            page = pdf[i]
            try:
                width, height = page.get_size()
                scale = min(3.0, 1536 / max(right - left, top - bottom))
                bitmap = page.render(scale=scale, crop=(left, bottom, width - right, height - top))
                try:
                    image = bitmap.to_pil()
                    image_path = directory / f"page-{i + 1}.png"
                    image.save(image_path, format="PNG")
                    image_width, image_height = image.size
                    image.close()
                finally:
                    bitmap.close()
                image_bytes = image_path.read_bytes()
                if len(image_bytes) > 4 * 1024 * 1024:
                    raise AppError(413, "image_too_large", "A rendered image exceeds 4 MiB.")
                visuals.append(
                    Visual(
                        page=i + 1,
                        image_path=str(image_path),
                        width=image_width,
                        height=image_height,
                        context=cleaned[i][:1000],
                        digest=hashlib.sha256(image_bytes).hexdigest(),
                    )
                )
            finally:
                page.close()
    if not sources and not visuals:
        raise AppError(422, "empty_document", "The PDF contains no usable text or images.")
    return ParsedDocument(sources=sources, visuals=visuals)


def vision_tokens(width: int, height: int) -> int:
    scale = min(1, 2048 / max(width, height), 768 / min(width, height))
    return 2833 + 5667 * math.ceil(int(width * scale) / 512) * math.ceil(int(height * scale) / 512)


async def parse_document(path: Path, kind: str, settings: Settings) -> ParsedDocument:
    output = path.parent / "parsed.json"
    safe_settings = settings.model_dump(exclude={"openai_api_key"})
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "app.ingestion",
        str(path),
        str(output),
        kind,
        json.dumps(safe_settings),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env={k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"},
    )
    try:
        async with asyncio.timeout(settings.parsing_timeout + settings.rendering_timeout):
            await process.wait()
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode != 0 or not output.exists():
        raise AppError(422, "invalid_document", "Document parsing failed or exceeded resources.")
    payload = json.loads(output.read_text())
    if "error" in payload:
        raise AppError(**payload["error"])
    return ParsedDocument.model_validate(payload)


def worker() -> None:
    path, output, kind, config = sys.argv[1:]
    if sys.platform == "linux":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    settings = Settings(_env_file=None, **json.loads(config))
    try:
        document = (
            pdf_document(Path(path), Path(output).parent, settings)
            if kind == "pdf"
            else json_sources(Path(path).read_bytes(), settings)
        )
        result = document.model_dump()
    except AppError as exc:
        result = {"error": {"status": exc.status, "code": exc.code, "message": exc.message}}
    except Exception:
        result = {
            "error": {"status": 422, "code": "invalid_document", "message": "Cannot read document."}
        }
    Path(output).write_text(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    worker()

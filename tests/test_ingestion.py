import csv
import io
import json
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from app.ingestion import (
    json_sources,
    parse_document,
    pdf_document,
    questions_from_bytes,
    vision_tokens,
)
from app.models import AppError
from scripts.convert_sample import convert
from tests.conftest import pdf_bytes


@pytest.mark.parametrize("data", [b"{", b"\xff", b"[NaN]", b"[Infinity]"])
def test_invalid_json(data, settings):
    with pytest.raises(AppError, match="valid UTF-8 JSON") as error:
        questions_from_bytes(data, settings)
    assert error.value.status == 400


@pytest.mark.parametrize("value", [[], {}, [None], [" "], [{"question": "Q?"}], [3]])
def test_question_schema(value, settings):
    with pytest.raises(AppError) as error:
        questions_from_bytes(json.dumps(value).encode(), settings)
    assert error.value.code == "invalid_questions"


def test_question_bounds_and_original_strings(settings):
    assert questions_from_bytes(b'[" Q? ", "Q?"]', settings) == [" Q? ", "Q?"]
    for value in [["Q?"] * 31, ["x" * 2001]]:
        with pytest.raises(AppError):
            questions_from_bytes(json.dumps(value).encode(), settings)


def test_json_records_and_nested_pointers(settings):
    record = {"question": "Is encryption enabled?", "answer": "Yes", "comments": "At rest"}
    parsed = json_sources(json.dumps([record]).encode(), settings)
    assert parsed.sources[0].source_path == "/0"
    assert parsed.sources[0].context == record["question"]
    assert '"answer": "Yes"' in parsed.sources[0].text
    large = {"a/b~c": "x" * 6500}
    parsed = json_sources(json.dumps(large).encode(), settings)
    assert parsed.sources[0].source_path == "/a~1b~0c"


@pytest.mark.parametrize("value", [[], {}, {"x": None}, 3, "abc", [1, False]])
def test_empty_or_non_document(value, settings):
    with pytest.raises(AppError):
        json_sources(json.dumps(value).encode(), settings)


def test_depth_is_checked_even_after_valid_text(settings):
    nested = {"valid": "text", "deep": [[[["other"]]]]}
    settings.max_json_depth = 3
    with pytest.raises(AppError) as error:
        json_sources(json.dumps(nested).encode(), settings)
    assert error.value.code == "json_too_deep"


@pytest.mark.parametrize("image,vector", [(False, False), (True, False), (False, True)])
def test_real_pdf_native_and_visual_selection(tmp_path, settings, image, vector):
    path = tmp_path / "fixture.pdf"
    path.write_bytes(
        pdf_bytes(text="The following diagram describes hosting.", image=image, vector=vector)
    )
    document = pdf_document(path, tmp_path, settings)
    assert document.sources[0].page == 1
    assert len(document.visuals) == int(image or vector)
    if document.visuals:
        visual = document.visuals[0]
        assert visual.page == 1
        assert max(visual.width, visual.height) <= 1536
        assert Path(visual.image_path).read_bytes().startswith(b"\x89PNG")


def test_scanned_page_and_visual_cap(tmp_path, settings):
    path = tmp_path / "scan.pdf"
    path.write_bytes(pdf_bytes(text="", image=True, pages=2))
    document = pdf_document(path, tmp_path, settings)
    assert not document.sources
    assert len(document.visuals) == 2
    settings.max_visual_pages = 1
    with pytest.raises(AppError) as error:
        pdf_document(path, tmp_path, settings)
    assert error.value.code == "too_many_visuals"


def test_encrypted_and_page_limit(tmp_path, settings):
    path = tmp_path / "encrypted.pdf"
    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(pdf_bytes())))
    writer.encrypt("private")
    writer.write(path)
    with pytest.raises(AppError) as error:
        pdf_document(path, tmp_path, settings)
    assert error.value.code == "encrypted_pdf"
    path.write_bytes(pdf_bytes(pages=2))
    settings.max_pages = 1
    with pytest.raises(AppError) as error:
        pdf_document(path, tmp_path, settings)
    assert error.value.code == "too_many_pages"


def test_repeated_substantive_text_is_preserved(tmp_path, settings):
    path = tmp_path / "repeated.pdf"
    path.write_bytes(pdf_bytes(pages=3))
    parsed = pdf_document(path, tmp_path, settings)
    assert len(parsed.sources) == 3
    assert all("GCP" in source.text for source in parsed.sources)


async def test_worker_parses_and_rejects_corrupt_pdf(tmp_path, settings):
    path = tmp_path / "document.pdf"
    path.write_bytes(pdf_bytes())
    result = await parse_document(path, "pdf", settings)
    assert "GCP" in result.sources[0].text
    path.write_bytes(b"%PDF-broken")
    with pytest.raises(AppError) as error:
        await parse_document(path, "pdf", settings)
    assert error.value.code == "invalid_document"


async def test_worker_timeout(tmp_path, settings):
    path = tmp_path / "document.json"
    path.write_text('["hello"]')
    settings.parsing_timeout = settings.rendering_timeout = 0.000001
    with pytest.raises(TimeoutError):
        await parse_document(path, "json", settings)


def test_vision_estimate():
    assert vision_tokens(512, 512) == 8500
    assert vision_tokens(1536, 1024) == 36835


def test_csv_conversion_preserves_all_fields(tmp_path):
    source, target = tmp_path / "source.csv", tmp_path / "document.json"
    with source.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["", "id", "question", "answer", "comments", "confidence"])
        for i in range(19):
            writer.writerow([i, f"id-{i}", "Question?", "Yes", 'A, "quote"\nline', "high"])
    assert convert(source, target) == 19
    records = json.loads(target.read_text())
    assert list(records[0]) == ["id", "question", "answer", "comments", "confidence"]
    assert records[-1]["id"] == "id-18"
    assert records[0]["comments"] == 'A, "quote"\nline'
    with pytest.raises(FileExistsError):
        convert(source, target)

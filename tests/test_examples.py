"""Check the shipped examples offline, not the semantic quality of live model answers."""

import json
from pathlib import Path

import pytest
from pypdf import PdfReader

from app.ingestion import json_sources, parse_document, questions_from_bytes
from app.models import AnswerOutput, AppError, Evidence, VisionOutput
from app.retrieval import retrieve_all
from scripts.create_example_cases import generate
from tests.conftest import FakeEmbeddings

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CASES = EXAMPLES / "cases"
GENERATED_ERRORS = [
    ("example-fixtures/questions-invalid-utf8.json", "questions", 400, "invalid_json"),
    ("example-fixtures/questions-too-many-31.json", "questions", 422, "invalid_questions"),
    ("example-fixtures/questions-too-long.json", "questions", 422, "invalid_questions"),
    ("example-fixtures/questions-too-large.json", "questions", 413, "file_too_large"),
    ("example-fixtures/document-empty-bytes.json", "document", 422, "empty_file"),
    ("example-fixtures/document-too-deep.json", "document", 413, "json_too_deep"),
    ("example-fixtures/document-too-large.json", "document", 413, "file_too_large"),
    ("example-fixtures/corrupt.pdf", "document", 422, "invalid_document"),
    ("example-fixtures/not-a-pdf.pdf", "document", 415, "invalid_pdf_signature"),
    ("pdf/encrypted.pdf", "document", 422, "encrypted_pdf"),
    ("pdf/blank.pdf", "document", 422, "empty_document"),
]


def upload_paths(questions: Path, document: Path):
    return {
        "questions": (questions.name, questions.read_bytes(), "application/json"),
        "document": (
            document.name,
            document.read_bytes(),
            "application/pdf" if document.suffix == ".pdf" else "application/json",
        ),
    }


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    directory = tmp_path_factory.mktemp("example-cases")
    generate(directory, include_large=True)
    return directory


@pytest.mark.parametrize(
    "folder", sorted(path.name for path in CASES.iterdir() if (path / "document.json").exists())
)
def test_json_example_inputs_and_reference_locations(folder, settings):
    directory = CASES / folder
    questions = questions_from_bytes((directory / "questions.json").read_bytes(), settings)
    document = json_sources((directory / "document.json").read_bytes(), settings)
    expected = json.loads((directory / "expected.json").read_text())
    assert expected["http_status"] == 200
    assert len(expected["results"]) == len(questions)
    pointers = {source.source_path for source in document.sources}
    for result in expected["results"]:
        assert set(result["evidence_paths"]) <= pointers
        if result["status"] == "not_found":
            assert result["evidence_paths"] == []
    contexts = retrieve_all(document.sources, questions, FakeEmbeddings(), settings)
    assert len(contexts) == len(questions)
    assert all(contexts)
    assert not document.visuals


@pytest.mark.parametrize(
    "name,field,status,code",
    [
        ("questions-empty.json", "questions", 422, "invalid_questions"),
        ("questions-object.json", "questions", 422, "invalid_questions"),
        ("questions-records.json", "questions", 422, "invalid_questions"),
        ("questions-mixed.json", "questions", 422, "invalid_questions"),
        ("questions-blank.json", "questions", 422, "invalid_questions"),
        ("questions-malformed.json", "questions", 400, "invalid_json"),
        ("document-malformed.json", "document", 400, "invalid_json"),
        ("document-empty.json", "document", 422, "empty_document"),
        ("document-scalar.json", "document", 422, "invalid_document"),
        ("document-no-text.json", "document", 422, "empty_document"),
        ("document-unsupported.txt", "document", 415, "unsupported_file"),
    ],
)
def test_static_invalid_upload_examples(client, service, name, field, status, code):
    paths = {"questions": EXAMPLES / "questions.json", "document": EXAMPLES / "document.json"}
    paths[field] = EXAMPLES / "failures" / name
    response = client.post("/qa", files=upload_paths(**paths))
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code
    assert not service.provider.answers
    assert not service.provider.images


@pytest.mark.parametrize("name,field,status,code", GENERATED_ERRORS)
def test_generated_invalid_upload_examples(client, service, generated, name, field, status, code):
    paths = {"questions": EXAMPLES / "questions.json", "document": EXAMPLES / "document.json"}
    paths[field] = generated / name
    response = client.post("/qa", files=upload_paths(**paths))
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code
    assert not service.provider.answers
    assert not service.provider.images


def test_thirty_questions_accepted_and_deduplicated(client, service, generated):
    questions = generated / "example-fixtures/questions-limit-30.json"
    response = client.post("/qa", files=upload_paths(questions, EXAMPLES / "document.json"))
    assert response.status_code == 200
    assert len(response.json()["results"]) == 30
    assert len(service.provider.answers) == 1


async def test_native_and_scanned_pdf_evidence(generated, settings):
    native = await parse_document(generated / "pdf/text-only.pdf", "pdf", settings)
    assert [source.page for source in native.sources] == [1, 2]
    assert "Microsoft Azure" in native.sources[0].text
    assert "without undue delay" in native.sources[1].text
    assert not native.visuals
    scan = await parse_document(generated / "pdf/scanned-policy.pdf", "pdf", settings)
    assert not scan.sources
    assert len(scan.visuals) == 1
    assert scan.visuals[0].page == 1
    for kind in ("text", "scanned"):
        questions = questions_from_bytes(
            (EXAMPLES / f"pdf/questions-{kind}.json").read_bytes(), settings
        )
        expected = json.loads((EXAMPLES / f"pdf/expected-{kind}.json").read_text())
        assert len(questions) == len(expected["results"])
    assert PdfReader(generated / "pdf/encrypted.pdf").is_encrypted


def test_scanned_example_through_api(client, service, generated, monkeypatch):
    # Scripted observations test the plumbing, not whether a live model reads the pixels correctly.
    service.provider.visual_output = VisionOutput(
        kind="content",
        observations=[
            "Cloud provider: Amazon Web Services (AWS). Hosting region: Ireland.",
            "Backup retention: 45 days. Administrator MFA: Required.",
        ],
    )

    async def answer(question, chunks):
        if "SLA" in question:
            return AnswerOutput(
                status="not_found", answer="Not found in document", missing_details=[], evidence=[]
            )
        phrase = (
            "45 days"
            if "retention" in question
            else "Administrator MFA: Required"
            if "MFA" in question
            else "Amazon Web Services (AWS). Hosting region: Ireland"
        )
        chunk = next(c for c in chunks if phrase in c.text)
        return AnswerOutput(
            status="answered",
            answer=phrase,
            missing_details=[],
            evidence=[Evidence(chunk_id=chunk.id, excerpt=phrase)],
        )

    monkeypatch.setattr(service.provider, "answer", answer)
    response = client.post(
        "/qa",
        files=upload_paths(
            EXAMPLES / "pdf/questions-scanned.json", generated / "pdf/scanned-policy.pdf"
        ),
    )
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert [r["status"] for r in results] == ["answered", "answered", "answered", "not_found"]
    assert all(r["citations"][0]["source_type"] == "image" for r in results[:3])
    assert all(r["citations"][0]["page"] == 1 for r in results[:3])
    assert len(service.provider.images) == 1


def test_example_generator_never_overwrites(generated):
    target = generated / "example-fixtures/questions-too-long.json"
    target.write_text("user edit")
    try:
        generate(generated)
        assert target.read_text() == "user edit"
    finally:
        target.write_text(json.dumps(["x" * 2001]))


def test_same_question_with_different_documents(client, service, monkeypatch):
    async def answer(question, chunks):
        chunk = chunks[0]
        is_azure = "Azure" in chunk.text
        phrase = "Microsoft Azure" if is_azure else "AWS in Frankfurt"
        return AnswerOutput(
            status="partial" if is_azure else "answered",
            answer=phrase,
            missing_details=["the geographic region"] if is_azure else [],
            evidence=[Evidence(chunk_id=chunk.id, excerpt=phrase)],
        )

    monkeypatch.setattr(service.provider, "answer", answer)
    directory = CASES / "request-isolation"
    for variant in ("a", "b"):
        expected = json.loads((directory / f"expected-{variant}.json").read_text())
        response = client.post(
            "/qa",
            files=upload_paths(
                directory / "questions.json", directory / f"document-{variant}.json"
            ),
        )
        assert response.status_code == expected["http_status"]
        result = response.json()["results"][0]
        assert result["status"] == expected["results"][0]["status"]
        if variant == "b":
            assert "Frankfurt" not in json.dumps(result)


@pytest.mark.parametrize(
    "failure,status,code",
    [
        ("unavailable", 503, "provider_unavailable"),
        ("timeout", 504, "provider_timeout"),
        ("bad_quote", 502, "invalid_citation"),
    ],
)
def test_operational_failure_examples(client, service, monkeypatch, failure, status, code):
    async def broken_answer(question, chunks):
        if failure == "bad_quote":
            return AnswerOutput(
                status="answered",
                answer="Invented AWS answer",
                missing_details=[],
                evidence=[Evidence(chunk_id=chunks[0].id, excerpt="FABRICATED")],
            )
        raise AppError(status, code, "Simulated provider failure; no remote call was made.")

    monkeypatch.setattr(service.provider, "answer", broken_answer)
    response = client.post(
        "/qa", files=upload_paths(EXAMPLES / "questions.json", EXAMPLES / "document.json")
    )
    assert response.status_code == status
    assert response.json()["error"]["code"] == "all_questions_failed"
    assert all(
        result["status"] == "error" and result["error"]["code"] == code
        for result in response.json()["results"]
    )

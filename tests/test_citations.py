import asyncio
import json

import pytest
from pydantic import ValidationError

from app.evidence import Passage, build_passages
from app.ingestion import json_sources
from app.models import AnswerOutput, AppError, Source
from app.retrieval import Chunk
from app.service import grounded_result
from tests.test_provider import completion, provider_with_transport

TLS_TEXT = (
    "User entities access their instance using standard web browsers utilizing Transport Layer "
    "Security (“TLS”) 1.2 or above for encrypted communications."
)
HOSTING_TEXT = (
    "The application runs in Google Cloud Platform (GCP) utilizing Virtual Machines, "
    "Storage and Database services."
)
COMMENTS = (
    "We do not have a dedicated sanctions compliance officer; however, we have a compliance "
    "program in place that restricts activities with sanctioned countries."
)


def selection(ids, answer="TLS 1.2 or above."):
    return {"status": "answered", "answer": answer, "missing_details": [], "evidence_ids": ids}


def passage_for(text, *, chunk_id="c33", page=16):
    return build_passages([Chunk(chunk_id, text, Source(text=text, page=page))])


def allowed_ids(schema):
    items = schema["properties"]["evidence_ids"]["items"]
    if "$ref" in items:
        items = schema["$defs"][items["$ref"].split("/")[-1]]
    return items["enum"]


@pytest.mark.parametrize(
    "source,answer",
    [
        (TLS_TEXT, TLS_TEXT.replace("User entities", "Users")),
        (HOSTING_TEXT, "The service is hosted on GCP."),
        ("No; annual review is required.", "No."),
        ("Company’s controls apply.", "The company's controls apply."),
    ],
)
async def test_paraphrased_answers_cannot_rewrite_server_citations(settings, source, answer):
    passages = passage_for(source)
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return completion(selection([passages[0].id], answer))

    provider = provider_with_transport(settings, handler)
    try:
        output = await provider.answer("Question?", passages)
    finally:
        await provider.close()
    result = grounded_result("Question?", output, passages)
    assert len(calls) == 1  # No repair/retry call is needed.
    assert result.answer == answer
    assert result.citations[0].excerpt == source
    assert result.citations[0].page == 16
    assert set(output.model_dump()) == {"status", "answer", "missing_details", "evidence_ids"}


async def test_json_negative_answer_keeps_original_record_and_context(settings):
    record = {
        "question": "Is there a dedicated sanctions compliance officer?",
        "answer": "No",
        "comments": COMMENTS,
    }
    source = json_sources(json.dumps([record]).encode(), settings).sources[0]
    chunk = Chunk("c11", source.text, source, "Compliance questionnaire")
    passages = build_passages([chunk])
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion(selection([passages[0].id], "No"))

    provider = provider_with_transport(settings, handler)
    try:
        output = await provider.answer(record["question"], passages)
    finally:
        await provider.close()
    result = grounded_result(record["question"], output, passages)
    assert result.citations[0].excerpt == source.text
    assert result.citations[0].source_path == "/0"
    assert "No, we do not" not in result.citations[0].excerpt
    payload = json.loads(requests[0]["messages"][1]["content"])
    group = payload["evidence"][0]
    assert group["context"] == "Compliance questionnaire"
    assert group["passages"] == [{"evidence_id": "c11:p0", "text": source.text}]
    assert "record's question" in requests[0]["messages"][0]["content"]


async def test_request_schema_limits_ids_and_server_keeps_correct_location(settings):
    passages = passage_for(HOSTING_TEXT, chunk_id="c32") + passage_for(
        "GCP is listed as a cloud hosting provider.", chunk_id="c35", page=17
    )
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion(selection(["c32:p0"], "GCP hosts the application."))

    provider = provider_with_transport(settings, handler)
    try:
        output = await provider.answer("Which cloud providers?", passages)
    finally:
        await provider.close()
    schema = requests[0]["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"status", "answer", "missing_details", "evidence_ids"}
    assert allowed_ids(schema) == ["c32:p0", "c35:p0"]
    assert "excerpt" not in json.dumps(schema)
    assert "chunk_id" not in json.dumps(schema)
    result = grounded_result("Which cloud providers?", output, passages)
    assert result.citations[0].excerpt == HOSTING_TEXT
    assert result.citations[0].page == 16


@pytest.mark.parametrize("bad_id", ["c33", "c32:p0", "c33:p99", "", "invented:p0"])
async def test_sdk_rejects_ids_outside_this_question_enum(settings, bad_id):
    provider = provider_with_transport(settings, lambda _: completion(selection([bad_id])))
    try:
        with pytest.raises(AppError) as error:
            await provider.answer("Which TLS versions?", passage_for(TLS_TEXT))
        assert error.value.code == "invalid_model_output"
    finally:
        await provider.close()


@pytest.mark.parametrize(
    "excerpt", [TLS_TEXT.replace("User entities", "Users"), "No, " + COMMENTS, HOSTING_TEXT]
)
async def test_old_free_text_citation_contract_is_rejected(settings, excerpt):
    payload = selection(["c33:p0"])
    payload["evidence"] = [{"chunk_id": "c35", "excerpt": excerpt}]
    with pytest.raises(ValidationError):
        AnswerOutput.model_validate(payload)
    provider = provider_with_transport(settings, lambda _: completion(payload))
    try:
        with pytest.raises(AppError) as error:
            await provider.answer("Question?", passage_for(TLS_TEXT))
        assert error.value.code == "invalid_model_output"
    finally:
        await provider.close()


def test_duplicate_ids_produce_one_citation_and_public_schema_is_unchanged():
    passages = passage_for(TLS_TEXT)
    output = AnswerOutput(**selection(["c33:p0", "c33:p0"]))
    result = grounded_result("TLS?", output, passages)
    assert len(result.citations) == 1
    assert result.model_dump(exclude_none=True) == {
        "question": "TLS?",
        "answer": "TLS 1.2 or above.",
        "status": "answered",
        "citations": [{"source_type": "text", "page": 16, "excerpt": TLS_TEXT}],
    }


@pytest.mark.parametrize("text", ["Invented quote.", " "])
def test_corrupt_server_passage_cannot_bypass_source_check(text):
    passage = passage_for(TLS_TEXT)[0]
    corrupt = Passage(passage.id, text, passage.chunk)
    with pytest.raises(AppError) as error:
        grounded_result("TLS?", AnswerOutput(**selection([passage.id])), [corrupt])
    assert error.value.code == "invalid_citation"


async def test_empty_evidence_never_calls_model(settings):
    def unexpected_request(_):
        pytest.fail("No model call is allowed without passages.")

    provider = provider_with_transport(settings, unexpected_request)
    try:
        output = await provider.answer("Question?", [])
        assert output.status == "not_found" and output.evidence_ids == []
    finally:
        await provider.close()


async def test_concurrent_questions_keep_independent_enums(settings):
    requests = []
    inputs = [passage_for(TLS_TEXT, chunk_id="c1"), passage_for(HOSTING_TEXT, chunk_id="c2")]

    async def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        evidence_id = payload["evidence"][0]["passages"][0]["evidence_id"]
        schema = body["response_format"]["json_schema"]["schema"]
        assert allowed_ids(schema) == [evidence_id]
        requests.append(body)
        await asyncio.sleep(0)
        return completion(selection([evidence_id]))

    provider = provider_with_transport(settings, handler)
    try:
        outputs = await asyncio.gather(*(provider.answer("Question?", p) for p in inputs))
    finally:
        await provider.close()
    assert [output.evidence_ids for output in outputs] == [["c1:p0"], ["c2:p0"]]
    assert len(requests) == 2

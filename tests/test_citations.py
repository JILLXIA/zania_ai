import json

import pytest

from app.ingestion import json_sources
from app.models import AnswerOutput, AppError, Evidence
from app.retrieval import Chunk
from app.service import grounded_result
from tests.test_provider import completion, provider_with_transport

QUESTION = "Is there a dedicated sanctions compliance officer?"
COMMENTS = (
    "We do not have a dedicated sanctions compliance officer; however, we have a compliance "
    "program in place that restricts activities with sanctioned countries. Our internal audit "
    "and compliance department oversees regulatory and compliance issues."
)
SHORT_QUOTE = "We do not have a dedicated sanctions compliance officer"
MERGED_QUOTE = "No, " + COMMENTS[0].lower() + COMMENTS[1:]


@pytest.fixture
def sanctions_chunk(settings):
    # Reproduce the reported field-boundary issue without private sample files or record IDs.
    document = [{"question": QUESTION, "answer": "No", "comments": COMMENTS}]
    source = json_sources(json.dumps(document).encode(), settings).sources[0]
    return Chunk("c11", source.text, source, source.context)


@pytest.mark.parametrize(
    "quotes,valid",
    [
        pytest.param(["No"], True, id="literal-no"),
        pytest.param([COMMENTS], True, id="literal-comments"),
        pytest.param([SHORT_QUOTE], True, id="continuous-span"),
        pytest.param(["No", COMMENTS], True, id="separate-fields-separate-quotes"),
        pytest.param([MERGED_QUOTE], False, id="reported-merged-fields"),
        pytest.param(["No."], False, id="added-period"),
        pytest.param([SHORT_QUOTE + "."], False, id="semicolon-rewritten-as-period"),
        pytest.param([SHORT_QUOTE[0].lower() + SHORT_QUOTE[1:]], False, id="changed-case"),
    ],
)
def test_json_citations_require_exact_source_spans(sanctions_chunk, quotes, valid):
    output = AnswerOutput(
        status="answered",
        answer="No",
        missing_details=[],
        evidence=[Evidence(chunk_id="c11", excerpt=quote) for quote in quotes],
    )
    if not valid:
        with pytest.raises(AppError) as error:
            grounded_result(QUESTION, output, [sanctions_chunk])
        assert error.value.code == "invalid_citation"
        return
    result = grounded_result(QUESTION, output, [sanctions_chunk])
    assert result.status == "answered" and result.answer == "No"
    assert [c.excerpt for c in result.citations] == quotes
    assert all(c.source_path == "/0" and c.source_type == "text" for c in result.citations)


async def test_provider_sends_literal_json_citation_rules(settings, sanctions_chunk):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        return completion(
            {
                "status": "answered",
                "answer": "No",
                "missing_details": [],
                "evidence": [{"chunk_id": "c11", "excerpt": SHORT_QUOTE}],
            }
        )

    provider = provider_with_transport(settings, handler)
    try:
        output = await provider.answer(QUESTION, [sanctions_chunk])
    finally:
        await provider.close()

    assert len(requests) == 1
    prompt = requests[0]["messages"][0]["content"]
    assert "Never combine JSON field values into one excerpt" in prompt
    assert "Preserve capitalization and punctuation" in prompt
    assert "separate evidence entries" in prompt
    assert "Example for citation formatting only" in prompt
    payload = json.loads(requests[0]["messages"][1]["content"])
    assert payload["question"] == QUESTION
    assert payload["evidence"][0]["text"] == sanctions_chunk.text
    assert MERGED_QUOTE not in payload["evidence"][0]["text"]
    result = grounded_result(QUESTION, output, [sanctions_chunk])
    assert result.answer == "No" and result.citations[0].excerpt == SHORT_QUOTE

import pytest

from app.evidence import build_passages
from app.models import Source
from app.retrieval import Chunk


@pytest.mark.parametrize(
    "text",
    [
        "User entities use TLS 1.2 or above. No TLS 1.0 support.",
        '{"question": "Is there an officer?", "answer": "No", "comments": "We review controls."}',
        "Long sentence without punctuation " * 100,
        "审计记录：不共享数据。" * 300,
        "One sentence.\n\nAnother sentence with  multiple spaces.\tAnd tabs.",
        "x" * 16000,
        "A. " * 5300,
        "  \n\t ",
        "",
    ],
)
def test_passages_are_bounded_exact_slices_with_no_lost_content(text):
    chunk = Chunk("c33", text, Source(text=text, page=16), "context-only label")
    passages = build_passages([chunk])
    assert len(passages) <= 208  # Below the structured-output enum limit, even for tiny sentences.
    assert len({p.id for p in passages}) == len(passages)
    cursor = 0
    for passage in passages:
        assert passage.chunk is chunk
        assert passage.text and len(passage.text) <= 600
        start = text.index(passage.text, cursor)
        assert not text[cursor:start].strip()
        cursor = start + len(passage.text)
    assert not text[cursor:].strip()


def test_passage_ids_are_stable_and_do_not_cross_sources():
    chunks = [
        Chunk("c32", "Cloud hosting on GCP.", Source(text="Cloud hosting on GCP.", page=16)),
        Chunk("c35", "GCP is a provider.", Source(text="GCP is a provider.", page=17)),
    ]
    first, second = build_passages(chunks), build_passages(chunks)
    assert [p.id for p in first] == ["c32:p0", "c35:p0"]
    assert first == second
    assert [p.chunk.source.page for p in first] == [16, 17]


def test_only_the_retrieved_chunk_is_citable_not_hidden_source_or_context():
    chunk = Chunk(
        "c1",
        "Visible passage.",
        Source(text="Visible passage. Hidden qualifier.", page=2),
        "Context-only title",
    )
    passages = build_passages([chunk])
    assert [p.text for p in passages] == ["Visible passage."]


def test_whole_sentence_boundary_is_preferred_when_available():
    first = "The service provides customers with encrypted transport and controlled access. " * 3
    second = "User entities use TLS 1.2 or above for encrypted communications. " * 8
    text = first + second
    passages = build_passages([Chunk("c0", text, Source(text=text, page=1))])
    assert len(passages) > 1
    assert passages[0].text.endswith(".")
    assert all("1.2" in p.text for p in passages)

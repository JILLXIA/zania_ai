import json

import numpy as np
import pytest

import app.retrieval as retrieval
from app.models import Source
from app.retrieval import Chunk, lexical_tokens, reciprocal_rank_fusion, retrieve_all
from tests.conftest import FakeEmbeddings


class OrderedEmbeddings(FakeEmbeddings):
    """Dense ranking prefers earlier sources, regardless of their words."""

    def embed_documents(self, texts):
        self.document_calls += 1
        return np.asarray([[1, i / 10] for i in range(len(texts))], dtype="float32")

    def embed_queries(self, texts):
        return np.asarray([[1, 0] for _ in texts], dtype="float32")


def test_tokenization_preserves_identifiers_and_negation():
    assert lexical_tokens('Control CC6.1: "TLS-1.2"; No, NOT enabled.') == [
        "control",
        "cc6.1",
        "tls-1.2",
        "no",
        "not",
        "enabled",
    ]


def test_rrf_rewards_agreement_and_uses_both_rankings():
    # Both lists rank 2 second, so it outranks either list's first result.
    assert reciprocal_rank_fusion([0, 2, 3], [1, 2, 4]) == [2, 0, 1, 3, 4]
    assert reciprocal_rank_fusion([3, 1], []) == [3, 1]
    assert reciprocal_rank_fusion([0, 0, 2], [1, 2]) == [2, 0, 1]
    assert reciprocal_rank_fusion([], []) == []


def test_bm25_recovers_exact_term_outside_dense_candidates(settings, caplog):
    sources = [Source(text=f"General policy section {i}.", page=i + 1) for i in range(13)]
    target = Source(text="Control ZXY-917 requires a quarterly review.", page=14)
    sources.append(target)
    embeddings = OrderedEmbeddings()
    # The target ranks 14th in dense search, outside its top-12 candidate pool.
    with caplog.at_level("INFO", logger="zania"):
        contexts = retrieve_all(sources, ["zxy-917", "quarterly review"], embeddings, settings)
    for context in contexts:
        assert context[1].source == target
        assert context[1].text == target.text
        assert len(context) == 7
    assert embeddings.document_calls == 1
    events = [json.loads(record.message) for record in caplog.records]
    hybrid = [event for event in events if event["stage"] == "hybrid_retrieval"]
    assert len(hybrid) == 2
    assert hybrid[0]["dense_candidates"] == 12
    assert hybrid[0]["lexical_candidates"] == 1
    assert hybrid[0]["selected_chunks"] == 7
    assert "zxy-917" not in json.dumps(events).lower()


@pytest.mark.parametrize("question", ["no_matching_keyword", "?!"])
def test_no_lexical_match_keeps_dense_results(settings, question):
    sources = [Source(text=f"Source policy {i}.", page=i + 1) for i in range(8)]
    context = retrieve_all(sources, [question], OrderedEmbeddings(), settings)[0]
    assert [chunk.source.page for chunk in context] == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("texts", [["!!!", "..."], ["yes"], ["same", "same"]])
def test_tiny_or_tokenless_corpus_does_not_break_retrieval(settings, texts):
    sources = [Source(text=text, page=i + 1) for i, text in enumerate(texts)]
    context = retrieve_all(sources, ["yes"], OrderedEmbeddings(), settings)[0]
    assert context
    assert all(chunk.text in texts for chunk in context)


def test_empty_sources_skip_embedding(settings):
    embeddings = OrderedEmbeddings()
    assert retrieve_all([], ["One?", "Two?"], embeddings, settings) == [[], []]
    assert embeddings.document_calls == 0


def test_expanded_duplicates_do_not_use_up_result_slots(settings):
    sources = [Source(text="Repeated policy. " * 130, page=1)]
    sources.extend(Source(text=f"Other section {i}.", page=i + 2) for i in range(6))
    context = retrieve_all(sources, ["other section"], OrderedEmbeddings(), settings)[0]
    assert len(context) >= 6
    assert len({chunk.text for chunk in context}) == len(context)
    assert sum(chunk.source.page == 1 for chunk in context) == 1


def test_hybrid_context_preserves_byte_limit_and_source_metadata(settings):
    sources = [
        Source(text=f"Control ZXY-917 section {i}. " + "审计" * 620, source_path=f"/{i}")
        for i in range(10)
    ]
    contexts = retrieve_all(sources, ["ZXY-917"], FakeEmbeddings(), settings)
    context = contexts[0]
    assert context
    assert sum(len(c.text.encode()) + len(c.context.encode()) + 120 for c in context) <= 16000
    assert len(context) <= 8
    for chunk in context:
        assert chunk.source.source_path is not None
        assert chunk.text in sources[int(chunk.source.source_path[1:])].text


@pytest.mark.parametrize("repetitions,merged", [(80, True), (400, False)])
def test_fusion_merges_only_expandable_source_hits(settings, monkeypatch, repetitions, merged):
    source = Source(text="Policy details. " * repetitions + "ZXY-917 is archived.", page=1)
    sources = [source] + [Source(text=f"General section {i}.", page=i + 2) for i in range(13)]
    embeddings = OrderedEmbeddings()

    def embed_documents(texts):
        # Dense misses the final keyword-bearing chunk, but finds the source's first chunk.
        return np.asarray(
            [[0, 1] if "ZXY-917" in text else [1, i / 10] for i, text in enumerate(texts)],
            dtype="float32",
        )

    rankings = []

    def capture_fusion(*args):
        rankings.extend(args)
        return reciprocal_rank_fusion(*args)

    monkeypatch.setattr(embeddings, "embed_documents", embed_documents)
    monkeypatch.setattr(retrieval, "reciprocal_rank_fusion", capture_fusion)
    context = retrieve_all(sources, ["ZXY-917"], embeddings, settings)[0]
    assert (rankings[0][0] == rankings[1][0]) is merged
    assert any("ZXY-917" in chunk.text for chunk in context)


def test_full_source_over_budget_keeps_matching_chunk(settings, monkeypatch):
    chunks = []
    for i in range(4):
        quote = f"Important qualifier {i}."
        source = Source(text=quote + " padding" * 590, page=i + 1)
        chunks.append(Chunk(f"c{i}", quote, source))
    monkeypatch.setattr(retrieval, "build_chunks", lambda *args: chunks)
    context = retrieve_all(
        [chunk.source for chunk in chunks], ["unmatched"], OrderedEmbeddings(), settings
    )[0]
    assert context[0].text == chunks[0].source.text
    assert context[1].text == chunks[1].source.text
    assert context[2].text == chunks[2].source.text
    assert context[3].text == chunks[3].text
    assert context[3].id == chunks[3].id
    assert sum(len(c.text.encode()) + 120 for c in context) <= 16000


def test_lexical_index_is_request_local(settings):
    embeddings = OrderedEmbeddings()
    first = [Source(text="ZXY-917 is confidential.", source_path="/private")]
    second = [Source(text="Unrelated employee training.", source_path="/training")]
    retrieve_all(first, ["ZXY-917"], embeddings, settings)
    context = retrieve_all(second, ["ZXY-917"], embeddings, settings)[0]
    assert [chunk.source.source_path for chunk in context] == ["/training"]
    assert all("confidential" not in chunk.text for chunk in context)

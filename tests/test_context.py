"""Context assembly and the citation map that lets an answer point at a section."""

from __future__ import annotations

from conftest import ACME, ACME_EU
from tenant_fence import RetrievedChunk, build_context
from tenant_fence.context import DEFAULT_HEADER


def chunk(
    doc_id: str, text: str, *, title: str = "", uri: str = "", rank: int = 1
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}#0",
        doc_id=doc_id,
        text=text,
        scope=ACME,
        title=title,
        source_uri=uri,
        rank=rank,
    )


def test_markers_are_assigned_per_chunk_in_rank_order() -> None:
    context = build_context([chunk("a", "first"), chunk("b", "second")])
    assert "[1] a" in context.text
    assert "[2] b" in context.text
    assert [citation.marker for citation in context.citations] == ["[1]", "[2]"]


def test_citation_map_resolves_a_marker_to_its_source() -> None:
    context = build_context(
        [chunk("policy-7", "text", title="Refund policy", uri="kb://acme/policy-7")]
    )
    citation = context.citation_map["[1]"]
    assert (citation.doc_id, citation.title, citation.source_uri) == (
        "policy-7",
        "Refund policy",
        "kb://acme/policy-7",
    )
    assert citation.chunk_id == "policy-7#0"


def test_citation_carries_the_scope_label_for_the_audit_trail() -> None:
    eu_chunk = RetrievedChunk(chunk_id="d#0", doc_id="d", text="t", scope=ACME_EU)
    assert build_context([eu_chunk]).citations[0].scope_label == "acme/eu/*"


def test_the_context_contains_exactly_the_chunks_it_was_given() -> None:
    context = build_context([chunk("a", "alpha text"), chunk("b", "beta text")])
    assert "alpha text" in context.text
    assert "beta text" in context.text
    assert context.text.count("[") == 2


def test_a_chunk_title_falls_back_to_the_document_id() -> None:
    assert "[1] bare-doc" in build_context([chunk("bare-doc", "text")]).text


def test_empty_input_returns_the_header_alone() -> None:
    context = build_context([])
    assert context.text == DEFAULT_HEADER
    assert context.citations == ()


def test_max_chars_drops_whole_chunks_and_counts_them() -> None:
    chunks = [chunk("a", "x" * 200), chunk("b", "y" * 200)]
    context = build_context(chunks, max_chars=len(DEFAULT_HEADER) + 240)
    assert context.dropped == 1
    assert len(context.citations) == 1
    assert "y" * 200 not in context.text, "a dropped chunk left its text in the context"


def test_a_dropped_chunk_never_leaves_a_truncated_fragment_behind() -> None:
    chunks = [chunk("a", "short"), chunk("b", "z" * 500)]
    context = build_context(chunks, max_chars=len(DEFAULT_HEADER) + 60)
    assert "z" not in context.text, (
        "part of a dropped chunk survived; half a passage still looks citable and invites "
        "an answer that cites a sentence the context no longer contains"
    )
    assert context.dropped == 1


def test_surviving_markers_keep_their_retrieval_numbers() -> None:
    chunks = [chunk("a", "q" * 400), chunk("b", "short")]
    context = build_context(chunks, max_chars=len(DEFAULT_HEADER) + 60)
    assert [citation.marker for citation in context.citations] == ["[2]"], (
        "markers were renumbered after a drop; marker [1] in an answer and marker [1] in "
        "the log would then refer to different sections"
    )


def test_doc_ids_are_deduplicated_in_marker_order() -> None:
    first = chunk("same", "one")
    second = RetrievedChunk(chunk_id="same#1", doc_id="same", text="two", scope=ACME)
    context = build_context([first, second, chunk("other", "three")])
    assert context.doc_ids == ("same", "other")

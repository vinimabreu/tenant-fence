"""Retriever behaviour: entitled chunks, stamped provenance, counted refusals."""

from __future__ import annotations

import pytest

from conftest import ACME, BELMONT, document, grant, principal
from tenant_fence import (
    AuditLog,
    FencedRetriever,
    InMemoryIndex,
    Principal,
    ScopePattern,
)


def test_retriever_returns_only_entitled_chunks(
    two_tenant_index: InMemoryIndex, acme_agent: Principal
) -> None:
    result = FencedRetriever(two_tenant_index).retrieve(acme_agent, "refund invoice")
    assert result.doc_ids
    for chunk in result.chunks:
        assert chunk.scope.customer == "acme", (
            f"belmont content reached an acme principal: {chunk.doc_id} scoped "
            f"{chunk.scope.label}"
        )


def test_retriever_stamps_rank_and_the_retrieving_principal(
    two_tenant_index: InMemoryIndex, acme_agent: Principal
) -> None:
    result = FencedRetriever(two_tenant_index).retrieve(acme_agent, "refund invoice vat")
    assert [chunk.rank for chunk in result.chunks] == list(range(1, len(result.chunks) + 1))
    for chunk in result.chunks:
        assert chunk.retrieved_by == "agent-acme"
        assert chunk.score > 0.0


def test_retriever_reports_what_the_fence_excluded(
    two_tenant_index: InMemoryIndex, acme_agent: Principal
) -> None:
    result = FencedRetriever(two_tenant_index).retrieve(acme_agent, "refund invoice")
    assert result.excluded == 1, (
        "the count of chunks the fence removed was lost; without it an empty or thin "
        "result is indistinguishable from a filter that matches nothing"
    )
    assert result.considered == 2


def test_retriever_with_no_grants_returns_nothing_and_says_so(
    two_tenant_index: InMemoryIndex, ungranted: Principal
) -> None:
    result = FencedRetriever(two_tenant_index).retrieve(ungranted, "refund invoice")
    assert result.chunks == ()
    assert result.excluded == 3
    assert result.applied_filter == "allow=[] deny=[]"


def test_retriever_writes_one_audit_record_per_query(
    two_tenant_index: InMemoryIndex, acme_agent: Principal
) -> None:
    log = AuditLog()
    retriever = FencedRetriever(two_tenant_index, audit=log)
    retriever.retrieve(acme_agent, "refund")
    retriever.retrieve(acme_agent, "vat")
    assert len(log) == 2
    assert [record.query for record in log.records] == ["refund", "vat"]
    assert log.records[0].applied_filter == "allow=[acme/*/*] deny=[]"


def test_audit_record_carries_the_returned_documents_and_the_refusal_count(
    two_tenant_index: InMemoryIndex, acme_agent: Principal
) -> None:
    result = FencedRetriever(two_tenant_index).retrieve(acme_agent, "refund invoice")
    assert result.record.doc_ids_returned == result.doc_ids
    assert result.record.refused_count == result.excluded


def test_doc_ids_are_deduplicated_in_rank_order() -> None:
    index = InMemoryIndex()
    index.add(document("multi", "refund policy one\n\nrefund policy two", ACME))
    index.add(document("other", "refund policy three", BELMONT))
    agent = principal("a", grant(ScopePattern(customer="acme")))
    result = FencedRetriever(index).retrieve(agent, "refund policy")
    assert len(result.chunks) == 2
    assert result.doc_ids == ("multi",)


def test_k_defaults_to_the_retriever_setting_and_can_be_overridden() -> None:
    index = InMemoryIndex()
    for number in range(6):
        index.add(document(f"acme-{number}", f"refund policy {number}", ACME))
    agent = principal("a", grant(ScopePattern(customer="acme")))
    retriever = FencedRetriever(index, k=2)
    assert len(retriever.retrieve(agent, "refund policy").chunks) == 2
    assert len(retriever.retrieve(agent, "refund policy", k=5).chunks) == 5


def test_retrieval_is_always_audited_even_without_an_explicit_log(
    two_tenant_index: InMemoryIndex, acme_agent: Principal
) -> None:
    retriever = FencedRetriever(two_tenant_index)
    retriever.retrieve(acme_agent, "refund")
    assert len(retriever.audit) == 1, (
        "a retriever built without an explicit log ran an unrecorded query; the audit "
        "log defaults to present, never to absent"
    )


@pytest.mark.parametrize("k", [-1, -3, -10])
def test_a_negative_k_is_refused_at_the_retriever_boundary(
    two_tenant_index: InMemoryIndex, acme_agent: Principal, k: int
) -> None:
    """A negative k reaches a slice bound and shortens the caller's own results."""
    with pytest.raises(ValueError, match="k must not be negative"):
        FencedRetriever(two_tenant_index).retrieve(acme_agent, "refund invoice", k=k)
    with pytest.raises(ValueError, match="k must not be negative"):
        FencedRetriever(two_tenant_index, k=k)


def test_the_withheld_count_reaches_the_caller_on_no_attribute_of_the_result(
    two_tenant_index: InMemoryIndex, acme_agent: Principal
) -> None:
    """excluded is a global count, so a tenant reading it measures its neighbours.

    Withholding it from ``excluded`` and leaving it on ``record.refused_count``
    would withhold nothing. The record is the audit evidence, so serializing it
    is the natural thing for an integrator to do, and the tenant then reads the
    same global number from the same response one attribute further along.
    """
    log = AuditLog()
    retriever = FencedRetriever(two_tenant_index, audit=log, disclose_excluded=False)
    result = retriever.retrieve(acme_agent, "refund invoice")

    assert result.excluded is None, (
        f"the caller was told excluded={result.excluded} after asking for it to be "
        "withheld; the count is the size of everything this tenant cannot see, and "
        "polling it on a schedule measures another tenant's growth rate"
    )
    assert result.record.refused_count is None, (
        f"the caller's copy of the audit record carries refused_count="
        f"{result.record.refused_count}; the number was withheld from one attribute and "
        "handed back on the next, so a tenant that serializes the record it was given "
        "still measures the corpus it cannot see"
    )
    assert result.chunks, "nothing came back, so the disclosure was never exercised"


def test_withholding_the_count_from_the_caller_keeps_it_for_the_operator(
    two_tenant_index: InMemoryIndex, acme_agent: Principal
) -> None:
    """The other half: redaction may not blind the log it is protecting the tenant from."""
    log = AuditLog()
    result = FencedRetriever(
        two_tenant_index, audit=log, disclose_excluded=False
    ).retrieve(acme_agent, "refund invoice")

    (kept,) = log.records
    assert kept.refused_count == 1, (
        f"the log holds refused_count={kept.refused_count}; suppressing the caller facing "
        "count also blinded the operator, and the record is where the true number has to "
        "survive"
    )
    assert kept.sequence == result.record.sequence, (
        "the redacted copy and the record in the log do not share a sequence number, so "
        "an operator holding one cannot find the other"
    )


def test_the_retriever_exposes_no_public_bulk_read(
    two_tenant_index: InMemoryIndex, acme_agent: Principal
) -> None:
    """A request handler holds the retriever, so what hangs off it is one hop away.

    The claim is about the public surface, and it is worth being exact about
    what that does and does not cover. ``retriever._index`` and
    ``retriever.index._index`` are the real index and will read the whole corpus
    for anyone who reaches through the underscore; Python offers no way to
    prevent that, and claiming otherwise would be the kind of unchecked promise
    this package is written against. What is enforced, and asserted here, is
    that nothing a handler, a serialiser or a debug view finds by walking public
    attributes hands back another tenant's material.
    """
    retriever = FencedRetriever(two_tenant_index)
    surfaces = {"retriever": retriever, "retriever.index": retriever.index}

    for label, surface in surfaces.items():
        for name in dir(surface):
            if name.startswith("_"):
                continue
            attribute = getattr(surface, name, None)
            if not isinstance(attribute, tuple | list):
                continue
            scopes = {
                item.scope.customer for item in attribute if hasattr(item, "scope")
            }
            assert not scopes, (
                f"{label}.{name} hands back content scoped to {sorted(scopes)} with no "
                "principal and no predicate; the whole argument of this package is that a "
                "leak should require the filter to be wrong rather than require every "
                "caller to remember not to touch something"
            )

    assert not hasattr(retriever.index, "chunks"), (
        "the retriever hands out an index that can be read whole, with no principal and "
        "no predicate; a debug endpoint or an admin view reaches every tenant's text "
        "through it"
    )
    assert retriever.chunks_for(acme_agent), (
        "the fenced enumeration path returned nothing, so the unfenced one was removed "
        "without leaving a replacement"
    )


def test_chunks_for_enumerates_only_what_the_principal_may_see(
    two_tenant_index: InMemoryIndex, acme_agent: Principal, belmont_agent: Principal
) -> None:
    """The fenced replacement for reading the index whole."""
    retriever = FencedRetriever(two_tenant_index)

    mine = retriever.chunks_for(acme_agent)
    theirs = retriever.chunks_for(belmont_agent)

    assert {chunk.doc_id for chunk in mine} == {"acme-refunds", "acme-eu-vat"}
    assert {chunk.doc_id for chunk in theirs} == {"belmont-refunds"}
    for chunk in mine:
        assert chunk.scope.customer == "acme", (
            f"enumeration handed an acme principal {chunk.doc_id} at scope "
            f"{chunk.scope.label}; it must apply the same predicate the search path does"
        )
    assert retriever.chunks_for(Principal(principal_id="agent-new")) == (), (
        "an ungranted principal enumerated the corpus; deny by default has to hold on "
        "every path that reads content, not only on search"
    )

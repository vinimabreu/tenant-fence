"""The redundant check. These tests simulate the upstream bug it exists to catch."""

from __future__ import annotations

import pytest

from conftest import ACME, ACME_EU, BELMONT, document, grant, principal
from tenant_fence import (
    AuditEvent,
    AuditLog,
    FenceBreach,
    FencedRetriever,
    Guard,
    InMemoryIndex,
    Principal,
    RetrievedChunk,
    Scope,
    ScopeFilter,
    ScopePattern,
    SearchResult,
    is_allowed,
    scope_filter,
)


def chunk(doc_id: str, scope: Scope, text: str = "text") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=f"{doc_id}#0", doc_id=doc_id, text=text, scope=scope)


def test_guard_passes_entitled_chunks_through_unchanged(acme_agent: Principal) -> None:
    chunks = (chunk("acme-1", ACME), chunk("acme-2", ACME))
    assert Guard().check(acme_agent, chunks) == chunks


def test_guard_raises_when_a_foreign_chunk_reaches_it(acme_agent: Principal) -> None:
    with pytest.raises(FenceBreach) as caught:
        Guard().check(acme_agent, [chunk("belmont-secret", BELMONT)], query="refund")

    breach = caught.value
    assert breach.doc_id == "belmont-secret"
    assert breach.scope == BELMONT
    assert breach.principal_id == "agent-acme"


def test_breach_message_names_the_document_the_scope_and_the_principal(
    acme_agent: Principal,
) -> None:
    with pytest.raises(FenceBreach) as caught:
        Guard().check(acme_agent, [chunk("belmont-secret", BELMONT)])
    message = str(caught.value)
    for fragment in ("belmont-secret", "belmont/*/*", "agent-acme", "allow=[acme/*/*]"):
        assert fragment in message, (
            f"the breach message left out {fragment!r}; an alert that does not name the "
            "document, its scope and the principal cannot be triaged"
        )


def test_breach_is_recorded_before_the_exception_escapes(acme_agent: Principal) -> None:
    log = AuditLog()
    with pytest.raises(FenceBreach):
        Guard(audit=log).check(acme_agent, [chunk("belmont-secret", BELMONT)], query="refund")

    breaches = log.breaches()
    assert len(breaches) == 1, (
        "the breach was not written to the audit log; evidence that only exists in a "
        "traceback disappears the moment someone adds an except clause"
    )
    assert breaches[0].event is AuditEvent.BREACH
    assert "belmont-secret" in breaches[0].detail
    assert breaches[0].principal_id == "agent-acme"


def test_guard_refuses_the_whole_response_rather_than_dropping_the_bad_chunk(
    acme_agent: Principal,
) -> None:
    mixed = [chunk("acme-ok", ACME), chunk("belmont-secret", BELMONT)]
    with pytest.raises(FenceBreach):
        Guard().check(acme_agent, mixed)


def test_guard_uses_the_same_rule_as_the_index_predicate(acme_eu_agent: Principal) -> None:
    predicate = scope_filter(acme_eu_agent)
    for scope in (ACME, BELMONT):
        assert predicate(scope) is False
        assert is_allowed(acme_eu_agent, scope) is False
        with pytest.raises(FenceBreach):
            Guard().check(acme_eu_agent, [chunk("d", scope)])


def test_guard_catches_an_index_that_ignores_the_predicate(acme_agent: Principal) -> None:
    """The refactor this layer exists for: an index that stops honouring the filter."""

    class BrokenIndex(InMemoryIndex):
        def search(self, query: str, *, predicate: ScopeFilter, k: int = 5) -> SearchResult:
            del predicate
            return super().search(query, predicate=lambda _scope: True, k=k)

    index = BrokenIndex()
    index.add(document("acme-1", "refund policy", ACME))
    index.add(document("belmont-1", "refund policy", BELMONT))

    log = AuditLog()
    retriever = FencedRetriever(index, audit=log)
    with pytest.raises(FenceBreach, match="belmont-1"):
        retriever.retrieve(acme_agent, "refund policy")
    assert len(log.breaches()) == 1, (
        "an index that silently stopped filtering produced no breach record; the guard "
        "is the layer that notices when the index changes underneath it"
    )


def test_guard_holds_when_a_deny_is_added_after_indexing() -> None:
    revoked = principal(
        "a",
        grant(ScopePattern(customer="acme"), deny=(ScopePattern(customer="acme", site="eu"),)),
    )
    with pytest.raises(FenceBreach):
        Guard().check(revoked, [chunk("eu-doc", ACME_EU)])


def test_guard_creates_its_own_log_when_none_is_given(acme_agent: Principal) -> None:
    guard = Guard()
    with pytest.raises(FenceBreach):
        guard.check(acme_agent, [chunk("belmont-secret", BELMONT)])
    assert len(guard.audit.breaches()) == 1

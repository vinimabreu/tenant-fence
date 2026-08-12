"""The last check before chunks leave the retriever.

This is deliberate redundancy. The index already filtered the candidate set with
the same predicate, so in a correct system this check never fires. It stays
because the cost is one comparison per chunk and the alternative is trusting an
index that a future refactor may change: a new scorer, a cache in front of
search, a reranker that merges results from two calls, a "temporary" branch that
bypasses the predicate. Any of those can reintroduce a foreign chunk, and this
is the layer that notices.

It fails loud. A violation raises :class:`FenceBreach` naming the document, its
scope and the principal, after writing a breach record to the audit log. It does
not filter the offending chunk out and continue, because a silent drop hides a
real bug in the layer above: the system would keep serving correct looking
answers while the index quietly hands out other tenants' material on every
query, and nobody would find out until the one path that lacks this guard is
added.
"""

from __future__ import annotations

from collections.abc import Sequence

from .audit import AuditLog
from .entitlement import describe_filter, is_allowed
from .models import Principal, RetrievedChunk, Scope


class FenceBreach(Exception):
    """Raised when a chunk reaches the guard that the principal may not see.

    Carries the structured detail as attributes as well as in the message, so a
    handler can route on the offending document without parsing a string.

    The message is for the operator. It names another customer's document id and
    tenant label, so a handler must not render it to the account that triggered
    it, for the same reason
    :meth:`tenant_fence.audit.AuditLog.for_principal` redacts the matching
    record: refusing a disclosure and then printing it in the error is not a
    refusal. Return an opaque failure to the caller and keep this for the log.
    """

    def __init__(self, *, principal: Principal, doc_id: str, chunk_id: str, scope: Scope) -> None:
        self.principal_id = principal.principal_id
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.scope = scope
        super().__init__(
            f"fence breach: principal {principal.principal_id!r} "
            f"({principal.kind.value}) is not entitled to doc_id={doc_id!r} "
            f"chunk_id={chunk_id!r} scope={scope.label!r}; "
            f"effective entitlement was {describe_filter(principal)}"
        )


class Guard:
    """Re-checks retrieved chunks against the principal's entitlement.

    Uses :func:`tenant_fence.entitlement.is_allowed`, the same function that
    builds the index predicate. Checking with a second, independent copy of the
    rule would only prove that the copy agrees with itself; checking with the
    same rule proves that the chunk in hand still satisfies the rule that was
    supposed to have been applied upstream.
    """

    def __init__(self, *, audit: AuditLog | None = None) -> None:
        self.audit = AuditLog() if audit is None else audit

    def check(
        self, principal: Principal, chunks: Sequence[RetrievedChunk], *, query: str = ""
    ) -> tuple[RetrievedChunk, ...]:
        """Return the chunks unchanged, or raise on the first one that fails.

        Returning the input rather than a filtered copy is the contract: this
        function either passes everything through or refuses the whole response.
        There is no third outcome where the caller gets a partial answer built
        from a set that was quietly repaired.
        """
        for chunk in chunks:
            if not is_allowed(principal, chunk.scope):
                self.audit.record_breach(
                    principal=principal,
                    query=query,
                    applied_filter=describe_filter(principal),
                    doc_id=chunk.doc_id,
                    scope=chunk.scope,
                )
                raise FenceBreach(
                    principal=principal,
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    scope=chunk.scope,
                )
        return tuple(chunks)

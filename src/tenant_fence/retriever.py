"""Principal in, entitled chunks out.

The retriever owns the sequence: compile the principal into a predicate, hand
that predicate to the index so it filters before scoring, stamp provenance on
what comes back, run the guard, write the audit record.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .audit import AuditLog, without_refusal_count
from .entitlement import ScopeFilter, describe_filter, scope_filter
from .guard import Guard
from .index import EnumerableIndex, SearchableIndex, SearchResult, require_valid_k
from .models import AuditRecord, Chunk, Principal, RetrievedChunk


def distinct_doc_ids(chunks: Sequence[RetrievedChunk]) -> tuple[str, ...]:
    """Distinct document ids, in rank order.

    Used by both the audit record and :attr:`FenceResult.doc_ids`. Writing it
    twice would let the log and the caller disagree about what was returned,
    which is the same drift this package refuses to allow between the index
    filter and the guard.
    """
    seen: list[str] = []
    for chunk in chunks:
        if chunk.doc_id not in seen:
            seen.append(chunk.doc_id)
    return tuple(seen)


@dataclass(frozen=True)
class FenceResult:
    """Entitled chunks plus the numbers that explain them.

    ``excluded`` is carried all the way out to the caller rather than being
    swallowed by the index, because "your filter is silently matching nothing"
    and "there is genuinely nothing here" produce the same empty list. With the
    count in hand: excluded high and returned zero means the fence is working and
    the corpus has nothing for this tenant; excluded equal to the whole index
    means the grants are probably wrong; both zero means the index is empty.

    That count is a global one, so it is also a volumetric disclosure: a tenant
    reading it learns the size of everything it cannot see, and reading it
    repeatedly learns the growth rate. It is ``None`` when the retriever was
    built with ``disclose_excluded=False``.

    ``record`` is the caller's copy of the audit record, and withholding has to
    cover it too. It is the natural thing for an integrator to serialize (it is
    the evidence), so leaving the true count on ``record.refused_count`` while
    nulling ``excluded`` would hand the tenant the same global number by the
    same request, one attribute further along. When the count is withheld this
    copy carries ``refused_count=None`` and the log keeps the record with the
    real number, joined by ``record.sequence``.
    """

    chunks: tuple[RetrievedChunk, ...]
    applied_filter: str
    considered: int
    excluded: int | None
    record: AuditRecord

    @property
    def doc_ids(self) -> tuple[str, ...]:
        """Distinct document ids in the result, in rank order."""
        return distinct_doc_ids(self.chunks)


class _SearchOnlyIndex:
    """The index as the retriever hands it out: one public method, no bulk read.

    A request handler holds the retriever, so anything reachable from the
    retriever is reachable from the request. An index usually exposes some form
    of "list everything you hold", which is one attribute access away from every
    tenant's text with no principal and no predicate involved: a debug endpoint,
    an admin view, a health check reporting corpus size. This package's argument
    is that a leak should require the filter to be wrong rather than require
    every caller to remember something, so the thing a caller would have to
    remember not to touch is not on the public surface.

    The limit of that, stated because the alternative is a claim this cannot
    keep: the real index is still held in the private attribute ``_index``, here
    and on the retriever itself, so ``retriever.index._index.chunks`` and
    ``retriever._index.chunks`` both read the whole corpus. Python has no way to
    make an object unreachable from the object that uses it, and a wrapper that
    stored the index in a closure would only move the same reference into
    ``__closure__``. What is removed is the accident: a public bulk read on the
    object a handler holds, and anything that walks public attributes. A caller
    that reaches through an underscore, or hand writes a permissive predicate
    and passes it to :meth:`search`, is writing the leak on purpose.
    """

    __slots__ = ("_index",)

    def __init__(self, index: SearchableIndex) -> None:
        self._index = index

    def search(self, query: str, *, predicate: ScopeFilter, k: int) -> SearchResult:
        return self._index.search(query, predicate=predicate, k=k)


class FencedRetriever:
    """Retrieval with entitlement applied before scoring and re-checked after.

    The audit log is not optional. It defaults to a fresh in memory log rather
    than to ``None``, so there is no configuration in which queries run
    unrecorded; the only choice a caller has is where the records go, by passing
    a log with a sink attached.

    The index is held privately. What is exposed under :attr:`index` is a
    search only view; enumeration goes through :meth:`chunks_for`, which takes a
    principal.
    """

    def __init__(
        self,
        index: SearchableIndex,
        *,
        audit: AuditLog | None = None,
        guard: Guard | None = None,
        k: int = 5,
        disclose_excluded: bool = True,
    ) -> None:
        self._index = index
        self._view = _SearchOnlyIndex(index)
        self.audit = AuditLog() if audit is None else audit
        # A Guard reports breaches to the AuditLog it was constructed with, and
        # Guard() builds a fresh private one. Accepting an injected guard as it
        # came would send the single event the log exists for (a cross-tenant
        # chunk refused) to a log nobody reads, while the configured log, the one
        # with the durable sink, ends the request holding no record that the
        # request happened at all. Rebinding is the quiet fix; silently
        # reporting elsewhere is not.
        if guard is not None:
            guard.audit = self.audit
        self.guard = Guard(audit=self.audit) if guard is None else guard
        self.k = require_valid_k(k)
        self._disclose_excluded = disclose_excluded

    @property
    def index(self) -> SearchableIndex:
        """The index, as a search only view. See :class:`_SearchOnlyIndex`."""
        return self._view

    def retrieve(self, principal: Principal, query: str, *, k: int | None = None) -> FenceResult:
        """Search on behalf of ``principal`` and return only what it may see.

        With ``disclose_excluded=False`` the refusal count is stripped from
        both places it would otherwise reach the caller: :attr:`
        FenceResult.excluded` and the ``refused_count`` of the record handed
        back. The record kept in the log is untouched.
        """
        window = require_valid_k(self.k if k is None else k)
        predicate = scope_filter(principal)
        applied_filter = describe_filter(principal)
        result = self._index.search(query, predicate=predicate, k=window)

        stamped = tuple(
            chunk.model_copy(update={"rank": rank, "retrieved_by": principal.principal_id})
            for rank, chunk in enumerate(result.chunks, start=1)
        )
        checked = self.guard.check(principal, stamped, query=query)

        record = self.audit.record_query(
            principal=principal,
            query=query,
            applied_filter=applied_filter,
            doc_ids=distinct_doc_ids(checked),
            refused_count=result.excluded,
        )
        return FenceResult(
            chunks=checked,
            applied_filter=applied_filter,
            considered=result.considered,
            excluded=result.excluded if self._disclose_excluded else None,
            record=record if self._disclose_excluded else without_refusal_count(record),
        )

    def chunks_for(self, principal: Principal) -> tuple[Chunk, ...]:
        """Enumerate what one principal may see, in index order.

        The fenced replacement for reaching into the index and reading it whole.
        It applies the same predicate the search path applies, from the same
        :func:`~tenant_fence.entitlement.is_allowed`, so a debug view or an admin
        screen built on it cannot show a chunk the search path would refuse.
        """
        if not isinstance(self._index, EnumerableIndex):
            raise TypeError(
                f"{type(self._index).__name__} cannot enumerate its chunks; "
                "chunks_for needs an index exposing a 'chunks' property"
            )
        allowed = scope_filter(principal)
        return tuple(chunk for chunk in self._index.chunks if allowed(chunk.scope))

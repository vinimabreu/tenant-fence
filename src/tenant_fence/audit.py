"""Append only audit log for retrieval, with an injected clock and a pluggable sink.

The bar this has to clear is a real one from vendor security questionnaires:
answer "what did this customer's account see last Tuesday" from the log alone,
without replaying queries against an index whose contents have since changed.
That means recording the applied filter, not just the result, because a filter
that matched nothing and a corpus that held nothing produce identical result
sets and very different conclusions.

The log holds records in memory and forwards each one to an optional sink. The
sink is where durability belongs (a file, a table, a log shipper); keeping it
behind a protocol keeps this module free of I/O and free of dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Protocol

from .models import AuditEvent, AuditRecord, Principal, Scope

Clock = Callable[[], datetime]
"""Injected time source. Tests pass a fixed one; production passes the system one."""

REDACTED_BREACH_DETAIL = (
    "guard refused a chunk this account is not entitled to; the document and its scope "
    "belong to another account and are recorded in the operator view of record {sequence}"
)
"""What a tenant reads where an operator reads the offending document id."""


def system_clock() -> datetime:
    """Timezone aware wall clock.

    This is the single place in the package that reads the current time. Every
    other module takes a :data:`Clock`, so an audit trail can be reproduced
    exactly in a test and there is one line to change if the deployment needs a
    different time source.
    """
    return datetime.now(tz=UTC)


def _redacted(record: AuditRecord) -> AuditRecord:
    """Strip the operator only detail from a record leaving for its subject."""
    if record.event is not AuditEvent.BREACH or not record.detail:
        return record
    return record.model_copy(
        update={"detail": REDACTED_BREACH_DETAIL.format(sequence=record.sequence)}
    )


def without_refusal_count(record: AuditRecord) -> AuditRecord:
    """A copy of ``record`` with the refusal count removed rather than zeroed.

    ``refused_count`` is a global number: it counts everything the fence
    withheld from this query across the whole index, so a tenant reading it
    learns the size of what it cannot see and a tenant reading it hourly learns
    its neighbours' growth rate.
    :class:`~tenant_fence.retriever.FencedRetriever` built with
    ``disclose_excluded=False`` hands the caller this copy, and the log keeps
    the record with the real number, joined by
    :attr:`~tenant_fence.models.AuditRecord.sequence`.

    ``None``, never ``0``. A zero is a statement that nothing was refused, and a
    redaction that lies to the reader is worse than the disclosure it prevents.
    """
    return record.model_copy(update={"refused_count": None})


class AuditSink(Protocol):
    """Where records go for durable keeping.

    One method, called once per record, in order. Implementations must not
    mutate the record: it is frozen, and the log's own copy is the same object.
    """

    def write(self, record: AuditRecord) -> None: ...


class AuditLog:
    """An append only sequence of :class:`~tenant_fence.models.AuditRecord`.

    There is no delete, no update and no truncate, and the records themselves
    are frozen. A log that can be quietly rewritten answers nothing when it
    matters. :meth:`records` hands back a tuple rather than the internal list so
    a caller cannot append to it or reorder it from the outside.
    """

    def __init__(self, *, clock: Clock = system_clock, sink: AuditSink | None = None) -> None:
        self._records: list[AuditRecord] = []
        self._clock = clock
        self._sink = sink

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[AuditRecord]:
        return iter(self._records)

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        """Every record written so far, oldest first."""
        return tuple(self._records)

    def record_query(
        self,
        *,
        principal: Principal,
        query: str,
        applied_filter: str,
        doc_ids: tuple[str, ...],
        refused_count: int,
    ) -> AuditRecord:
        """Append one query record and return it."""
        return self._append(
            AuditRecord(
                principal_id=principal.principal_id,
                query=query,
                applied_filter=applied_filter,
                timestamp=self._clock(),
                event=AuditEvent.QUERY,
                doc_ids_returned=doc_ids,
                refused_count=refused_count,
            )
        )

    def record_breach(
        self,
        *,
        principal: Principal,
        query: str,
        applied_filter: str,
        doc_id: str,
        scope: Scope,
    ) -> AuditRecord:
        """Append one breach record and return it.

        Written before the exception is raised, so the evidence survives whatever
        the caller does with the error. A breach that is only visible in a
        traceback is a breach that disappears the moment someone adds an
        ``except`` clause.

        ``detail`` names the refused document and its scope, which on a
        cross-tenant breach belong to a different account from the one this
        record is filed under. That is correct for the operator and wrong for
        the subject, so :meth:`for_principal` redacts it on the way out.

        The document id is quoted with ``!r`` and the scope arrives escaped
        from :func:`~tenant_fence.models.render_segment`, because both are
        chosen by whoever filed the document and the line has structure. An
        unquoted id spelled ``benign scope=acme/*/* and doc_id=stolen`` renders
        a second ``doc_id=`` and a second ``scope=`` inside one record, and the
        line stops mapping onto the event it describes.
        """
        return self._append(
            AuditRecord(
                principal_id=principal.principal_id,
                query=query,
                applied_filter=applied_filter,
                timestamp=self._clock(),
                event=AuditEvent.BREACH,
                doc_ids_returned=(),
                refused_count=0,
                detail=f"guard rejected doc_id={doc_id!r} scope={scope.label}",
            )
        )

    def _append(self, record: AuditRecord) -> AuditRecord:
        numbered = record.model_copy(update={"sequence": len(self._records) + 1})
        self._records.append(numbered)
        if self._sink is not None:
            self._sink.write(numbered)
        return numbered

    def for_principal(
        self, principal_id: str, *, operator: bool = False
    ) -> tuple[AuditRecord, ...]:
        """One account's own slice of the log, safe to hand to that account.

        This is the natural building block for a customer facing access history,
        and the question this module exists to answer is phrased per account, so
        the default has to be the tenant safe version.

        Breach records are the reason. A breach is filed under the principal
        that triggered it and its ``detail`` names the document and the scope
        that were refused, which on a cross-tenant breach is another customer's
        document id and tenant label. Exporting that verbatim means the layer
        that refused the disclosure performs it: agent-acme reads "guard
        rejected doc_id=belmont-ledger scope=belmont/*/*" in its own history and
        learns that a belmont document exists and what it is called. The
        identifiers stay on the record for the operator; what leaves here is
        redacted, with :attr:`~tenant_fence.models.AuditRecord.sequence` as the
        join key back to the full record.

        Pass ``operator=True`` for the unredacted slice. It is a separate,
        explicit request rather than the default, because the dangerous call is
        the one that reads naturally.
        """
        mine = (record for record in self._records if record.principal_id == principal_id)
        if operator:
            return tuple(mine)
        return tuple(_redacted(record) for record in mine)

    def between(self, start: datetime, end: datetime) -> tuple[AuditRecord, ...]:
        """Records with ``start <= timestamp < end``.

        Half open on purpose, so day sized windows tile without double counting
        the record that lands exactly on midnight.
        """
        return tuple(record for record in self._records if start <= record.timestamp < end)

    def breaches(self) -> tuple[AuditRecord, ...]:
        """Every recorded guard rejection."""
        return tuple(record for record in self._records if record.event is AuditEvent.BREACH)

    def doc_ids_seen_by(
        self, principal_id: str, *, start: datetime | None = None, end: datetime | None = None
    ) -> set[str]:
        """Answer the questionnaire question: which documents did this account see.

        Optionally bounded by a time window. Returns document ids rather than
        chunk ids because the question is always asked about documents, and the
        answer has to be readable by someone who is not an engineer.
        """
        seen: set[str] = set()
        for record in self._records:
            if record.principal_id != principal_id or record.event is not AuditEvent.QUERY:
                continue
            if start is not None and record.timestamp < start:
                continue
            if end is not None and record.timestamp >= end:
                continue
            seen.update(record.doc_ids_returned)
        return seen

"""The log has to answer 'what did this account see, and when' without replaying anything."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from conftest import BELMONT, FixedClock
from tenant_fence import AuditEvent, AuditLog, AuditRecord, Principal


def write_query(log: AuditLog, principal: Principal, query: str, *docs: str) -> AuditRecord:
    return log.record_query(
        principal=principal,
        query=query,
        applied_filter="allow=[acme/*/*] deny=[]",
        doc_ids=docs,
        refused_count=2,
    )


def test_clock_is_injected_and_never_read_inline(
    clock: FixedClock, acme_agent: Principal
) -> None:
    log = AuditLog(clock=clock)
    first = write_query(log, acme_agent, "refund")
    second = write_query(log, acme_agent, "vat")
    assert first.timestamp == datetime(2026, 3, 3, 9, 0, tzinfo=UTC)
    assert second.timestamp == datetime(2026, 3, 3, 9, 1, tzinfo=UTC)


def test_records_are_returned_as_an_immutable_snapshot(acme_agent: Principal) -> None:
    log = AuditLog()
    write_query(log, acme_agent, "refund", "acme-1")
    snapshot = log.records
    write_query(log, acme_agent, "vat", "acme-2")
    assert isinstance(snapshot, tuple)
    assert len(snapshot) == 1, "the caller held a live view of the log and could reorder it"
    assert len(log.records) == 2


def test_the_log_exposes_no_way_to_delete_or_rewrite_a_record(acme_agent: Principal) -> None:
    log = AuditLog()
    write_query(log, acme_agent, "refund", "acme-1")
    for forbidden in ("delete", "clear", "remove", "truncate", "update"):
        assert not hasattr(log, forbidden), (
            f"AuditLog exposes {forbidden!r}; an append only log that can be rewritten "
            "answers nothing under scrutiny"
        )
    with pytest.raises(ValidationError):
        log.records[0].query = "something else"  # type: ignore[misc]


def test_sink_receives_every_record_in_order(acme_agent: Principal) -> None:
    class CollectingSink:
        def __init__(self) -> None:
            self.written: list[AuditRecord] = []

        def write(self, record: AuditRecord) -> None:
            self.written.append(record)

    sink = CollectingSink()
    log = AuditLog(sink=sink)
    write_query(log, acme_agent, "refund")
    write_query(log, acme_agent, "vat")
    assert [record.query for record in sink.written] == ["refund", "vat"]


def test_len_and_iteration(acme_agent: Principal) -> None:
    log = AuditLog()
    write_query(log, acme_agent, "refund")
    write_query(log, acme_agent, "vat")
    assert len(log) == 2
    assert [record.query for record in log] == ["refund", "vat"]


def test_for_principal_separates_accounts(
    acme_agent: Principal, belmont_agent: Principal
) -> None:
    log = AuditLog()
    write_query(log, acme_agent, "refund", "acme-1")
    write_query(log, belmont_agent, "refund", "belmont-1")
    assert [record.principal_id for record in log.for_principal("agent-acme")] == ["agent-acme"]


def test_a_per_account_export_never_names_the_document_it_refused(
    clock: FixedClock, acme_agent: Principal
) -> None:
    """A breach is filed under the caller and describes somebody else's document.

    ``for_principal`` is how a customer facing access history gets built, so the
    default it returns has to be the version that account may read. The refused
    document id and tenant label stay on the record for the operator.
    """
    log = AuditLog(clock=clock)
    log.record_breach(
        principal=acme_agent,
        query="settlement ledger",
        applied_filter="allow=[acme/*/*] deny=[]",
        doc_id="belmont-ledger",
        scope=BELMONT,
    )

    (exported,) = log.for_principal("agent-acme")
    assert "belmont" not in exported.detail, (
        f"the acme account's own audit slice reads {exported.detail!r}, naming a belmont "
        "document id and the belmont tenant label; the layer that refused the disclosure "
        "performed it instead"
    )
    assert exported.event is AuditEvent.BREACH, "the refusal itself must stay visible"
    assert str(exported.sequence) in exported.detail, (
        "the redacted record carries no join key, so an operator answering a support "
        "ticket about it cannot find the full record"
    )

    (operator_view,) = log.for_principal("agent-acme", operator=True)
    assert "belmont-ledger" in operator_view.detail, (
        "redaction reached the operator view; the identifiers have to survive somewhere "
        "or an incident cannot be reconstructed at all"
    )


def test_every_record_is_numbered_in_append_order(
    clock: FixedClock, acme_agent: Principal
) -> None:
    log = AuditLog()
    first = write_query(log, acme_agent, "refund")
    second = write_query(log, acme_agent, "vat")
    assert (first.sequence, second.sequence) == (1, 2)
    assert [record.sequence for record in log.records] == [1, 2]


def test_between_is_half_open_so_day_windows_tile(
    clock: FixedClock, acme_agent: Principal
) -> None:
    log = AuditLog(clock=clock)
    first = write_query(log, acme_agent, "one")
    second = write_query(log, acme_agent, "two")
    window = log.between(first.timestamp, second.timestamp)
    assert [record.query for record in window] == ["one"], (
        "the window included the record sitting exactly on its upper bound; half open "
        "windows are what keep consecutive day queries from double counting"
    )


def test_doc_ids_seen_by_answers_the_questionnaire_question(
    clock: FixedClock, acme_agent: Principal, belmont_agent: Principal
) -> None:
    log = AuditLog(clock=clock)
    start = clock.now
    write_query(log, acme_agent, "refund", "acme-1", "acme-2")
    write_query(log, belmont_agent, "refund", "belmont-1")
    write_query(log, acme_agent, "vat", "acme-2", "acme-3")
    cutoff = clock.now

    assert log.doc_ids_seen_by("agent-acme") == {"acme-1", "acme-2", "acme-3"}
    assert log.doc_ids_seen_by("agent-acme", start=start, end=cutoff) == {
        "acme-1",
        "acme-2",
        "acme-3",
    }
    assert log.doc_ids_seen_by("agent-acme", start=cutoff) == set()
    assert log.doc_ids_seen_by("agent-belmont") == {"belmont-1"}


def test_breach_records_are_separated_from_query_records(
    clock: FixedClock, acme_agent: Principal
) -> None:
    log = AuditLog(clock=clock)
    write_query(log, acme_agent, "refund", "acme-1")
    log.record_breach(
        principal=acme_agent,
        query="refund",
        applied_filter="allow=[acme/*/*] deny=[]",
        doc_id="belmont-secret",
        scope=BELMONT,
    )
    assert len(log) == 2
    assert len(log.breaches()) == 1
    assert log.breaches()[0].event is AuditEvent.BREACH
    assert log.doc_ids_seen_by("agent-acme") == {"acme-1"}, (
        "a breach record leaked into the 'documents seen' answer; a refused document is "
        "not a document the account saw"
    )


def test_a_query_that_returned_nothing_still_records_the_filter_it_applied(
    acme_agent: Principal,
) -> None:
    log = AuditLog()
    record = log.record_query(
        principal=acme_agent,
        query="refund",
        applied_filter="allow=[] deny=[]",
        doc_ids=(),
        refused_count=17,
    )
    assert record.doc_ids_returned == ()
    assert record.refused_count == 17
    assert record.applied_filter == "allow=[] deny=[]", (
        "an empty result was logged without the filter that produced it; a filter that "
        "matched nothing and a corpus that held nothing look identical otherwise"
    )


def test_window_query_over_a_realistic_day_boundary(acme_agent: Principal) -> None:
    tuesday = datetime(2026, 3, 3, 8, 0, tzinfo=UTC)
    clock = FixedClock(start=tuesday, step=timedelta(hours=8))
    log = AuditLog(clock=clock)
    write_query(log, acme_agent, "tuesday morning", "acme-1")
    write_query(log, acme_agent, "tuesday afternoon", "acme-2")
    write_query(log, acme_agent, "wednesday midnight exactly", "acme-3")

    day_start = datetime(2026, 3, 3, tzinfo=UTC)
    day_end = day_start + timedelta(days=1)
    assert log.doc_ids_seen_by("agent-acme", start=day_start, end=day_end) == {"acme-1", "acme-2"}
    assert len(log.between(day_start, day_end)) == 2, (
        "the query that landed exactly on the following midnight was counted against "
        "Tuesday; that is how a document gets reported under two different days"
    )


def test_record_query_returns_the_record_it_appended(acme_agent: Principal) -> None:
    log = AuditLog()
    record = write_query(log, acme_agent, "refund", "acme-1")
    assert log.records[-1] is record
    assert record.principal_id == "agent-acme"

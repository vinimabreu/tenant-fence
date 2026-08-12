"""The example has to keep working, because a broken example is a broken README."""

from __future__ import annotations

import pytest
from examples.support_kb import (
    ACME_AGENT,
    ACME_EU_CONTRACTOR,
    BELMONT_AGENT,
    CORPUS,
    NEW_AGENT,
    echo_markers,
    main,
)

from tenant_fence import AuditLog, DocumentStatus, FencedPipeline, InMemoryIndex


@pytest.fixture
def example_log() -> AuditLog:
    index = InMemoryIndex()
    index.add_all(CORPUS)
    log = AuditLog()
    pipeline = FencedPipeline(index, audit=log, k=3)
    for principal in (ACME_AGENT, ACME_EU_CONTRACTOR, BELMONT_AGENT, NEW_AGENT):
        pipeline.answer(principal, "What is the refund window?", generate=echo_markers)
    return log


def test_the_example_runs(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    assert "agent-acme" in capsys.readouterr().out


def test_the_example_shows_the_count_withheld_from_the_caller_but_kept_in_the_log(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The walkthrough demonstrates the disclosure choice, so it has to be a real one."""
    main()
    out = capsys.readouterr().out
    assert "caller reads   refused_count=None" in out, (
        "the example prints a refusal count to a caller that asked for it to be withheld"
    )
    assert "log still has  refused_count=1" in out, (
        "the example shows the operator's log losing the count as well, which would make "
        "the redaction blind the wrong reader"
    )


def test_each_account_only_ever_saw_its_own_documents(example_log: AuditLog) -> None:
    assert example_log.doc_ids_seen_by("agent-acme") == {
        "acme-refunds",
        "acme-eu-vat",
        "acme-us-payroll",
    }
    assert example_log.doc_ids_seen_by("agent-belmont") == {"belmont-refunds"}
    assert example_log.doc_ids_seen_by("agent-unconfigured") == set(), (
        "an account that was never granted anything still read documents"
    )


def test_a_site_grant_with_a_deny_reads_neither_the_sibling_site_nor_the_parent(
    example_log: AuditLog,
) -> None:
    assert example_log.doc_ids_seen_by("contractor-eu") == {"acme-eu-vat"}, (
        "the eu contractor read outside its site: acme-us-payroll is explicitly denied "
        "and acme-refunds sits at the customer level, above the granted site"
    )


def test_the_draft_document_never_appears_in_any_result(example_log: AuditLog) -> None:
    seen = {doc for record in example_log.records for doc in record.doc_ids_returned}
    assert "belmont-draft-pricing" not in seen, (
        "an unapproved draft was retrievable; draft content must never enter the index"
    )


def test_the_draft_is_absent_from_the_index_not_merely_a_poor_match() -> None:
    """The draft has to be missing, not just unlucky against one question.

    The test above runs the example's single question, "What is the refund
    window?", which shares no token with the draft's text, so it scores zero and
    would be dropped even if unapproved content were indexed. That makes it pass
    for a reason unrelated to the property it names. This one removes both
    escapes: it reads the index directly, and it queries the draft using the
    draft's own wording, asked by the tenant that owns it, so the status gate is
    the only thing standing between the caller and the text.
    """
    index = InMemoryIndex()
    index.add_all(CORPUS)
    draft = next(doc for doc in CORPUS if doc.status is DocumentStatus.DRAFT)

    stored = [chunk.doc_id for chunk in index.chunks]
    assert draft.doc_id not in stored, (
        f"{draft.doc_id} has status {draft.status.value} and was still turned into a chunk "
        f"({stored}); unapproved content must never enter the index, because content that is "
        "absent cannot be reached by a cache, a reranker or a future query path"
    )

    pipeline = FencedPipeline(index, k=3)
    answer = pipeline.answer(BELMONT_AGENT, draft.text, generate=echo_markers)
    assert draft.doc_id not in answer.record.doc_ids_returned, (
        f"querying the draft's own wording returned {draft.doc_id} to "
        f"{BELMONT_AGENT.principal_id}; the draft is that tenant's own material, so "
        "entitlement never applies here and only the approved-only gate withholds it"
    )

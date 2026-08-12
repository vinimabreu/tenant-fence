"""A shared support knowledge base serving two tenants and one internal account.

Runs offline. The generator is a stub that echoes the citation markers it was
given, because the point of the example is which sections reach the prompt, not
what a model would say about them.

    python examples/support_kb.py
"""

from __future__ import annotations

import re

from tenant_fence import (
    AuditLog,
    Document,
    DocumentStatus,
    FencedPipeline,
    Grant,
    InMemoryIndex,
    Principal,
    PrincipalKind,
    Scope,
    ScopePattern,
)

CORPUS: tuple[Document, ...] = (
    Document(
        doc_id="acme-refunds",
        title="Acme refund window",
        source_uri="kb://acme/refunds",
        text="Acme customers may request a refund within 30 days of the invoice date.",
        scope=Scope(customer="acme"),
    ),
    Document(
        doc_id="acme-eu-vat",
        title="Acme eu vat handling",
        source_uri="kb://acme/eu/vat",
        text="Acme eu invoices carry vat at the rate of the billing country.",
        scope=Scope(customer="acme", site="eu"),
    ),
    Document(
        doc_id="acme-us-payroll",
        title="Acme us payroll export",
        source_uri="kb://acme/us/payroll",
        text="Acme us payroll exports run on the last business day of the month.",
        scope=Scope(customer="acme", site="us"),
    ),
    Document(
        doc_id="belmont-refunds",
        title="Belmont refund window",
        source_uri="kb://belmont/refunds",
        text=(
            "Belmont customers may request a refund within 7 days of the invoice date, "
            "at a negotiated rate of 12 percent."
        ),
        scope=Scope(customer="belmont"),
    ),
    Document(
        doc_id="belmont-draft-pricing",
        title="Belmont pricing rewrite",
        source_uri="kb://belmont/pricing",
        text="Draft pricing that has not been approved and must never be retrievable.",
        scope=Scope(customer="belmont"),
        status=DocumentStatus.DRAFT,
    ),
)

ACME_AGENT = Principal(
    principal_id="agent-acme",
    kind=PrincipalKind.CUSTOMER,
    grants=(Grant(label="acme, every site", allow=(ScopePattern(customer="acme"),)),),
)

ACME_EU_CONTRACTOR = Principal(
    principal_id="contractor-eu",
    kind=PrincipalKind.CUSTOMER,
    grants=(
        Grant(
            label="acme eu only, us payroll revoked",
            allow=(ScopePattern(customer="acme", site="eu"),),
            deny=(ScopePattern(customer="acme", site="us"),),
        ),
    ),
)

BELMONT_AGENT = Principal(
    principal_id="agent-belmont",
    kind=PrincipalKind.CUSTOMER,
    grants=(Grant(label="belmont", allow=(ScopePattern(customer="belmont"),)),),
)

NEW_AGENT = Principal(principal_id="agent-unconfigured", kind=PrincipalKind.CUSTOMER)

INTERNAL = Principal(
    principal_id="staff-1",
    kind=PrincipalKind.INTERNAL,
    grants=(Grant(label="all tenants", allow=(ScopePattern(),)),),
)


def echo_markers(prompt: str) -> str:
    """A stub generator: reports which markers it was handed, nothing more."""
    markers = re.findall(r"\[\d+\]", prompt)
    return f"Answered from {len(markers)} entitled section(s): {' '.join(markers)}"


def main() -> None:
    index = InMemoryIndex()
    index.add_all(CORPUS)
    log = AuditLog()
    pipeline = FencedPipeline(index, audit=log, k=3)

    question = "What is the refund window?"
    for principal in (ACME_AGENT, ACME_EU_CONTRACTOR, BELMONT_AGENT, NEW_AGENT, INTERNAL):
        answer = pipeline.answer(principal, question, generate=echo_markers)
        print(f"\n{principal.principal_id} ({principal.kind.value})")
        print(f"  filter    {answer.record.applied_filter}")
        print(f"  returned  {list(answer.record.doc_ids_returned)}")
        print(f"  refused   {answer.record.refused_count} chunk(s) outside the fence")
        print(f"  in prompt {list(answer.context_doc_ids)}, dropped for length: {answer.dropped}")
        print(f"  answer    {answer.text}")

    print("\nThe same question with the refusal count withheld from the caller:")
    withholding = FencedPipeline(index, audit=log, k=3, disclose_excluded=False)
    withheld = withholding.answer(ACME_AGENT, question, generate=echo_markers)
    print(f"  caller reads   refused_count={withheld.record.refused_count}")
    print(f"  log still has  refused_count={log.records[-1].refused_count}")
    print(f"  joined on      sequence={withheld.record.sequence}")

    print("\nDocuments each account saw, straight from the audit log:")
    for principal_id in ("agent-acme", "contractor-eu", "agent-belmont", "agent-unconfigured"):
        print(f"  {principal_id:20} {sorted(log.doc_ids_seen_by(principal_id))}")

    draft = "belmont-draft-pricing"
    everything_returned = {doc for record in log.records for doc in record.doc_ids_returned}
    print(f"\nDraft document retrievable by anyone: {draft in everything_returned}")

    withdrawn = CORPUS[0].model_copy(update={"status": DocumentStatus.RETIRED})
    index.add(withdrawn)
    after = pipeline.answer(ACME_AGENT, question, generate=echo_markers)
    print(
        f"\nAfter withdrawing {withdrawn.doc_id}, the same account retrieves "
        f"{list(after.record.doc_ids_returned)}"
    )


if __name__ == "__main__":
    main()

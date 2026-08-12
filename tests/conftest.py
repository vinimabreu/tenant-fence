"""Shared fixtures. Synthetic tenants, invented names, no real company anywhere."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tenant_fence import (
    Document,
    DocumentStatus,
    Grant,
    InMemoryIndex,
    Principal,
    PrincipalKind,
    Scope,
    ScopePattern,
)

ACME = Scope(customer="acme")
ACME_EU = Scope(customer="acme", site="eu")
ACME_EU_BILLING = Scope(customer="acme", site="eu", system="billing")
ACME_US = Scope(customer="acme", site="us")
BELMONT = Scope(customer="belmont")
BELMONT_EU = Scope(customer="belmont", site="eu")


def grant(*allow: ScopePattern, deny: tuple[ScopePattern, ...] = (), label: str = "") -> Grant:
    """Build a grant without repeating the tuple ceremony in every test."""
    return Grant(label=label, allow=allow, deny=deny)


def principal(
    principal_id: str,
    *grants: Grant,
    kind: PrincipalKind = PrincipalKind.CUSTOMER,
) -> Principal:
    return Principal(principal_id=principal_id, kind=kind, grants=grants)


def document(
    doc_id: str,
    text: str,
    scope: Scope,
    *,
    title: str = "",
    source_uri: str = "",
    status: DocumentStatus = DocumentStatus.APPROVED,
    version: int = 1,
) -> Document:
    return Document(
        doc_id=doc_id,
        text=text,
        scope=scope,
        title=title,
        source_uri=source_uri,
        status=status,
        version=version,
    )


@pytest.fixture
def acme_agent() -> Principal:
    """Sees every site and system of acme."""
    return principal("agent-acme", grant(ScopePattern(customer="acme"), label="acme"))


@pytest.fixture
def belmont_agent() -> Principal:
    """Sees every site and system of belmont."""
    return principal("agent-belmont", grant(ScopePattern(customer="belmont"), label="belmont"))


@pytest.fixture
def acme_eu_agent() -> Principal:
    """Sees only the eu site of acme."""
    return principal("agent-acme-eu", grant(ScopePattern(customer="acme", site="eu")))


@pytest.fixture
def ungranted() -> Principal:
    """A principal that was created and never granted anything."""
    return principal("agent-new")


@pytest.fixture
def internal_staff() -> Principal:
    """Internal support, granted across every tenant."""
    return principal(
        "staff-1",
        grant(ScopePattern(), label="all tenants"),
        kind=PrincipalKind.INTERNAL,
    )


@pytest.fixture
def two_tenant_index() -> InMemoryIndex:
    """One shared index holding both tenants' material."""
    index = InMemoryIndex()
    index.add_all(
        [
            document(
                "acme-refunds",
                "Acme customers may request a refund within 30 days of the invoice date.",
                ACME,
                title="Acme refund window",
                source_uri="kb://acme/refunds",
            ),
            document(
                "acme-eu-vat",
                "Acme eu invoices carry vat at the rate of the billing country.",
                ACME_EU,
                title="Acme eu vat",
                source_uri="kb://acme/eu/vat",
            ),
            document(
                "belmont-refunds",
                "Belmont customers may request a refund within 7 days of the invoice date.",
                BELMONT,
                title="Belmont refund window",
                source_uri="kb://belmont/refunds",
            ),
        ]
    )
    return index


class FixedClock:
    """A clock that starts at a known instant and advances one minute per read.

    Advancing on read keeps records strictly ordered in time without any test
    having to sleep, which is what makes the time window queries testable.
    """

    def __init__(self, start: datetime | None = None, step: timedelta | None = None) -> None:
        self.now = start or datetime(2026, 3, 3, 9, 0, tzinfo=UTC)
        self.step = step or timedelta(minutes=1)

    def __call__(self) -> datetime:
        current = self.now
        self.now = current + self.step
        return current


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock()


class RecordingGenerator:
    """A fake generator that keeps every prompt it was handed."""

    def __init__(self, reply: str = "answer [1]") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply

    @property
    def last_prompt(self) -> str:
        return self.prompts[-1]


@pytest.fixture
def generator() -> RecordingGenerator:
    return RecordingGenerator()

"""Construction time guarantees: canonical ids, no half specified scopes, frozen records."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tenant_fence import (
    AuditRecord,
    Chunk,
    DenyPattern,
    Document,
    DocumentStatus,
    Grant,
    Principal,
    Scope,
    ScopePattern,
    normalise_id,
)

ACME = Scope(customer="acme")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("acme", "acme"),
        ("ACME", "acme"),
        ("  acme  ", "acme"),
        ("\tAcme\n", "acme"),
        ("\x0bacme\x0c", "acme"),
        ("ACME_CORP", "acme_corp"),
    ],
)
def test_normalise_id_trims_ascii_whitespace_then_folds_ascii_case(
    raw: str, expected: str
) -> None:
    assert normalise_id(raw) == expected, (
        f"id {raw!r} did not canonicalise to {expected!r}; inconsistent ids mean the same "
        "tenant can end up stored under two keys, and the fence compares keys"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "\u212aestrel",  # KELVIN SIGN, which casefolds and lowers to ASCII 'k'
        "gro\u00df",  # SHARP S, which casefolds to 'ss'
        "\ufb01nch",  # LIGATURE FI, which casefolds to 'fi'
        "vanti\u017f",  # LONG S, which casefolds to 's'
        "\u0130stanbul",  # DOTTED CAPITAL I, whose lowering is locale dependent
        "\u00c4rger",  # A DIAERESIS, a genuine case pair but outside ASCII
    ],
)
def test_normalise_id_leaves_every_non_ascii_character_exactly_as_written(raw: str) -> None:
    """The fold is A-Z only, because every wider fold is many-to-one somewhere.

    ``str.casefold`` and ``str.lower`` map U+212A onto 'k', U+00DF onto 'ss',
    U+FB01 onto 'fi' and U+017F onto 's'. Each of those merges two separately
    registered tenants into one entitlement key at construction, before any
    matching happens, so the filter, the guard and the audit log all agree the
    resulting cross-tenant read is legitimate.
    """
    assert normalise_id(raw) == raw, (
        f"id {raw!r} canonicalised to {normalise_id(raw)!r}; a fold that changes a "
        "non-ASCII character can map it onto a character another tenant already uses, and "
        "two tenants sharing one key is a cross-tenant read no later layer can catch"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "kronos\u3000",  # IDEOGRAPHIC SPACE
        "kronos\u2007",  # FIGURE SPACE
        "kronos\u00a0",  # NO-BREAK SPACE
        "kronos\u1680",  # OGHAM SPACE MARK
        "kronos\u205f",  # MEDIUM MATHEMATICAL SPACE
        "kronos\u0085",  # NEXT LINE
        "kronos\u2028",  # LINE SEPARATOR
    ],
)
def test_normalise_id_trims_no_unicode_whitespace_beyond_ascii(raw: str) -> None:
    """A bare ``str.strip()`` removes all of these, and removing them merges tenants.

    ``"kronos"`` and ``"kronos" + U+3000`` are two rows in any registry that
    enforces uniqueness on the raw string. Trimming the second character folds
    them onto one entitlement key here, and every layer downstream then agrees
    that a grant on one reads the documents of the other.
    """
    assert normalise_id(raw) != normalise_id("kronos"), (
        f"id {raw!r} canonicalised to {normalise_id(raw)!r}, the same key as the plain "
        "tenant 'kronos'; trimming a Unicode whitespace character merges two separately "
        "registered tenants exactly the way a Unicode case fold does"
    )


@pytest.mark.parametrize(
    "raw",
    ["acme corp", "acme\u3000corp", "acme\u200b", "acme\x1f", "\u00a0"],
)
def test_an_identifier_that_is_not_fully_visible_is_refused(raw: str) -> None:
    with pytest.raises(ValidationError, match="may not contain whitespace"):
        Scope(customer=raw)


def test_normalise_id_alone_never_raises_because_it_only_canonicalises() -> None:
    """The trim and the refusal are separate functions on purpose.

    ``normalise_id`` is exported and callable on anything, including a string a
    caller is about to reject itself. Keeping the refusal in
    ``require_visible_id`` means the canonical form of a hostile id is still
    inspectable, which is what the tests above compare.
    """
    assert normalise_id(" KRONOS\u3000 ") == "kronos\u3000"


def test_scope_ids_are_canonicalised_on_construction() -> None:
    scope = Scope(customer="  ACME  ", site=" EU ", system="Billing")
    assert (scope.customer, scope.site, scope.system) == ("acme", "eu", "billing")


def test_two_spellings_of_the_same_id_are_the_same_scope() -> None:
    assert Scope(customer=" ACME ") == Scope(customer="acme"), (
        "the same tenant written two ways produced two distinct scopes; content filed "
        "under one spelling would be invisible to a grant written with the other"
    )


def test_scope_is_frozen() -> None:
    scope = Scope(customer="acme")
    with pytest.raises(ValidationError):
        scope.customer = "belmont"  # type: ignore[misc]


def test_scope_is_hashable_so_it_can_key_a_set() -> None:
    assert len({Scope(customer="acme"), Scope(customer="ACME"), Scope(customer="belmont")}) == 2


def test_scope_rejects_empty_customer() -> None:
    with pytest.raises(ValidationError, match="customer must not be empty"):
        Scope(customer="   ")


def test_scope_rejects_empty_site_because_it_would_widen_the_grant() -> None:
    with pytest.raises(ValidationError, match="never an empty string"):
        Scope(customer="acme", site="  ")


def test_scope_rejects_a_system_without_a_site() -> None:
    with pytest.raises(ValidationError, match="system cannot be named while site is None"):
        Scope(customer="acme", system="billing")


def test_scope_label_marks_open_levels_with_a_star() -> None:
    assert Scope(customer="acme").label == "acme/*/*"
    assert Scope(customer="acme", site="eu").label == "acme/eu/*"
    assert Scope(customer="acme", site="eu", system="billing").label == "acme/eu/billing"


def test_pattern_label_marks_wildcards() -> None:
    assert ScopePattern().label == "*/*/*"
    assert ScopePattern(customer="acme").label == "acme/*/*"


def test_pattern_rejects_a_site_without_a_customer() -> None:
    with pytest.raises(ValidationError, match="cannot name a site or system while customer is"):
        ScopePattern(site="eu")


def test_pattern_rejects_a_system_without_a_site() -> None:
    with pytest.raises(ValidationError, match="cannot name a system while site is open"):
        ScopePattern(customer="acme", system="billing")


def test_a_deny_pattern_may_name_a_system_across_every_site() -> None:
    """The one hierarchy rule a revocation needs relaxed rather than enforced."""
    everywhere = DenyPattern(customer="acme", system="billing")
    assert everywhere.label == "acme/*/billing"


def test_a_deny_pattern_still_cannot_name_a_site_without_a_customer() -> None:
    with pytest.raises(ValidationError, match="cannot name a site or system while customer is"):
        DenyPattern(site="eu")


def test_a_system_level_deny_is_stored_widened_to_every_site() -> None:
    """What the log renders has to be what the fence enforces."""
    granted = Grant(
        allow=(ScopePattern(customer="acme"),),
        deny=(ScopePattern(customer="acme", site="eu", system="billing"),),
    )
    assert [pattern.label for pattern in granted.deny] == ["acme/*/billing"], (
        f"the deny is stored as {[pattern.label for pattern in granted.deny]}; a deny that "
        "reads as one site and is enforced across all of them, or the reverse, makes the "
        "audit log describe a revocation that never happened"
    )


def test_a_deny_pattern_cannot_be_smuggled_into_an_allow_list() -> None:
    """DenyPattern subclasses ScopePattern, so the type system alone would let it in."""
    with pytest.raises(ValidationError, match="allow pattern cannot name a system"):
        Grant(allow=(DenyPattern(customer="acme", system="billing"),))


def test_a_grant_rebuilt_around_the_validators_is_refused_by_the_principal() -> None:
    """``model_copy`` skips field validators, so the widening has to be re-checked."""
    valid = Grant(
        allow=(ScopePattern(customer="acme"),),
        deny=(ScopePattern(customer="acme", site="eu", system="billing"),),
    )
    unwidened = valid.model_copy(
        update={"deny": (ScopePattern(customer="acme", site="eu", system="billing"),)}
    )
    with pytest.raises(ValidationError, match="deny naming a system must leave the site open"):
        Principal(principal_id="agent-acme", grants=(unwidened,))


@pytest.mark.parametrize(
    ("customer", "expected"),
    [
        ("*", "\\*/*/*"),
        ("acme/eu", "acme\\/eu/*/*"),
        ("acme\\eu", "acme\\\\eu/*/*"),
    ],
)
def test_an_identifier_cannot_forge_the_rendering_of_an_open_level(
    customer: str, expected: str
) -> None:
    """A tenant that names itself '*' must not read as the all tenants grant."""
    assert Scope(customer=customer).label == expected, (
        f"tenant {customer!r} renders as {Scope(customer=customer).label!r}; an identifier "
        "that reproduces the structural characters of a label makes every audit line and "
        "every breach message about it ambiguous, and a reviewer cannot then tell one "
        "document's own scope from a grant across every tenant"
    )
    assert Scope(customer=customer).label != ScopePattern().label


def test_document_defaults_to_approved() -> None:
    assert Document(doc_id="d1", text="x", scope=Scope(customer="acme")).status is (
        DocumentStatus.APPROVED
    )


def test_document_rejects_empty_doc_id() -> None:
    with pytest.raises(ValidationError, match="doc_id must not be empty"):
        Document(doc_id="  ", text="x", scope=Scope(customer="acme"))


def test_document_id_is_stripped_but_never_casefolded() -> None:
    document = Document(doc_id="  Policy-A  ", text="x", scope=Scope(customer="acme"))
    assert document.doc_id == "Policy-A", (
        "doc_id was folded; it is an identity key, not an entitlement coordinate, and "
        "folding it can merge two genuinely distinct documents"
    )


def test_document_version_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Document(doc_id="d1", text="x", scope=Scope(customer="acme"), version=0)


def test_chunk_cannot_be_built_without_a_scope() -> None:
    with pytest.raises(ValidationError):
        Chunk(chunk_id="d1#0", doc_id="d1", text="x")  # type: ignore[call-arg]


def test_the_cache_key_of_two_tenants_sharing_a_doc_id_is_not_the_same_key() -> None:
    """chunk_id is what a cache reaches for, and tenants choose their own doc ids."""
    mine = Chunk(chunk_id="refund-policy#0", doc_id="refund-policy", text="30 days", scope=ACME)
    theirs = Chunk(
        chunk_id="refund-policy#0",
        doc_id="refund-policy",
        text="7 days",
        scope=Scope(customer="belmont"),
    )
    assert mine.chunk_id == theirs.chunk_id, "the premise of this test has changed"
    assert mine.cache_key != theirs.cache_key, (
        f"both tenants' chunks key on {mine.cache_key!r}; a retrieval cache, a dedup pass "
        "or a citation table keyed on it would serve acme's text under belmont's key, "
        "downstream of a fence that did its job"
    )
    assert mine.cache_key == "acme/*/*::refund-policy#0"


def test_a_crafted_scope_segment_cannot_forge_another_chunks_cache_key() -> None:
    """The separator is ``::``, and a tenant chosen identifier may contain a colon.

    Not a cross-customer break: the first two unescaped slashes still pin the
    customer and the site, and both chunks below belong to acme. It crosses the
    system level, which this package does enforce with grants and denies, and it
    breaks it in the one place the docstring tells people to rely on. A
    retrieval cache, a dedup pass or a vector store upsert keyed on
    ``cache_key`` would serve one system's text under the other system's key,
    downstream of a fence that did its job.
    """
    forged = Chunk(
        chunk_id="p#0",
        doc_id="p",
        text="chiller runbook",
        scope=Scope(customer="acme", site="bexley", system="chiller::x"),
    )
    target = Chunk(
        chunk_id="x::p#0",
        doc_id="x",
        text="site wide runbook",
        scope=Scope(customer="acme", site="bexley", system="chiller"),
    )
    assert forged.cache_key != target.cache_key, (
        f"two different (scope, chunk) pairs both key on {forged.cache_key!r}; the system "
        "segment reproduced the separator, so a cache keyed on cache_key holds one entry "
        "for two systems and hands the wrong text back"
    )
    assert "::" not in forged.scope.label, (
        f"the scope label {forged.scope.label!r} still contains the raw separator, so the "
        "split between the label and the chunk id is ambiguous no matter what the keys "
        "compare as today"
    )


def test_a_grant_built_without_validation_is_refused_by_the_principal() -> None:
    """``model_construct`` skips validation outright, so the principal re-checks.

    Pydantic re-runs an after validator on a nested model instance today, which
    catches this, but the package pins the behaviour rather than resting on it:
    if that default ever changes, an allow pattern naming a system under an open
    site becomes live, which means that system at every site of the customer,
    and no test would turn red.
    """
    bypassed = ScopePattern.model_construct(customer="acme", site=None, system="billing")
    smuggled = Grant.model_construct(label="acme billing", allow=(bypassed,))

    with pytest.raises(ValidationError, match="allow pattern cannot name a system"):
        Principal(principal_id="agent-acme", grants=(smuggled,))


def test_a_deny_built_without_validation_is_refused_by_the_principal() -> None:
    pinned = ScopePattern.model_construct(customer="acme", site="eu", system="billing")
    smuggled = Grant.model_construct(label="acme, eu billing revoked", deny=(pinned,))

    with pytest.raises(ValidationError, match="deny naming a system must leave the site open"):
        Principal(principal_id="agent-acme", grants=(smuggled,))


def test_principal_defaults_to_no_grants() -> None:
    assert Principal(principal_id="new-agent").grants == (), (
        "a principal was created holding entitlement nobody granted it; the default "
        "must be deny by default"
    )


def test_principal_rejects_an_empty_id() -> None:
    with pytest.raises(ValidationError, match="principal_id must not be empty"):
        Principal(principal_id="  ")


def test_audit_record_is_frozen() -> None:
    record = AuditRecord(
        principal_id="agent-1",
        query="q",
        applied_filter="allow=[acme/*/*] deny=[]",
        timestamp=datetime(2026, 3, 3, tzinfo=UTC),
    )
    with pytest.raises(ValidationError):
        record.doc_ids_returned = ("tampered",)  # type: ignore[misc]

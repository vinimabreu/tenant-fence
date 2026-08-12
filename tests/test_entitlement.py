"""The matching table. These tests are the specification of who may see what."""

from __future__ import annotations

import pytest

from conftest import (
    ACME,
    ACME_EU,
    ACME_EU_BILLING,
    ACME_US,
    BELMONT,
    grant,
    principal,
)
from tenant_fence import (
    DenyPattern,
    Principal,
    PrincipalKind,
    Scope,
    ScopePattern,
    describe_filter,
    effective_allows,
    is_allowed,
    is_denied,
    pattern_matches,
    scope_filter,
    segment_matches,
)

INTERNAL = PrincipalKind.INTERNAL


def test_principal_with_no_grants_sees_nothing() -> None:
    nobody = principal("agent-new")
    for scope in (ACME, ACME_EU, ACME_EU_BILLING, BELMONT):
        assert is_allowed(nobody, scope) is False, (
            f"a principal with zero grants was allowed to read {scope.label}; deny by "
            "default is broken, which means every unconfigured account reads everything"
        )


def test_grant_on_a_customer_covers_every_site_and_system() -> None:
    agent = principal("a", grant(ScopePattern(customer="acme")))
    assert is_allowed(agent, ACME)
    assert is_allowed(agent, ACME_EU)
    assert is_allowed(agent, ACME_EU_BILLING)


def test_grant_on_a_customer_covers_no_other_customer() -> None:
    agent = principal("a", grant(ScopePattern(customer="acme")))
    assert is_allowed(agent, BELMONT) is False, (
        "an acme grant read belmont content; this is the cross-customer leak the "
        "package exists to prevent"
    )


def test_grant_naming_a_site_does_not_cover_another_site() -> None:
    agent = principal("a", grant(ScopePattern(customer="acme", site="eu")))
    assert is_allowed(agent, ACME_EU)
    assert is_allowed(agent, ACME_US) is False, (
        "a grant scoped to acme/eu read acme/us content; site isolation is broken"
    )


def test_grant_naming_a_site_does_not_cover_the_customer_level_document() -> None:
    agent = principal("a", grant(ScopePattern(customer="acme", site="eu")))
    assert is_allowed(agent, ACME) is False, (
        "a grant on one site read a document filed at the customer level; that document "
        "is not confined to the granted site, so it must not be visible"
    )


def test_grant_naming_a_system_covers_only_that_system() -> None:
    agent = principal("a", grant(ScopePattern(customer="acme", site="eu", system="billing")))
    assert is_allowed(agent, ACME_EU_BILLING)
    assert is_allowed(agent, ACME_EU) is False
    assert is_allowed(agent, Scope(customer="acme", site="eu", system="support")) is False


def test_internal_wildcard_grant_sees_every_customer() -> None:
    """The one principal kind a cross-tenant grant is honoured for."""
    staff = principal("staff", grant(ScopePattern(), label="all tenants"), kind=INTERNAL)
    assert is_allowed(staff, ACME)
    assert is_allowed(staff, BELMONT)


def test_the_same_wildcard_grant_opens_nothing_for_a_customer_principal() -> None:
    """A cross-tenant allow on a tenant account is a loading bug, so it fails closed."""
    customer = principal("agent-acme", grant(ScopePattern(), label="all tenants"))
    for scope in (ACME, BELMONT):
        assert is_allowed(customer, scope) is False, (
            f"a principal declared kind=customer read {scope.label} through an allow "
            "pattern with an open customer; all three pattern levels default to None and "
            "None means any, so a grant record that lost its customer field during "
            "deserialisation turns a single tenant account into a cross tenant one"
        )
    assert describe_filter(customer) == "allow=[*/*/*] deny=[]", (
        "the grant stopped being legible in the audit log; refusing to honour it is only "
        "half the control, the other half is a reviewer being able to see it was issued"
    )


@pytest.mark.parametrize(
    "impostor",
    ["acme-corp", "acme_corp", "acmecorp", "acme.corp", "acm", "acmex", "xacme"],
)
def test_exact_segment_match_rejects_prefix_confusion(impostor: str) -> None:
    """The spaced spelling, 'acme corp', is not here because it cannot be built.

    An identifier carrying whitespace is refused at construction rather than
    matched and rejected; see
    ``test_an_identifier_carrying_whitespace_or_an_invisible_character_is_refused``.
    """
    agent = principal("a", grant(ScopePattern(customer="acme")))
    assert is_allowed(agent, Scope(customer=impostor)) is False, (
        f"a grant on 'acme' matched tenant {impostor!r}; prefix or substring confusion "
        "at the fence is a breach, matching must be exact per segment"
    )


@pytest.mark.parametrize(
    ("pattern", "value", "expected"),
    [
        (None, "acme", True),
        (None, None, True),
        ("acme", "acme", True),
        ("acme", "acme-corp", False),
        ("acme", "acme ", False),
        ("acme", "ACME_corp", False),
        ("acme", "ACME", False),
        ("acme", None, False),
        ("eu", "eu-west", False),
    ],
)
def test_segment_matches_table(pattern: str | None, value: str | None, expected: bool) -> None:
    assert segment_matches(pattern, value) is expected, (
        f"segment_matches({pattern!r}, {value!r}) should be {expected}; this function is "
        "raw equality on already canonical ids and must never become fuzzy"
    )


def test_segment_matches_does_not_normalise_its_inputs() -> None:
    assert segment_matches("acme", " acme ") is False, (
        "the matcher normalised its own input; normalisation belongs at construction "
        "only, or the fence silently forgives whatever a caller failed to canonicalise"
    )


def test_named_pattern_does_not_match_an_absent_segment() -> None:
    assert pattern_matches(ScopePattern(customer="acme", site="eu"), ACME) is False


def test_deny_beats_allow_in_the_same_grant() -> None:
    agent = principal(
        "a",
        grant(ScopePattern(customer="acme"), deny=(ScopePattern(customer="acme", site="eu"),)),
    )
    assert is_allowed(agent, ACME_US)
    assert is_denied(agent, ACME_EU)
    assert is_allowed(agent, ACME_EU) is False, "deny lost to allow inside a single grant"


def test_deny_beats_allow_across_separate_grants() -> None:
    agent = principal(
        "a",
        grant(deny=(ScopePattern(customer="acme", site="eu"),), label="revocation"),
        grant(ScopePattern(customer="acme"), label="broad access"),
    )
    assert is_allowed(agent, ACME_EU) is False, (
        "a deny in one grant was undone by an allow in another; a revocation that a "
        "second grant can override is not a revocation"
    )
    assert is_allowed(agent, ACME_US)


def test_deny_at_the_customer_level_closes_every_site_below_it() -> None:
    agent = principal(
        "a",
        grant(ScopePattern(customer="acme"), deny=(ScopePattern(customer="acme"),)),
    )
    for scope in (ACME, ACME_EU, ACME_EU_BILLING):
        assert is_allowed(agent, scope) is False


def test_deny_on_a_site_leaves_the_rest_of_the_customer_open() -> None:
    agent = principal(
        "a",
        grant(
            ScopePattern(customer="acme"),
            deny=(ScopePattern(customer="acme", site="eu", system="billing"),),
        ),
    )
    assert is_allowed(agent, ACME_EU)
    assert is_allowed(agent, ACME_EU_BILLING) is False


def test_a_revoked_system_stays_revoked_at_a_site_created_afterwards() -> None:
    """The site an operator could not know about when writing the revocation.

    A system level deny had to name a site, because an allow pattern may not
    leave the middle level open. Enforced literally, the revocation covers
    today's sites and silently exempts every site created after it was written,
    so the account reads acme/apac/billing the day apac exists.
    """
    agent = principal(
        "agent-acme",
        grant(
            ScopePattern(customer="acme"),
            deny=(ScopePattern(customer="acme", site="eu", system="billing"),),
        ),
    )
    future = Scope(customer="acme", site="apac", system="billing")
    assert is_allowed(agent, future) is False, (
        f"the billing system at the newly created site {future.label} is readable by a "
        "principal whose billing access was revoked; revocation fails open on a schedule "
        "nobody is watching"
    )
    assert is_allowed(agent, Scope(customer="acme", site="apac")) is True, (
        "the revocation closed the whole new site, not the revoked system within it; "
        "widening a deny past the level it names revokes work nobody asked to revoke"
    )


def test_a_deny_written_directly_across_every_site_matches_the_widened_form() -> None:
    """DenyPattern is the canonical spelling of what the widening produces."""
    written = principal(
        "a", grant(ScopePattern(customer="acme"), deny=(DenyPattern(customer="acme", system="hr"),))
    )
    enumerated = principal(
        "b",
        grant(
            ScopePattern(customer="acme"),
            deny=(ScopePattern(customer="acme", site="eu", system="hr"),),
        ),
    )
    for scope in (
        Scope(customer="acme", site="eu", system="hr"),
        Scope(customer="acme", site="us", system="hr"),
    ):
        assert is_allowed(written, scope) is is_allowed(enumerated, scope) is False
    assert describe_filter(written) == describe_filter(enumerated), (
        "the two spellings of one revocation render differently in the log, so an "
        "operator comparing two accounts cannot tell they hold the same entitlement"
    )


def test_a_principal_kind_nobody_has_written_yet_reads_no_cross_tenant_grant() -> None:
    """The rule is an allow list, so a third kind starts with nothing.

    ``PrincipalKind`` has two members today and the interesting failure is the
    day it has three. A service account added as ``SERVICE = "service"`` would
    inherit cross-tenant reach on a ``*/*/*`` allow under a deny list check
    (``is CUSTOMER``): no error, no failing test, and an audit line rendering
    the grant as legitimate. The kind is forced in here with
    ``model_construct`` because the enum cannot be extended from a test, and the
    point is precisely what happens to a value this module has never seen.
    """
    future_kind = Principal.model_construct(
        principal_id="svc-future",
        kind="service",
        grants=(grant(ScopePattern()),),
    )

    assert list(effective_allows(future_kind)) == [], (
        "a principal of an unrecognised kind was handed the cross-tenant allow; the check "
        "has to name the one kind that may hold it, not the kinds that may not, or every "
        "future member of the enum starts life reading every tenant in the index"
    )
    assert is_allowed(future_kind, ACME) is False
    assert describe_filter(future_kind) == "allow=[*/*/*] deny=[]", (
        "the grant stopped being legible in the log; refusing to honour it is only half "
        "the control"
    )


def test_case_and_whitespace_variants_resolve_to_the_same_tenant() -> None:
    agent = principal("a", grant(ScopePattern(customer="  ACME  ")))
    assert is_allowed(agent, Scope(customer="acme"))


def test_scope_filter_agrees_with_is_allowed_on_every_scope() -> None:
    agent = principal(
        "a",
        grant(ScopePattern(customer="acme"), deny=(ScopePattern(customer="acme", site="us"),)),
    )
    predicate = scope_filter(agent)
    for scope in (ACME, ACME_EU, ACME_EU_BILLING, ACME_US, BELMONT):
        assert predicate(scope) == is_allowed(agent, scope), (
            f"the compiled index predicate and the guard rule disagreed on {scope.label}; "
            "they must be the same rule or the index and the guard will drift apart"
        )


def test_describe_filter_shows_deny_by_default_explicitly() -> None:
    assert describe_filter(principal("agent-new")) == "allow=[] deny=[]", (
        "an ungranted principal did not render as an empty entitlement; the audit log "
        "has to distinguish 'entitled to nothing' from 'found nothing'"
    )


def test_describe_filter_lists_allow_and_deny_patterns() -> None:
    agent = principal(
        "a",
        grant(ScopePattern(customer="acme"), deny=(ScopePattern(customer="acme", site="eu"),)),
    )
    assert describe_filter(agent) == "allow=[acme/*/*] deny=[acme/eu/*]"

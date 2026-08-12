"""The matching rules. Pure functions, no I/O, no state.

This module is the only place in the package that decides whether a principal
may see a scope. The index filter and the guard both call into it, so there is
one rule with two callers rather than two implementations of one rule. Two
implementations always drift, and the drift is only discovered by the customer
who received someone else's document.

The four rules, in evaluation order:

1. Deny wins. A deny pattern anywhere in the principal's grants closes the scope
   even if another grant opens it.
2. A cross-tenant allow (``customer=None``) is honoured only for an INTERNAL
   principal. Held by a CUSTOMER principal it opens nothing.
3. Allow is explicit. A scope is visible only if some allow pattern matches it.
4. Everything else is invisible. A principal with no grants sees nothing.

Matching is exact per segment. Identifiers arrive already canonical (see
:func:`tenant_fence.models.normalise_id`), so these functions compare with
``==`` and never re-normalise. That is deliberate: a matcher that normalises its
own inputs is forgiving of whatever the caller forgot to canonicalise, and
"forgiving" at the fence means ``acme`` eventually matching ``acme-corp``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from .models import Principal, PrincipalKind, Scope, ScopePattern

ScopeFilter = Callable[[Scope], bool]
"""A predicate over scopes, handed to an index so it can filter before scoring."""


def segment_matches(pattern: str | None, value: str | None) -> bool:
    """Match one segment of a pattern against one segment of a scope.

    ``None`` in the pattern is a wildcard and matches anything, including a
    scope segment that is itself ``None``.

    A named pattern segment matches only an identical scope segment. It does not
    match a prefix, a suffix, a different case, or a value with surrounding
    whitespace, and it does not match ``None``: a grant on site ``eu`` does not
    cover a document filed at the customer level with no site, because that
    document is not confined to ``eu``.

    No normalisation happens here. Both sides were canonicalised at construction.
    """
    if pattern is None:
        return True
    return pattern == value


def pattern_matches(pattern: ScopePattern, scope: Scope) -> bool:
    """True when every segment of ``pattern`` matches the same segment of ``scope``."""
    return (
        segment_matches(pattern.customer, scope.customer)
        and segment_matches(pattern.site, scope.site)
        and segment_matches(pattern.system, scope.system)
    )


def is_denied(principal: Principal, scope: Scope) -> bool:
    """True when any deny pattern in any of the principal's grants covers ``scope``.

    Deny is evaluated across the whole principal, not per grant. A deny in one
    grant closes a scope that a different grant opens. The alternative, scoping
    deny to its own grant, means a revocation can be undone by adding a second
    grant, which is a revocation that does not revoke.
    """
    return any(
        pattern_matches(pattern, scope) for grant in principal.grants for pattern in grant.deny
    )


def effective_allows(principal: Principal) -> Iterator[ScopePattern]:
    """The allow patterns this principal may actually use, in grant order.

    A pattern with ``customer=None`` is a grant across every tenant in the
    index. Held by a principal declared as CUSTOMER it is refused here, and the
    reason is the shape of the accident that produces one. All three levels of
    :class:`~tenant_fence.models.ScopePattern` default to ``None`` and ``None``
    means "any", so a grant record deserialised from a source that dropped its
    ``customer`` key, or set it to null, becomes ``*/*/*``: a single tenant
    account silently turns into a cross-tenant one with no error raised
    anywhere. The empty string is already refused at construction for exactly
    this reason (see ``_canonical_segment``); an absent key reaches the same
    widening by a different road, and it has to fail closed the same way.

    Refusing at match time rather than at construction is deliberate. A grant
    loader that produces this shape is broken, and the account has to read
    nothing until someone fixes it; raising instead would take down the request
    path on a configuration error, and the grant as written stays legible in the
    audit log either way, which is what a review needs to see.

    An INTERNAL principal is unaffected. Staff accounts are where a legitimate
    cross-tenant grant lives, and the log renders it as ``*/*/*``.

    The test is written as an allow list (``is not INTERNAL``) rather than as a
    deny list (``is CUSTOMER``), and the difference only shows up the day
    somebody adds a third member to :class:`~tenant_fence.models.PrincipalKind`.
    A deny list would hand every principal of that new kind a cross-tenant read
    on a ``*/*/*`` allow, with no error, no failing test and an audit line
    rendering the grant as legitimate. An allow list gives the new kind nothing
    until someone writes it down.
    """
    for grant in principal.grants:
        for pattern in grant.allow:
            if pattern.customer is None and principal.kind is not PrincipalKind.INTERNAL:
                continue
            yield pattern


def is_allowed(principal: Principal, scope: Scope) -> bool:
    """The whole rule: deny wins, then allow must be explicit, else invisible."""
    if is_denied(principal, scope):
        return False
    return any(pattern_matches(pattern, scope) for pattern in effective_allows(principal))


def scope_filter(principal: Principal) -> ScopeFilter:
    """Compile a principal into the predicate an index applies before scoring.

    The predicate closes over :func:`is_allowed`, the same function the guard
    calls. That is the whole point of this function existing: the filter and the
    final check cannot disagree, because they are the same code.
    """

    def allowed(scope: Scope) -> bool:
        return is_allowed(principal, scope)

    return allowed


def describe_filter(principal: Principal) -> str:
    """Render the effective entitlement as one line for the audit log.

    A principal with no grants renders as ``allow=[] deny=[]``, so deny by
    default is visible in the log rather than inferred from an empty result set.
    Reading "this account was entitled to nothing" beats guessing why a query
    returned zero rows.

    Grants are rendered as they were written, not as they were enforced. A
    CUSTOMER principal holding ``*/*/*`` reads nothing (see
    :func:`effective_allows`) and still logs ``allow=[*/*/*]``, because the
    review control is being able to see the misconfiguration in the log. A
    filter line that quietly showed the enforced subset would hide the very
    thing a reviewer is looking for.

    The line has syntax (``[``, ``]``, ``, `` and ``=``) and the identifiers
    inside it are chosen by tenants, so every segment arrives escaped from
    :func:`~tenant_fence.models.render_segment`. Unescaped, a customer id
    spelled ``acme,belmont/*/*]deny=[belmont`` renders a ``deny=[...]`` clause
    the principal does not hold, and a reviewer or a log parser reads a
    revocation that was never issued. Identifiers cannot contain whitespace
    either (see :func:`~tenant_fence.models.require_visible_id`), so ``
    deny=`` cannot be spelled at all.
    """
    allow = [pattern.label for grant in principal.grants for pattern in grant.allow]
    deny = [pattern.label for grant in principal.grants for pattern in grant.deny]
    return f"allow=[{', '.join(allow)}] deny=[{', '.join(deny)}]"

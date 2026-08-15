"""Value objects that carry entitlement through the retrieval path.

Every type here is a plain, serializable value object with no external
dependency, so the whole package runs offline with no API key.

Three design decisions in this module are load bearing for the security property:

1. Identifiers are canonicalised once, at construction, by :func:`normalise_id`.
   Nothing downstream re-normalises. Matching is then exact equality on values
   that are already canonical. Normalising in two places is how the two copies
   drift apart, and a fence that is forgiving in one of them is not a fence.

2. Canonicalisation trims ASCII whitespace and folds ASCII case, and does
   nothing else. Any transform that maps two different identifiers onto one
   merges two tenants into one entitlement key, which is a cross-tenant read
   that no later layer can catch. Whatever survives the trim has to be visible:
   :func:`require_visible_id` refuses an identifier carrying whitespace or a
   non-printable character rather than folding it onto its neighbour.

3. Scopes are hierarchical and cannot be partially specified. ``site`` may only
   be named when ``customer`` is named, and ``system`` may only be named when
   ``site`` is named. A half specified scope reads like a narrow grant and
   behaves like a broad one, which is exactly the confusion that leaks data.
   :class:`DenyPattern` relaxes exactly one of those rules, for the reason
   written on the class.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from string import ascii_lowercase, ascii_uppercase
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

_ASCII_CASE_FOLD = str.maketrans(ascii_uppercase, ascii_lowercase)
"""A-Z to a-z and nothing else. See :func:`normalise_id` for why nothing else."""

ASCII_WHITESPACE = " \t\n\r\v\f"
"""The only characters :func:`normalise_id` trims. See the function for why only these."""

_LABEL_ESCAPES = str.maketrans({character: f"\\{character}" for character in "\\/*:,[]="})
"""Every character a label or an audit line uses structurally. See :func:`render_segment`."""


def normalise_id(value: str) -> str:
    """Return the canonical form of an entitlement identifier.

    The rule is trim ASCII whitespace, then fold ASCII case. ``" ACME "`` and
    ``"acme"`` are the same tenant; ``"acme-corp"`` is a different one and always
    will be.

    The trim is ASCII only, and that is load bearing rather than fussy.
    ``str.strip()`` with no argument removes every Unicode whitespace character,
    which merges identifiers the same way a Unicode case fold does: an upstream
    registry enforcing uniqueness on the raw string accepts ``"kronos"`` and
    ``"kronos" + U+3000`` (IDEOGRAPHIC SPACE) as two tenants, and a full strip
    folds them onto one entitlement key before a single comparison happens.
    U+2007, U+00A0, U+1680, U+205F, U+0085 and U+2028 all take the same road.
    Trimming ``" \\t\\n\\r\\v\\f"`` and nothing else cannot merge two identifiers
    that differ anywhere outside those six characters.

    What is left after the trim still has to be a visible identifier;
    :func:`require_visible_id` refuses the rest at construction, so the pair
    above is a loud error rather than a second invisible tenant.

    Case folding is deliberately limited to ``A-Z``. ``str.casefold`` and
    ``str.lower`` both apply the Unicode case folding tables, which are
    many-to-one across characters that are not case variants of each other.
    Four entries, written as code points because a glyph nobody can see is the
    whole trick: U+212A KELVIN SIGN folds to ASCII ``k``, U+00DF SHARP S folds
    to ``ss``, U+FB01 LIGATURE FI folds to ``fi``, U+017F LONG S folds to
    ``s``. Any of those merges two separately registered tenants into one key
    before a single comparison happens, and every layer downstream then agrees
    that the read is legitimate. Folding only ASCII case cannot merge two
    distinct characters, because ``A-Z`` and ``a-z`` are genuine case pairs and
    nothing else is touched.

    The cost of that choice is stated plainly: a non-ASCII identifier is never
    folded at all, so ``"Ärger"`` and ``"ärger"`` are two tenants. That fails
    closed (content filed under one spelling is invisible to a grant written
    with the other) rather than open, and it is the direction this package
    always takes. Issue ASCII identifiers upstream.
    """
    return value.strip(ASCII_WHITESPACE).translate(_ASCII_CASE_FOLD)


def require_visible_id(value: str, *, level: str) -> str:
    """Return ``value`` if every character in it is visible, raise if one is not.

    Applied to an entitlement identifier after :func:`normalise_id` has trimmed
    it. Whatever survives that trim may not contain whitespace of any kind, nor
    a character the Unicode database classes as Other or Separator: control
    codes, the line and paragraph separators, and the format characters
    (U+200B ZERO WIDTH SPACE, U+00AD SOFT HYPHEN, U+FEFF, U+200E LEFT TO RIGHT
    MARK).

    Refusing is the point, and refusing loudly is the rest of it. An identifier
    carrying an invisible character is either a paste accident or an
    impersonation of a tenant that already exists, and the two cannot be told
    apart from here. Accepting it would file the content under a key nobody can
    type, read or spot in an audit line next to the tenant it shadows; trimming
    the character away would merge two separately registered tenants into one
    entitlement key, which is the cross-tenant read this package exists to
    prevent. A ``ValueError`` at construction is the only option that leaves a
    human able to see what happened.

    The error names the code point rather than printing the character, because
    the whole problem is that printing it shows nothing.
    """
    for position, character in enumerate(value):
        if character.isspace() or not character.isprintable():
            raise ValueError(
                f"{level} carries U+{ord(character):04X} at position {position} "
                f"({value!r}); an entitlement identifier may not contain whitespace or a "
                "non-printable character, because an id nobody can see is either a paste "
                "accident or an impersonation of the tenant it renders identically to"
            )
    return value


def render_segment(value: str | None) -> str:
    """Render one scope segment for a label, escaping the structural characters.

    ``None`` renders as ``*``, the open level. Everything the labels and the
    audit line use structurally is escaped: ``\\``, ``/``, ``*``, ``:``, ``,``,
    ``[``, ``]`` and ``=``. Tenants choose their own identifiers, and an
    unescaped one forges the syntax around it in three places:

    1. A tenant that registers the id ``*`` produces the label ``*/*/*``, byte
       identical to the rendering of a grant across every tenant, so a reviewer
       reading an audit line or a breach message cannot tell one document's own
       scope from a maximal entitlement.
    2. ``,``, ``[``, ``]`` and ``=`` are the syntax of
       :func:`tenant_fence.entitlement.describe_filter`. Unescaped, an id
       spelled ``acme,belmont/*/*]deny=[belmont`` renders a ``deny=[...]``
       clause the principal does not hold.
    3. ``:`` is half of the ``::`` separator in
       :attr:`Chunk.cache_key`, where an unescaped one lets two different
       (scope, chunk) pairs produce one identical key.

    Access control is unaffected either way, since matching compares the
    segments and never the rendering. Audit legibility and key uniqueness are
    not.
    """
    if value is None:
        return "*"
    return value.translate(_LABEL_ESCAPES)


def _canonical_segment(value: str | None, *, level: str) -> str | None:
    """Canonicalise one optional scope segment, refusing the empty string.

    An empty string is rejected rather than coerced to ``None``. At every level
    ``None`` means "any", so silently turning ``""`` into ``None`` would widen a
    grant from one site to every site of that customer. A widening typo must
    fail loudly at construction, not quietly at query time.
    """
    if value is None:
        return None
    folded = normalise_id(value)
    if not folded:
        raise ValueError(
            f"{level} was empty after normalisation; use None to mean 'any {level}', "
            "never an empty string, because the two are not the same entitlement"
        )
    return require_visible_id(folded, level=level)


class TenantCollision(ValueError):
    """Raised when a registration lands on a canonical key that is already minted.

    The message always carries the spelling that arrived, the spelling that got
    there first and the shared key, written with ``repr`` so a pair that renders
    identically on screen still reads as two different strings. A collision is
    only ever resolved by a human, and the human can only resolve what they can
    see.
    """


class TenantRegistry:
    """Write-time uniqueness for tenant identifiers: the unique index, made explicit.

    :func:`normalise_id` is deliberately many-to-one across exactly two things,
    ASCII case and edge whitespace, so that a sloppy reference to an existing
    tenant resolves to that tenant. Right for a reference, wrong for a
    registration. When two separately minted identifiers land on one canonical
    key, every layer downstream agrees they are the same tenant, and the
    disagreement about who owns the key surfaces at query time, where it is a
    cross-tenant read.

    This class moves that discovery to write time, where it is a validation
    message. :meth:`register` mints a canonical key exactly once; any second
    registration that lands on a minted key is refused with
    :class:`TenantCollision`, whether it is the same spelling twice (a retry or
    a duplicate) or a different spelling folding onto it (the dangerous case).
    A reader's review of the public write-up asked for exactly this and said it
    plainly: the second ``strasse-werke`` should fail loudly at write time
    instead of silently sharing an entitlement.

    The registry does not replace the fold, and it does not catch the pairs the
    fold deliberately keeps apart: ``"Straße-Werke"`` and ``"strasse-werke"``
    mint two distinct keys here, two tenants, isolated in both directions. What
    it catches is every pair the fold merges. If that merge was not an
    intentional reference to the existing tenant, registration is the only
    moment a human is present to notice, and this is the refusal that makes
    them look.
    """

    def __init__(self) -> None:
        self._spellings: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._spellings)

    def __contains__(self, spelling: str) -> bool:
        """Whether ``spelling`` resolves to a minted key.

        Normalises first, so any spelling that would reference the tenant
        answers True. Lookups never mint and never refuse; only
        :meth:`register` does either.
        """
        return normalise_id(spelling) in self._spellings

    def spelling_of(self, key: str) -> str:
        """The exact spelling that minted ``key``; ``KeyError`` if nothing has."""
        return self._spellings[key]

    def register(self, spelling: str) -> str:
        """Mint the canonical key for ``spelling``, exactly once, and return it.

        The key is the :func:`normalise_id` form, checked by
        :func:`require_visible_id` the same way a :class:`Scope` customer is,
        so nothing can be minted here that a scope would refuse to name.
        """
        folded = normalise_id(spelling)
        if not folded:
            raise ValueError(
                "a tenant identifier must not be empty after trimming; "
                "there is no anonymous tenant to mint"
            )
        key = require_visible_id(folded, level="customer")
        first = self._spellings.get(key)
        if first is None:
            self._spellings[key] = spelling
            return key
        if first == spelling:
            raise TenantCollision(
                f"tenant {spelling!r} is already registered; a registry mints a key "
                "exactly once, and a second identical registration is either a retry "
                "that deserves a clear answer or two callers who both believe they "
                "created this tenant"
            )
        raise TenantCollision(
            f"tenant spelling {spelling!r} normalises to {key!r}, which is already "
            f"registered as {first!r}. Two distinct spellings are claiming one "
            "entitlement key. A reference to the existing tenant needs no "
            "registration, and a genuinely new tenant needs an identifier that does "
            "not fold onto a minted one"
        )


class DocumentStatus(StrEnum):
    """Publication state of a document.

    APPROVED is the only retrievable state. DRAFT and RETIRED content stays out
    of the index entirely, so no query path can reach it by accident.
    """

    APPROVED = "approved"
    DRAFT = "draft"
    RETIRED = "retired"


class PrincipalKind(StrEnum):
    """Who is asking.

    CUSTOMER is scoped to the tenants named in its grants. INTERNAL is a staff
    account that may hold a cross customer grant.

    The kind grants nothing on its own: only grants grant. It does take one
    thing away. An allow pattern with an open ``customer`` is a grant across
    every tenant in the index, and a CUSTOMER principal may not hold one; see
    :func:`tenant_fence.entitlement.is_allowed` for the enforcement and the
    reason. Deny patterns are never restricted by kind, because narrowing a
    revocation is the unsafe direction.
    """

    INTERNAL = "internal"
    CUSTOMER = "customer"


class Scope(BaseModel):
    """The entitlement coordinates of a piece of content.

    Three levels, narrowing left to right: ``customer`` (always present),
    ``site`` and ``system``. A document filed at ``acme`` with no site belongs to
    the customer as a whole. A document filed at ``acme / eu / billing`` belongs
    to one system at one site.

    Identifiers are normalised on construction with :func:`normalise_id` (trim
    ASCII whitespace, then fold ASCII case) and then checked by
    :func:`require_visible_id`. This is deliberate and it is documented here
    because id confusion IS the attack surface of a multi-tenant index:
    ``"Acme "`` and ``"acme"`` must resolve to one tenant, while ``"acme-corp"``
    must never resolve to ``"acme"``. Matching downstream is exact equality, so
    everything that enters the system has to be canonical before it is compared,
    and the canonicalisation itself has to be injective on anything that is not
    an ASCII case pair.
    """

    model_config = ConfigDict(frozen=True)

    customer: str
    site: str | None = None
    system: str | None = None

    @field_validator("customer")
    @classmethod
    def _canonical_customer(cls, value: str) -> str:
        folded = normalise_id(value)
        if not folded:
            raise ValueError("customer must not be empty; every document belongs to a tenant")
        return require_visible_id(folded, level="customer")

    @field_validator("site", "system")
    @classmethod
    def _canonical_optional(cls, value: str | None, info: ValidationInfo) -> str | None:
        return _canonical_segment(value, level=info.field_name or "segment")

    @model_validator(mode="after")
    def _check_hierarchy(self) -> Scope:
        if self.site is None and self.system is not None:
            raise ValueError(
                "system cannot be named while site is None; a system lives inside a site, "
                "and a scope that skips a level is ambiguous about what it covers"
            )
        return self

    @property
    def label(self) -> str:
        """Human readable ``customer/site/system`` form, ``*`` for an open level."""
        return "/".join(
            (render_segment(self.customer), render_segment(self.site), render_segment(self.system))
        )


class ScopePattern(BaseModel):
    """A scope with wildcards, used to express what a principal may see.

    ``None`` at a level means "any value at this level". ``customer=None`` is
    therefore a grant across every tenant in the index, which is only ever
    legitimate for an internal principal; it renders as ``*/*/*`` in the audit
    log so a review can spot it, and
    :func:`tenant_fence.entitlement.is_allowed` refuses to honour it for a
    principal declared as :attr:`PrincipalKind.CUSTOMER`.

    This is a separate type from :class:`Scope` on purpose. A document's scope
    must name a customer, a pattern need not. Reusing one type for both would
    make it possible to construct a document that belongs to nobody, and content
    that belongs to nobody is content the fence cannot reason about.

    Two hierarchy rules apply, and a pattern may not narrow a level while
    leaving a broader level open. ``customer=None, site="eu"`` would mean "the
    eu site of every tenant", which reads like a scoping mistake in every real
    system, so it is rejected instead of honoured. ``customer="acme",
    system="billing"`` is rejected for the same reason on an allow: it reads
    narrow (one system) and would behave broad (that system at every site).

    A deny needs the second rule relaxed rather than enforced. That is what
    :class:`DenyPattern` is for.
    """

    model_config = ConfigDict(frozen=True)

    may_leave_site_open: ClassVar[bool] = False
    """Whether a pattern of this class may name a system while the site is open."""

    customer: str | None = None
    site: str | None = None
    system: str | None = None

    @field_validator("customer", "site", "system")
    @classmethod
    def _canonical(cls, value: str | None, info: ValidationInfo) -> str | None:
        return _canonical_segment(value, level=info.field_name or "segment")

    @model_validator(mode="after")
    def _check_hierarchy(self) -> ScopePattern:
        if self.customer is None and (self.site is not None or self.system is not None):
            raise ValueError(
                "a pattern cannot name a site or system while customer is open; "
                "that grants the same site across every tenant"
            )
        if self.site is None and self.system is not None and not self.may_leave_site_open:
            raise ValueError("a pattern cannot name a system while site is open")
        return self

    @property
    def label(self) -> str:
        """Human readable ``customer/site/system`` form, ``*`` for a wildcard."""
        return "/".join(
            (render_segment(self.customer), render_segment(self.site), render_segment(self.system))
        )


class DenyPattern(ScopePattern):
    """A pattern for a revocation, which may name a system across every site.

    An allow that names a system while leaving the site open reads narrow and
    behaves broad, so :class:`ScopePattern` refuses it. A deny is the mirror
    image: refusing it forces the operator to enumerate today's sites, and every
    site created after the revocation was written then escapes it. The
    revocation fails open on a schedule nobody is watching, which is worse than
    revoking one site too many.

    ``DenyPattern(customer="acme", system="billing")`` is therefore legal and
    means "the billing system of acme, at every site it ever has". It is the
    canonical form: :class:`Grant` rewrites any deny that names a system into
    exactly this shape, so the pattern stored, the pattern rendered in the audit
    log and the pattern enforced are the same object.
    """

    may_leave_site_open: ClassVar[bool] = True


def _check_patterns(
    allow: tuple[ScopePattern, ...], deny: tuple[ScopePattern, ...]
) -> None:
    """Re-apply every hierarchy rule to one grant's two pattern lists.

    A standalone function because two models call it. :class:`Grant` runs it on
    itself, and :class:`Principal` runs it again over every grant it is handed.
    The second pass is the one that matters: ``Grant.model_construct`` skips
    validation entirely, and a grant built that way, or rebuilt by
    ``model_copy``, reaches a principal carrying patterns no validator has ever
    seen. Pydantic happens to re-run an after validator on a nested model
    instance today, which catches it, but that is a default this package has no
    control over. Re-checking from the principal makes the defence the
    package's own.

    Nothing is repaired here, only refused. A validator that silently fixes an
    object built by bypassing validation teaches the next caller that the
    bypass is fine.
    """
    for pattern in allow:
        if pattern.customer is None and (pattern.site is not None or pattern.system is not None):
            raise ValueError(
                f"an allow pattern cannot name a site or system while the customer is open "
                f"({pattern.label}); that grants the same site across every tenant"
            )
        if pattern.site is None and pattern.system is not None:
            raise ValueError(
                f"an allow pattern cannot name a system while the site is open "
                f"({pattern.label}); it reads as one system and grants that system at "
                "every site of the customer. Only a deny may leave that level open"
            )
    for pattern in deny:
        if pattern.customer is None and (pattern.site is not None or pattern.system is not None):
            raise ValueError(
                f"a deny pattern cannot name a site or system while the customer is open "
                f"({pattern.label}); revoking one site name across every tenant is not a "
                "revocation anybody wrote on purpose"
            )
        if pattern.site is not None and pattern.system is not None:
            raise ValueError(
                f"a deny naming a system must leave the site open ({pattern.label}); "
                "a deny pinned to one site exempts every site created after it was "
                "written. Write DenyPattern(customer=..., system=...) instead"
            )


class Grant(BaseModel):
    """One coherent block of entitlement: what it opens and what it closes.

    ``allow`` lists the scope patterns this grant opens. ``deny`` lists patterns
    that are closed no matter what any allow says, including allows in other
    grants held by the same principal. Deny is evaluated first and wins, always,
    because the only safe reading of two conflicting rules is the restrictive
    one.

    A grant with an empty ``allow`` list opens nothing. That is not a degenerate
    case to guard against, it is the default the whole package is built on.

    A deny that names a system is widened to that system at every site of the
    named customer, and stored widened. See :meth:`_widen_system_denies`.
    """

    model_config = ConfigDict(frozen=True)

    label: str = ""
    allow: tuple[ScopePattern, ...] = ()
    deny: tuple[ScopePattern, ...] = ()

    @field_validator("deny")
    @classmethod
    def _widen_system_denies(cls, patterns: tuple[ScopePattern, ...]) -> tuple[ScopePattern, ...]:
        """Rewrite ``acme/eu/billing`` as ``acme/*/billing`` in a deny list.

        A system level deny had to name a site, because an allow pattern rejects
        an open middle level. Enforcing it site by site means the deny covers
        the sites that existed when it was written and silently exempts every
        site created afterwards, so revoking billing at eu and us leaves billing
        at apac readable the day apac is created. Widening at construction keeps
        the stored pattern, the rendered label and the enforced rule identical,
        so the audit log shows the real reach of the revocation instead of the
        shape the type system forced.
        """
        return tuple(
            pattern
            if pattern.system is None
            else DenyPattern(customer=pattern.customer, system=pattern.system)
            for pattern in patterns
        )

    @model_validator(mode="after")
    def _validate_patterns(self) -> Grant:
        """Re-check both lists, including on a grant that skipped the validators.

        The allow half closes the door :class:`DenyPattern` opens. A deny
        pattern is a :class:`ScopePattern` subclass, so nothing in the type
        system stops one from being dropped into an ``allow`` list, where a
        named system under an open site means "that system at every site": it
        reads narrow and behaves broad, which is exactly the confusion the
        hierarchy rule exists to prevent.

        :class:`Principal` runs the same check again over the grants it is
        given, so a grant that reached it without passing through here is
        refused too. See :func:`_check_patterns`.
        """
        _check_patterns(self.allow, self.deny)
        return self


class Principal(BaseModel):
    """The caller, and everything the fence knows about what it may see.

    ``grants`` defaults to empty, so a principal constructed with no explicit
    entitlement sees nothing at all. Deny by default is the point: a bug in the
    code that loads grants produces an account that reads zero documents, which
    someone notices in a minute, instead of an account that reads every
    document, which nobody notices until a customer does.

    Every grant handed in is re-checked here against the pattern hierarchy
    rules, so a :class:`Grant` built by ``model_construct`` or rebuilt by
    ``model_copy`` cannot carry a widened pattern into a live principal. See
    :func:`_check_patterns`.
    """

    model_config = ConfigDict(frozen=True)

    principal_id: str
    kind: PrincipalKind = PrincipalKind.CUSTOMER
    grants: tuple[Grant, ...] = ()

    @field_validator("principal_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        trimmed = value.strip(ASCII_WHITESPACE)
        if not trimmed:
            raise ValueError("principal_id must not be empty; the audit log needs a subject")
        return trimmed

    @model_validator(mode="after")
    def _check_grants(self) -> Principal:
        for granted in self.grants:
            _check_patterns(granted.allow, granted.deny)
        return self


class Document(BaseModel):
    """A source document, filed under exactly one scope.

    ``doc_id`` is opaque: leading and trailing ASCII whitespace is trimmed and
    nothing else is touched, because it is an identity key rather than an
    entitlement coordinate, and any transform that maps two spellings onto one
    could merge two genuinely distinct documents. The trim is ASCII only for the
    same reason it is in :func:`normalise_id`, and it matters beyond tidiness
    here: :meth:`~tenant_fence.index.InMemoryIndex.add` is an upsert keyed on
    ``doc_id``, so two ids that collapsed onto one key would let a document
    filed by one tenant evict a document filed by another.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    text: str
    scope: Scope
    status: DocumentStatus = DocumentStatus.APPROVED
    version: int = Field(default=1, ge=1)
    title: str = ""
    source_uri: str = ""

    @field_validator("doc_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        trimmed = value.strip(ASCII_WHITESPACE)
        if not trimmed:
            raise ValueError("doc_id must not be empty; provenance depends on it")
        return trimmed


class Chunk(BaseModel):
    """A retrievable passage, carrying its document's scope and provenance.

    ``scope`` is required, not optional. A chunk that has lost its scope cannot
    be checked by the guard, and an unverifiable chunk in a multi-tenant context
    is a leak waiting for a refactor. Making the field mandatory means that bug
    is a construction error rather than a runtime surprise.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    scope: Scope
    ordinal: int = Field(default=0, ge=0)
    title: str = ""
    source_uri: str = ""
    version: int = Field(default=1, ge=1)

    @property
    def cache_key(self) -> str:
        """The tenant qualified key anything outside this package must key on.

        ``chunk_id`` is ``doc_id#ordinal``, and tenants in a shared knowledge
        base choose their own document ids: two of them will file a
        ``refund-policy`` sooner rather than later.
        :class:`~tenant_fence.index.InMemoryIndex` keeps ``doc_id`` unique
        inside one index, so ``chunk_id`` is safe there, but a retrieval cache,
        a dedup pass, a citation table or a vector store upsert that spans
        indexes or outlives one has no such guarantee. Keying those on
        ``chunk_id`` serves one tenant's text under another tenant's key,
        downstream of a fence that did its job. Key them on this instead.

        The separator is ``::`` and :func:`render_segment` escapes ``:`` along
        with the rest of the structural characters, so no scope label can
        contain an unescaped ``::`` and the split between the label and the
        chunk id is unambiguous. Without that escape the pair is forgeable from
        an identifier: a chunk at ``acme/bexley/chiller::x`` with chunk id
        ``p#0`` and a chunk at ``acme/bexley/chiller`` with chunk id ``x::p#0``
        both key on ``acme/bexley/chiller::x::p#0``, which is two systems of one
        customer sharing a cache entry.
        """
        return f"{self.scope.label}::{self.chunk_id}"


class RetrievedChunk(Chunk):
    """A chunk that came back from a search, with its retrieval provenance.

    ``score`` is stamped by the index, ``rank`` and ``retrieved_by`` by the
    retriever. Keeping the retrieving principal on the object means the guard
    and the audit log are reading the same record the caller will read.
    """

    score: float = 0.0
    rank: int = Field(default=0, ge=0)
    retrieved_by: str = ""


class AuditEvent(StrEnum):
    """What kind of thing the audit record describes."""

    QUERY = "query"
    BREACH = "breach"


class AuditRecord(BaseModel):
    """One immutable line in the audit log.

    Frozen on purpose: an append only log whose entries can be edited in place
    answers no questions under scrutiny. ``timestamp`` is supplied by an
    injected clock, never read inline, so the log is reproducible in tests and
    the time source is a single named dependency in production.

    ``detail`` is an operator field. On a breach it names the document and the
    scope that were refused, which for a cross-tenant breach means it names
    another customer's document, so it is redacted on the way out of
    :meth:`tenant_fence.audit.AuditLog.for_principal`. ``sequence`` is the join
    key between the redacted record a tenant may read and the full record an
    operator reads.

    ``refused_count`` is ``None`` only on a copy the count was withheld from
    (see :func:`tenant_fence.audit.without_refusal_count`). The record kept in
    the log always carries the real number, and ``sequence`` joins the two.
    ``None`` rather than ``0`` because zero is a claim, and the claim would be
    false.

    ``doc_ids_returned`` is what the fence returned to its caller. When that
    caller then trims the context (``FencedPipeline(max_context_chars=...)``),
    the documents that reached the prompt are a subset of this list, so the
    "which documents did this account see" answer over-reports rather than
    under-reports. :attr:`tenant_fence.pipeline.Answer.context_doc_ids` is the
    narrower list.
    """

    model_config = ConfigDict(frozen=True)

    principal_id: str
    query: str
    applied_filter: str
    timestamp: datetime
    event: AuditEvent = AuditEvent.QUERY
    doc_ids_returned: tuple[str, ...] = ()
    refused_count: Annotated[int, Field(ge=0)] | None = 0
    detail: str = ""
    sequence: int = Field(default=0, ge=0)

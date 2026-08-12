"""Attacks on the candidate set. These tests try to break the fence, not describe it.

Everything here is written from the attacker's side of the wall. The question is
never "does the happy path work", it is "given a corpus I control, can I make the
caller receive material it may not see, or lose material it may".

The corpus below is the weapon. One tenant, ``brightmoor``, owns twelve keyword
dense documents about escalation. The victim, ``alderpath``, owns one short
document that actually answers the question. On a global scoring pass the twelve
foreign chunks take the entire top twelve, so the victim's own answer sits at
global rank thirteen and dies in any window smaller than that. Filtering after
the ranking cannot recover it; filtering before the ranking never loses it.

Tests that pass here are permanent regression guards: each failure message names
the document, the scope and the tenant that leaked, so a future refactor that
reintroduces the bug reads as an incident report rather than as an assertion
error. Tests that fail here are real defects, listed by name in the module that
reported them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar
from warnings import catch_warnings, simplefilter

import pytest

from conftest import document, grant, principal
from tenant_fence import (
    AuditEvent,
    AuditLog,
    Chunk,
    FenceBreach,
    FencedRetriever,
    InMemoryIndex,
    PostFilterIndex,
    Principal,
    PrincipalKind,
    RetrievedChunk,
    Scope,
    ScopeFilter,
    ScopePattern,
    SearchResult,
    scope_filter,
)

VICTIM = Scope(customer="alderpath")
DOMINANT = Scope(customer="brightmoor")

QUERY = "escalation path stalled shipment claim"
VICTIM_ANSWER = "alderpath-escalation"

FOREIGN_TEXT = (
    "Escalation. Stalled shipment escalation path. Shipment claim escalation "
    "for a stalled shipment claim, escalation path, stalled claim escalation."
)

IndexT = TypeVar("IndexT", bound=InMemoryIndex)


def victim() -> Principal:
    """The caller under attack: entitled to every scope of one tenant, nothing else."""
    return principal("agent-alderpath", grant(ScopePattern(customer="alderpath")))


VICTIM_FILTER = scope_filter(victim())

# The unfenced predicate, used to stand in for an index that stopped filtering.
# The principal behind it is INTERNAL because a cross-tenant allow is refused for
# a CUSTOMER principal (see entitlement.effective_allows): a grant loader that
# drops the customer field must fail closed, so */*/* only means "everything" for
# a staff account.
ALWAYS: ScopeFilter = scope_filter(
    principal("root", grant(ScopePattern()), kind=PrincipalKind.INTERNAL)
)


def build_post_filter(factory: Callable[[], IndexT]) -> IndexT:
    """Construct a post-filter index without the construction warning failing the run."""
    with catch_warnings():
        simplefilter("ignore", UserWarning)
        return factory()


def dominance_corpus(index: IndexT, *, foreign: int = 12) -> IndexT:
    """A shared index where one tenant's material dominates the query lexically.

    The victim owns three documents: one that answers the question, one that
    partially matches, and one that shares no term with the query at all. The
    third is there so the honest count tests can tell "entitled" from "entitled
    and matching", which are different numbers and are reported separately.
    """
    index.add(
        document(
            VICTIM_ANSWER,
            "Escalation path for a stalled shipment claim: open a claim ticket, then escalate.",
            VICTIM,
            title="Alderpath escalation path",
        )
    )
    index.add(
        document(
            "alderpath-claim-form",
            "The claim form must be attached before a shipment claim is reviewed.",
            VICTIM,
            title="Alderpath claim form",
        )
    )
    index.add(
        document(
            "alderpath-hours",
            "Support desk opens 09:00 and closes 17:00 on weekdays.",
            VICTIM,
            title="Alderpath support hours",
        )
    )
    for number in range(foreign):
        index.add(
            document(
                f"brightmoor-escalation-{number}",
                FOREIGN_TEXT,
                DOMINANT,
                title=f"Brightmoor escalation {number}",
            )
        )
    return index


def foreign_leaks(chunks: tuple[RetrievedChunk, ...] | tuple[Chunk, ...]) -> list[str]:
    """Every chunk in ``chunks`` that does not belong to the victim, as readable evidence."""
    return [
        f"{chunk.doc_id} (scope {chunk.scope.label})"
        for chunk in chunks
        if chunk.scope.customer != VICTIM.customer
    ]


class _Recorder:
    """Records what reached the scorer and what was turned into a result object.

    Two seams matter, and they fail differently. ``_score`` is where a foreign
    chunk starts costing the caller its window. ``_top_k`` is where foreign text
    is copied into an object a reranker or a cache could hold on to. A leak at
    the first seam degrades the answer, a leak at the second one hands the text
    to whatever runs next.
    """

    scored: list[Chunk]
    promoted: list[RetrievedChunk]

    def _score(self, query: str, candidates: list) -> list:  # type: ignore[type-arg]
        self.scored.extend(entry.chunk for entry in candidates)
        return super()._score(query, candidates)  # type: ignore[misc]

    def _top_k(self, scored: list, k: int) -> tuple[RetrievedChunk, ...]:  # type: ignore[type-arg]
        ranked: tuple[RetrievedChunk, ...] = super()._top_k(scored, k)  # type: ignore[misc]
        self.promoted.extend(ranked)
        return ranked


class WatchedPreFilterIndex(_Recorder, InMemoryIndex):
    """The correct index, wiretapped at both seams."""

    def __init__(self) -> None:
        super().__init__()
        self.scored = []
        self.promoted = []


class WatchedPostFilterIndex(_Recorder, PostFilterIndex):
    """The wrong index, wiretapped at the same two seams."""

    def __init__(self) -> None:
        super().__init__()
        self.scored = []
        self.promoted = []


class RerankIndexThatForgotTheFilter:
    """A plausible refactor that reintroduces the leak: a second, unfenced pass.

    This is not a strawman. Merging a keyword pass with a vector pass, or adding
    a cache in front of search, is exactly how the predicate gets dropped from
    one of two branches while the code still reads as if it were fenced.
    """

    def __init__(self, inner: InMemoryIndex) -> None:
        self._inner = inner

    def search(self, query: str, *, predicate: ScopeFilter, k: int = 5) -> SearchResult:
        fenced = self._inner.search(query, predicate=predicate, k=k)
        unfenced = self._inner.search(query, predicate=ALWAYS, k=k)
        merged = (*fenced.chunks, *unfenced.chunks)[:k]
        return SearchResult(
            chunks=merged,
            considered=fenced.considered,
            excluded=fenced.excluded,
            indexed=fenced.indexed,
        )


# --------------------------------------------------------------------------------------
# Lexical dominance: the caller's own answer against a foreign tenant's keyword wall
# --------------------------------------------------------------------------------------


def test_lexical_dominance_leaves_the_post_filter_caller_with_nothing() -> None:
    """Twelve foreign chunks eat the window; the post-filtered caller gets zero results."""
    post = dominance_corpus(build_post_filter(PostFilterIndex))
    pre = dominance_corpus(InMemoryIndex())

    starved = post.search(QUERY, predicate=VICTIM_FILTER, k=3)
    fenced = pre.search(QUERY, predicate=VICTIM_FILTER, k=3)

    assert starved.chunks == (), (
        "the post-filter index returned results for this corpus, so the corpus no longer "
        "demonstrates dominance and the comparison below proves nothing; rebuild it so the "
        "foreign tenant occupies the whole top k"
    )
    returned = [chunk.doc_id for chunk in fenced.chunks]
    assert returned and returned[0] == VICTIM_ANSWER, (
        f"pre-filtering did not return the caller's own answer: expected {VICTIM_ANSWER!r} at "
        f"rank 1, got {returned}; an empty list means a foreign tenant's keyword volume "
        "starved the caller out of a window that should never have been shared"
    )
    assert not foreign_leaks(fenced.chunks), (
        f"the pre-filtered index handed tenant alderpath material from another tenant: "
        f"{foreign_leaks(fenced.chunks)}"
    )


def test_the_scorer_is_never_shown_a_chunk_the_caller_may_not_read() -> None:
    """The ordering property, checked at the seam rather than inferred from the output."""
    index = dominance_corpus(WatchedPreFilterIndex())
    index.search(QUERY, predicate=VICTIM_FILTER, k=3)

    assert not foreign_leaks(tuple(index.scored)), (
        f"the scorer was handed {foreign_leaks(tuple(index.scored))} while serving tenant "
        "alderpath; unentitled chunks reached the ranking stage, so the filter is running "
        "after scoring and k is being spent on data the caller may not see"
    )
    assert len(index.scored) == 3, (
        f"the scorer saw {len(index.scored)} chunks; the victim owns exactly 3, so any other "
        "number means the candidate set was not the caller's own material"
    )


def test_no_foreign_chunk_is_ever_turned_into_a_result_object() -> None:
    """Nothing downstream can hold foreign text, because no foreign object is built."""
    index = dominance_corpus(WatchedPreFilterIndex())
    index.search(QUERY, predicate=VICTIM_FILTER, k=3)

    assert not foreign_leaks(tuple(index.promoted)), (
        f"result objects were constructed for {foreign_leaks(tuple(index.promoted))} while "
        "serving tenant alderpath; a reranker, a cache or a merge step running at this seam "
        "would be holding another tenant's text even though the caller never received it"
    )


def test_post_filtering_materialises_foreign_text_before_the_filter_runs() -> None:
    """The demonstration: the wrong order builds the foreign objects, then hides them."""
    index = dominance_corpus(build_post_filter(WatchedPostFilterIndex))
    result = index.search(QUERY, predicate=VICTIM_FILTER, k=3)

    assert foreign_leaks(tuple(index.promoted)), (
        "the post-filter index did not materialise foreign chunks here, so this test no "
        "longer demonstrates the pre-filter difference it exists to demonstrate"
    )
    assert all(FOREIGN_TEXT in chunk.text for chunk in index.promoted), (
        "the materialised objects do not carry the foreign body text, so the demonstration "
        "understates what a reranker at this seam would be holding"
    )
    assert result.chunks == (), (
        "the caller received chunks, which is not the point being made here; the point is "
        "that the foreign objects existed at all before the filter removed them"
    )


# --------------------------------------------------------------------------------------
# Starvation and the statistical side channel
# --------------------------------------------------------------------------------------


def test_a_foreign_document_evicts_the_callers_answer_from_the_global_top_k() -> None:
    """Name the eviction precisely: the victim's answer is at global rank 13, not 1."""
    index = dominance_corpus(InMemoryIndex())
    everything = [chunk.doc_id for chunk in index.search(QUERY, predicate=ALWAYS, k=50).chunks]
    global_rank = everything.index(VICTIM_ANSWER) + 1

    assert global_rank > 3, (
        f"the victim's answer is at global rank {global_rank}, inside a k=3 window, so this "
        "corpus does not exercise eviction any more"
    )
    fenced = index.search(QUERY, predicate=VICTIM_FILTER, k=3)
    assert [chunk.doc_id for chunk in fenced.chunks][0] == VICTIM_ANSWER, (
        f"the caller's own answer sits at global rank {global_rank} but must sit at rank 1 "
        "within its own candidate set; if it does not, foreign documents are competing for "
        "a window that should contain only the caller's material"
    )


@pytest.mark.parametrize("foreign", [0, 1, 60])
def test_foreign_corpus_volume_cannot_move_the_callers_scores(foreign: int) -> None:
    """Document frequency is a side channel. It must be computed over the caller's set only."""
    alone = dominance_corpus(InMemoryIndex(), foreign=0)
    crowded = dominance_corpus(InMemoryIndex(), foreign=foreign)

    baseline = [
        (chunk.doc_id, chunk.score)
        for chunk in alone.search(QUERY, predicate=VICTIM_FILTER, k=10).chunks
    ]
    observed = [
        (chunk.doc_id, chunk.score)
        for chunk in crowded.search(QUERY, predicate=VICTIM_FILTER, k=10).chunks
    ]
    assert observed == baseline, (
        f"adding {foreign} chunks belonging to tenant brightmoor changed what tenant "
        f"alderpath sees: {baseline} became {observed}; corpus statistics computed across "
        "tenants let a caller measure another tenant's document count through its own scores"
    )


def test_post_filter_scores_carry_another_tenants_corpus_statistics() -> None:
    """The same side channel, open, in the arrangement this repo argues against."""
    quiet = build_post_filter(PostFilterIndex)
    dominance_corpus(quiet, foreign=0)
    loud = build_post_filter(PostFilterIndex)
    dominance_corpus(loud, foreign=60)

    quiet_score = quiet.search(QUERY, predicate=VICTIM_FILTER, k=10_000).chunks[0].score
    loud_score = loud.search(QUERY, predicate=VICTIM_FILTER, k=10_000).chunks[0].score

    assert quiet_score != loud_score, (
        "the post-filter index scored the victim's document identically with and without a "
        "foreign corpus, so this test no longer shows the statistical channel it documents"
    )
    assert loud_score < quiet_score, (
        f"expected the foreign corpus to depress the victim's score ({quiet_score} became "
        f"{loud_score}); the direction matters because it is what makes the caller's own "
        "document fall out of a fixed score threshold it used to clear"
    )


# --------------------------------------------------------------------------------------
# k overflow and pagination
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [4, 25, 10_000])
def test_asking_for_more_than_the_entitlement_holds_stays_honest(k: int) -> None:
    """Overflowing k must not reach past the fence to fill the window."""
    index = dominance_corpus(InMemoryIndex())
    result = index.search(QUERY, predicate=VICTIM_FILTER, k=k)

    assert not foreign_leaks(result.chunks), (
        f"asking for k={k} pulled foreign material in to fill the window: "
        f"{foreign_leaks(result.chunks)}"
    )
    assert len(result.chunks) == 2, (
        f"k={k} returned {len(result.chunks)} chunks; the victim owns 3 chunks of which 2 "
        "match the query, so any larger number means unmatched or foreign material was used "
        "as padding"
    )
    assert result.considered == 3, (
        f"considered reported {result.considered} for k={k}; it must report the caller's own "
        "candidate count (3), independently of k, or the caller cannot tell a thin corpus "
        "from a narrow window"
    )
    assert result.indexed == 15
    assert result.excluded == 12


def test_paging_by_growing_k_only_appends_and_never_admits_a_foreign_chunk() -> None:
    """There is no cursor API, so pagination is growing k. That must be a stable prefix."""
    index = dominance_corpus(InMemoryIndex())
    previous: list[str] = []
    for k in range(1, 9):
        page = [chunk.doc_id for chunk in index.search(QUERY, predicate=VICTIM_FILTER, k=k).chunks]
        assert page[: len(previous)] == previous, (
            f"page k={k} returned {page}, which is not an extension of the previous page "
            f"{previous}; a caller paging through results would see rows appear, vanish or "
            "reorder between requests, and could never prove it saw the whole set"
        )
        assert not [doc_id for doc_id in page if not doc_id.startswith("alderpath")], (
            f"page k={k} contained a foreign document: {page}"
        )
        previous = page


def test_post_filter_pagination_cannot_reach_the_callers_own_material() -> None:
    """Under post-filtering the caller must guess the global corpus size to see anything."""
    index = dominance_corpus(build_post_filter(PostFilterIndex))
    reachable = {
        k: len(index.search(QUERY, predicate=VICTIM_FILTER, k=k).chunks) for k in range(1, 13)
    }
    assert set(reachable.values()) == {0}, (
        f"expected every window up to k=12 to be empty under post-filtering, got {reachable}; "
        "this corpus is meant to prove that the caller's own material is unreachable until k "
        "exceeds the foreign document count"
    )
    assert len(index.search(QUERY, predicate=VICTIM_FILTER, k=13).chunks) == 1, (
        "the victim's answer did not appear at k=13 either; the demonstration depends on the "
        "answer sitting at exactly global rank 13"
    )


def test_negative_k_must_not_silently_shorten_the_callers_own_results() -> None:
    """A negative k is fed straight into a Python slice bound. Nothing rejects it."""
    index = dominance_corpus(InMemoryIndex())
    entitled = len(index.search(QUERY, predicate=VICTIM_FILTER, k=10_000).chunks)

    outcome: SearchResult | None
    try:
        outcome = index.search(QUERY, predicate=VICTIM_FILTER, k=-1)
    except ValueError:
        outcome = None

    if outcome is not None:
        assert len(outcome.chunks) == entitled, (
            f"k=-1 returned {len(outcome.chunks)} of the caller's {entitled} entitled chunks "
            f"and reported considered={outcome.considered}, excluded={outcome.excluded}, so "
            "nothing in the result says material was dropped; a negative k must be rejected "
            "or clamped, never used as a negative slice bound, because silently returning "
            "fewer results than the caller can see is the exact failure this package argues "
            "post-filtering commits"
        )


# --------------------------------------------------------------------------------------
# Empty entitlement
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "caller"),
    [
        ("no grants at all", principal("agent-new")),
        ("a grant that opens nothing", principal("agent-empty", grant())),
        (
            "an allow cancelled by a deny",
            principal(
                "agent-revoked",
                grant(
                    ScopePattern(customer="alderpath"),
                    deny=(ScopePattern(customer="alderpath"),),
                ),
            ),
        ),
        (
            "a deny in a second grant closing the first",
            principal(
                "agent-split",
                grant(ScopePattern()),
                grant(deny=(ScopePattern(),)),
            ),
        ),
        (
            "a grant on a tenant that is not in the index",
            principal("agent-elsewhere", grant(ScopePattern(customer="nowhere-industries"))),
        ),
    ],
)
@pytest.mark.parametrize("k", [1, 5, 10_000])
def test_an_empty_entitlement_returns_empty_and_never_everything(
    label: str, caller: Principal, k: int
) -> None:
    """Deny by default, under every shape of empty grant and every window size."""
    index = dominance_corpus(InMemoryIndex())
    result = index.search(QUERY, predicate=scope_filter(caller), k=k)

    assert result.chunks == (), (
        f"a principal with {label} received {[chunk.doc_id for chunk in result.chunks]} at "
        f"k={k}; an entitlement that opens nothing must return nothing, and a fence that "
        "falls open when the grants are empty fails exactly when the grant loader has a bug"
    )
    assert result.considered == 0
    assert result.excluded == result.indexed == 15, (
        f"excluded={result.excluded} indexed={result.indexed} for {label}; the caller cannot "
        "tell 'the fence removed everything' from 'the index is empty' unless both numbers "
        "are reported honestly"
    )


def test_an_empty_entitlement_through_the_retriever_returns_empty_and_records_it() -> None:
    """The same property one layer up, where the audit log has to agree with the result."""
    log = AuditLog()
    retriever = FencedRetriever(dominance_corpus(InMemoryIndex()), audit=log, k=5)
    result = retriever.retrieve(principal("agent-new"), QUERY)

    assert result.chunks == (), (
        f"an ungranted principal retrieved {result.doc_ids} through FencedRetriever"
    )
    assert result.excluded == 15
    assert result.record.doc_ids_returned == ()
    assert result.record.refused_count == 15, (
        f"the audit record says {result.record.refused_count} chunks were refused; it must "
        "say 15, or a review reading the log cannot distinguish an account that was entitled "
        "to nothing from a query against an empty corpus"
    )
    assert result.applied_filter == "allow=[] deny=[]", (
        f"the log recorded the effective entitlement as {result.applied_filter!r}; deny by "
        "default has to be legible in the log rather than inferred from an empty result"
    )


# --------------------------------------------------------------------------------------
# Making a misconfigured filter visible instead of silent
# --------------------------------------------------------------------------------------


def test_the_retriever_reports_how_many_candidates_the_fence_excluded() -> None:
    """The count is the difference between a working fence and a broken one."""
    log = AuditLog()
    retriever = FencedRetriever(dominance_corpus(InMemoryIndex()), audit=log, k=3)
    result = retriever.retrieve(victim(), QUERY)

    assert result.excluded == 12, (
        f"the retriever reported excluded={result.excluded} against a corpus holding 12 "
        "foreign chunks; if this number is wrong nobody can audit the fence from the outside, "
        "and a filter that quietly matches nothing looks identical to a thin corpus"
    )
    assert result.considered == 3
    assert result.record.refused_count == result.excluded, (
        f"the audit log recorded refused_count={result.record.refused_count} while the caller "
        f"was told excluded={result.excluded}; the log and the caller must not disagree about "
        "what the fence did"
    )


def test_a_misconfigured_filter_is_distinguishable_from_an_empty_index() -> None:
    """Three empty result sets, three different causes, three different number triples."""
    populated = dominance_corpus(InMemoryIndex())

    typo = principal("agent-typo", grant(ScopePattern(customer="alderpth")))

    working = populated.search(QUERY, predicate=VICTIM_FILTER, k=3)
    misconfigured = populated.search(QUERY, predicate=scope_filter(typo), k=3)
    empty = InMemoryIndex().search(QUERY, predicate=VICTIM_FILTER, k=3)

    signature = {
        "working": (len(working.chunks), working.considered, working.excluded, working.indexed),
        "misconfigured": (
            len(misconfigured.chunks),
            misconfigured.considered,
            misconfigured.excluded,
            misconfigured.indexed,
        ),
        "empty": (len(empty.chunks), empty.considered, empty.excluded, empty.indexed),
    }
    assert signature["working"] == (2, 3, 12, 15)
    assert signature["misconfigured"] == (0, 0, 15, 15), (
        f"a filter with a typo in the tenant id produced {signature['misconfigured']}; it must "
        "report excluded equal to indexed with considered zero, which is the fingerprint of "
        "grants that match nothing rather than of a corpus that holds nothing"
    )
    assert signature["empty"] == (0, 0, 0, 0), (
        f"an empty index produced {signature['empty']}; it must be all zeros, otherwise the "
        "two failure modes above cannot be told apart from the numbers alone"
    )
    assert len(set(signature.values())) == 3, (
        f"two of the three causes produced the same numbers: {signature}; the exclusion count "
        "only earns its place if it separates them"
    )


def test_post_filtering_under_reports_what_the_fence_actually_excluded() -> None:
    """The wrong order cannot even count honestly: the number is capped at k."""
    index = dominance_corpus(build_post_filter(PostFilterIndex), foreign=60)
    result = index.search(QUERY, predicate=VICTIM_FILTER, k=3)

    assert result.excluded == 3, (
        f"expected the post-filter index to report excluded={3}, capped at k, got "
        f"{result.excluded}; this test exists to pin the under-report"
    )
    truthful = dominance_corpus(InMemoryIndex(), foreign=60).search(
        QUERY, predicate=VICTIM_FILTER, k=3
    )
    assert truthful.excluded == 60, (
        f"the pre-filtered index reported excluded={truthful.excluded} against 60 foreign "
        "chunks; the whole value of the count is that it is the real number, so an operator "
        "reading it can see the fence working"
    )


def test_the_audit_log_inherits_the_post_filter_under_report() -> None:
    """A dishonest count does not stop at the index. It is what the reviewer will read."""
    log = AuditLog()
    index = dominance_corpus(build_post_filter(PostFilterIndex), foreign=60)
    retriever = FencedRetriever(index, audit=log, k=3)
    retriever.retrieve(victim(), QUERY)

    (record,) = log.records
    assert record.refused_count == 3, (
        f"the audit record wrote refused_count={record.refused_count}; the point being pinned "
        "is that post-filtering writes the truncated count into the permanent log, so an "
        "investigation reading it a year later sees 3 where the fence handled 60"
    )


# --------------------------------------------------------------------------------------
# Defence in depth: what happens when a future refactor drops the predicate
# --------------------------------------------------------------------------------------


def test_an_index_that_forgets_the_filter_is_stopped_by_the_guard() -> None:
    """The redundant check has to fire, name the document, and land in the log."""
    log = AuditLog()
    forgetful = RerankIndexThatForgotTheFilter(dominance_corpus(InMemoryIndex()))
    retriever = FencedRetriever(forgetful, audit=log, k=3)  # type: ignore[arg-type]

    with pytest.raises(FenceBreach) as raised:
        retriever.retrieve(victim(), QUERY)

    breach = raised.value
    assert breach.scope.customer == "brightmoor", (
        f"the guard raised on scope {breach.scope.label!r}, not on the foreign tenant's; a "
        "guard that fires on the wrong chunk is worse than none, because it points the "
        "investigation at the wrong document"
    )
    assert breach.doc_id.startswith("brightmoor-escalation"), (
        f"the breach names doc_id={breach.doc_id!r}, which is not the leaked document"
    )
    assert breach.principal_id == "agent-alderpath"
    assert "brightmoor" in str(breach), (
        "the exception message does not name the leaking tenant, so an on-call reading it "
        "cannot tell whose data was about to be served"
    )
    (recorded,) = log.breaches()
    assert recorded.event is AuditEvent.BREACH
    assert breach.doc_id in recorded.detail, (
        f"the audit record detail {recorded.detail!r} does not name the offending document; "
        "a breach visible only in a traceback disappears the moment someone adds an except"
    )


def test_post_filtering_never_trips_the_guard_so_the_damage_stays_silent() -> None:
    """Post-filtering does not raise. It returns a plausible, wrong, empty answer."""
    log = AuditLog()
    index = dominance_corpus(build_post_filter(PostFilterIndex))
    retriever = FencedRetriever(index, audit=log, k=3)
    result = retriever.retrieve(victim(), QUERY)

    assert result.chunks == ()
    assert log.breaches() == (), (
        "the guard fired on the post-filter path; the property being pinned is that it "
        "cannot fire, because the foreign chunks were removed before it ran, which is why "
        "starvation is invisible to every alarm in the system"
    )
    (record,) = log.records
    assert record.event is AuditEvent.QUERY
    assert record.doc_ids_returned == (), (
        "the caller was recorded as having seen documents it did not receive"
    )


def test_a_denied_subscope_never_reaches_the_scorer() -> None:
    """Deny has to be applied at the candidate stage, not cleaned up afterwards."""
    index = WatchedPreFilterIndex()
    eu = Scope(customer="alderpath", site="eu")
    us = Scope(customer="alderpath", site="us")
    index.add(document("alderpath-eu-open", "escalation path eu", eu))
    index.add(document("alderpath-us-shut", "escalation path us", us))

    caller = principal(
        "agent-alderpath-eu",
        grant(
            ScopePattern(customer="alderpath"),
            deny=(ScopePattern(customer="alderpath", site="us"),),
        ),
    )
    result = index.search("escalation path", predicate=scope_filter(caller), k=5)

    scored_ids = [chunk.doc_id for chunk in index.scored]
    assert scored_ids == ["alderpath-eu-open"], (
        f"the scorer was handed {scored_ids}; a denied sub-scope must be removed before "
        "scoring, or the denied document competes for the window and its ranking statistics "
        "influence what the caller does see"
    )
    assert [chunk.doc_id for chunk in result.chunks] == ["alderpath-eu-open"]
    assert result.excluded == 1


# --------------------------------------------------------------------------------------
# Breaches found. These fail, and they are meant to.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filed_as", "granted_to"),
    [
        # U+00DF LATIN SMALL LETTER SHARP S casefolds to the two characters "ss".
        ("Stra\u00dfe-Werke", "strasse-werke"),
        # U+212A KELVIN SIGN casefolds to a plain "k".
        ("A\u212aME-Logistics", "akme-logistics"),
        # U+FB01 LATIN SMALL LIGATURE FI casefolds to the two characters "fi".
        ("\ufb01nch-labs", "finch-labs"),
    ],
    ids=["sharp-s", "kelvin-sign", "fi-ligature"],
)
def test_case_folding_must_not_merge_two_distinct_tenant_ids(
    filed_as: str, granted_to: str
) -> None:
    """normalise_id casefolds, and casefold collapses characters that are not case variants."""
    index = InMemoryIndex()
    index.add(
        document(
            "payroll-q3",
            "Payroll run for the quarter three cycle, headcount and gross totals.",
            Scope(customer=filed_as),
            title="Payroll quarter three",
        )
    )
    intruder = principal("agent-intruder", grant(ScopePattern(customer=granted_to)))
    result = index.search("payroll run quarter three", predicate=scope_filter(intruder), k=5)

    assert result.chunks == (), (
        f"a principal granted customer={granted_to!r} read a document filed at "
        f"customer={filed_as!r}: {[chunk.doc_id for chunk in result.chunks]}. These are two "
        "different tenant identifiers, not two spellings of one, and normalise_id folded them "
        "into the same key. The docstring on normalise_id promises the opposite, that ids "
        "differing by more than case 'stay distinct, which is the safe direction'; casefold "
        "does not honour that promise"
    )


def test_two_tenants_using_the_same_doc_id_do_not_collide_in_chunk_ids() -> None:
    """chunk_id is doc_id#ordinal with no tenant in it. Tenants pick their own doc ids."""
    index = InMemoryIndex()
    index.add(document("refund-policy", "Alderpath refunds within 30 days.", VICTIM))
    index.add(document("refund-policy", "Brightmoor refunds within 7 days.", DOMINANT))

    by_id: dict[str, list[str]] = {}
    for chunk in index.chunks:
        by_id.setdefault(chunk.chunk_id, []).append(chunk.scope.label)
    collisions = {chunk_id: scopes for chunk_id, scopes in by_id.items() if len(scopes) > 1}

    assert not collisions, (
        f"the same chunk_id identifies chunks belonging to different tenants: {collisions}. "
        "chunk_id is the natural key for a retrieval cache, a dedup pass or a citation map, "
        "and two tenants are free to name a document 'refund-policy'. Any component keyed on "
        "chunk_id will serve alderpath's text under brightmoor's key, downstream of a fence "
        "that did its job"
    )


def test_retiring_an_indexed_document_makes_it_unretrievable() -> None:
    """DocumentStatus.RETIRED is checked at write time only, and there is no way to unwrite."""
    index = InMemoryIndex()
    approved = document("clinical-note", "Escalation path for the stalled claim.", VICTIM)
    index.add(approved)
    index.add(approved.model_copy(update={"status": "retired"}))

    result = index.search("escalation stalled claim", predicate=VICTIM_FILTER, k=5)
    assert result.chunks == (), (
        f"a document retired after indexing is still retrievable: "
        f"{[chunk.doc_id for chunk in result.chunks]}. InMemoryIndex exposes add and add_all "
        "and nothing else, so there is no eviction path at all: retiring, deleting or "
        "correcting a document is impossible without discarding the whole index. The "
        "DocumentStatus docstring states that retired content 'stays out of the index "
        "entirely, so no query path can reach it by accident', which holds only for content "
        "that was never approved"
    )


def test_refiling_a_document_to_another_tenant_revokes_the_old_tenants_access() -> None:
    """Tenant reassignment leaves a live copy under the old scope, readable forever."""
    index = InMemoryIndex()
    original = document("migrated-runbook", "Escalation path for a stalled shipment claim.", VICTIM)
    index.add(original)
    index.add(original.model_copy(update={"scope": DOMINANT}))

    result = index.search(QUERY, predicate=VICTIM_FILTER, k=5)
    assert result.chunks == (), (
        f"after the document was re-filed to tenant brightmoor, tenant alderpath still reads "
        f"it: {[(chunk.chunk_id, chunk.scope.label) for chunk in result.chunks]}. The index "
        "now holds two live copies of migrated-runbook under two different scopes, and the "
        "fence correctly serves the stale one to the tenant that no longer owns the document"
    )


def test_no_public_bulk_read_is_reachable_from_the_retriever() -> None:
    """The request handler holds the retriever. One public attribute away is the corpus.

    Scoped deliberately, because the wider claim is not one this package can
    keep. ``retriever._index.chunks`` and ``retriever.index._index.chunks``
    return every tenant's material, and no arrangement of Python objects
    prevents that: a wrapper holding the index in a closure moves the same
    reference into ``__closure__``, and the retriever needs the index to do its
    job. What is enforced is the public surface, which is what a debug endpoint,
    an admin view, a health check or a serialiser walking attributes actually
    reaches for. The reader deserves the boundary drawn where it really sits.
    """
    retriever = FencedRetriever(dominance_corpus(InMemoryIndex()), k=3)

    for label, surface in (("retriever", retriever), ("retriever.index", retriever.index)):
        for name in dir(surface):
            if name.startswith("_"):
                continue
            attribute = getattr(surface, name, None)
            if not isinstance(attribute, tuple | list):
                continue
            leaked = foreign_leaks(tuple(item for item in attribute if hasattr(item, "scope")))
            assert not leaked, (
                f"a caller holding only the retriever read {leaked} through {label}.{name}, "
                "with no principal and no predicate. The package's argument is that a leak "
                "should require the filter to be wrong rather than require every downstream "
                "caller to remember something; a public bulk read hanging off the retriever "
                "is exactly the thing a caller has to remember not to touch"
            )

    fenced = retriever.chunks_for(principal("victim", grant(ScopePattern(customer="alderpath"))))
    assert fenced and not foreign_leaks(fenced), (
        "the fenced enumeration path either returned nothing or returned another tenant's "
        "chunks; removing the unfenced read is only half of it, the replacement has to work"
    )

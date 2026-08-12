"""Index behaviour, and the side by side proof that filter ordering is the whole game."""

from __future__ import annotations

import pytest

from conftest import ACME, ACME_EU, BELMONT, document, grant, principal
from tenant_fence import (
    Document,
    DocumentStatus,
    InMemoryIndex,
    PostFilterIndex,
    PrincipalKind,
    Scope,
    ScopePattern,
    scope_filter,
    tokenise,
)
from tenant_fence.index import cosine, split_passages

# INTERNAL, because a cross-tenant allow opens nothing for a CUSTOMER principal.
ALWAYS = scope_filter(principal("root", grant(ScopePattern()), kind=PrincipalKind.INTERNAL))

ACME_ONLY = scope_filter(principal("acme", grant(ScopePattern(customer="acme"))))


def starvation_corpus(index: InMemoryIndex) -> InMemoryIndex:
    """One thin acme document against three keyword dense belmont documents.

    Belmont repeats the query terms, so on a global scoring pass its chunks take
    the whole top three and the single acme chunk never makes the window.
    """
    index.add(
        document(
            "acme-refund-policy",
            "The acme refund policy allows 30 days.",
            ACME,
            title="Acme refund policy",
        )
    )
    for number in range(3):
        index.add(
            document(
                f"belmont-refund-{number}",
                "Belmont refund policy. The refund policy covers refund requests, "
                "refund windows and refund policy exceptions under the refund policy.",
                BELMONT,
                title=f"Belmont refund policy {number}",
            )
        )
    return index


def build_post_filter_index() -> PostFilterIndex:
    with pytest.warns(UserWarning, match="filters after scoring"):
        return PostFilterIndex()


def test_post_filter_index_warns_loudly_on_construction() -> None:
    with pytest.warns(UserWarning, match="never to serve traffic"):
        PostFilterIndex()


def test_tokenise_splits_on_non_alphanumeric_and_casefolds() -> None:
    assert tokenise("Refund-Policy, 30 days!") == ["refund", "policy", "30", "days"]


def test_split_passages_drops_empty_blocks() -> None:
    assert split_passages("one\n\n\n  \n\ntwo\n") == ["one", "two"]
    assert split_passages("   ") == []


def test_cosine_returns_zero_for_a_zero_vector() -> None:
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="vector length mismatch"):
        cosine([1.0], [1.0, 2.0])


@pytest.mark.parametrize("status", [DocumentStatus.DRAFT, DocumentStatus.RETIRED])
def test_draft_and_retired_documents_are_not_retrievable(status: DocumentStatus) -> None:
    index = InMemoryIndex()
    stored = index.add(
        document("unpublished", "refund policy secrets", ACME, title="Draft", status=status)
    )
    result = index.search("refund policy", predicate=ALWAYS, k=5)
    assert stored == 0
    assert len(index) == 0
    assert result.chunks == (), (
        f"a document with status {status.value} was retrievable; unapproved content must "
        "never enter the index, or a future code path will surface it"
    )


def test_approved_documents_are_indexed() -> None:
    index = InMemoryIndex()
    assert index.add(document("d1", "first passage\n\nsecond passage", ACME)) == 2


def test_remove_evicts_every_chunk_of_one_document() -> None:
    index = InMemoryIndex()
    index.add(document("d1", "refund policy one\n\nrefund policy two", ACME))
    index.add(document("d2", "refund policy three", ACME))

    assert index.remove("d1") == 2
    assert index.remove("d1") == 0, "a second removal reported work it did not do"
    assert [chunk.doc_id for chunk in index.chunks] == ["d2"]
    assert index.search("refund policy", predicate=ACME_ONLY, k=5).chunks[0].doc_id == "d2"


def test_re_adding_a_document_replaces_it_instead_of_duplicating_it() -> None:
    """An append only index cannot express a correction, only an accumulation."""
    index = InMemoryIndex()
    index.add(document("d1", "refund policy allows 30 days", ACME))
    index.add(document("d1", "refund policy allows 14 days", ACME))

    texts = [chunk.text for chunk in index.chunks]
    assert texts == ["refund policy allows 14 days"], (
        f"the index holds {texts}; a corrected document left its superseded text in place, "
        "so a query can still retrieve the wrong version and cite it as current"
    )
    assert len(index) == 1


def test_a_documents_chunk_count_shrinking_does_not_leave_orphans() -> None:
    """Re-adding a shorter document must not leave the tail chunks of the longer one."""
    index = InMemoryIndex()
    index.add(document("d1", "alpha passage\n\nbeta passage\n\ngamma passage", ACME))
    index.add(document("d1", "alpha passage", ACME))

    assert [chunk.chunk_id for chunk in index.chunks] == ["d1#0"], (
        f"chunks {[chunk.chunk_id for chunk in index.chunks]} survived a rewrite that "
        "removed their text; an orphaned chunk is content nobody can find to withdraw"
    )


def test_empty_document_produces_no_chunks() -> None:
    assert InMemoryIndex().add(document("empty", "   \n\n  ", ACME)) == 0


def test_chunks_carry_scope_and_provenance_from_the_document() -> None:
    index = InMemoryIndex()
    index.add(
        Document(
            doc_id="d1",
            text="alpha\n\nbeta",
            scope=ACME_EU,
            title="Title",
            source_uri="kb://acme/eu/d1",
            version=4,
        )
    )
    first, second = index.chunks
    assert first.chunk_id == "d1#0"
    assert second.ordinal == 1
    for chunk in (first, second):
        assert chunk.scope == ACME_EU, "a chunk lost its scope, so the guard cannot check it"
        assert chunk.source_uri == "kb://acme/eu/d1"
        assert chunk.version == 4


def test_pre_filter_returns_the_callers_own_top_k() -> None:
    index = starvation_corpus(InMemoryIndex())
    result = index.search("refund policy", predicate=ACME_ONLY, k=3)
    assert [chunk.doc_id for chunk in result.chunks] == ["acme-refund-policy"], (
        "the caller's own matching document did not come back; with the filter applied "
        "before scoring, foreign documents cannot compete for the window"
    )
    assert result.excluded == 3
    assert result.considered == 1


def test_post_filter_starves_the_caller_out_of_its_own_document() -> None:
    index = starvation_corpus(build_post_filter_index())
    result = index.search("refund policy", predicate=ACME_ONLY, k=3)
    assert result.chunks == (), (
        "the post-filter index happened to return something here; the corpus is meant to "
        "prove the starvation case, where foreign documents occupy the whole top k"
    )
    good = starvation_corpus(InMemoryIndex()).search("refund policy", predicate=ACME_ONLY, k=3)
    assert good.chunks, (
        "the caller's own document is retrievable when the filter runs first and vanishes "
        "when it runs last; that difference is the bug this repo is about"
    )


def test_post_filter_silently_returns_fewer_than_k() -> None:
    index = build_post_filter_index()
    index.add(document("acme-1", "refund policy one", ACME))
    index.add(document("acme-2", "refund policy two", ACME))
    index.add(document("belmont-1", "refund policy three", BELMONT))
    index.add(document("belmont-2", "refund policy four", BELMONT))

    result = index.search("refund policy", predicate=ACME_ONLY, k=4)
    assert len(result.chunks) < 4, (
        "expected the post-filter index to return fewer than the requested k; that silent "
        "shortfall is invisible to a caller and is why filtering after ranking is wrong"
    )

    pre_filtered = InMemoryIndex()
    pre_filtered.add(document("acme-1", "refund policy one", ACME))
    pre_filtered.add(document("acme-2", "refund policy two", ACME))
    pre_filtered.add(document("belmont-1", "refund policy three", BELMONT))
    pre_filtered.add(document("belmont-2", "refund policy four", BELMONT))
    assert len(pre_filtered.search("refund policy", predicate=ACME_ONLY, k=4).chunks) == 2


def test_the_scorer_never_sees_a_chunk_the_caller_may_not_read() -> None:
    """Ordering, checked at the seam rather than inferred from the output."""
    seen: list[str] = []

    class SpyIndex(InMemoryIndex):
        def _score(self, query: str, candidates: list) -> list:  # type: ignore[type-arg,override]
            seen.extend(entry.chunk.doc_id for entry in candidates)
            return super()._score(query, candidates)

    index = starvation_corpus(SpyIndex())
    index.search("refund policy", predicate=ACME_ONLY, k=3)
    assert seen == ["acme-refund-policy"], (
        f"the scorer was handed {seen}; unentitled chunks reached the ranking stage, which "
        "means the filter is running after scoring and k is being spent on foreign data"
    )


def test_excluded_count_separates_an_empty_corpus_from_a_working_fence() -> None:
    index = starvation_corpus(InMemoryIndex())
    nothing = index.search("refund policy", predicate=scope_filter(principal("new")), k=3)
    assert nothing.chunks == ()
    assert nothing.excluded == 4, (
        "an ungranted principal got an empty result with no exclusion count; the caller "
        "cannot then tell a working fence from a broken filter or an empty index"
    )
    assert nothing.considered == 0
    assert nothing.indexed == 4

    empty = InMemoryIndex().search("refund policy", predicate=ALWAYS, k=3)
    assert (empty.excluded, empty.indexed) == (0, 0)


def test_k_is_honoured_after_filtering() -> None:
    index = InMemoryIndex()
    for number in range(6):
        index.add(document(f"acme-{number}", f"refund policy number {number}", ACME))
    assert len(index.search("refund policy", predicate=ACME_ONLY, k=4).chunks) == 4


def test_zero_overlap_queries_return_nothing() -> None:
    index = InMemoryIndex()
    index.add(document("acme-1", "refund policy", ACME))
    assert index.search("unrelated terminology", predicate=ACME_ONLY, k=3).chunks == ()


def test_ranking_is_deterministic_across_runs() -> None:
    index = InMemoryIndex()
    for number in range(5):
        index.add(document(f"acme-{number}", "refund policy identical text", ACME))
    first = [chunk.chunk_id for chunk in index.search("refund", predicate=ACME_ONLY, k=5).chunks]
    second = [chunk.chunk_id for chunk in index.search("refund", predicate=ACME_ONLY, k=5).chunks]
    assert first == second == sorted(first), (
        "tied scores did not break deterministically; a retrieval layer that reorders "
        "under a tie makes every incident harder to reconstruct"
    )


def test_injected_embedder_is_used_and_still_pre_filtered() -> None:
    def embed(text: str) -> list[float]:
        tokens = set(tokenise(text))
        return [1.0 if term in tokens else 0.0 for term in ("refund", "vat", "payroll")]

    index = InMemoryIndex(embed=embed)
    index.add(document("acme-refund", "refund terms", ACME))
    index.add(document("acme-payroll", "payroll terms", ACME))
    index.add(document("belmont-refund", "refund terms", BELMONT))

    result = index.search("refund", predicate=ACME_ONLY, k=2)
    assert [chunk.doc_id for chunk in result.chunks] == ["acme-refund"]
    assert result.excluded == 1, "the vector path skipped the fence that the lexical path applies"


def test_search_ignores_documents_of_a_sibling_site() -> None:
    index = InMemoryIndex()
    index.add(document("eu", "refund policy eu", Scope(customer="acme", site="eu")))
    index.add(document("us", "refund policy us", Scope(customer="acme", site="us")))
    eu_only = scope_filter(principal("a", grant(ScopePattern(customer="acme", site="eu"))))
    result = index.search("refund policy", predicate=eu_only, k=5)
    assert [chunk.doc_id for chunk in result.chunks] == ["eu"]

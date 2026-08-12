"""In memory retrieval, with entitlement applied before anything is scored.

Two indexes live here. :class:`InMemoryIndex` filters the candidate set first
and scores what is left. :class:`PostFilterIndex` does it the common way, scores
everything and filters the survivors, and is kept only so the test suite can
demonstrate the failure instead of describing it.

Scoring defaults to a small BM25 implemented locally, so the package has no
model dependency and no network call. Vector search is supported by injecting an
``embed`` callable; the ordering property is identical either way, because the
filter runs before the scorer regardless of which scorer it is.
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .entitlement import ScopeFilter
from .models import Chunk, Document, DocumentStatus, RetrievedChunk

Embedder = Callable[[str], Sequence[float]]
"""Injected text to vector function. Nothing in this package provides one."""

_TOKEN_SPLIT = re.compile(r"[^0-9a-z]+")
_PASSAGE_SPLIT = re.compile(r"\n\s*\n")


def tokenise(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens. Deterministic, no deps.

    The ``casefold`` here is scoring, not identity, and the two must not be
    unified. Folding aggressively is right for matching words inside one
    caller's own candidate set and wrong for identifiers, where a many-to-one
    fold merges two tenants; see :func:`tenant_fence.models.normalise_id`.
    """
    return [token for token in _TOKEN_SPLIT.split(text.casefold()) if token]


def split_passages(text: str) -> list[str]:
    """Split a document into passages on blank lines, keeping order.

    A document with no blank lines becomes a single passage. An empty document
    becomes no passages at all, which keeps empty text out of the index rather
    than storing a chunk that can never match anything.
    """
    passages = [block.strip() for block in _PASSAGE_SPLIT.split(text)]
    return [block for block in passages if block]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, returning 0.0 for a zero vector instead of dividing by it."""
    if len(left) != len(right):
        raise ValueError(f"vector length mismatch: {len(left)} against {len(right)}")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def require_valid_k(k: int) -> int:
    """Return ``k`` if it is a usable window size, raise if it is not.

    A negative ``k`` reaching a slice bound is the post-filter failure mode
    reproduced inside the pre-filtered index: ``kept[:-3]`` quietly drops the
    three lowest ranked chunks the caller was entitled to, and nothing in
    :class:`SearchResult` distinguishes it from a thin corpus, because
    ``considered`` and ``excluded`` are computed before truncation and report
    the healthy numbers. Refusing is the honest option. Clamping to zero would
    trade a silent shortfall for a silent emptiness.
    """
    if k < 0:
        raise ValueError(
            f"k must not be negative, got {k}; a negative k is used as a slice bound and "
            "silently drops the caller's own lowest ranked results with nothing in the "
            "result saying material was withheld"
        )
    return k


@dataclass(frozen=True)
class SearchResult:
    """What a search returned, plus what it refused.

    ``excluded`` is not decoration. A caller that gets an empty result set cannot
    otherwise tell "the fence removed 4,000 chunks that belong to other tenants"
    from "this index is empty" from "my filter is broken and matches nothing".
    Those three have very different fixes, so the count travels with the result.

    Disclosure, stated because the caller here is usually a tenant rather than
    an operator: ``excluded`` is ``len(index) - len(own candidates)`` and
    ``indexed`` is the whole index, so both are global counts. A tenant that
    polls one cheap query an hour reads the total corpus size and the growth
    rate of every other tenant's material. No document text is disclosed, and
    the counts do not vary with the query text (pinned by
    ``test_the_excluded_count_is_not_an_oracle_for_the_neighbours_content``), but
    volume is information. :class:`~tenant_fence.retriever.FencedRetriever` and
    :class:`~tenant_fence.pipeline.FencedPipeline` both take
    ``disclose_excluded=False``, which strips the number from the result and
    from the audit record they hand back, while the record kept in the
    :class:`~tenant_fence.audit.AuditLog` still carries it for the operator.
    """

    chunks: tuple[RetrievedChunk, ...] = ()
    considered: int = 0
    excluded: int = 0
    indexed: int = 0


class SearchableIndex(Protocol):
    """The one method the retriever depends on.

    Written as a protocol so the retriever can be pointed at the deliberately
    wrong index in a test and behave identically, which is what makes the leak
    demonstration honest: same retriever, same principal, different ordering.
    """

    def search(self, query: str, *, predicate: ScopeFilter, k: int) -> SearchResult: ...


@runtime_checkable
class EnumerableIndex(Protocol):
    """An index that can list what it holds.

    Kept separate from :class:`SearchableIndex` so that enumeration is something
    a caller has to ask for by type, and so
    :meth:`tenant_fence.retriever.FencedRetriever.chunks_for` can refuse politely
    when the index behind it cannot enumerate.
    """

    @property
    def chunks(self) -> tuple[Chunk, ...]: ...


def _promote(chunk: Chunk, score: float) -> RetrievedChunk:
    """Copy an indexed chunk into a retrieved one, carrying scope and provenance.

    Every field is named explicitly rather than splatted from a dump, so adding a
    provenance field to :class:`~tenant_fence.models.Chunk` without carrying it
    through here is a type error rather than a field that quietly goes missing
    on the way to the caller.
    """
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        scope=chunk.scope,
        ordinal=chunk.ordinal,
        title=chunk.title,
        source_uri=chunk.source_uri,
        version=chunk.version,
        score=score,
    )


@dataclass
class _Entry:
    """An indexed chunk with its precomputed representations."""

    chunk: Chunk
    tokens: list[str] = field(default_factory=list)
    vector: Sequence[float] | None = None


class InMemoryIndex:
    """A lexical (or injected vector) index that filters before it scores.

    Only documents with status ``approved`` are indexed at all. Draft and retired
    content is never turned into chunks, so no query path, cache or rerank step
    can surface it later.

    ``doc_id`` is the identity of a document inside one index, and :meth:`add`
    is an upsert on it rather than an append. That is what makes withdrawal and
    re-filing expressible at all; see the method for what an append only index
    gets wrong.
    """

    def __init__(
        self,
        *,
        embed: Embedder | None = None,
        k1: float = 1.2,
        b: float = 0.75,
        min_score: float = 0.0,
    ) -> None:
        self._entries: list[_Entry] = []
        self._indexed_doc_ids: set[str] = set()
        self._embed = embed
        self._k1 = k1
        self._b = b
        self._min_score = min_score

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        """Everything currently indexed. Inspection only, in index order.

        Unfenced by design and not reachable from
        :class:`~tenant_fence.retriever.FencedRetriever`, which holds its index
        privately and enumerates through
        :meth:`~tenant_fence.retriever.FencedRetriever.chunks_for`. Reach for
        this one only where you already hold the whole corpus, such as an
        ingestion job or a test.
        """
        return tuple(entry.chunk for entry in self._entries)

    def add(self, document: Document) -> int:
        """Index one document, returning the number of chunks stored.

        Returns 0 for anything that is not approved, and evicts whatever that
        document already had in the index. Filtering status at write time rather
        than read time means unapproved content is absent, and absent content
        cannot leak through a code path that forgot to check.

        This is an upsert keyed on ``doc_id``, and both halves are load bearing.
        An append only version cannot express two ordinary operations and gets
        both of them wrong in the same direction:

        1. Withdrawal. Re-adding an indexed document as RETIRED did nothing at
           all, so content pulled from the knowledge base (a retention window, a
           deletion request, a piece of guidance that turned out to be wrong)
           kept being retrieved and kept being fed to the generator as current.
        2. Re-filing. Moving a document to another tenant (an account migration,
           a filing error corrected, an acquisition split) left the old copy in
           place under the old scope, so the fence went on correctly serving the
           stale copy to the tenant that no longer owned the document.

        Keying on ``doc_id`` also keeps ``chunk_id`` unique inside one index,
        which is what a retrieval cache or a dedup pass downstream assumes it
        can rely on. Across indexes, use
        :attr:`~tenant_fence.models.Chunk.cache_key`.
        """
        self.remove(document.doc_id)
        if document.status is not DocumentStatus.APPROVED:
            return 0
        passages = split_passages(document.text)
        for ordinal, passage in enumerate(passages):
            chunk = Chunk(
                chunk_id=f"{document.doc_id}#{ordinal}",
                doc_id=document.doc_id,
                text=passage,
                scope=document.scope,
                ordinal=ordinal,
                title=document.title,
                source_uri=document.source_uri,
                version=document.version,
            )
            self._entries.append(
                _Entry(
                    chunk=chunk,
                    tokens=tokenise(passage),
                    vector=self._embed(passage) if self._embed is not None else None,
                )
            )
        if passages:
            self._indexed_doc_ids.add(document.doc_id)
        return len(passages)

    def add_all(self, documents: Iterable[Document]) -> int:
        """Index many documents, returning the total number of chunks stored."""
        return sum(self.add(document) for document in documents)

    def remove(self, doc_id: str) -> int:
        """Evict every chunk of one document, returning how many were removed.

        The explicit half of the upsert. Without a removal path the only way to
        withdraw or correct a document is to discard the index and rebuild it,
        which is not an operation anyone performs under a deletion request at
        four in the afternoon.
        """
        if doc_id not in self._indexed_doc_ids:
            return 0
        before = len(self._entries)
        self._entries = [entry for entry in self._entries if entry.chunk.doc_id != doc_id]
        self._indexed_doc_ids.discard(doc_id)
        return before - len(self._entries)

    def search(self, query: str, *, predicate: ScopeFilter, k: int = 5) -> SearchResult:
        """Return the caller's own top ``k``, never a global top ``k`` trimmed down.

        The order of the next three steps is the entire security property of this
        package, so it is spelled out rather than left to be inferred.

        The predicate runs FIRST, over the raw entry list, before any score is
        computed and before any truncation. Three consequences follow, and each
        is a bug in the post-filter arrangement:

        1. ``k`` means ``k`` of the caller's own material. Filtering after
           truncation silently returns fewer results than asked for, and the
           caller cannot distinguish that from a thin corpus.
        2. A popular foreign document cannot starve the caller's own relevant
           document out of the candidate set, because it was never a candidate.
        3. A future code path that forgets to filter cannot leak, because there
           is nothing to forget: no unentitled chunk exists downstream of this
           line. Correctness stops depending on every caller remembering.

        Corpus statistics are computed over the filtered candidates too, not over
        the whole index. That keeps scores comparable within the caller's own
        material, and it keeps document frequencies from other tenants out of the
        arithmetic, which is a small statistical side channel but a real one.

        ``k`` is validated before anything else. See :func:`require_valid_k`.
        """
        require_valid_k(k)
        candidates = [entry for entry in self._entries if predicate(entry.chunk.scope)]
        excluded = len(self._entries) - len(candidates)
        if not candidates:
            return SearchResult(
                chunks=(), considered=0, excluded=excluded, indexed=len(self._entries)
            )

        scored = self._score(query, candidates)
        ranked = self._top_k(scored, k)
        return SearchResult(
            chunks=ranked,
            considered=len(candidates),
            excluded=excluded,
            indexed=len(self._entries),
        )

    def _top_k(self, scored: list[tuple[_Entry, float]], k: int) -> tuple[RetrievedChunk, ...]:
        """Sort, cut to ``k``, and stamp the score onto each surviving chunk.

        Ties break on ``chunk_id`` so that two runs over the same corpus return
        the same order. A retrieval layer that reorders under a tie makes every
        downstream test flaky and every incident harder to reconstruct.
        """
        kept = [pair for pair in scored if pair[1] > self._min_score]
        kept.sort(key=lambda pair: (-pair[1], pair[0].chunk.chunk_id))
        return tuple(_promote(entry.chunk, score) for entry, score in kept[:k])

    def _score(self, query: str, candidates: list[_Entry]) -> list[tuple[_Entry, float]]:
        if self._embed is not None:
            query_vector = self._embed(query)
            return [(entry, self._vector_score(query_vector, entry)) for entry in candidates]
        return self._bm25(query, candidates)

    @staticmethod
    def _vector_score(query_vector: Sequence[float], entry: _Entry) -> float:
        if entry.vector is None:
            return 0.0
        return cosine(query_vector, entry.vector)

    def _bm25(self, query: str, candidates: list[_Entry]) -> list[tuple[_Entry, float]]:
        """BM25 over the candidate set, implemented locally to keep the dep budget at one."""
        query_terms = set(tokenise(query))
        if not query_terms:
            return [(entry, 0.0) for entry in candidates]

        total = len(candidates)
        lengths = [len(entry.tokens) for entry in candidates]
        average_length = sum(lengths) / total if total else 0.0

        frequency: Counter[str] = Counter()
        for entry in candidates:
            frequency.update(set(entry.tokens) & query_terms)

        scored: list[tuple[_Entry, float]] = []
        for entry, length in zip(candidates, lengths, strict=True):
            counts = Counter(entry.tokens)
            score = 0.0
            for term in query_terms:
                term_frequency = counts[term]
                if term_frequency == 0:
                    continue
                document_frequency = frequency[term]
                idf = math.log(
                    1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                normaliser = self._k1 * (
                    1.0 - self._b + self._b * (length / average_length if average_length else 1.0)
                )
                score += idf * (term_frequency * (self._k1 + 1.0)) / (term_frequency + normaliser)
            scored.append((entry, score))
        return scored


class PostFilterIndex(InMemoryIndex):
    """The wrong one, kept on purpose. Never use this in production.

    This is the arrangement most multi-tenant RAG code ships with: score the
    whole corpus, take the global top ``k``, then drop the chunks the caller is
    not entitled to. It is wrong in three ways that a code review does not catch
    and a test does:

    1. The caller gets fewer than ``k`` results with no signal that it happened.
    2. Another tenant's popular document starves the caller's own relevant
       document out of the top ``k``, so the answer degrades because of data the
       caller cannot see and cannot mention.
    3. Every downstream path has to remember the post-filter. A new endpoint, a
       cache, a reranker, and the filter is gone.

    The leak is proved, not asserted, by
    ``tests/test_index.py::test_post_filter_starves_the_caller_out_of_its_own_document``
    and its neighbours in the same file. Delete this class and those tests can
    only describe the bug in prose, which is why it stays.

    Constructing it emits a ``UserWarning``, because a class that exists to be
    wrong should be hard to reach for by accident.
    """

    def __init__(
        self,
        *,
        embed: Embedder | None = None,
        k1: float = 1.2,
        b: float = 0.75,
        min_score: float = 0.0,
    ) -> None:
        super().__init__(embed=embed, k1=k1, b=b, min_score=min_score)
        warnings.warn(
            "PostFilterIndex filters after scoring and leaks by construction; "
            "it exists to be tested against, never to serve traffic",
            UserWarning,
            stacklevel=2,
        )

    def search(self, query: str, *, predicate: ScopeFilter, k: int = 5) -> SearchResult:
        """Score globally, truncate to ``k``, then filter the survivors.

        Read the order and compare it against :meth:`InMemoryIndex.search`. The
        predicate is the last thing applied instead of the first, so ``k`` is
        spent on documents the caller may not see, and whatever is left over is
        what the caller gets.
        """
        require_valid_k(k)
        scored = self._score(query, self._entries)
        ranked = self._top_k(scored, k)
        kept = tuple(chunk for chunk in ranked if predicate(chunk.scope))
        return SearchResult(
            chunks=kept,
            considered=len(self._entries),
            excluded=len(ranked) - len(kept),
            indexed=len(self._entries),
        )

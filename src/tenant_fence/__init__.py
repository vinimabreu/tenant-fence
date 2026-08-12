"""tenant-fence: permission-aware retrieval for a shared knowledge base.

One index, many customers, and a retrieval path where entitlement is applied
before the candidate set is scored. Cross-customer leakage is prevented by
construction rather than by a post-filter that every future caller has to
remember.

Everything is injected: the embedder and the generator are plain callables, so
the package has one runtime dependency (pydantic) and the whole suite runs
offline, deterministic, with no API key.
"""

from __future__ import annotations

from .audit import (
    REDACTED_BREACH_DETAIL,
    AuditLog,
    AuditSink,
    Clock,
    system_clock,
    without_refusal_count,
)
from .context import AssembledContext, Citation, build_context
from .entitlement import (
    ScopeFilter,
    describe_filter,
    effective_allows,
    is_allowed,
    is_denied,
    pattern_matches,
    scope_filter,
    segment_matches,
)
from .guard import FenceBreach, Guard
from .index import (
    Embedder,
    EnumerableIndex,
    InMemoryIndex,
    PostFilterIndex,
    SearchableIndex,
    SearchResult,
    require_valid_k,
    tokenise,
)
from .models import (
    AuditEvent,
    AuditRecord,
    Chunk,
    DenyPattern,
    Document,
    DocumentStatus,
    Grant,
    Principal,
    PrincipalKind,
    RetrievedChunk,
    Scope,
    ScopePattern,
    normalise_id,
    render_segment,
    require_visible_id,
)
from .pipeline import NO_CONTEXT_ANSWER, Answer, FencedPipeline, Generator
from .retriever import FencedRetriever, FenceResult

__version__ = "0.1.0"

__all__ = [
    "NO_CONTEXT_ANSWER",
    "REDACTED_BREACH_DETAIL",
    "Answer",
    "AssembledContext",
    "AuditEvent",
    "AuditLog",
    "AuditRecord",
    "AuditSink",
    "Chunk",
    "Citation",
    "Clock",
    "DenyPattern",
    "Document",
    "DocumentStatus",
    "Embedder",
    "EnumerableIndex",
    "FenceBreach",
    "FenceResult",
    "FencedPipeline",
    "FencedRetriever",
    "Generator",
    "Grant",
    "Guard",
    "InMemoryIndex",
    "PostFilterIndex",
    "Principal",
    "PrincipalKind",
    "RetrievedChunk",
    "Scope",
    "ScopeFilter",
    "ScopePattern",
    "SearchResult",
    "SearchableIndex",
    "__version__",
    "build_context",
    "describe_filter",
    "effective_allows",
    "is_allowed",
    "is_denied",
    "normalise_id",
    "pattern_matches",
    "render_segment",
    "require_valid_k",
    "require_visible_id",
    "scope_filter",
    "segment_matches",
    "system_clock",
    "tokenise",
    "without_refusal_count",
]

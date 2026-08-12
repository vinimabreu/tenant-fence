"""End to end: entitlement, pre-filtered retrieval, guard, context, generation.

The generator is a plain ``(str) -> str`` callable, injected by the caller. There
is no SDK, no client, no API key and no network anywhere in this package, which
is what lets the entire test suite run offline and deterministically. Wire your
own model in at the edge; everything the fence guarantees happens before the
prompt is built.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .audit import AuditLog
from .context import AssembledContext, Citation, build_context
from .index import SearchableIndex
from .models import AuditRecord, Principal, RetrievedChunk
from .retriever import FencedRetriever

Generator = Callable[[str], str]
"""Injected text in, text out. The only place a model would ever be called."""

DEFAULT_TEMPLATE = """Answer the question using only the context sections below.
Cite the marker of every section you use. If the context does not answer the
question, say that it does not.

{context}

Question: {question}
Answer:"""

NO_CONTEXT_ANSWER = "No material you are entitled to see answers this question."


@dataclass(frozen=True)
class Answer:
    """What the pipeline produced, and the evidence for how it produced it.

    ``dropped`` is the number of entitled chunks that were retrieved and then
    left out of the context because ``max_context_chars`` was reached. It is on
    the response for the same reason
    :attr:`~tenant_fence.index.SearchResult.excluded` is: a caller that gets
    fewer sections than it asked for cannot otherwise tell "the corpus is thin"
    from "material was withheld", and this package's opening argument is that
    silently returning less than ``k`` is a bug rather than a detail. A pipeline
    that counted the shortfall internally and dropped it on the way out would be
    committing the failure the rest of the package is about.

    ``chunks`` is everything retrieval returned, so with ``dropped`` above zero
    it is a superset of what reached the prompt. :attr:`context_doc_ids` is the
    list of documents the generator actually saw.
    """

    text: str
    citations: tuple[Citation, ...]
    record: AuditRecord
    chunks: tuple[RetrievedChunk, ...] = ()
    prompt: str = ""
    refused: bool = False
    dropped: int = 0

    @property
    def citation_map(self) -> dict[str, Citation]:
        """Marker to citation, for resolving what the answer cited."""
        return {citation.marker: citation for citation in self.citations}

    @property
    def context_doc_ids(self) -> tuple[str, ...]:
        """Distinct documents that reached the prompt, in marker order.

        Narrower than ``record.doc_ids_returned`` whenever ``dropped`` is above
        zero: the audit record names what the fence returned, this names what
        the generator was shown.
        """
        seen: list[str] = []
        for citation in self.citations:
            if citation.doc_id not in seen:
                seen.append(citation.doc_id)
        return tuple(seen)


class FencedPipeline:
    """Question in, grounded answer out, with entitlement enforced throughout."""

    def __init__(
        self,
        index: SearchableIndex,
        *,
        audit: AuditLog | None = None,
        k: int = 5,
        max_context_chars: int | None = None,
        template: str = DEFAULT_TEMPLATE,
        disclose_excluded: bool = True,
    ) -> None:
        """Wire the fenced retriever this pipeline answers from.

        ``disclose_excluded`` is passed straight through to
        :class:`~tenant_fence.retriever.FencedRetriever`. It is repeated here
        rather than left to callers who build their own retriever because this
        class is the documented entry point, and a disclosure choice that can
        only be made on the path most people do not take is not a choice
        anybody makes.
        """
        self.audit = AuditLog() if audit is None else audit
        self.retriever = FencedRetriever(
            index, audit=self.audit, k=k, disclose_excluded=disclose_excluded
        )
        self.max_context_chars = max_context_chars
        self.template = template

    def answer(self, principal: Principal, question: str, *, generate: Generator) -> Answer:
        """Run the full path for one question.

        When the fence returns nothing, the generator is not called at all and a
        fixed refusal comes back. Calling a model with an empty context is how a
        system ends up answering from training data and presenting it as the
        customer's own material, which is a quieter version of the failure this
        package exists to prevent.

        Whatever the context assembly leaves out is reported on
        :attr:`Answer.dropped`, never swallowed.
        """
        result = self.retriever.retrieve(principal, question)
        if not result.chunks:
            return Answer(
                text=NO_CONTEXT_ANSWER,
                citations=(),
                record=result.record,
                chunks=(),
                prompt="",
                refused=True,
                dropped=0,
            )

        context = self._build(result.chunks)
        prompt = self.template.format(context=context.text, question=question)
        return Answer(
            text=generate(prompt),
            citations=context.citations,
            record=result.record,
            chunks=result.chunks,
            prompt=prompt,
            refused=False,
            dropped=context.dropped,
        )

    def _build(self, chunks: tuple[RetrievedChunk, ...]) -> AssembledContext:
        return build_context(chunks, max_chars=self.max_context_chars)

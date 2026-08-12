"""The demo is the README, so the demo has to be a test.

Every number printed by ``examples/maintenance_kb.py`` is pasted verbatim into
the README. A demo that drifts turns the README into a claim nobody checked, so
each property the write-up rests on is asserted here, and each failure message
names what leaked rather than saying an assertion failed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from examples.maintenance_kb import (
    ARLEN_OPS,
    BEXLEY_OPS,
    CORPUS,
    INJECTION,
    NADIA,
    OWEN,
    PELLWORTH_OPS,
    QUESTION,
    UNSCOPED,
    K,
    build_index,
    build_post_filter_index,
    main,
    report_sections,
)

from tenant_fence import (
    AuditLog,
    DocumentStatus,
    FencedPipeline,
    FencedRetriever,
    InMemoryIndex,
    Principal,
)

CUSTOMER_OF = {document.doc_id: document.scope.customer for document in CORPUS}

README = Path(__file__).resolve().parent.parent / "README.md"
BANNER = "tenant-fence  |  shared equipment maintenance knowledge base"


@pytest.fixture
def index() -> InMemoryIndex:
    return build_index()


@pytest.fixture
def retriever(index: InMemoryIndex) -> FencedRetriever:
    return FencedRetriever(index, audit=AuditLog(), k=K)


def customers_in(doc_ids: tuple[str, ...]) -> set[str]:
    """The tenants a result set touches, resolved through the corpus."""
    return {CUSTOMER_OF[doc_id] for doc_id in doc_ids}


def test_the_demo_runs(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    out = capsys.readouterr().out
    assert "shared equipment maintenance knowledge base" in out
    assert "guard breaches: 0" in out


def test_the_demo_output_is_byte_identical_across_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The README pastes one capture of this. Two runs have to agree on it."""
    main()
    first = capsys.readouterr().out
    main()
    second = capsys.readouterr().out
    assert first == second, (
        "the demo is not deterministic, so the output pasted into the README is a "
        "snapshot of one run rather than a description of the code"
    )


@pytest.mark.skipif(not README.exists(), reason="README is not part of an installed package")
def test_the_readme_pastes_this_runs_real_output(capsys: pytest.CaptureFixture[str]) -> None:
    """The README quotes this demo verbatim, so the quote is checked, not trusted.

    A captured block in a write-up rots the first time somebody tunes a scorer,
    and a rotted block is worse than no block: it is a number a reader has no
    way to distrust. Comparing the two here means the README cannot drift
    without turning the build red.
    """
    main()
    live = capsys.readouterr().out
    match = re.search(rf"```\n({re.escape(BANNER)}\n.*?)```", README.read_text(), re.DOTALL)
    assert match is not None, (
        f"the README no longer contains a fenced block starting with {BANNER!r}; "
        "either the demo capture was removed or the banner changed"
    )
    assert match.group(1) == live, (
        "the output pasted into the README is not what the demo prints today. "
        "Re-run `python -m examples.maintenance_kb` and paste the new capture, and "
        "read the diff first: a number moved, and the write-up explains that number"
    )


@pytest.mark.skipif(not README.exists(), reason="README is not part of an installed package")
def test_every_test_the_readme_names_actually_exists() -> None:
    """The README says "the names are the specification", so the names have to resolve.

    Forty-odd properties in that write-up are sold as verifiable by running one
    named test. A renamed or deleted test turns each of those lines into a claim
    the reader cannot check and would not know to distrust, which is the same
    failure mode as a stale output block one section above.
    """
    cited = set(re.findall(r"`(test_[a-z0-9_]+)`", README.read_text()))
    defined: set[str] = set()
    for path in sorted((README.parent / "tests").glob("test_*.py")):
        defined |= set(re.findall(r"^def (test_[a-z0-9_]+)", path.read_text(), re.MULTILINE))

    assert cited, "the README stopped naming any test, so this check proves nothing"
    assert cited <= defined, (
        f"the README names {sorted(cited - defined)}, which no test file defines. Every "
        "property in that table is sold as runnable with `pytest -k <name>`, so a name "
        "that does not resolve is a claim the reader cannot check"
    )


@pytest.mark.parametrize(
    ("principal", "customer"),
    [(BEXLEY_OPS, "harrowgate"), (PELLWORTH_OPS, "pellworth"), (ARLEN_OPS, "vantis")],
)
def test_a_customer_account_never_reads_another_customers_document(
    retriever: FencedRetriever, principal: Principal, customer: str
) -> None:
    returned = retriever.retrieve(principal, QUESTION, k=K).doc_ids
    foreign = {doc_id for doc_id in returned if CUSTOMER_OF[doc_id] != customer}
    assert not foreign, (
        f"{principal.principal_id} is entitled to {customer} only and received "
        f"{sorted(foreign)}, which belong to another customer of the same index"
    )


def test_the_internal_escalation_account_reads_across_customers(
    retriever: FencedRetriever,
) -> None:
    """The other half of the property: the fence must not break the job it exists for."""
    returned = retriever.retrieve(NADIA, QUESTION, k=K).doc_ids
    assert len(customers_in(returned)) > 1, (
        "the on-call escalation account holds a cross-customer grant and still read "
        f"one customer only ({sorted(customers_in(returned))}); a fence that blocks the "
        "account it was meant to open is a broken fence, not a strict one"
    )


def test_an_internal_account_is_still_bound_by_its_own_grants(
    retriever: FencedRetriever,
) -> None:
    """INTERNAL is not a bypass. Only grants grant."""
    reachable = {chunk.doc_id for chunk in retriever.chunks_for(OWEN)}
    pellworth = {doc_id for doc_id in reachable if CUSTOMER_OF[doc_id] == "pellworth"}
    assert not pellworth, (
        "tech-owen is granted harrowgate and vantis and reached pellworth documents "
        f"{sorted(pellworth)}; PrincipalKind.INTERNAL widened a grant it must not touch"
    )


def test_a_customer_account_holding_a_cross_tenant_grant_reads_nothing(
    retriever: FencedRetriever,
) -> None:
    result = retriever.retrieve(UNSCOPED, QUESTION, k=K)
    assert result.chunks == (), (
        "svc-unscoped is a CUSTOMER principal whose allow pattern lost its customer key "
        f"and became */*/*; it read {list(result.doc_ids)} instead of failing closed"
    )
    assert "*/*/*" in result.applied_filter, (
        "the misconfigured grant was refused and then hidden from the audit log; a "
        "reviewer needs to see that a cross-tenant grant was issued to a customer account"
    )


def test_the_widened_deny_covers_the_boiler_at_every_pellworth_site(
    retriever: FencedRetriever,
) -> None:
    """A system deny is stored widened, so a site added tomorrow is already covered."""
    reachable = {chunk.doc_id for chunk in retriever.chunks_for(PELLWORTH_OPS)}
    boilers = {
        document.doc_id
        for document in CORPUS
        if document.scope.customer == "pellworth" and document.scope.system == "boiler"
    }
    assert len(boilers) >= 2, "the corpus no longer has a boiler at two pellworth sites"
    assert not (reachable & boilers), (
        f"the boiler deny leaked at one or more sites: {sorted(reachable & boilers)} "
        "stayed readable, which is how a revocation quietly exempts every site created "
        "after it was written"
    )


def test_the_draft_document_never_enters_the_index(index: InMemoryIndex) -> None:
    drafts = {d.doc_id for d in CORPUS if d.status is DocumentStatus.DRAFT}
    indexed = {chunk.doc_id for chunk in index.chunks}
    assert drafts, "the corpus lost its unapproved document, so this proves nothing"
    assert not (drafts & indexed), (
        f"unapproved content is retrievable: {sorted(drafts & indexed)} was chunked into "
        "the index, so every downstream path can surface it"
    )


def test_the_post_filter_index_starves_the_customer_of_its_own_answer(
    index: InMemoryIndex,
) -> None:
    """The README's central number. If this stops holding, the README is wrong."""
    wrong_index, warning = build_post_filter_index()
    assert "leaks by construction" in warning, (
        "PostFilterIndex stopped warning on construction, so the wrong index is now "
        "reachable by accident"
    )
    before = FencedRetriever(index, audit=AuditLog(), k=K).retrieve(BEXLEY_OPS, QUESTION, k=K)
    after = FencedRetriever(wrong_index, audit=AuditLog(), k=K).retrieve(BEXLEY_OPS, QUESTION, k=K)
    assert len(before.chunks) == K, (
        f"ops-bexley no longer gets its own top {K} from the correct index, so the "
        "comparison the README draws has nothing to compare against"
    )
    assert len(after.chunks) < len(before.chunks), (
        "the post-filter arrangement returned as much as the pre-filter one, so the "
        "corpus no longer demonstrates the failure the README describes"
    )
    lost = set(before.doc_ids) - set(after.doc_ids)
    assert "hg-bexley-hhp" in lost, (
        "the demo no longer shows the caller's own answering document being starved out "
        f"of the window; it lost {sorted(lost)} instead"
    )


def test_the_post_filter_index_still_does_not_leak_across_customers(
    index: InMemoryIndex,
) -> None:
    """The wrong index is wrong about the window, not about entitlement.

    Worth pinning separately. The README argues the post-filter fails on result
    quality and on every path that forgets it, not that this particular class
    hands over foreign text; overstating that would be the easy sale and the
    dishonest one.
    """
    wrong_index, _ = build_post_filter_index()
    retriever = FencedRetriever(wrong_index, audit=AuditLog(), k=K)
    returned = retriever.retrieve(BEXLEY_OPS, QUESTION, k=K).doc_ids
    foreign = {doc_id for doc_id in returned if CUSTOMER_OF[doc_id] != "harrowgate"}
    assert not foreign, f"the post-filter handed ops-bexley {sorted(foreign)}"


def test_the_customer_can_tell_an_empty_answer_from_a_fenced_one(
    retriever: FencedRetriever,
) -> None:
    result = retriever.retrieve(BEXLEY_OPS, QUESTION, k=K)
    assert result.excluded is not None and result.excluded > 0, (
        "the excluded count is missing, so a caller cannot distinguish 'the fence "
        "removed material' from 'the index is empty' from 'my grants match nothing'"
    )
    assert result.considered + result.excluded == len(retriever.chunks_for(NADIA)), (
        "considered plus excluded no longer accounts for the whole index, so one of "
        "the two numbers the caller is given is wrong"
    )


def test_the_injected_instruction_reaches_the_prompt(index: InMemoryIndex) -> None:
    """Stated first, because the honest claim depends on it being true.

    The package does not remove injected text and does not claim to. Asserting
    that it arrives keeps the next test from passing for the wrong reason.
    """
    pipeline = FencedPipeline(index, audit=AuditLog(), k=K)
    answer = pipeline.answer(BEXLEY_OPS, QUESTION, generate=report_sections)
    assert INJECTION in answer.prompt, (
        "the injected passage did not reach the prompt, so the next assertion would "
        "prove nothing about what the fence withstands"
    )


def test_the_injected_instruction_has_nothing_foreign_to_exfiltrate(
    index: InMemoryIndex,
) -> None:
    pipeline = FencedPipeline(index, audit=AuditLog(), k=K)
    answer = pipeline.answer(BEXLEY_OPS, QUESTION, generate=report_sections)
    named = {
        document.doc_id
        for document in CORPUS
        if document.scope.customer in {"pellworth", "vantis"}
        and document.status is DocumentStatus.APPROVED
    }
    assert named, "the injection asks for documents that are not in the corpus"
    in_prompt = {doc_id for doc_id in named if doc_id in answer.prompt}
    scopes = {citation.scope_label for citation in answer.citations}
    assert not in_prompt, (
        f"a document the injected instruction asked for reached the prompt: "
        f"{sorted(in_prompt)}. The instruction did not need to work, because the "
        "content was in the context window before the model read a word of it"
    )
    assert scopes == {"harrowgate/bexley/chiller"}, (
        f"the context spans {sorted(scopes)}; ops-bexley is entitled to harrowgate "
        "bexley only, so anything else here is another customer's material"
    )


def test_the_demo_records_no_guard_breach(index: InMemoryIndex) -> None:
    """The guard is redundancy. In a correct run it must stay silent."""
    log = AuditLog()
    retriever = FencedRetriever(index, audit=log, k=K)
    for principal in (NADIA, OWEN, BEXLEY_OPS, PELLWORTH_OPS, ARLEN_OPS, UNSCOPED):
        retriever.retrieve(principal, QUESTION, k=K)
    assert log.breaches() == (), (
        "the guard rejected a chunk the index had already filtered, which means the "
        f"index and the guard disagree: {[record.detail for record in log.breaches()]}"
    )


def test_the_audit_log_answers_what_each_account_saw(index: InMemoryIndex) -> None:
    log = AuditLog()
    retriever = FencedRetriever(index, audit=log, k=K)
    for principal in (BEXLEY_OPS, PELLWORTH_OPS, UNSCOPED):
        retriever.retrieve(principal, QUESTION, k=K)
    assert customers_in(tuple(log.doc_ids_seen_by("ops-bexley"))) == {"harrowgate"}
    assert customers_in(tuple(log.doc_ids_seen_by("ops-pellworth"))) == {"pellworth"}
    assert log.doc_ids_seen_by("svc-unscoped") == set(), (
        "an account entitled to nothing has a document history, so either the fence or "
        "the log is telling a story the other one does not agree with"
    )

import asyncio
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cli import collect
from evaluate import (
    EvaluationConfig, FactRule, GroundTruth, ScoringRubric, evaluate, score_run, summarize, validate_inputs,
)
from llm import ScriptedLLM
from models import (
    CauseCategory, ExecutionBudget, ExecutionTrace, Hypothesis, InvestigationReport,
    InvestigationRun, ModelCallTrace, OperationFailure, SimulationSettings, ToolFamily, ToolQuery,
)
from tools import DATA_DIR, SimulatedTools, query_key

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def scoring():
    return (
        GroundTruth.model_validate_json((DATA_DIR / "evaluation" / "ground_truth.json").read_text()),
        ScoringRubric.model_validate_json((DATA_DIR / "evaluation" / "rubric.json").read_text()),
    )


def make_run(tools, catalog, index=0, hypothesis=None, *, inconclusive=False):
    context = catalog.incidents[index].context
    snapshot = asyncio.run(collect(tools, context))
    return InvestigationRun(
        report=InvestigationReport(
            incident_id=context.incident_id, outcome="inconclusive" if inconclusive else "ranked",
            hypotheses=(hypothesis,) if hypothesis else (),
            limitations=("Missing dependency spans prevent distinguishing causes",) if inconclusive else (),
        ),
        trace=ExecutionTrace(
            approach="baseline", execution_mode="mocked", model="scripted-test-double",
            budget=ExecutionBudget(), termination_reason="baseline_finished",
            failures=(), model_calls=(), state=snapshot,
        ),
    )


def pool_hypothesis(**changes):
    return Hypothesis.model_validate({
        "hypothesis_id": "h1", "cause_category": "connection_capacity",
        "claim": "The reduction of checkout pool capacity from 128 to 16 caused connection acquisition waits",
        "confidence": 0.8,
        "supporting_evidence_ids": ["inc-001:changes:r01", "inc-001:logs:r01", "inc-001:metrics:r01"],
        **changes,
    })


def score(run, catalog, scoring, index=0):
    truth, rubric = scoring
    return score_run(run, truth.cases[index], rubric.cases[index], catalog, elapsed=0.1)


def test_rubric_integrity_and_distinguishing_facts_for_all_ten_cases(catalog, scoring):
    truth, rubric = scoring
    validate_inputs(catalog, truth, rubric)
    assert len(rubric.cases) == 10
    for expected, rules in zip(truth.cases, rubric.cases):
        if expected.outcome == "resolved":
            recognized = {fact.fact_id for fact in rules.facts if fact.matches(expected.root_cause["description"])}
            assert set(rules.required_fact_ids) <= recognized
        else:
            assert not rules.required_fact_ids


def test_invalid_private_patterns_fail_validation_instead_of_being_ignored():
    with pytest.raises(ValueError, match="invalid fact pattern"):
        FactRule(fact_id="bad", patterns=("[",))


def test_root_accuracy_and_fact_grounding_not_just_citation_validity(tools, catalog, scoring):
    run = make_run(tools, catalog, hypothesis=pool_hypothesis())
    result = score(run, catalog, scoring)
    assert result.top_ranked_correct is True
    assert result.grounding.citation_existence == 1
    assert result.grounding.citation_availability == 1
    assert result.grounding.support_relationship_precision == 1
    assert result.grounding.rubric_fact_support == 1
    assert result.provider_input_tokens is None
    assert not result.provider_usage_complete
    wrong_evidence = make_run(tools, catalog, hypothesis=pool_hypothesis(
        supporting_evidence_ids=["inc-001:metrics:r03"],
    ))
    result = score(wrong_evidence, catalog, scoring)
    assert result.top_ranked_correct is True  # Accuracy and evidence support are independent.
    assert result.grounding.citation_existence == 1
    assert result.grounding.support_relationship_precision == 0
    assert result.grounding.rubric_fact_support == 0


@pytest.mark.parametrize("changes", [
    {"cause_category": "name_resolution"},
    {"claim": "Checkout pool capacity was increased"},
    {"claim": "Checkout pool was not reduced"},
])
def test_correct_category_alone_or_negated_cause_does_not_earn_accuracy(tools, catalog, scoring, changes):
    run = make_run(tools, catalog, hypothesis=pool_hypothesis(**changes))
    assert score(run, catalog, scoring).top_ranked_correct is False


def test_nonexistent_and_unavailable_citations_are_distinct(tools, catalog, scoring):
    run = make_run(tools, catalog, hypothesis=pool_hypothesis(
        supporting_evidence_ids=["invented", "inc-001:changes:r01"],
    ))
    snapshot = run.trace.state.model_copy(update={
        "evidence": tuple(item for item in run.trace.state.evidence if item.evidence_id != "inc-001:changes:r01"),
    })
    run = run.model_copy(update={"trace": run.trace.model_copy(update={"state": snapshot})})
    result = score(run, catalog, scoring).grounding
    assert result.citation_existence == 0.5
    assert result.citation_availability == 0
    assert result.support_relationship_precision == 0


def test_historical_similarity_is_not_current_dns_evidence(tools, catalog, scoring):
    run = make_run(tools, catalog, 4, Hypothesis(
        hypothesis_id="h1", confidence=0.8, cause_category=CauseCategory.NAME_RESOLUTION,
        claim="Inventory DNS resolver domain changed to legacy.internal causing name resolution failure",
        supporting_evidence_ids=("inc-005:historical_incidents:r01",),
    ))
    result = score(run, catalog, scoring, 4)
    assert result.top_ranked_correct
    assert result.grounding.citation_existence == 1
    assert result.grounding.support_relationship_precision == 0


def test_joint_witness_set_requires_both_configuration_and_runtime(tools, catalog, scoring):
    hypothesis = Hypothesis(
        hypothesis_id="h1", confidence=0.8, cause_category=CauseCategory.QUEUE_DELIVERY,
        claim="The queue visibility lease was shorter than processing time",
        supporting_evidence_ids=("inc-006:changes:r01",),
    )
    run = make_run(tools, catalog, 5, hypothesis)
    result = score(run, catalog, scoring, 5)
    assert result.top_ranked_correct
    assert result.grounding.support_relationship_precision == 1
    assert result.grounding.rubric_fact_support == 0
    hypothesis = hypothesis.model_copy(update={
        "supporting_evidence_ids": ("inc-006:changes:r01", "inc-006:metrics:r01"),
    })
    run = run.model_copy(update={"report": run.report.model_copy(update={"hypotheses": (hypothesis,)})})
    assert score(run, catalog, scoring, 5).grounding.rubric_fact_support == 1


def test_counterevidence_polarity_matters(tools, catalog, scoring):
    hypothesis = Hypothesis(
        hypothesis_id="h1", confidence=0.3,
        claim="All search regions failed",
        supporting_evidence_ids=("inc-004:logs:r02",),
    )
    run = make_run(tools, catalog, 3, hypothesis)
    assert score(run, catalog, scoring, 3).grounding.support_relationship_precision == 0
    opposite = hypothesis.model_copy(update={
        "supporting_evidence_ids": (), "contradicting_evidence_ids": ("inc-004:logs:r02",),
    })
    run = run.model_copy(update={"report": run.report.model_copy(update={"hypotheses": (opposite,)})})
    assert score(run, catalog, scoring, 3).grounding.contradiction_relationship_precision == 1


def test_unrecognized_claims_and_no_citations_are_not_perfect_grounding(tools, catalog, scoring):
    run = make_run(tools, catalog, hypothesis=Hypothesis(
        hypothesis_id="h1", claim="An unknown intermittent scheduler problem", confidence=0.3,
    ))
    result = score(run, catalog, scoring).grounding
    assert result.claims_without_rubric_match == 1
    assert result.citation_existence is None
    assert result.rubric_fact_support is None


def test_abstention_on_answerable_case_is_in_accuracy_denominator(tools, catalog, scoring):
    run = make_run(tools, catalog, inconclusive=True)
    result = score(run, catalog, scoring)
    assert result.abstained
    assert result.top_ranked_correct is False
    assert result.intentionally_inconclusive_handled is None


def test_intentional_inconclusive_abstention_requires_reason_and_available_analysis(tools, catalog, scoring):
    run = make_run(tools, catalog, 8, inconclusive=True)
    assert score(run, catalog, scoring, 8).intentionally_inconclusive_handled is True
    failed = run.model_copy(update={"trace": run.trace.model_copy(update={
        "termination_reason": "reporting_unavailable",
        "failures": (OperationFailure(
            task_id="report", operation="llm", code="authentication",
            message="Unavailable", retryable=False, attempt=1,
        ),),
    })})
    result = score(failed, catalog, scoring, 8)
    assert result.abstained  # A fallback is still reported as abstention.
    assert result.intentionally_inconclusive_handled is False  # Not a correct analytical decision.


def test_provider_tokens_include_failed_calls_and_unknown_usage_is_not_zero(tools, catalog, scoring):
    run = make_run(tools, catalog, hypothesis=pool_hypothesis())
    calls = (
        ModelCallTrace(task_id="reason", role="reasoning", attempt=1, system_prompt="", user_prompt="",
                       response_schema="AgentResult", error="Invalid response", input_tokens=10,
                       output_tokens=3, usage_measured=True),
        ModelCallTrace(task_id="reason", role="reasoning", attempt=2, system_prompt="", user_prompt="",
                       response_schema="AgentResult", input_tokens=11, output_tokens=5, usage_measured=True),
        ModelCallTrace(task_id="report", role="reporting", attempt=1, system_prompt="", user_prompt="",
                       response_schema="InvestigationReport", error="Unknown usage", reserved_tokens=100),
    )
    usage = run.trace.state.usage.model_copy(update={
        "llm_calls": 3, "input_tokens": 21, "output_tokens": 8,
        "tokens": 129, "unmeasured_token_reservations": 100,
    })
    run = run.model_copy(update={"trace": run.trace.model_copy(update={
        "execution_mode": "live", "model_calls": calls,
        "state": run.trace.state.model_copy(update={"usage": usage}),
    })})
    result = score(run, catalog, scoring)
    assert result.provider_input_tokens == 21
    assert result.provider_output_tokens == 8
    assert result.unmeasured_llm_calls == 1
    assert result.unmeasured_token_reservations == 100
    assert not result.provider_usage_complete


def test_inconsistent_usage_counters_are_rejected(tools, catalog, scoring):
    run = make_run(tools, catalog, hypothesis=pool_hypothesis())
    run = run.model_copy(update={"trace": run.trace.model_copy(update={
        "state": run.trace.state.model_copy(update={
            "usage": run.trace.state.usage.model_copy(update={"tool_calls": 999}),
        }),
    })})
    with pytest.raises(ValueError, match="tool call count"):
        score(run, catalog, scoring)


def test_all_unknown_provider_usage_is_null_not_zero(tools, catalog, scoring):
    run = make_run(tools, catalog, hypothesis=pool_hypothesis())
    call = ModelCallTrace(
        task_id="report", role="reporting", attempt=1, system_prompt="", user_prompt="",
        response_schema="InvestigationReport", error="Connection failed", reserved_tokens=100,
    )
    snapshot = run.trace.state.model_copy(update={
        "usage": run.trace.state.usage.model_copy(update={
            "llm_calls": 1, "unmeasured_token_reservations": 100, "tokens": 100,
        }),
    })
    run = run.model_copy(update={"trace": run.trace.model_copy(update={
        "execution_mode": "live", "model_calls": (call,), "state": snapshot,
    })})
    result = score(run, catalog, scoring)
    assert result.provider_input_tokens is None
    assert result.provider_output_tokens is None
    assert result.unmeasured_llm_calls == 1
    assert not result.provider_usage_complete


def test_query_keyed_failures_ignore_query_family_and_parallel_order(catalog):
    context = catalog.incidents[6].context
    settings = SimulationSettings.model_validate_json((DATA_DIR / "simulation.json").read_text())
    broad = ToolQuery(incident_id=context.incident_id, family=ToolFamily.METRICS, time_range=context.time_range)
    scoped = broad.model_copy(update={"services": ("catalog",)})
    padding = broad.model_copy(update={"family": ToolFamily.LOGS})

    async def sequential():
        tools = SimulatedTools(catalog, settings)
        history = {}
        for query in (padding, broad, broad, padding, scoped, scoped):
            result = await tools.execute(query)
            history.setdefault(query_key(query), []).append(result.status)
        return history

    async def parallel():
        tools = SimulatedTools(catalog, settings)
        history = {}
        for queries in ((scoped, padding, broad), (broad, scoped, padding)):
            results = await asyncio.gather(*(tools.execute(query) for query in queries))
            for query, result in zip(queries, results):
                history.setdefault(query_key(query), []).append(result.status)
        return history

    left, right = asyncio.run(sequential()), asyncio.run(parallel())
    assert left == right
    assert left[query_key(broad)] == ["error", "complete"]
    assert left[query_key(scoped)] == ["error", "complete"]
    assert left[query_key(padding)] == ["complete", "complete"]


def test_evaluation_includes_all_ten_incidents_both_approaches_and_repetitions(tmp_path, catalog, scoring):
    truth, rubric = scoring
    config = EvaluationConfig(mode="mocked", model="scripted-test-double", repeats=2)
    seen = []

    def factory(approach, context, repetition):
        seen.append((approach, context.incident_id, repetition))
        return ScriptedLLM({})  # Explicit failures, not invented benchmark predictions.

    output = tmp_path / "plumbing"
    rows = asyncio.run(evaluate(
        config, catalog, SimulationSettings(latency_ms=0), truth, rubric, output, factory,
    ))
    assert len(rows) == len(seen) == 40
    assert len(set(seen)) == 40
    assert all(row.metrics.terminal_failure for row in rows)
    assert all(row.metrics.top_ranked_correct is False for row in rows if row.metrics.answerable)
    assert all(row.metrics.provider_input_tokens is None for row in rows)
    saved = json.loads((output / "results.json").read_text())
    assert len(saved["runs"]) == 40
    assert saved["manifest"]["failure_policy"] == "normalized-query-first-n-attempts-v1"
    assert saved["manifest"]["config"]["budget"] == config.budget.model_dump(mode="json")
    assert len(saved["manifest"]["run_order"]) == 40
    assert "MOCKED PLUMBING ONLY" in (output / "summary.md").read_text()
    assert "No live results" in (output / "summary.md").read_text()
    assert len(list(csv.DictReader((output / "results.csv").open()))) == 40
    assert len((output / "runs.jsonl").read_text().splitlines()) == 40
    assert len(list((output / "runs").glob("*"))) == 40
    for approach in ("multi-agent", "baseline"):
        summaries = saved["comparison"][approach]
        assert summaries["per_repetition"][0]["answerable_runs"] == 9
        assert summaries["per_repetition"][1]["answerable_runs"] == 9
        assert summaries["terminal_failures"] == 20
        assert summaries["repetition_variability"]["top_ranked_root_cause_accuracy"]["n"] == 2


def test_repeated_results_report_variability_and_private_rubric_never_enters_prompts(tmp_path, catalog, scoring):
    truth, rubric = scoring
    fixture_directory = ROOT / "tests" / "fixtures"
    config = EvaluationConfig(
        mode="mocked", model="scripted-test-double", repeats=2, incident_ids=("inc-001",),
    )
    prompts = []

    class SpyLLM(ScriptedLLM):
        async def respond(self, task_id, system, user, schema, max_output_tokens):
            prompts.append(json.loads(user))
            return await super().respond(task_id, system, user, schema, max_output_tokens)

    def factory(approach, context, repetition):
        name = "multi_agent" if approach == "multi-agent" else "baseline"
        replies = json.loads((fixture_directory / f"{name}.json").read_text())
        if approach == "baseline" and repetition == 2:
            for value in (
                replies["baseline-2"][0]["hypotheses"][0],
                replies["baseline-2"][0]["report"]["hypotheses"][0],
            ):
                value["cause_category"] = "unknown"
        return SpyLLM(replies)

    rows = asyncio.run(evaluate(
        config, catalog, SimulationSettings(latency_ms=0), truth, rubric, tmp_path / "repeat", factory,
    ))
    summary = summarize(rows)
    assert summary["baseline"]["repetition_variability"]["top_ranked_root_cause_accuracy"]["sample_stddev"] > 0
    assert summary["multi-agent"]["repetition_variability"]["top_ranked_root_cause_accuracy"]["sample_stddev"] == 0
    assert summary["baseline"]["per_incident_variability"]["inc-001"]["tool_calls"]["n"] == 2
    for payload in prompts:
        serialized = json.dumps(payload)
        for forbidden in ("expected_category", "required_fact_ids", "root_cause", "challenges"):
            assert f'"{forbidden}"' not in serialized
        assert "limited-facts-v1" not in serialized
    artifacts = list((tmp_path / "repeat" / "runs").glob("*"))
    assert len(artifacts) == 4
    assert all(json.loads((path / "trace.json").read_text())["budget"] == config.budget.model_dump(mode="json") for path in artifacts)


def test_single_repetition_has_no_variability_estimate():
    from evaluate import distribution

    assert distribution([0.5])["sample_stddev"] is None
    assert distribution([None])["mean"] is None
    assert distribution([None])["missing"] == 1


def test_paid_approval_precedes_factory_or_filesystem_access(tmp_path, catalog, scoring):
    truth, rubric = scoring

    def forbidden(*args):
        raise AssertionError("must not create a live client")

    with pytest.raises(PermissionError, match="paid-run approval"):
        asyncio.run(evaluate(
            EvaluationConfig(mode="live"), catalog, SimulationSettings(), truth, rubric,
            tmp_path / "paid", forbidden,
        ))
    assert not (tmp_path / "paid").exists()


def test_per_run_exceptions_are_saved_not_excluded(tmp_path, catalog, scoring):
    truth, rubric = scoring

    def broken(*args):
        raise RuntimeError("Injected run initialization failure")

    rows = asyncio.run(evaluate(
        EvaluationConfig(mode="mocked", model="scripted-test-double", incident_ids=("inc-001",)),
        catalog, SimulationSettings(), truth, rubric, tmp_path / "exceptions", broken,
    ))
    assert len(rows) == 2
    assert all(row.metrics.status == "failed" and row.error for row in rows)
    assert all(row.metrics.abstained is None for row in rows)
    assert all(row.metrics.tool_calls is None for row in rows)
    assert all(row.metrics.top_ranked_correct is False for row in rows)


def test_budget_exhaustion_is_explicit_in_results(tmp_path, catalog, scoring):
    truth, rubric = scoring
    config = EvaluationConfig(
        mode="mocked", model="scripted-test-double", incident_ids=("inc-001",),
        budget=ExecutionBudget(max_tokens=1),
    )
    rows = asyncio.run(evaluate(
        config, catalog, SimulationSettings(), truth, rubric, tmp_path / "budget",
        lambda *args: ScriptedLLM({}),
    ))
    assert all(row.metrics.budget_exhausted for row in rows)
    assert all(row.metrics.status == "budget_exhausted" for row in rows)
    assert all(row.metrics.llm_calls == 0 for row in rows)


def test_existing_output_directory_is_preserved(tmp_path, catalog, scoring):
    truth, rubric = scoring
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "unrelated.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        asyncio.run(evaluate(
            EvaluationConfig(mode="mocked", model="scripted-test-double", incident_ids=("inc-001",)),
            catalog, SimulationSettings(), truth, rubric, output, lambda *args: ScriptedLLM({}),
        ))
    assert marker.read_text() == "preserve"


def test_evaluation_cli_refuses_paid_execution_without_approval(tmp_path):
    result = subprocess.run([
        sys.executable, str(ROOT / "evaluate.py"), "--mode", "live",
        "--output-dir", str(tmp_path / "live"),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "--allow-paid" in result.stderr
    assert not (tmp_path / "live").exists()


def test_offline_cli_saves_labeled_json_csv_summary_and_missing_scripts_fail_preflight(tmp_path):
    script_directory = tmp_path / "scripts"
    script_directory.mkdir()
    for approach, fixture in (("multi-agent", "multi_agent"), ("baseline", "baseline")):
        (script_directory / f"inc-001.{approach}.json").write_text(
            (ROOT / "tests" / "fixtures" / f"{fixture}.json").read_text()
        )
    command = [
        sys.executable, str(ROOT / "evaluate.py"), "--mode", "mocked",
        "--mock-script-dir", str(script_directory), "--incidents", "inc-001",
        "--repeats", "2", "--output-dir", str(tmp_path / "cli"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Mocked plumbing only" in result.stdout
    assert "no live benchmark" in result.stdout
    assert len(json.loads((tmp_path / "cli" / "results.json").read_text())["runs"]) == 4
    assert (tmp_path / "cli" / "manifest.json").exists()
    (script_directory / "inc-001.baseline.json").unlink()
    command[-1] = str(tmp_path / "incomplete")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / "incomplete").exists()

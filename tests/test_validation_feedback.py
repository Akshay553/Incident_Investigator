"""Focused contract/recovery checks with scripted replies, never live reasoning."""

import asyncio
import copy
import json
from pathlib import Path

import pytest

from baseline import run_baseline
from investigation import Coordinator, ModelOperation, run_multi_agent
from llm import LLMFailure, ModelResponse, ScriptedLLM
from models import ExecutionBudget, InitialPlan, InvestigationReport, Role

FIXTURES = Path(__file__).parent / "fixtures"


def replies(name="multi_agent"):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def prepare(tools, catalog, responses):
    coordinator = Coordinator(tools, ScriptedLLM({"review": responses}), catalog.incidents[0].context)
    asyncio.run(coordinator.evidence_tasks(InitialPlan.model_validate(replies()["plan"][0]).tasks))
    return coordinator


@pytest.mark.parametrize("stage,blocking", [
    ("mechanism", ["A trigger remains unknown"]), ("unknown", []),
    ("initiating_cause", ["A competing observation has not been resolved"]),
])
def test_confidence_is_preserved_in_accepted_analysis_and_inconclusive_fallback(tools, catalog, stage, blocking):
    response = replies()["reasoning-0"][0]
    response["task_id"] = "review"
    response["hypotheses"] = response["hypotheses"][:1]
    response["hypotheses"][0].update({
        "confidence": 0.99, "causal_stage": stage, "blocking_uncertainties": blocking,
    })
    coordinator = prepare(tools, catalog, [response])
    result = asyncio.run(coordinator.model(ModelOperation("review", Role.REASONING, Role.REASONING)))
    assert coordinator.llm.calls == ["review"]
    assert not coordinator.failures
    assert result.hypotheses[0].confidence == 0.99
    final = coordinator.finalize(
        "multi-agent", None, "critic_unavailable", fallback_hypotheses=result.hypotheses,
    )
    assert final.report.outcome == "inconclusive"
    assert final.report.hypotheses == result.hypotheses
    assert any("runtime fallback" in text for text in final.report.limitations)


def test_all_order_violations_have_paths_and_correct_without_score_edits(tools, catalog):
    rejected = replies()["reasoning-0"][0]
    rejected["task_id"] = "review"
    third = copy.deepcopy(rejected["hypotheses"][1])
    third.update({"hypothesis_id": "h3", "confidence": 0.99})
    rejected["hypotheses"][0]["confidence"] = 0.2
    rejected["hypotheses"][1]["confidence"] = 0.8
    rejected["hypotheses"].append(third)
    corrected = copy.deepcopy(rejected)
    corrected["hypotheses"].sort(key=lambda item: item["confidence"], reverse=True)
    corrected["limitations"] = ["Repaired descending order; scores and causal uncertainties are unchanged"]
    coordinator = prepare(tools, catalog, [rejected, corrected])
    result = asyncio.run(coordinator.model(ModelOperation("review", Role.REASONING, Role.REASONING)))
    feedback = json.loads(coordinator.model_calls[1].user_prompt)["response_correction"]
    assert len(feedback["validation_errors"]) == 2
    assert "hypotheses[1].confidence (h2)" in feedback["validation_errors"][0]
    assert "hypotheses[2].confidence (h3)" in feedback["validation_errors"][1]
    assert {item.hypothesis_id: item.confidence for item in result.hypotheses} == {
        "h1": 0.2, "h2": 0.8, "h3": 0.99,
    }
    assert coordinator.state.task("review").attempts == 2


def test_ranked_mechanism_and_blocking_gap_are_both_reported_then_finish_inconclusive(tools, catalog):
    responses = replies("baseline")
    rejected = responses["baseline-1"][0]
    rejected["hypotheses"][0]["confidence"] = 0.95
    rejected.update({
        "action": "finish", "queries": [],
        "report": {
            "incident_id": "inc-001", "outcome": "ranked",
            "hypotheses": copy.deepcopy(rejected["hypotheses"]),
        },
    })
    corrected = copy.deepcopy(rejected)
    corrected["rationale"] = "Observed mechanism confidence is not initiating-cause sufficiency; the settings gap remains"
    corrected["report"]["outcome"] = "inconclusive"
    corrected["report"]["limitations"] = ["The initiating cause and client settings remain unverified"]
    responses["baseline-1"] = [rejected, corrected]
    run = asyncio.run(run_baseline(tools, ScriptedLLM(responses), catalog.incidents[0].context))
    feedback = json.loads(run.trace.model_calls[-1].user_prompt)["response_correction"]
    assert len(feedback["validation_errors"]) == 2
    assert "report.hypotheses[0].causal_stage (h1)" in feedback["validation_errors"][0]
    assert "report.hypotheses[0].blocking_uncertainties (h1)" in feedback["validation_errors"][1]
    assert run.trace.termination_reason == "baseline_finished"
    assert run.report.outcome == "inconclusive"
    assert run.report.hypotheses[0].confidence == 0.95
    assert run.report.hypotheses[0].blocking_uncertainties == tuple(
        rejected["hypotheses"][0]["blocking_uncertainties"],
    )


def test_nested_report_order_error_uses_same_single_correction_and_precise_paths(tools, catalog):
    responses = replies("baseline")
    step = responses["baseline-1"][0]
    candidates = copy.deepcopy(replies()["reasoning-0"][0]["hypotheses"])
    candidates[1]["confidence"] = 0.95
    step.update({
        "action": "finish", "queries": [], "hypotheses": candidates,
        "report": {
            "incident_id": "inc-001", "outcome": "inconclusive", "hypotheses": copy.deepcopy(candidates),
            "limitations": ["Initiating causes remain unresolved"],
        },
    })
    corrected = copy.deepcopy(step)
    corrected["rationale"] = "Reordered unchanged candidates consistently in the step and report"
    corrected["hypotheses"].sort(key=lambda item: item["confidence"], reverse=True)
    corrected["report"]["hypotheses"] = copy.deepcopy(corrected["hypotheses"])
    responses["baseline-1"] = [step, corrected]
    llm = ScriptedLLM(responses)
    run = asyncio.run(run_baseline(tools, llm, catalog.incidents[0].context))
    assert llm.calls == ["baseline-0", "baseline-1", "baseline-1"]
    assert run.trace.termination_reason == "baseline_finished"
    feedback = json.loads(run.trace.model_calls[-1].user_prompt)["response_correction"]
    assert any("report.hypotheses[1].confidence" in text for text in feedback["validation_errors"])
    assert run.report.hypotheses == InvestigationReport.model_validate(corrected["report"]).hypotheses


def repeated_queries(partial=False):
    initial = replies("baseline")["baseline-0"][0]
    if partial:
        for query in initial["queries"]:
            query["time_range"] = {"start": "2026-01-15T09:50:00Z", "end": "2026-01-15T10:00:00Z"}
    rejected = {
        "action": "investigate", "rationale": "Accidentally repeated collected queries",
        "queries": copy.deepcopy(initial["queries"]), "hypotheses": [],
    }
    finish = {
        "action": "finish", "queries": [], "hypotheses": [],
        "rationale": "Use existing coverage; required causal observations remain unavailable in this investigation",
        "report": {
            "incident_id": "inc-001", "outcome": "inconclusive", "hypotheses": [],
            "limitations": ["Available observations do not establish an initiating cause"],
        },
    }
    return initial, rejected, finish


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("resolution", ["new_query", "finish"])
def test_repeated_queries_receive_existing_results_then_choose_new_query_or_finish(tools, catalog, partial, resolution):
    initial, rejected, finish = repeated_queries(partial)
    corrected = copy.deepcopy(finish)
    if resolution == "new_query":
        corrected = {
            "action": "investigate", "hypotheses": [],
            "rationale": "The previous reads are already executed; inspect a new configuration query",
            "queries": [copy.deepcopy(replies("baseline")["baseline-1"][0]["queries"][0])],
        }
    llm = ScriptedLLM({
        "baseline-0": [initial], "baseline-1": [rejected, corrected], "baseline-2": [finish],
    })
    respond = llm.respond
    seen_feedback = []

    async def inspect(task_id, system, user, schema, maximum):
        feedback = json.loads(user).get("response_correction")
        if feedback:
            assert len(feedback["already_executed_queries"]) == 2
            assert len(feedback["validation_errors"]) == 2
            for index, existing in enumerate(feedback["already_executed_queries"]):
                assert existing["field_path"] == f"queries[{index}]"
                result = existing["result"]
                assert result["query"] == {**initial["queries"][index], "services": [], "limit": 100}
                assert result["status"] == ("partial" if partial else "complete")
                assert result["source"] and result["coverage"] and result["collected_at"]
                assert not result["error"]
                if partial:
                    assert result["coverage"]["gaps"] and result["evidence"] == []
            seen_feedback.append(feedback)
        return await respond(task_id, system, user, schema, maximum)

    llm.respond = inspect
    run = asyncio.run(run_baseline(tools, llm, catalog.incidents[0].context))
    assert len(seen_feedback) == 1
    assert run.trace.termination_reason == "baseline_finished"
    assert run.report.outcome == "inconclusive"
    assert run.trace.state.usage.tool_calls == (3 if resolution == "new_query" else 2)
    accepted = next(task for task in run.trace.state.tasks if task.spec.task_id == "baseline-1")
    assert accepted.status == "succeeded" and accepted.attempts == 2
    assert json.loads(accepted.output_json)["action"] == corrected["action"]
    assert not any("runtime fallback" in text for text in run.report.limitations)


def test_second_repeated_proposal_stops_without_reexecuting_or_third_correction(tools, catalog):
    initial, rejected, finish = repeated_queries(True)
    llm = ScriptedLLM({"baseline-0": [initial], "baseline-1": [rejected, rejected, finish]})
    run = asyncio.run(run_baseline(tools, llm, catalog.incidents[0].context))
    assert llm.calls == ["baseline-0", "baseline-1", "baseline-1"]
    assert run.trace.state.usage.tool_calls == 2
    assert run.trace.termination_reason == "baseline_analysis_unavailable"
    failed = next(task for task in run.trace.state.tasks if task.spec.task_id == "baseline-1")
    assert failed.output_json is None and failed.attempts == 2


@pytest.mark.parametrize("budget", [ExecutionBudget(max_llm_calls=2), ExecutionBudget(operation_attempts=1)])
def test_repeated_query_correction_shares_request_and_attempt_budgets(tools, catalog, budget):
    initial, rejected, finish = repeated_queries()
    llm = ScriptedLLM({"baseline-0": [initial], "baseline-1": [rejected, finish]})
    run = asyncio.run(run_baseline(tools, llm, catalog.incidents[0].context, budget))
    assert llm.calls == ["baseline-0", "baseline-1"]
    assert run.trace.state.usage.tool_calls == 2
    assert any(item.code == "budget_exhausted" for item in run.trace.failures)


def test_unsafe_scope_prevents_correction_even_with_repeated_query_and_all_errors_are_visible(tools, catalog):
    initial, rejected, finish = repeated_queries()
    rejected["queries"][1]["incident_id"] = "inc-002"
    llm = ScriptedLLM({"baseline-0": [initial], "baseline-1": [rejected, finish]})
    run = asyncio.run(run_baseline(tools, llm, catalog.incidents[0].context))
    assert llm.calls == ["baseline-0", "baseline-1"]
    assert "queries[0]" in run.trace.model_calls[-1].error
    assert "queries[1].incident_id" in run.trace.model_calls[-1].error
    assert run.trace.state.usage.tool_calls == 2


def test_critic_repeated_query_is_corrected_before_followup_execution(tools, catalog):
    responses = replies()
    corrected = copy.deepcopy(responses["critic-0"][0])
    rejected = copy.deepcopy(corrected)
    rejected["follow_ups"][0].update({
        "target_role": "observability", "rationale": "Try the collected logs again",
        "query": copy.deepcopy(responses["plan"][0]["tasks"][0]["query"]),
    })
    responses["critic-0"] = [rejected, corrected]
    llm = ScriptedLLM(responses)
    run = asyncio.run(run_multi_agent(tools, llm, catalog.incidents[0].context))
    assert run.report.outcome == "ranked"
    assert run.trace.state.usage.tool_calls == 6
    calls = [call for call in run.trace.model_calls if call.task_id == "critic-0"]
    assert len(calls) == 2 and calls[1].correction_of_attempt == 1
    feedback = json.loads(calls[1].user_prompt)["response_correction"]
    assert feedback["already_executed_queries"][0]["field_path"] == "follow_ups[0].query"
    assert feedback["already_executed_queries"][0]["task_id"] == "initial-logs"
    followup = next(task for task in run.trace.state.tasks if task.spec.task_id == "followup-1-0")
    assert followup.spec.role == Role.DEPENDENCY and followup.spec.query.family == "changes"


def test_reporting_keeps_acquired_evidence_when_validated_reasoning_is_unavailable(tools, catalog):
    responses = replies()
    responses["reasoning-0"] = [LLMFailure("refusal", "Injected unavailable Reasoning")]
    llm = ScriptedLLM(responses)
    respond = llm.respond
    captured = []

    async def report_with_failure_context(task_id, system, user, schema, maximum):
        if task_id == "report":
            payload = json.loads(user)
            assert payload["validated_analysis_available"] is False
            assert payload["current_hypotheses"] == []
            assert payload["evidence"]
            assert payload["tool_observations"]
            assert any("Validated Reasoning analysis unavailable; acquired evidence ledger contains" in text
                       for text in payload["runtime_limitations"])
            assert payload["analysis_failures"] == [{
                "task_id": "reasoning-0", "role": "reasoning",
                "status": "failed", "failure": "Injected unavailable Reasoning",
            }]
            assert "no accepted hypotheses" in system
            captured.append(payload)
            llm.calls.append(task_id)
            return ModelResponse(InvestigationReport(
                incident_id="inc-001", outcome="inconclusive",
                limitations=("Validated Reasoning unavailable; acquired tool evidence is retained",),
            ).model_dump_json(), 0, 0)
        return await respond(task_id, system, user, schema, maximum)

    llm.respond = report_with_failure_context
    run = asyncio.run(run_multi_agent(tools, llm, catalog.incidents[0].context))
    assert llm.calls.count("reasoning-0") == 1
    assert "critic-0" not in llm.calls
    assert run.trace.termination_reason == "reasoning_unavailable"
    assert captured[0]["evidence"] == [item.model_dump(mode="json") for item in run.trace.state.evidence]
    assert len(captured[0]["tool_observations"]) == len(run.trace.state.tool_results)
    assert any("acquired tool evidence" in text for text in run.report.limitations)
    assert next(task for task in run.trace.state.tasks if task.spec.task_id == "report").status == "succeeded"

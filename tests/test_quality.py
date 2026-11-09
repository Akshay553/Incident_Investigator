"""Scripted quality-policy regressions, not evidence of live model performance."""

import asyncio
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents import STANDARDS, prompt
from baseline import run_baseline
from investigation import Coordinator, run_multi_agent, unique_limitations
from llm import ScriptedLLM, structured_schema
from models import (
    AgentResult, ContinueDecision, ExecutionBudget, Hypothesis, InvestigationReport,
    Role, TaskSpec, ToolFamily, ToolQuery,
)

FIXTURES = Path(__file__).parent / "fixtures"


def replies(name="multi_agent"):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def investigate(tools, catalog, responses, approach="multi-agent", budget=None):
    runner = run_multi_agent if approach == "multi-agent" else run_baseline
    return asyncio.run(runner(
        tools, ScriptedLLM(responses), catalog.incidents[0].context, budget,
    ))


def finish_before_followup(responses):
    responses["decision-0"] = [{
        "action": "finish", "rationale": "The requested observation remains unavailable in this run",
        "deferred_follow_ups": [{
            "follow_up_index": 0, "reason": "The investigator stopped before inspecting settings",
        }],
    }]
    responses["report"][0]["hypotheses"] = copy.deepcopy(
        responses["reasoning-0"][0]["hypotheses"][:1],
    )
    responses["report"][0]["outcome"] = "inconclusive"
    responses["report"][0]["limitations"] = ["The initiating cause remains unresolved"]


def test_critical_feedback_acquires_downstream_evidence_and_revises_candidates(tools, catalog):
    responses = replies()
    for task in responses["plan"][0]["tasks"][:2]:
        task["query"]["services"] = ["checkout"]
    responses["observability-0"][0]["findings"] = [{
        "claim": "Checkout waits for connections",
        "supporting_evidence_ids": ["inc-001:logs:r01", "inc-001:metrics:r01"],
    }]
    initial_alternative = responses["reasoning-0"][0]["hypotheses"][1]
    initial_alternative.update({
        "claim": "Downstream processing failure may be responsible for client waits",
        "supporting_evidence_ids": ["inc-001:logs:r01"],
        "contradicting_evidence_ids": ["inc-001:dependencies:r01"],
    })
    criticism = responses["critic-0"][0]
    criticism["objections"][0]["follow_up_indices"] = [0, 1, 2]
    criticism["objections"][0]["reason"] = (
        "Verify changed client settings and downstream observations before attributing client waits"
    )
    for family in ("metrics", "logs"):
        request = copy.deepcopy(criticism["follow_ups"][0])
        request.update({
            "target_role": "observability",
            "rationale": f"Inspect downstream {family} to discriminate database trouble from local waits",
            "evidence_ids": ["inc-001:dependencies:r01"],
        })
        request["query"].update({"family": family, "services": ["orders-db"]})
        criticism["follow_ups"].append(request)
    responses["decision-0"][0]["selected_follow_ups"] = [0, 1, 2]
    responses["observability-1"] = [{
        "task_id": "observability-1", "role": "observability",
        "findings": [{
            "claim": "Downstream samples show healthy processing despite caller waits",
            "supporting_evidence_ids": ["inc-001:logs:r02", "inc-001:metrics:r03"],
        }],
    }]
    alternative = copy.deepcopy(initial_alternative)
    alternative["confidence"] = 0.1
    alternative["contradicting_evidence_ids"] += ["inc-001:logs:r02", "inc-001:metrics:r03"]
    responses["reasoning-1"][0]["hypotheses"].append(alternative)
    responses["reasoning-1"][0]["findings"][0]["claim"] = (
        "The configuration observation explains local waits; downstream samples weaken the competing cause"
    )
    responses["report"][0]["hypotheses"] = copy.deepcopy(
        responses["reasoning-1"][0]["hypotheses"],
    )
    run = investigate(tools, catalog, responses)
    assert not run.trace.failures
    assert run.report.outcome == "ranked"
    assert run.trace.execution_mode == "mocked"
    assert run.trace.state.usage.tool_calls == 8
    tasks = {task.spec.task_id: task for task in run.trace.state.tasks}
    assert tasks["followup-1-1"].spec.query.services == ("orders-db",)
    assert tasks["followup-1-2"].spec.query.services == ("orders-db",)
    assert tasks["followup-1-1"].spec.depends_on == ("decision-0", "reasoning-0")
    outputs = {item.task_id: item for item in run.trace.state.agent_results}
    before, after = outputs["reasoning-0"].hypotheses, outputs["reasoning-1"].hypotheses
    assert before[0].causal_stage == "mechanism"
    assert before[0].blocking_uncertainties
    assert after[0].causal_stage == "initiating_cause"
    assert not after[0].blocking_uncertainties
    assert set(before[0].supporting_evidence_ids) <= set(after[0].supporting_evidence_ids)
    assert set(before[1].contradicting_evidence_ids) <= set(after[1].contradicting_evidence_ids)
    assert after[1].confidence < before[1].confidence
    assert run.report.hypotheses == after
    decision = next(call for call in run.trace.model_calls if call.task_id == "decision-0")
    payload = json.loads(decision.user_prompt)
    assert payload["current_follow_ups"][1]["query"]["services"] == ["orders-db"]
    assert payload["current_follow_ups"][1]["index"] == 1
    critic_payload = json.loads(next(
        call.user_prompt for call in run.trace.model_calls if call.task_id == "critic-0"
    ))
    assert "orders-db" in critic_payload["observed_service_scopes"]
    assert critic_payload["observed_dependency_edges"][0]["evidence_id"] == "inc-001:dependencies:r01"
    assert not any("Unresolved critical" in item for item in run.report.limitations)


@pytest.mark.parametrize("change,message", [
    ("missing_disposition", "targeted query"),
    ("invalid_index", "unknown Critic follow-up"),
])
def test_invalid_critical_objection_is_visible_not_silently_accepted(tools, catalog, change, message):
    responses = replies()
    finish_before_followup(responses)
    criticism = responses["critic-0"][0]
    if change == "missing_disposition":
        criticism["follow_ups"] = []
        criticism["objections"][0]["follow_up_indices"] = []
    else:
        criticism["objections"][0]["follow_up_indices"] = [99]
    run = investigate(tools, catalog, responses)
    assert run.report.outcome == "inconclusive"
    assert run.trace.termination_reason == "critic_unavailable"
    assert run.trace.state.usage.tool_calls == 5
    assert any(message in failure.message for failure in run.trace.failures)


def test_uninvestigable_objection_explains_limitation_without_inventing_query(tools, catalog):
    responses = replies()
    finish_before_followup(responses)
    criticism = responses["critic-0"][0]
    criticism["follow_ups"] = []
    criticism["objections"][0].update({
        "follow_up_indices": [],
        "reason": "No retained observation tests the mitigation's effect",
        "unavailable_reason": "Read-only tools cannot perform or observe a new mitigation experiment",
    })
    responses["decision-0"][0]["deferred_follow_ups"] = []
    run = investigate(tools, catalog, responses)
    assert not run.trace.failures
    assert run.report.outcome == "inconclusive"
    assert run.trace.state.usage.tool_calls == 5
    assert any("Read-only tools cannot" in item for item in run.report.limitations)
    assert any("Unresolved critical objection for h1" in item for item in run.report.limitations)


def test_omitted_critical_query_requires_an_explicit_deferral(tools, catalog):
    responses = replies()
    finish_before_followup(responses)
    responses["decision-0"][0]["deferred_follow_ups"] = []
    run = investigate(tools, catalog, responses)
    assert run.report.outcome == "inconclusive"
    assert run.trace.termination_reason == "continuation_decision_unavailable"
    assert any("explicit deferral" in item.message for item in run.trace.failures)
    assert any("inspect settings" in item for item in run.report.limitations)


def test_explicit_deferral_does_not_resolve_critical_objection(tools, catalog):
    responses = replies()
    finish_before_followup(responses)
    run = investigate(tools, catalog, responses)
    assert not run.trace.failures
    assert run.report.outcome == "inconclusive"
    assert run.report.hypotheses[0].confidence == responses["reasoning-0"][0]["hypotheses"][0]["confidence"]
    assert any("stopped before inspecting settings" in item for item in run.report.limitations)


@pytest.mark.parametrize("budget,expected_calls,reason", [
    (ExecutionBudget(max_follow_up_rounds=0), 5, "round budget exhausted"),
    (ExecutionBudget(max_tool_calls=5), 5, "budget exhausted"),
])
def test_critical_query_blocked_by_budget_is_explained_and_cannot_bypass_it(tools, catalog, budget, expected_calls, reason):
    run = investigate(tools, catalog, replies(), budget=budget)
    assert run.report.outcome == "inconclusive"
    assert run.trace.state.usage.tool_calls == expected_calls
    assert any(reason in item for item in run.report.limitations)


@pytest.mark.parametrize("approach", ["multi-agent", "baseline"])
@pytest.mark.parametrize("stage,blocking", [
    ("mechanism", []), ("unknown", []), ("initiating_cause", ["A causal condition remains unobserved"]),
])
def test_both_approaches_preserve_confidence_for_causal_scrutiny_not_structural_rejection(tools, catalog, approach, stage, blocking):
    responses = replies("multi_agent" if approach == "multi-agent" else "baseline")
    output = responses["reasoning-0" if approach == "multi-agent" else "baseline-1"][0]
    output["hypotheses"][0].update({
        "causal_stage": stage, "blocking_uncertainties": blocking, "confidence": 0.95,
    })
    run = investigate(tools, catalog, responses, approach)
    assert not run.trace.failures
    assert run.report.outcome == "ranked"
    task_id = "reasoning-0" if approach == "multi-agent" else "baseline-1"
    initial = next(task for task in run.trace.state.tasks if task.spec.task_id == task_id)
    candidate = json.loads(initial.output_json)["hypotheses"][0]
    assert candidate["confidence"] == 0.95
    assert candidate["causal_stage"] == stage
    assert candidate["blocking_uncertainties"] == blocking


@pytest.mark.parametrize("approach", ["multi-agent", "baseline"])
def test_mechanism_only_candidate_cannot_be_an_ordinary_root_cause_success(tools, catalog, approach):
    responses = replies("multi_agent" if approach == "multi-agent" else "baseline")
    if approach == "multi-agent":
        finish_before_followup(responses)
        responses["critic-0"][0].update({"objections": [], "follow_ups": []})
        responses["decision-0"][0]["deferred_follow_ups"] = []
    else:
        candidate = responses["baseline-1"][0]["hypotheses"]
        responses["baseline-1"][0].update({
            "action": "finish", "queries": [],
            "report": {
                "incident_id": "inc-001", "outcome": "inconclusive", "hypotheses": copy.deepcopy(candidate),
                "limitations": ["The mechanism is observed but its initiating cause is unresolved"],
            },
        })
    run = investigate(tools, catalog, responses, approach)
    assert not run.trace.failures
    assert run.report.outcome == "inconclusive"
    assert run.report.hypotheses[0].causal_stage == "mechanism"
    assert run.report.hypotheses[0].blocking_uncertainties
    assert any("initiating cause remains unresolved" in item for item in run.report.limitations)


def test_baseline_can_follow_an_acquired_edge_and_revise_the_initiating_cause(tools, catalog):
    responses = replies("baseline")
    for query in responses["baseline-0"][0]["queries"]:
        query["services"] = ["checkout"]
    responses["baseline-3"] = responses.pop("baseline-2")
    responses["baseline-2"] = [{
        "action": "investigate",
        "rationale": "The acquired checkout to orders-db edge warrants downstream health checks",
        "hypotheses": copy.deepcopy(responses["baseline-1"][0]["hypotheses"]),
        "queries": [{
            "incident_id": "inc-001", "family": family, "services": ["orders-db"],
            "time_range": catalog.incidents[0].context.time_range.model_dump(mode="json"),
        } for family in ("logs", "metrics")],
    }]
    run = investigate(tools, catalog, responses, "baseline")
    assert not run.trace.failures
    assert run.report.outcome == "ranked"
    assert run.trace.state.usage.tool_calls == 7
    call = next(item for item in run.trace.model_calls if item.task_id == "baseline-2")
    payload = json.loads(call.user_prompt)
    assert "orders-db" in payload["observed_service_scopes"]
    assert payload["observed_dependency_edges"][0]["source_service"] == "checkout"
    assert run.report.hypotheses[0].causal_stage == "initiating_cause"
    assert not run.report.hypotheses[0].blocking_uncertainties


def test_unresolved_critical_high_confidence_is_retained_without_claiming_a_supported_outcome(tools, catalog):
    responses = replies()
    candidate = responses["reasoning-0"][0]["hypotheses"][0]
    candidate.update({"causal_stage": "initiating_cause", "blocking_uncertainties": [], "confidence": 0.95})
    finish_before_followup(responses)
    run = investigate(tools, catalog, responses)
    assert not run.trace.failures
    assert run.report.outcome == "inconclusive"
    assert run.report.hypotheses[0].confidence == 0.95
    original = next(item for item in run.trace.state.agent_results if item.task_id == "reasoning-0")
    assert original.hypotheses[0].confidence == 0.95
    assert original.hypotheses[0].supporting_evidence_ids == tuple(candidate["supporting_evidence_ids"])
    assert not any("withheld" in item for item in run.report.limitations)
    assert any("Unresolved critical objection" in item for item in run.report.limitations)


def test_report_limitations_are_canonical_and_deduplicated_without_merging_distinct_gaps(tools, catalog):
    responses = replies()
    responses["report"][0]["limitations"] += [
        "  No mitigation experiment has confirmed recovery. ",
        "NO MITIGATION EXPERIMENT HAS CONFIRMED RECOVERY",
        "No mitigation experiment has confirmed recovery in the dependent service",
    ]
    run = investigate(tools, catalog, responses)
    assert not run.trace.failures
    assert run.report.outcome == "ranked"
    assert run.report.limitations.count("No mitigation experiment has confirmed recovery") == 1
    assert "No mitigation experiment has confirmed recovery in the dependent service" in run.report.limitations
    assert run.report.limitations == unique_limitations(run.report.limitations)
    call = next(item for item in run.trace.model_calls if item.task_id == "report")
    canonical = json.loads(call.user_prompt)["runtime_limitations"]
    assert "No mitigation experiment has confirmed recovery" in canonical
    assert "Mocked execution: scripted responses, not real model reasoning" in canonical


@pytest.mark.parametrize("incident_index", range(10))
def test_service_scopes_derive_only_from_context_and_acquired_evidence(tools, catalog, incident_index):
    context = catalog.incidents[incident_index].context
    coordinator = Coordinator(tools, ScriptedLLM({}), context)
    _, before = prompt(Role.REASONING, "before", context, coordinator.state.snapshot(), coordinator.state.budget)
    assert json.loads(before)["observed_service_scopes"] == sorted(set(context.affected_services))
    assert json.loads(before)["observed_dependency_edges"] == []
    asyncio.run(coordinator.evidence_tasks((TaskSpec(
        task_id="inspect-edge", role=Role.DEPENDENCY,
        query=ToolQuery(incident_id=context.incident_id, family=ToolFamily.DEPENDENCIES, time_range=context.time_range),
    ),)))
    snapshot = coordinator.state.snapshot()
    system, after = prompt(Role.REASONING, "after", context, snapshot, coordinator.state.budget)
    _, baseline = prompt("baseline", "after", context, snapshot, coordinator.state.budget)
    expected = set(context.affected_services)
    for item in snapshot.evidence:
        expected.update((item.record.service, item.record.target_service))
    payload = json.loads(after)
    assert payload["observed_service_scopes"] == sorted(expected)
    assert payload["observed_dependency_edges"] == json.loads(baseline)["observed_dependency_edges"]
    assert len(payload["observed_dependency_edges"]) == len(snapshot.evidence)
    assert STANDARDS in system
    assert all(edge["evidence_id"] in {item.evidence_id for item in snapshot.evidence}
               for edge in payload["observed_dependency_edges"])


def test_quality_fields_are_in_strict_provider_schemas_and_invalid_dispositions_fail():
    hypothesis = structured_schema(Hypothesis)
    assert {"causal_stage", "blocking_uncertainties"} <= set(hypothesis["required"])
    critic = structured_schema(AgentResult)
    assert {"follow_up_indices", "unavailable_reason"} <= set(critic["$defs"]["Objection"]["required"])
    decision = structured_schema(ContinueDecision)
    assert "deferred_follow_ups" in decision["required"]
    with pytest.raises(ValidationError, match="cannot also be selected"):
        ContinueDecision(
            action="continue", rationale="Invalid overlap", selected_follow_ups=(0,),
            deferred_follow_ups=({"follow_up_index": 0, "reason": "Deferred"},),
        )
    with pytest.raises(ValidationError):
        AgentResult(
            task_id="critic", role=Role.CRITIC,
            objections=({"hypothesis_id": "h1", "reason": "Invalid index", "follow_up_indices": [-1]},),
        )


def test_original_pilot_reports_remain_readable_without_rewriting_legacy_fields():
    pilot = FIXTURES.parent.parent / "outputs" / "pilot-live-20261108" / "runs"
    for approach in ("multi-agent", "baseline"):
        path = pilot / f"inc-001.{approach}.001" / "report.json"
        if not path.exists():
            pytest.skip("Historical live pilot artifacts are not bundled with this checkout")
        raw = path.read_bytes()
        report = InvestigationReport.model_validate_json(raw)
        assert all(item.causal_stage == "unknown" for item in report.hypotheses)
        assert path.read_bytes() == raw

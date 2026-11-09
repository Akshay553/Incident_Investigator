"""Real-size offline prompt/recovery regression replay; never a live benchmark."""

import asyncio
import copy
import json
from pathlib import Path

import pytest

from agents import ACCESS, SCHEMAS, STANDARDS, finish_examples, prompt
from baseline import run_baseline
from investigation import Coordinator, run_multi_agent
from llm import ModelResponse, ScriptedLLM, structured_schema
from models import (
    BaselineStep, ExecutionBudget, FailureRule, InitialPlan, InvestigationReport,
    Role, SimulationSettings, TaskSpec, TaskState, ToolFamily, ToolQuery, TimeRange,
)
from tools import SimulatedTools

FIXTURES = Path(__file__).parent / "fixtures"


def recorded(approach):
    return json.loads((FIXTURES / "recovery_replay.json").read_text(encoding="utf-8"))[approach]


def responses(approach):
    return {
        task: [item["response_json"] for item in calls]
        for task, calls in recorded(approach).items()
    }


def gathered(tools, catalog):
    coordinator = Coordinator(tools, ScriptedLLM({}), catalog.incidents[0].context)
    plan = InitialPlan.model_validate_json(responses("multi-agent")["plan"][0])
    asyncio.run(coordinator.evidence_tasks(plan.tasks))
    return coordinator


@pytest.mark.parametrize("critical", [False, True])
@pytest.mark.parametrize("round_limit", [0, 2])
def test_empty_current_round_records_runtime_finish_without_using_stale_indices(tools, catalog, critical, round_limit):
    replies = responses("multi-agent")
    criticism = json.loads(replies["critic-0"][0])
    criticism["follow_ups"] = []
    if critical:
        criticism["objections"][0].update({
            "follow_up_indices": [],
            "unavailable_reason": "Read-only tools cannot conduct a new mitigation experiment",
        })
    else:
        criticism["objections"] = []
    replies["critic-0"] = [criticism]
    candidate = json.loads(replies["reasoning-0"][-1])["hypotheses"]
    replies["report"] = [{
        "incident_id": "inc-001", "outcome": "inconclusive", "hypotheses": candidate,
        "limitations": ["The accepted candidate remains provisional"],
    }]
    llm = ScriptedLLM(replies)
    run = asyncio.run(run_multi_agent(
        tools, llm, catalog.incidents[0].context, ExecutionBudget(max_follow_up_rounds=round_limit),
    ))
    assert "decision-0" not in llm.calls
    decision = next(x for x in run.trace.state.tasks if x.spec.task_id == "decision-0")
    assert decision.status == "succeeded" and decision.attempts == 0
    output = json.loads(decision.output_json)
    assert output["selected_follow_ups"] == output["deferred_follow_ups"] == []
    assert "Runtime decision" in output["rationale"]
    assert decision.spec.depends_on == ("critic-0",)
    assert run.report.outcome == "inconclusive"
    if critical:
        assert any("Unresolved critical objection" in x for x in run.report.limitations)
    assert not any(x.task_id.startswith("decision-") for x in run.trace.model_calls)


def test_finish_examples_validate_against_full_provider_schema(catalog):
    schema = structured_schema(BaselineStep)
    for example in finish_examples(catalog.incidents[0].context):
        step = BaselineStep.model_validate(example)
        assert set(example) == set(schema["required"])
        assert set(example["report"]) == set(
            schema["$defs"]["InvestigationReport"]["required"]
        )
        assert set(example["hypotheses"][0]) == set(schema["$defs"]["Hypothesis"]["required"])
        assert step.hypotheses == step.report.hypotheses
        assert bool(step.hypotheses[0].blocking_uncertainties) == (step.report.outcome == "inconclusive")
        assert 0 <= step.hypotheses[0].confidence <= 1


def test_valid_inconclusive_baseline_finish_preserves_provisional_candidate(tools, catalog):
    replies = json.loads((FIXTURES / "baseline.json").read_text(encoding="utf-8"))
    step = replies["baseline-1"][0]
    candidate = copy.deepcopy(step["hypotheses"])
    step.update({
        "action": "finish", "queries": [],
        "report": {
            "incident_id": "inc-001", "outcome": "inconclusive", "hypotheses": candidate,
            "limitations": ["The observed mechanism has an unresolved initiating-cause gap"],
        },
    })
    llm = ScriptedLLM(replies)
    run = asyncio.run(run_baseline(tools, llm, catalog.incidents[0].context))
    assert not run.trace.failures
    assert run.trace.termination_reason == "baseline_finished"
    assert run.report.outcome == "inconclusive"
    assert run.report.hypotheses == BaselineStep.model_validate(step).hypotheses
    assert llm.calls == ["baseline-0", "baseline-1"]
    assert not any("runtime fallback" in x for x in run.report.limitations)


def test_recorded_baseline_failure_can_receive_one_valid_inconclusive_mock_correction(tools, catalog):
    replies = responses("baseline")
    corrected = json.loads(replies["baseline-2"][1])
    corrected["report"]["outcome"] = "inconclusive"
    corrected["rationale"] = (
        "The blocking question remains unresolved, so retain the provisional candidate "
        "and finish inconclusive instead of claiming an established cause."
    )
    replies["baseline-2"][1] = corrected
    llm = ScriptedLLM(replies)
    run = asyncio.run(run_baseline(tools, llm, catalog.incidents[0].context))
    assert run.trace.termination_reason == "baseline_finished"
    assert run.report.hypotheses == BaselineStep.model_validate(corrected).hypotheses
    assert run.report.outcome == "inconclusive"
    calls = [x for x in run.trace.model_calls if x.task_id == "baseline-2"]
    assert len(calls) == 2 and calls[1].correction_of_attempt == 1
    assert calls[0].error and not calls[1].error
    assert json.loads(calls[0].response_json) == json.loads(recorded("baseline")["baseline-2"][0]["response_json"])


def test_unmodified_recorded_baseline_correction_still_fails_without_third_request(tools, catalog):
    llm = ScriptedLLM(responses("baseline"))
    run = asyncio.run(run_baseline(tools, llm, catalog.incidents[0].context))
    assert llm.calls.count("baseline-2") == 2
    assert run.trace.termination_reason == "baseline_analysis_unavailable"
    assert "report.hypotheses[0].blocking_uncertainties" in run.trace.model_calls[-1].error


def test_compact_reasoning_keeps_uncited_counterevidence_and_does_not_resend_archived_outputs(tools, catalog):
    coordinator = gathered(tools, catalog)
    snapshot = coordinator.state.snapshot()
    archive = TaskState(
        spec=TaskSpec(task_id="old-history", role=Role.ORCHESTRATOR),
        status="succeeded", output_json="ARCHIVE_ONLY " * 10000,
    )
    snapshot = snapshot.model_copy(update={"tasks": (*snapshot.tasks, archive)})
    candidate = json.loads(responses("multi-agent")["reasoning-0"][-1])["hypotheses"]
    candidate[0]["contradicting_evidence_ids"] = ["inc-001:logs:r03"]
    _, user = prompt(
        Role.REASONING, "reason", coordinator.context, snapshot, coordinator.state.budget,
        role=Role.REASONING, extra={"current_hypotheses": candidate},
    )
    payload = json.loads(user)
    assert "tasks" not in payload and "ARCHIVE_ONLY" not in user
    assert snapshot.tasks[-1].output_json == archive.output_json
    assert payload["current_hypotheses"] == candidate
    assert payload["evidence"] == [x.model_dump(mode="json") for x in snapshot.evidence]
    for expected, observation in zip(snapshot.tool_results, payload["tool_observations"]):
        assert observation["source"] == expected.source
        assert observation["query"] == expected.query.model_dump(mode="json")
        assert observation["coverage"] == expected.coverage.model_dump(mode="json")
        assert observation["evidence_ids"] == [x.evidence_id for x in expected.evidence]
    assert user == json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


@pytest.mark.parametrize("role", [Role.OBSERVABILITY, Role.DEPENDENCY])
def test_specialist_context_contains_only_its_observations_not_hypothesis_history(tools, catalog, role):
    coordinator = gathered(tools, catalog)
    _, user = prompt(role, "specialist", coordinator.context, coordinator.state.snapshot(),
                     coordinator.state.budget, role=role)
    payload = json.loads(user)
    assert all(ToolFamily(x["record"]["kind"]) in ACCESS[role] for x in payload["evidence"])
    assert all(ToolFamily(x["query"]["family"]) in ACCESS[role] for x in payload["tool_observations"])
    assert "tasks" not in payload and "current_hypotheses" not in payload
    assert "suggested_queries" not in payload


def test_reporting_context_preserves_both_citation_polarities_and_gap_provenance(tools, catalog):
    coordinator = gathered(tools, catalog)
    candidate = json.loads(responses("multi-agent")["reasoning-0"][-1])["hypotheses"]
    candidate[0]["contradicting_evidence_ids"] = ["inc-001:logs:r03"]
    _, user = prompt(
        Role.REPORTING, "report", coordinator.context, coordinator.state.snapshot(),
        coordinator.state.budget, role=Role.REPORTING,
        extra={"current_hypotheses": candidate, "runtime_limitations": ["A source gap remains"],
               "must_be_inconclusive": True},
    )
    payload = json.loads(user)
    assert payload["current_hypotheses"] == candidate
    assert payload["must_be_inconclusive"] and payload["runtime_limitations"] == ["A source gap remains"]
    refs = set(candidate[0]["supporting_evidence_ids"] + candidate[0]["contradicting_evidence_ids"])
    assert refs <= {x["evidence_id"] for x in payload["evidence"]}
    assert payload["evidence"] == [x.model_dump(mode="json") for x in coordinator.state.snapshot().evidence]
    assert all(x["source"] and x["query"] and x["coverage"] for x in payload["tool_observations"])
    assert "tasks" not in payload and "specialist_findings" not in payload


@pytest.mark.parametrize("status", ["not_queried", "failed", "covered_empty"])
def test_critic_sees_actual_query_status_not_inferred_source_unavailability(catalog, status):
    settings = SimulationSettings(failures=(
        FailureRule(incident_id="inc-001", family=ToolFamily.LOGS, retryable=False),
    )) if status == "failed" else SimulationSettings()
    coordinator = gathered(SimulatedTools(catalog, settings), catalog)
    if status == "covered_empty":
        asyncio.run(coordinator.evidence_tasks((TaskSpec(
            task_id="empty-db-window", role=Role.OBSERVABILITY,
            query=ToolQuery(
                incident_id="inc-001", family=ToolFamily.LOGS, services=("orders-db",),
                time_range=TimeRange(start="2026-01-15T10:15:00Z", end="2026-01-15T10:16:00Z"),
            ),
        ),)))
    _, user = prompt(
        Role.CRITIC, "critique", coordinator.context, coordinator.state.snapshot(),
        coordinator.state.budget, role=Role.CRITIC,
    )
    payload = json.loads(user)
    assert "orders-db" in payload["observed_service_scopes"]
    history = payload["tool_observations"]
    if status == "not_queried":
        assert not any(x["query"]["services"] == ["orders-db"] for x in history)
    elif status == "failed":
        failure = next(x for x in history if x["status"] == "error")
        assert failure["error"] and not failure["coverage"]["observed_ranges"]
    else:
        empty = next(x for x in history if x["query"]["services"] == ["orders-db"])
        assert empty["status"] == "complete" and empty["evidence_ids"] == []
        assert not empty["coverage"]["gaps"] and empty["coverage"]["returned_records"] == 0
    assert "NOT QUERIED" in STANDARDS and "QUERY FAILED" in STANDARDS


def test_saved_recovery_replays_through_reporting_with_actual_prompt_reservations(tools, catalog):
    fixture = recorded("multi-agent")

    class ReplayLLM(ScriptedLLM):
        """Old token subtotals are a pessimistic workload fixture, not new provider measurements."""
        def __init__(self):
            super().__init__(responses("multi-agent"))
            self.occurrences = {}

        async def respond(self, task_id, system, user, schema, maximum):
            if task_id == "report":
                payload = json.loads(user)
                report = InvestigationReport(
                    incident_id=payload["incident"]["incident_id"], outcome="inconclusive",
                    hypotheses=payload["current_hypotheses"],
                    limitations=payload["runtime_limitations"],
                )
                self.calls.append(task_id)
                return ModelResponse(report.model_dump_json(), 10000, 500)
            index = self.occurrences.get(task_id, 0)
            self.occurrences[task_id] = index + 1
            response = await super().respond(task_id, system, user, schema, maximum)
            saved = fixture[task_id][index]
            return ModelResponse(
                response.response_json, saved["recorded_input_tokens"],
                saved["recorded_output_tokens"],
            )

    llm = ReplayLLM()
    run = asyncio.run(run_multi_agent(tools, llm, catalog.incidents[0].context))
    assert run.trace.budget == ExecutionBudget()
    assert run.trace.execution_mode == "mocked"
    assert run.trace.termination_reason == "orchestrator_finished"
    assert run.report.outcome == "inconclusive"
    assert run.trace.state.usage.tool_calls == 6
    assert run.trace.state.usage.llm_calls == 10
    expected_tokens = 10500 + sum(
        fixture[call.task_id][call.attempt - 1]["recorded_input_tokens"]
        + fixture[call.task_id][call.attempt - 1]["recorded_output_tokens"]
        for call in run.trace.model_calls[:-1]
    )
    assert run.trace.state.usage.tokens == expected_tokens
    assert all(x.code != "budget_exhausted" for x in run.trace.failures)
    assert "decision-1" not in llm.calls
    assert llm.responses["decision-1"] == [fixture["decision-1"][0]["response_json"]]
    assert llm.calls[-1] == "report"
    roles = {x.spec.role for x in run.trace.state.tasks if x.output_json is not None}
    assert roles == set(Role)
    report = run.trace.model_calls[-1]
    assert not report.error
    for call in run.trace.model_calls:
        schema = InvestigationReport if call.task_id == "report" else (
            InitialPlan if call.task_id == "plan" else
            SCHEMAS[call.role]
        )
        exact_bound = (
            len(call.system_prompt.encode()) + len(call.user_prompt.encode())
            + len(json.dumps(structured_schema(schema)).encode()) + 1024 + 1500
        )
        assert call.reserved_tokens == exact_bound
    before_report = run.trace.state.usage.tokens - 10500
    assert before_report + report.reserved_tokens <= 120000
    paired = [(call, fixture[call.task_id][call.attempt - 1]) for call in run.trace.model_calls[:-1]]
    old_user = sum(saved["old_user_bytes"] for _, saved in paired)
    new_user = sum(len(call.user_prompt.encode()) for call, _ in paired)
    assert new_user < old_user * 0.6
    assert all(call.reserved_tokens < saved["old_reservation"] for call, saved in paired)
    print(json.dumps({
        "mode": "MOCKED REPLAY; not live reasoning or new measured provider usage",
        "paired_old_user_bytes": old_user,
        "paired_new_user_bytes": new_user,
        "reduction_percent": round((1 - new_user / old_user) * 100, 2),
        "report_user_bytes": len(report.user_prompt.encode()),
        "report_reservation": report.reserved_tokens,
        "tokens_charged_before_reporting_fixture": before_report,
        "token_ceiling": 120000,
        "calls": [
            {"task_id": call.task_id, "attempt": call.attempt,
             "user_bytes": len(call.user_prompt.encode()),
             "reservation": call.reserved_tokens}
            for call in run.trace.model_calls
        ],
    }))

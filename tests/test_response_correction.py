"""Bounded correction with scripted responses; no live reasoning or provider calls."""

import asyncio
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents import SCHEMAS, prompt
from baseline import run_baseline
from investigation import Coordinator, ModelOperation, run_multi_agent
from llm import LLMFailure, ModelResponse, ScriptedLLM, structured_schema
from models import ACCESS, ExecutionBudget, InitialPlan, Role, TaskSpec, ToolFamily
from tools import SimulatedTools

FIXTURES = Path(__file__).parent / "fixtures"


def replies(name="multi_agent"):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def invalid_plan():
    plan = replies()["plan"][0]
    plan["tasks"][-1]["role"] = "reasoning"
    return plan


def reasoning_reply():
    output = replies()["reasoning-0"][0]
    output["task_id"] = "review"
    output["hypotheses"][1]["confidence"] = 0.95
    output["hypotheses"][1]["blocking_uncertainties"] = ["Downstream processing has not been fully checked"]
    return output


def provisional_correction(rejected):
    corrected = copy.deepcopy(rejected)
    corrected["hypotheses"].sort(key=lambda item: item["confidence"], reverse=True)
    corrected["limitations"] = [
        "Corrected descending order without changing confidence or the unresolved causal gaps"
    ]
    return corrected


def prepare_reasoning(tools, catalog, responses):
    coordinator = Coordinator(tools, ScriptedLLM({"review": responses}), catalog.incidents[0].context)
    specs = InitialPlan.model_validate(replies()["plan"][0]).tasks
    asyncio.run(coordinator.evidence_tasks(specs))
    return coordinator


def test_plan_schema_and_runtime_permissions_share_one_specialist_matrix():
    schema = structured_schema(InitialPlan)
    assert set(schema["$defs"]["ObservabilityQuery"]["properties"]["family"]["enum"]) == {
        family.value for family in ACCESS[Role.OBSERVABILITY]
    }
    assert set(schema["$defs"]["DependencyQuery"]["properties"]["family"]["enum"]) == {
        family.value for family in ACCESS[Role.DEPENDENCY]
    }
    assert len(schema["properties"]["tasks"]["items"]["anyOf"]) == 2
    for role in Role:
        for family in ToolFamily:
            plan = replies()["plan"][0]
            plan["tasks"][0]["role"] = role.value
            plan["tasks"][0]["query"]["family"] = family.value
            if family in ACCESS.get(role, ()):
                assert InitialPlan.model_validate(plan).tasks[0].role == role
            else:
                with pytest.raises(ValidationError):
                    InitialPlan.model_validate(plan)


@pytest.mark.parametrize("wrong_role,index", [("reasoning", 4), ("observability", 2)])
def test_wrong_initial_assignment_is_corrected_before_any_tools_execute(catalog, wrong_role, index):
    responses = replies()
    valid = copy.deepcopy(responses["plan"][0])
    rejected = copy.deepcopy(valid)
    rejected["tasks"][index]["role"] = wrong_role
    responses["plan"] = [rejected, valid]
    llm = ScriptedLLM(responses)

    class GuardedTools(SimulatedTools):
        async def execute(self, query):
            assert llm.calls.count("plan") == 2
            return await super().execute(query)

    run = asyncio.run(run_multi_agent(GuardedTools(catalog), llm, catalog.incidents[0].context))
    assert run.report.outcome == "ranked"
    plan = next(task for task in run.trace.state.tasks if task.spec.task_id == "plan")
    assert plan.status == "succeeded"
    assert plan.attempts == 2
    assert run.trace.state.usage.llm_calls == 11
    calls = [call for call in run.trace.model_calls if call.task_id == "plan"]
    assert calls[0].correction_of_attempt is None
    assert calls[0].error
    assert calls[1].correction_of_attempt == 1
    assert not calls[1].error
    assert json.loads(calls[0].response_json) == rejected
    correction = json.loads(calls[1].user_prompt)["response_correction"]
    assert json.loads(correction["rejected_response"]) == rejected
    assert "owner is dependency" in correction["validation_errors"][0]
    assert correction["response_contract"] == structured_schema(InitialPlan)
    assert "reasoning" not in json.loads(plan.output_json)["tasks"][index]["role"]


def test_order_correction_retains_scores_and_real_gaps_and_only_commits_revalidated_output(tools, catalog):
    rejected = reasoning_reply()
    corrected = provisional_correction(rejected)
    coordinator = prepare_reasoning(tools, catalog, [rejected, corrected])
    llm = coordinator.llm
    respond = llm.respond

    async def observe(task_id, system, user, schema, maximum):
        payload = json.loads(user)
        assert coordinator.state.task(task_id).output_json is None
        assert not coordinator.state.snapshot().agent_results
        if "response_correction" in payload:
            assert json.loads(payload["response_correction"]["rejected_response"]) == rejected
            assert "hypotheses[1].confidence" in payload["response_correction"]["validation_errors"][0]
        return await respond(task_id, system, user, schema, maximum)

    llm.respond = observe
    output = asyncio.run(coordinator.model(ModelOperation("review", Role.REASONING, Role.REASONING)))
    assert output.hypotheses[0].confidence == 0.95
    assert output.hypotheses[0].blocking_uncertainties == tuple(
        rejected["hypotheses"][1]["blocking_uncertainties"]
    )
    assert coordinator.state.task("review").status == "succeeded"
    assert coordinator.state.task("review").attempts == 2
    assert len(coordinator.state.snapshot().agent_results) == 1
    assert coordinator.model_calls[1].correction_of_attempt == 1
    assert coordinator.failures[0].code == "invalid_response"
    assert not coordinator.failures[0].retryable
    first, second = [json.loads(call.user_prompt) for call in coordinator.model_calls]
    assert first["evidence"] == second["evidence"]
    assert first["tool_observations"] == second["tool_observations"]


def test_ordinary_uncertainty_can_be_explicitly_reclassified_without_deletion_or_confidence_change(tools, catalog):
    rejected = replies()["reasoning-1"][0]
    rejected["task_id"] = "review"
    hypothesis = rejected["hypotheses"][0]
    hypothesis.update({
        "causal_stage": "initiating_cause",
        "confidence": 0.95,
        "blocking_uncertainties": ["Other unobserved factors may have contributed"],
    })
    other = copy.deepcopy(replies()["reasoning-0"][0]["hypotheses"][1])
    other["confidence"] = 0.99
    rejected["hypotheses"].append(other)
    corrected = copy.deepcopy(rejected)
    corrected["hypotheses"][0].update({
        "blocking_uncertainties": [],
        "residual_uncertainties": hypothesis["blocking_uncertainties"],
    })
    corrected["limitations"] = [
        "There is no concrete competing-factor lead in the supplied observations; this possibility is residual, not a blocking gap"
    ]
    corrected["hypotheses"].sort(key=lambda item: item["confidence"], reverse=True)
    coordinator = prepare_reasoning(tools, catalog, [rejected, corrected])
    asyncio.run(coordinator.evidence_tasks((TaskSpec(
        task_id="verify-settings", role=Role.DEPENDENCY,
        query=replies()["critic-0"][0]["follow_ups"][0]["query"],
    ),)))
    output = asyncio.run(coordinator.model(ModelOperation("review", Role.REASONING, Role.REASONING)))
    candidate = next(item for item in output.hypotheses if item.hypothesis_id == hypothesis["hypothesis_id"])
    assert candidate.confidence == 0.95
    assert not candidate.blocking_uncertainties
    assert candidate.residual_uncertainties == tuple(hypothesis["blocking_uncertainties"])
    assert json.loads(coordinator.model_calls[0].response_json)["hypotheses"][0] == hypothesis


def test_baseline_uses_same_one_correction_policy_and_counts_it(tools, catalog):
    responses = replies("baseline")
    corrected = copy.deepcopy(responses["baseline-1"][0])
    corrected["rationale"] += "; the completed log query does not need repeating"
    rejected = copy.deepcopy(corrected)
    rejected["queries"].append(copy.deepcopy(responses["baseline-0"][0]["queries"][0]))
    rejected["rationale"] = "Accidentally proposed the completed log query again"
    responses["baseline-1"] = [rejected, corrected]
    llm = ScriptedLLM(responses)
    run = asyncio.run(run_baseline(tools, llm, catalog.incidents[0].context))
    assert run.report.outcome == "ranked"
    assert run.trace.state.usage.tool_calls == 5
    assert run.trace.state.usage.llm_calls == 4
    assert llm.calls == ["baseline-0", "baseline-1", "baseline-1", "baseline-2"]
    calls = [call for call in run.trace.model_calls if call.task_id == "baseline-1"]
    assert calls[1].correction_of_attempt == 1
    assert calls[0].error and not calls[1].error
    assert json.loads(calls[1].user_prompt)["response_correction"]["response_contract"] == structured_schema(
        SCHEMAS["baseline"]
    )


def test_second_invalid_response_terminates_without_third_request_or_partial_state(tools, catalog):
    llm = ScriptedLLM({"plan": [invalid_plan(), invalid_plan(), replies()["plan"][0]]})
    run = asyncio.run(run_multi_agent(tools, llm, catalog.incidents[0].context))
    assert llm.calls == ["plan", "plan"]
    assert run.report.outcome == "inconclusive"
    assert run.trace.termination_reason == "planning_unavailable"
    assert run.trace.state.usage.tool_calls == 0
    assert run.trace.state.usage.llm_calls == 2
    assert run.trace.state.tasks[0].status == "failed"
    assert run.trace.state.tasks[0].output_json is None
    assert len(run.trace.failures) == 2
    assert all(call.error for call in run.trace.model_calls)


@pytest.mark.parametrize("change", ["delete_gap", "change_claim", "drop_citation", "no_explanation"])
def test_correction_cannot_erase_uncertainty_change_diagnosis_or_drop_evidence(tools, catalog, change):
    rejected = reasoning_reply()
    corrected = provisional_correction(rejected)
    if change == "delete_gap":
        corrected["hypotheses"][0]["blocking_uncertainties"] = []
    elif change == "change_claim":
        corrected["hypotheses"][0]["claim"] = "A replacement unsupported diagnosis"
    elif change == "drop_citation":
        corrected["hypotheses"][1]["supporting_evidence_ids"] = []
    else:
        corrected["limitations"] = []
    coordinator = prepare_reasoning(tools, catalog, [rejected, corrected, provisional_correction(rejected)])
    output = asyncio.run(coordinator.model(ModelOperation("review", Role.REASONING, Role.REASONING)))
    assert output is None
    assert coordinator.llm.calls == ["review", "review"]
    assert coordinator.state.task("review").output_json is None
    assert not coordinator.state.snapshot().agent_results
    assert len(coordinator.failures) == 2


@pytest.mark.parametrize("budget", [
    ExecutionBudget(max_llm_calls=1), ExecutionBudget(operation_attempts=1),
])
def test_correction_obeys_request_and_attempt_caps(tools, catalog, budget):
    llm = ScriptedLLM({"plan": [invalid_plan(), replies()["plan"][0]]})
    run = asyncio.run(run_multi_agent(tools, llm, catalog.incidents[0].context, budget))
    assert llm.calls == ["plan"]
    assert run.report.outcome == "inconclusive"
    assert run.trace.state.usage.llm_calls == 1
    assert run.trace.state.usage.tool_calls == 0
    assert any(failure.code == "budget_exhausted" for failure in run.trace.failures)


def test_larger_correction_prompt_obeys_token_preflight(tools, catalog):
    context = catalog.incidents[0].context
    probe = Coordinator(tools, ScriptedLLM({}), context)
    probe.state.add_tasks(TaskSpec(task_id="plan", role=Role.ORCHESTRATOR))
    probe.state.start_task("plan")
    system, user = prompt("plan", "plan", context, probe.state.snapshot(), probe.state.budget, role=Role.ORCHESTRATOR)
    first_bound = len(system.encode()) + len(user.encode()) + len(json.dumps(structured_schema(InitialPlan)).encode()) + 1024 + 1500
    llm = ScriptedLLM({"plan": [invalid_plan(), replies()["plan"][0]]})
    run = asyncio.run(run_multi_agent(tools, llm, context, ExecutionBudget(max_tokens=first_bound + 100)))
    assert llm.calls == ["plan"]
    assert run.trace.state.usage.llm_calls == 1
    assert run.report.outcome == "inconclusive"
    assert any("token budget exhausted" in failure.message for failure in run.trace.failures)


def test_correction_cannot_start_after_deadline(monkeypatch, tools, catalog):
    clock = [0.0]
    monkeypatch.setattr("investigation.monotonic", lambda: clock[0])

    async def advance_between_requests(_):
        clock[0] = 2.0

    monkeypatch.setattr("investigation.asyncio.sleep", advance_between_requests)
    llm = ScriptedLLM({"plan": [invalid_plan(), replies()["plan"][0]]})
    run = asyncio.run(run_multi_agent(tools, llm, catalog.incidents[0].context, ExecutionBudget(max_elapsed_seconds=1)))
    assert llm.calls == ["plan"]
    assert run.trace.failures[0].code == "invalid_response"
    assert any("time budget exhausted" in failure.message for failure in run.trace.failures)
    assert run.report.outcome == "inconclusive"


def test_initial_transient_attempts_and_correction_share_the_same_total_limit(tools, catalog):
    responses = replies()
    responses["plan"] = [
        LLMFailure("http_429", "Transient rate limit", retryable=True),
        invalid_plan(), responses["plan"][0],
    ]
    llm = ScriptedLLM(responses)
    run = asyncio.run(run_multi_agent(tools, llm, catalog.incidents[0].context))
    plan = next(task for task in run.trace.state.tasks if task.spec.task_id == "plan")
    assert plan.attempts == 3
    assert run.report.outcome == "ranked"
    assert next(call for call in run.trace.model_calls if call.correction_of_attempt).correction_of_attempt == 2
    assert run.trace.state.usage.unmeasured_token_reservations > 0


def test_transient_failure_of_correction_is_not_a_second_correction_request(tools, catalog):
    llm = ScriptedLLM({"plan": [
        invalid_plan(), LLMFailure("http_429", "Correction request rate limited", retryable=True),
        replies()["plan"][0],
    ]})
    run = asyncio.run(run_multi_agent(tools, llm, catalog.incidents[0].context))
    assert llm.calls == ["plan", "plan"]
    assert run.report.outcome == "inconclusive"
    assert run.trace.model_calls[1].correction_of_attempt == 1
    assert run.trace.failures[-1].code == "http_429"


@pytest.mark.parametrize("code", ["authentication", "refusal", "incomplete_response", "budget_exhausted"])
def test_non_correctable_provider_errors_keep_their_failure_paths(tools, catalog, code):
    llm = ScriptedLLM({"plan": [
        LLMFailure(code, "Injected provider failure"), replies()["plan"][0],
    ]})
    run = asyncio.run(run_multi_agent(tools, llm, catalog.incidents[0].context))
    assert llm.calls == ["plan"]
    assert run.trace.failures[0].code == code
    assert run.trace.state.usage.tool_calls == 0
    assert run.report.outcome == "inconclusive"


def test_unsafe_query_is_not_corrected_even_when_ownership_is_wrong(tools, catalog):
    rejected = invalid_plan()
    rejected["tasks"][0]["query"]["incident_id"] = "inc-002"
    llm = ScriptedLLM({"plan": [rejected, replies()["plan"][0]]})
    run = asyncio.run(run_multi_agent(tools, llm, catalog.incidents[0].context))
    assert llm.calls == ["plan"]
    assert run.trace.state.usage.tool_calls == 0
    assert run.report.outcome == "inconclusive"


def test_unknown_evidence_is_not_corrected_even_when_confidence_is_inconsistent(tools, catalog):
    rejected = reasoning_reply()
    rejected["hypotheses"][0]["supporting_evidence_ids"] = ["unavailable"]
    coordinator = prepare_reasoning(tools, catalog, [rejected, provisional_correction(rejected)])
    assert asyncio.run(coordinator.model(ModelOperation("review", Role.REASONING, Role.REASONING))) is None
    assert coordinator.llm.calls == ["review"]
    assert "unknown evidence" in coordinator.failures[0].message


def test_failed_correction_preserves_previously_accepted_baseline_hypotheses(tools, catalog):
    responses = replies("baseline")
    rejected = copy.deepcopy(responses["baseline-2"][0])
    for candidate in (rejected["hypotheses"][0], rejected["report"]["hypotheses"][0]):
        candidate["blocking_uncertainties"] = ["An essential observation remains unavailable"]
    unsafe = copy.deepcopy(rejected)
    for candidate in (unsafe["hypotheses"][0], unsafe["report"]["hypotheses"][0]):
        candidate["supporting_evidence_ids"].append("unavailable")
    responses["baseline-2"] = [rejected, unsafe, responses["baseline-2"][0]]
    llm = ScriptedLLM(responses)
    run = asyncio.run(run_baseline(tools, llm, catalog.incidents[0].context))
    assert llm.calls.count("baseline-2") == 2
    assert run.report.outcome == "inconclusive"
    assert run.report.hypotheses[0].confidence == 0.45
    prior = next(task for task in run.trace.state.tasks if task.spec.task_id == "baseline-1")
    failed = next(task for task in run.trace.state.tasks if task.spec.task_id == "baseline-2")
    assert prior.output_json
    assert failed.output_json is None
    assert run.trace.state.usage.tool_calls == 5


def test_provider_reported_usage_includes_rejected_and_corrected_calls(tools, catalog):
    responses = replies()
    responses["plan"] = [invalid_plan(), responses["plan"][0]]

    class MeasuredLLM(ScriptedLLM):
        async def respond(self, *args):
            response = await super().respond(*args)
            return ModelResponse(response.response_json, input_tokens=19, output_tokens=7)

    run = asyncio.run(run_multi_agent(tools, MeasuredLLM(responses), catalog.incidents[0].context))
    assert run.report.outcome == "ranked"
    assert run.trace.state.usage.llm_calls == 11
    assert run.trace.state.usage.input_tokens == 11 * 19
    assert run.trace.state.usage.output_tokens == 11 * 7
    assert run.trace.state.usage.tokens == 11 * (19 + 7)
    assert all(call.usage_measured for call in run.trace.model_calls)


@pytest.mark.parametrize("resolution", ["retain_blocking", "reclassify_residual"])
def test_baseline_final_correction_preserves_uncertainty_and_matches_outcome(tools, catalog, resolution):
    responses = replies("baseline")
    rejected = copy.deepcopy(responses["baseline-2"][0])
    question = (
        "A concrete workload change has not been investigated"
        if resolution == "retain_blocking" else "Other unobserved factors may have contributed"
    )
    rejected["hypotheses"][0]["blocking_uncertainties"] = [question]
    rejected["report"]["hypotheses"] = copy.deepcopy(rejected["hypotheses"])
    corrected = copy.deepcopy(rejected)
    if resolution == "retain_blocking":
        corrected["hypotheses"][0]["confidence"] = 0.55
        corrected["report"]["outcome"] = "inconclusive"
        corrected["rationale"] = "A workload gap remains blocking; the initiating explanation is provisional, not established"
    else:
        corrected["hypotheses"][0]["blocking_uncertainties"] = []
        corrected["hypotheses"][0]["residual_uncertainties"] = [question]
        corrected["rationale"] = "The original evidence supplies no concrete concurrent-factor lead; this generic possibility is residual"
    corrected["report"]["hypotheses"] = copy.deepcopy(corrected["hypotheses"])
    responses["baseline-2"] = [rejected, corrected]
    llm = ScriptedLLM(responses)
    run = asyncio.run(run_baseline(tools, llm, catalog.incidents[0].context))
    candidate = run.report.hypotheses[0]
    assert run.report.outcome == ("inconclusive" if resolution == "retain_blocking" else "ranked")
    assert candidate.claim == rejected["hypotheses"][0]["claim"]
    assert question in (*candidate.blocking_uncertainties, *candidate.residual_uncertainties)
    assert candidate.confidence == (0.55 if resolution == "retain_blocking" else 0.88)
    assert llm.calls == ["baseline-0", "baseline-1", "baseline-2", "baseline-2"]
    assert next(task for task in run.trace.state.tasks if task.spec.task_id == "baseline-2").status == "succeeded"


def test_correction_that_only_lowers_confidence_but_keeps_invalid_ranked_outcome_is_rejected(tools, catalog):
    responses = replies("baseline")
    rejected = copy.deepcopy(responses["baseline-2"][0])
    rejected["hypotheses"][0]["blocking_uncertainties"] = ["A blocking observation is missing"]
    rejected["report"]["hypotheses"] = copy.deepcopy(rejected["hypotheses"])
    corrected = copy.deepcopy(rejected)
    corrected["hypotheses"][0]["confidence"] = 0.55
    corrected["report"]["hypotheses"] = copy.deepcopy(corrected["hypotheses"])
    corrected["rationale"] = "Confidence is provisional but the reported outcome was not corrected"
    responses["baseline-2"] = [rejected, corrected, responses["baseline-2"][0]]
    llm = ScriptedLLM(responses)
    run = asyncio.run(run_baseline(tools, llm, catalog.incidents[0].context))
    assert run.report.outcome == "inconclusive"
    assert llm.calls.count("baseline-2") == 2
    assert "report.hypotheses[0].blocking_uncertainties" in run.trace.failures[-1].message
    assert run.report.hypotheses[0].confidence == 0.45


def test_no_correction_dispatch_after_all_three_attempts_are_used(tools, catalog):
    llm = ScriptedLLM({"plan": [
        LLMFailure("http_429", "Transient", retryable=True),
        LLMFailure("http_429", "Transient", retryable=True),
        invalid_plan(), replies()["plan"][0],
    ]})
    run = asyncio.run(run_multi_agent(tools, llm, catalog.incidents[0].context))
    assert llm.calls == ["plan", "plan", "plan"]
    assert run.trace.state.usage.llm_calls == 3
    assert any("attempt limit exhausted" in failure.message for failure in run.trace.failures)
    assert not any(call.correction_of_attempt for call in run.trace.model_calls)
    assert run.report.outcome == "inconclusive"


def test_plan_correction_cannot_replace_the_rejected_queries(tools, catalog):
    corrected = replies()["plan"][0]
    corrected["tasks"][0]["query"]["services"] = ["checkout"]
    llm = ScriptedLLM({"plan": [invalid_plan(), corrected, replies()["plan"][0]]})
    run = asyncio.run(run_multi_agent(tools, llm, catalog.incidents[0].context))
    assert llm.calls == ["plan", "plan"]
    assert run.trace.state.usage.tool_calls == 0
    assert run.report.outcome == "inconclusive"
    assert "not replace tasks or queries" in run.trace.failures[-1].message


def test_unsafe_baseline_query_takes_precedence_over_correctable_confidence(tools, catalog):
    responses = replies("baseline")
    rejected = copy.deepcopy(responses["baseline-1"][0])
    rejected["hypotheses"][0]["confidence"] = 0.95
    rejected["queries"][0]["incident_id"] = "inc-002"
    responses["baseline-1"] = [rejected, responses["baseline-1"][0]]
    llm = ScriptedLLM(responses)
    run = asyncio.run(run_baseline(tools, llm, catalog.incidents[0].context))
    assert llm.calls == ["baseline-0", "baseline-1"]
    assert run.trace.state.usage.tool_calls == 2
    assert "cross-incident queries" in run.trace.failures[-1].message
    assert not any(call.correction_of_attempt for call in run.trace.model_calls)

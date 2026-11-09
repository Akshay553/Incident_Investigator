import asyncio
import copy
import json
from pathlib import Path

import pytest

from baseline import run_baseline
from investigation import Coordinator, ModelOperation, run_multi_agent
from llm import LLMFailure, ModelResponse, ScriptedLLM, structured_schema
from models import (
    AgentResult, ExecutionBudget, ExecutionTrace, InitialPlan, InvestigationReport,
    FailureRule, Role, SimulationSettings, TaskSpec, TimeRange, ToolFamily,
)
from tools import SimulatedTools

FIXTURES = Path(__file__).parent / "fixtures"


def script(name="multi_agent"):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def multi(tools, catalog, responses=None, budget=None):
    llm = ScriptedLLM(responses or script())
    run = asyncio.run(run_multi_agent(tools, llm, catalog.incidents[0].context, budget))
    return run, llm


def test_critic_feedback_causes_new_query_and_hypothesis_revision(tools, catalog):
    run, llm = multi(tools, catalog)
    assert run.report.outcome == "ranked"
    assert run.trace.execution_mode == "mocked"
    assert run.trace.termination_reason == "orchestrator_finished"
    assert run.trace.state.usage.tool_calls == 6
    assert run.trace.state.usage.llm_calls == 10
    assert run.trace.state.usage.tokens == 0
    assert not run.trace.failures
    assert {task.spec.role for task in run.trace.state.tasks} == set(Role)
    tasks = {task.spec.task_id: task for task in run.trace.state.tasks}
    initial = tasks["initial-changes"].spec.query
    followup = tasks["followup-1-0"].spec.query
    assert initial.time_range != followup.time_range
    assert tasks["followup-1-0"].spec.depends_on == ("decision-0", "reasoning-0")
    outputs = {result.task_id: result for result in run.trace.state.agent_results}
    assert outputs["reasoning-0"].hypotheses[0].confidence == 0.45
    assert outputs["reasoning-1"].hypotheses[0].confidence == 0.88
    assert "inc-001:changes:r01" not in outputs["reasoning-0"].hypotheses[0].supporting_evidence_ids
    assert "inc-001:changes:r01" in run.report.hypotheses[0].supporting_evidence_ids
    assert llm.calls.index("critic-0") < llm.calls.index("dependency-1") < llm.calls.index("reasoning-1")
    assert ExecutionTrace.model_validate_json(run.trace.model_dump_json()) == run.trace


def test_initial_evidence_and_specialist_models_really_run_concurrently(catalog):
    class BarrierTools(SimulatedTools):
        def __init__(self):
            super().__init__(catalog, SimulationSettings(latency_ms=0))
            self.initial_entered = 0
            self.barrier = asyncio.Event()

        async def execute(self, query):
            if self.initial_entered < 5:
                self.initial_entered += 1
                if self.initial_entered == 5:
                    self.barrier.set()
                await asyncio.wait_for(self.barrier.wait(), 1)
            return await super().execute(query)

    class BarrierLLM(ScriptedLLM):
        def __init__(self):
            super().__init__(script())
            self.summaries_entered = 0
            self.barrier = asyncio.Event()

        async def respond(self, task_id, *args):
            if task_id in {"observability-0", "dependency-0"}:
                self.summaries_entered += 1
                if self.summaries_entered == 2:
                    self.barrier.set()
                await asyncio.wait_for(self.barrier.wait(), 1)
            return await super().respond(task_id, *args)

    tools, llm = BarrierTools(), BarrierLLM()
    run = asyncio.run(run_multi_agent(tools, llm, catalog.incidents[0].context))
    assert tools.initial_entered == 5
    assert llm.summaries_entered == 2
    assert run.report.outcome == "ranked"


def test_no_fixed_followup_when_critic_requests_none(tools, catalog):
    responses = script()
    responses["plan"][0]["tasks"][2]["query"] = copy.deepcopy(
        responses["critic-0"][0]["follow_ups"][0]["query"]
    )
    responses["dependency-0"][0]["limitations"] = []
    responses["reasoning-0"][0]["hypotheses"] = copy.deepcopy(
        responses["reasoning-1"][0]["hypotheses"]
    )
    responses["critic-0"][0].update({"objections": [], "follow_ups": []})
    responses["decision-0"] = [{"action": "finish", "rationale": "No discriminating query requested"}]
    hypotheses = responses["reasoning-0"][0]["hypotheses"][:1]
    responses["report"][0]["hypotheses"] = hypotheses
    run, llm = multi(tools, catalog, responses)
    assert run.trace.state.usage.tool_calls == 5
    assert "dependency-1" not in llm.calls
    assert "reasoning-1" not in llm.calls
    assert run.report.outcome == "ranked"


@pytest.mark.parametrize("change,message", [
    ("duplicate", "duplicate"),
    ("wrong_role", "not permitted"),
    ("wrong_incident", "cross-incident"),
])
def test_invalid_followups_do_not_reach_tools(tools, catalog, change, message):
    responses = script()
    request = responses["critic-0"][0]["follow_ups"][0]
    if change == "duplicate":
        request["query"] = copy.deepcopy(responses["plan"][0]["tasks"][2]["query"])
    elif change == "wrong_role":
        request["target_role"] = "observability"
    else:
        request["query"]["incident_id"] = "inc-002"
    if change == "duplicate":
        responses["critic-0"].append(copy.deepcopy(responses["critic-0"][0]))
    responses["report"][0]["hypotheses"] = responses["reasoning-0"][0]["hypotheses"][:1]
    responses["report"][0]["outcome"] = "inconclusive"
    run, llm = multi(tools, catalog, responses)
    assert run.report.outcome == "inconclusive"
    assert run.trace.termination_reason == "critic_unavailable"
    assert run.trace.state.usage.tool_calls == 5
    assert any(
        message in failure.message
        for failure in run.trace.failures
    )
    assert "dependency-1" not in llm.calls


@pytest.mark.parametrize("change", ["evidence", "claim", "confidence"])
def test_reporting_cannot_add_evidence_or_change_reasoning(tools, catalog, change):
    responses = script()
    hypothesis = responses["report"][0]["hypotheses"][0]
    if change == "evidence":
        hypothesis["supporting_evidence_ids"].append("invented")
    elif change == "claim":
        hypothesis["claim"] = "An unsupported replacement diagnosis"
    else:
        hypothesis["confidence"] = 0.99
    run, _ = multi(tools, catalog, responses)
    assert run.report.outcome == "inconclusive"
    assert run.trace.termination_reason == "reporting_unavailable"
    assert run.trace.failures[-1].code == "invalid_response"
    assert any("runtime fallback" in item for item in run.report.limitations)


def test_critical_objection_cannot_be_ignored_as_ordinary_success(tools, catalog):
    responses = script()
    responses["decision-0"] = [{
        "action": "finish", "rationale": "Stop despite the unresolved objection",
        "deferred_follow_ups": [{
            "follow_up_index": 0,
            "reason": "Investigator chose to stop; the settings query is unperformed and uncertainty unresolved",
        }],
    }]
    responses["report"][0]["hypotheses"] = responses["reasoning-0"][0]["hypotheses"][:1]
    responses["report"][0]["outcome"] = "inconclusive"
    run, _ = multi(tools, catalog, responses)
    assert run.report.outcome == "inconclusive"
    assert any("inspect settings" in limitation for limitation in run.report.limitations)
    assert run.trace.termination_reason == "orchestrator_finished"


@pytest.mark.parametrize("task_id", ["plan", "reasoning-0", "critic-0", "report"])
def test_permanent_model_failure_is_not_retried_and_produces_limitation(tools, catalog, task_id):
    responses = script()
    responses[task_id] = [LLMFailure("authentication", "Provider authorization unavailable")]
    run, llm = multi(tools, catalog, responses)
    assert llm.calls.count(task_id) == 1
    assert run.report.outcome == "inconclusive"
    assert any("unavailable" in text.lower() for text in run.report.limitations)
    assert any(failure.code == "authentication" for failure in run.trace.failures)


@pytest.mark.parametrize("failures,expected_status,expected_attempts", [
    (1, "ranked", 2), (2, "ranked", 3), (3, "inconclusive", 3),
])
def test_transient_llm_retries_have_three_total_attempts(tools, catalog, failures, expected_status, expected_attempts):
    responses = script()
    responses["plan"] = [
        *(LLMFailure("http_429", "Transient rate limit", retryable=True) for _ in range(failures)),
        *responses["plan"],
    ]
    run, llm = multi(tools, catalog, responses)
    assert llm.calls.count("plan") == expected_attempts
    assert run.report.outcome == expected_status
    assert len([failure for failure in run.trace.failures if failure.task_id == "plan"]) == failures
    assert run.trace.state.usage.unmeasured_token_reservations > 0


def test_invalid_json_is_recorded_not_parsed_as_prose_or_retried(tools, catalog):
    responses = script()
    responses["plan"] = ["Here is my plan: inspect some logs.", *responses["plan"]]
    run, llm = multi(tools, catalog, responses)
    assert run.report.outcome == "inconclusive"
    assert llm.calls == ["plan"]
    assert run.trace.model_calls[0].response_json.startswith("Here is")
    assert run.trace.failures[0].code == "invalid_response"


def test_initial_plan_access_and_dependency_checks_are_runtime_enforced(tools, catalog):
    responses = script()
    responses["plan"][0]["tasks"][0]["depends_on"] = ["initial-metrics"]
    run, llm = multi(tools, catalog, responses)
    assert llm.calls == ["plan"]
    assert run.trace.state.usage.tool_calls == 0
    assert run.trace.failures[0].code == "invalid_response"


def test_followup_round_budget_stops_unresolved_analysis(tools, catalog):
    responses = script()
    responses["report"][0]["hypotheses"] = responses["reasoning-0"][0]["hypotheses"][:1]
    responses["report"][0]["outcome"] = "inconclusive"
    run, llm = multi(tools, catalog, responses, budget=ExecutionBudget(max_follow_up_rounds=0))
    assert run.report.outcome == "inconclusive"
    assert run.trace.termination_reason == "follow_up_round_limit"
    assert "decision-0" not in llm.calls
    assert run.trace.state.usage.tool_calls == 5


def test_full_two_round_limit_is_enforced(tools, catalog):
    responses = script()
    responses["critic-1"] = [copy.deepcopy(responses["critic-0"][0])]
    responses["critic-1"][0]["task_id"] = "critic-1"
    responses["critic-1"][0]["follow_ups"][0]["query"]["family"] = "metrics"
    responses["critic-1"][0]["follow_ups"][0]["target_role"] = "observability"
    responses["critic-1"][0]["follow_ups"][0]["depends_on"] = ["reasoning-1"]
    responses["decision-1"] = copy.deepcopy(responses["decision-0"])
    responses["observability-2"] = [{
        "task_id": "observability-2", "role": "observability",
        "limitations": ["Additional metric scope does not settle the remaining causal assumption"],
    }]
    responses["reasoning-2"] = copy.deepcopy(responses["reasoning-1"])
    responses["reasoning-2"][0]["task_id"] = "reasoning-2"
    responses["critic-2"] = [{
        "task_id": "critic-2", "role": "critic",
        "objections": [{
            "hypothesis_id": "h1", "reason": "A critical causal assumption remains", "critical": True,
            "unavailable_reason": "No remaining observation or tool query can test the causal assumption",
        }],
    }]
    run, llm = multi(tools, catalog, responses)
    assert run.trace.state.usage.tool_calls == 7
    assert run.trace.termination_reason == "orchestrator_finished"
    assert run.report.outcome == "inconclusive"
    assert "reasoning-2" in llm.calls
    assert "decision-2" not in llm.calls


def test_equal_total_ceilings_and_iterative_baseline_revision(tools, catalog):
    budget = ExecutionBudget()
    multi_run, _ = multi(tools, catalog, budget=budget)
    llm = ScriptedLLM(script("baseline"))
    baseline = asyncio.run(run_baseline(tools, llm, catalog.incidents[0].context, budget))
    assert baseline.trace.budget == multi_run.trace.budget == budget
    assert baseline.trace.execution_mode == "mocked"
    expected = InvestigationReport.model_validate(script("baseline")["baseline-2"][0]["report"])
    assert baseline.report.hypotheses == expected.hypotheses
    assert baseline.report.recommended_actions == expected.recommended_actions
    assert set(expected.limitations) <= set(baseline.report.limitations)
    assert baseline.trace.state.usage.tool_calls == 5
    assert baseline.trace.state.usage.llm_calls == 3
    assert llm.calls == ["baseline-0", "baseline-1", "baseline-2"]
    assert all(task.spec.max_attempts == 3 for task in baseline.trace.state.tasks)
    assert baseline.trace.state.usage.tokens == 0
    assert "Client connection capacity" in baseline.trace.state.tasks[3].output_json


def test_baseline_tools_execute_sequentially(catalog):
    class TrackingTools(SimulatedTools):
        def __init__(self):
            super().__init__(catalog, SimulationSettings(latency_ms=1))
            self.active = self.peak = 0

        async def execute(self, query):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                return await super().execute(query)
            finally:
                self.active -= 1

    tools = TrackingTools()
    run = asyncio.run(run_baseline(tools, ScriptedLLM(script("baseline")), catalog.incidents[0].context))
    assert tools.peak == 1
    assert run.report.outcome == "ranked"


def test_baseline_duplicate_query_is_rejected_without_spending_tool_calls(tools, catalog):
    responses = script("baseline")
    responses["baseline-1"][0]["queries"] = copy.deepcopy(responses["baseline-0"][0]["queries"])
    responses["baseline-1"].append(copy.deepcopy(responses["baseline-1"][0]))
    llm = ScriptedLLM(responses)
    run = asyncio.run(run_baseline(tools, llm, catalog.incidents[0].context))
    assert run.report.outcome == "inconclusive"
    assert run.trace.state.usage.tool_calls == 2
    assert run.trace.failures[-1].code == "invalid_response"
    assert "duplicate" in run.trace.failures[-1].message


@pytest.mark.parametrize("approach", ["multi-agent", "baseline"])
def test_total_model_call_and_token_budgets_are_enforced_before_dispatch(tools, catalog, approach):
    runner = run_multi_agent if approach == "multi-agent" else run_baseline
    responses = script() if approach == "multi-agent" else script("baseline")
    llm = ScriptedLLM(responses)
    run = asyncio.run(runner(tools, llm, catalog.incidents[0].context, ExecutionBudget(max_tokens=1)))
    assert llm.calls == []
    assert run.report.outcome == "inconclusive"
    assert run.trace.state.usage.llm_calls == 0
    assert run.trace.failures[0].code == "budget_exhausted"
    llm = ScriptedLLM(responses)
    run = asyncio.run(runner(tools, llm, catalog.incidents[0].context, ExecutionBudget(max_llm_calls=1)))
    assert len(llm.calls) == 1
    assert run.report.outcome == "inconclusive"
    assert run.trace.state.usage.llm_calls == 1


def test_model_usage_is_measured_even_when_schema_validation_fails(tools, catalog):
    class MeasuredLLM(ScriptedLLM):
        async def respond(self, *args):
            return ModelResponse("invalid", input_tokens=20, output_tokens=3)

    run = asyncio.run(run_multi_agent(tools, MeasuredLLM({}), catalog.incidents[0].context))
    assert run.trace.state.usage.tokens == 23
    assert run.trace.state.usage.input_tokens == 20
    assert run.trace.state.usage.output_tokens == 3
    assert run.trace.model_calls[0].usage_measured
    assert run.report.outcome == "inconclusive"


def test_incomplete_response_is_an_error_with_retained_usage(tools, catalog):
    class IncompleteLLM(ScriptedLLM):
        async def respond(self, *args):
            return ModelResponse(
                json.dumps(script()["plan"][0]), 20, 30,
                failure=LLMFailure("incomplete_response", "Output was truncated"),
            )

    run = asyncio.run(run_multi_agent(tools, IncompleteLLM({}), catalog.incidents[0].context))
    assert run.trace.state.usage.tokens == 50
    assert run.trace.failures[0].code == "incomplete_response"
    assert run.trace.state.usage.tool_calls == 0


def test_schema_required_fields_and_no_arbitrary_properties():
    for contract in (InitialPlan, AgentResult, InvestigationReport):
        schema = structured_schema(contract)

        def walk(value):
            if isinstance(value, dict):
                assert "default" not in value
                if value.get("type") == "object":
                    assert value["additionalProperties"] is False
                    assert set(value["required"]) == set(value["properties"])
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(schema)


def test_provider_requires_local_key_not_credentials_in_chat(monkeypatch):
    from llm import OpenAILLM

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="configure it locally"):
        OpenAILLM()


@pytest.mark.parametrize("failures,retryable,attempts", [(1, True, 2), (2, True, 3), (None, False, 1)])
def test_agent_tool_failure_recovery_and_terminal_limitations(catalog, failures, retryable, attempts):
    tools = SimulatedTools(catalog, SimulationSettings(
        latency_ms=0,
        failures=(FailureRule(
            incident_id="inc-001", family=ToolFamily.LOGS,
            failures=failures, retryable=retryable,
        ),),
    ))
    run, _ = multi(tools, catalog)
    task = next(task for task in run.trace.state.tasks if task.spec.task_id == "initial-logs")
    assert task.attempts == attempts
    assert run.trace.state.tool_results[task.result_indices[0]].status == "error"
    assert any(failure.operation == "tool" for failure in run.trace.failures)
    if retryable:
        assert run.report.outcome == "ranked"
        assert run.trace.state.tool_results[task.result_indices[-1]].status == "complete"
    else:
        assert run.report.outcome == "inconclusive"
        assert any("initial-logs" in text for text in run.report.limitations)


def test_model_wall_clock_deadline_is_bounded_and_recorded(tools, catalog):
    class SlowLLM(ScriptedLLM):
        async def respond(self, *args):
            await asyncio.sleep(1)
            return await super().respond(*args)

    run = asyncio.run(run_multi_agent(
        tools, SlowLLM(script()), catalog.incidents[0].context,
        ExecutionBudget(max_elapsed_seconds=0.05),
    ))
    assert run.report.outcome == "inconclusive"
    assert run.trace.state.usage.llm_calls == 1
    assert run.trace.state.usage.elapsed_seconds < 0.5
    assert run.trace.failures[0].code == "timeout"
    assert run.trace.state.usage.unmeasured_token_reservations > 0


def test_unknown_citations_and_wrong_roles_are_rejected(tools, catalog):
    for mutation in ("citation", "role"):
        responses = script()
        if mutation == "citation":
            responses["reasoning-0"][0]["hypotheses"][0]["supporting_evidence_ids"] = ["made-up"]
        else:
            responses["reasoning-0"][0]["role"] = "critic"
        responses["report"] = [{
            "incident_id": "inc-001", "outcome": "inconclusive",
            "limitations": ["Reasoning unavailable"],
        }]
        run, llm = multi(tools, catalog, responses)
        assert run.report.outcome == "inconclusive"
        assert run.trace.failures[0].code == "invalid_response"
        assert llm.calls.count("reasoning-0") == 1
        assert "critic-0" not in llm.calls


def test_semantically_identical_timezone_and_service_order_queries_are_duplicates(tools, catalog):
    coordinator = Coordinator(tools, ScriptedLLM({}), catalog.incidents[0].context)
    query = catalog.incidents[0].context
    from models import ToolQuery

    first = ToolQuery(
        incident_id=query.incident_id, family=ToolFamily.LOGS,
        time_range=query.time_range, services=("checkout", "orders-db"),
    )
    equivalent = first.model_copy(update={
        "services": ("orders-db", "checkout"),
        "time_range": TimeRange(
            start="2026-01-15T15:30:00+05:30", end="2026-01-15T16:00:00+05:30",
        ),
    })
    asyncio.run(coordinator.evidence_tasks((TaskSpec(
        task_id="once", role=Role.OBSERVABILITY, query=first,
    ),)))
    with pytest.raises(ValueError, match="duplicate"):
        coordinator.validate_query(equivalent, Role.OBSERVABILITY)


def test_failed_query_retry_flag_cannot_reset_three_attempt_ceiling(catalog):
    tools = SimulatedTools(catalog, SimulationSettings(
        latency_ms=0,
        failures=(FailureRule(incident_id="inc-001", family=ToolFamily.LOGS, failures=None),),
    ))
    coordinator = Coordinator(tools, ScriptedLLM({}), catalog.incidents[0].context)
    from agents import suggested_queries

    query = suggested_queries(catalog.incidents[0].context)[0]
    asyncio.run(coordinator.evidence_tasks((TaskSpec(
        task_id="failed", role=Role.OBSERVABILITY, query=query,
    ),)))
    assert coordinator.state.task("failed").attempts == 3
    with pytest.raises(ValueError, match="total attempt"):
        coordinator.validate_query(query, Role.OBSERVABILITY, retry=True)


def test_paid_adapter_wiring_uses_strict_schema_shared_client_and_no_sdk_retries(monkeypatch):
    from types import SimpleNamespace
    from llm import OpenAILLM
    import openai

    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs == {"max_retries": 0}
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                status="completed", output_text=json.dumps(script()["plan"][0]),
                usage=SimpleNamespace(input_tokens=120, output_tokens=50),
            )

        async def close(self):
            pass

    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key-for-unit-testing")
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)
    adapter = OpenAILLM()

    async def run():
        response = await adapter.respond("plan", "system", "user", InitialPlan, 1500)
        await adapter.close()
        return response

    response = asyncio.run(run())
    assert response.input_tokens == 120
    assert InitialPlan.model_validate_json(response.response_json)
    assert calls[0]["model"] == "gpt-4.1-mini"
    assert calls[0]["max_output_tokens"] == 1500
    assert calls[0]["store"] is False
    assert calls[0]["text"]["format"]["strict"] is True
    assert "tools" not in calls[0]


@pytest.mark.parametrize("status,retryable", [(401, False), (400, False), (429, True), (503, True)])
def test_paid_adapter_classifies_http_errors_without_network(monkeypatch, status, retryable):
    from types import SimpleNamespace
    from llm import OpenAILLM
    import httpx
    import openai

    class FakeClient:
        def __init__(self, **kwargs):
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **kwargs):
            raise openai.APIStatusError(
                "Synthetic API failure",
                response=httpx.Response(status, request=httpx.Request("POST", "https://example.invalid")),
                body=None,
            )

    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key-for-unit-testing")
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)
    adapter = OpenAILLM()
    with pytest.raises(LLMFailure) as failure:
        asyncio.run(adapter.respond("plan", "system", "user", InitialPlan, 1500))
    assert failure.value.retryable is retryable
    assert failure.value.code == f"http_{status}"


@pytest.mark.parametrize("runner", [run_multi_agent, run_baseline])
def test_cancellation_saves_a_trace_and_conservatively_charges_inflight_model_usage(tools, catalog, runner):
    class BlockingLLM(ScriptedLLM):
        def __init__(self):
            super().__init__({})
            self.started = asyncio.Event()

        async def respond(self, *args):
            self.started.set()
            await asyncio.sleep(10)

    llm = BlockingLLM()

    async def run():
        task = asyncio.create_task(runner(tools, llm, catalog.incidents[0].context))
        await llm.started.wait()
        task.cancel()
        return await task

    result = asyncio.run(run())
    assert result.report.outcome == "inconclusive"
    assert result.trace.termination_reason == "cancelled"
    assert result.trace.failures[-1].code == "cancelled"
    assert result.trace.state.usage.llm_calls == 1
    assert result.trace.state.usage.tokens > 0
    assert result.trace.state.usage.tokens == result.trace.state.usage.unmeasured_token_reservations
    assert len(result.trace.model_calls) == 1
    assert result.trace.model_calls[0].error == "LLM task cancelled"
    assert not result.trace.model_calls[0].usage_measured
    assert any("Mocked execution" in text for text in result.report.limitations)


def test_cancellation_records_all_running_tool_failures(catalog):
    class BlockingTools(SimulatedTools):
        def __init__(self):
            super().__init__(catalog, SimulationSettings(latency_ms=10_000))
            self.started = asyncio.Event()

        async def execute(self, query):
            self.started.set()
            return await super().execute(query)

    tools = BlockingTools()

    async def run():
        task = asyncio.create_task(run_multi_agent(
            tools, ScriptedLLM(script()), catalog.incidents[0].context,
        ))
        await tools.started.wait()
        task.cancel()
        return await task

    result = asyncio.run(run())
    assert result.report.outcome == "inconclusive"
    assert result.trace.termination_reason == "cancelled"
    assert result.trace.state.usage.tool_calls == 5
    assert len([failure for failure in result.trace.failures if failure.operation == "tool"]) == 5
    assert all(item.error.code == "cancelled" for item in result.trace.state.tool_results)

import asyncio

import pytest

from cli import collect
from investigation import BudgetExceeded, DependencyNotReady, InvestigationState, run_tool_task, run_tool_tasks
from models import (
    AgentResult, ExecutionBudget, FailureRule, Finding, FollowUpRequest, Role,
    SimulationSettings, TaskSpec, ToolFamily, ToolQuery,
)
from tools import SimulatedTools


def tool_spec(query, task_id="observe", depends_on=(), max_attempts=2):
    return TaskSpec(
        task_id=task_id, role=Role.OBSERVABILITY, query=query,
        depends_on=depends_on, max_attempts=max_attempts,
    )


def test_dependencies_release_only_after_success_and_support_structured_followups(tools, query):
    state = InvestigationState(query.incident_id)
    state.add_tasks(
        tool_spec(query),
        TaskSpec(task_id="reason", role=Role.REASONING, depends_on=("observe",)),
    )
    assert [spec.task_id for spec in state.ready_tasks()] == ["observe"]
    with pytest.raises(DependencyNotReady):
        state.start_task("reason")
    asyncio.run(run_tool_task(state, "observe", tools))
    assert [spec.task_id for spec in state.ready_tasks()] == ["reason"]
    state.start_task("reason")
    evidence_id = state.snapshot().evidence[0].evidence_id
    state.publish_agent_result(AgentResult(
        task_id="reason", role=Role.REASONING,
        findings=(Finding(claim="Connection acquisition waits observed",
                          supporting_evidence_ids=(evidence_id,)),),
        follow_ups=(FollowUpRequest(
            target_role=Role.DEPENDENCY, query=query.model_copy(update={"family": ToolFamily.CHANGES}),
            rationale="Check whether client settings changed before the waits",
            evidence_ids=(evidence_id,), depends_on=("observe", "reason"),
        ),),
    ))
    state.finish_task("reason")
    assert state.task("reason").status == "succeeded"
    assert state.snapshot().agent_results[0].follow_ups[0].depends_on == ("observe", "reason")


@pytest.mark.parametrize("specs", [
    (TaskSpec(task_id="a", role=Role.CRITIC, depends_on=("missing",)),),
    (TaskSpec(task_id="a", role=Role.CRITIC, depends_on=("a",)),),
    (
        TaskSpec(task_id="a", role=Role.CRITIC, depends_on=("b",)),
        TaskSpec(task_id="b", role=Role.REASONING, depends_on=("a",)),
    ),
    (TaskSpec(task_id="a", role=Role.CRITIC), TaskSpec(task_id="a", role=Role.CRITIC)),
])
def test_invalid_dependency_graphs_are_rejected_atomically(specs):
    state = InvestigationState("inc-001")
    with pytest.raises(ValueError):
        state.add_tasks(*specs)
    assert state.snapshot().tasks == ()


def test_cross_incident_queries_and_duplicate_task_ids_rejected(query):
    state = InvestigationState(query.incident_id)
    with pytest.raises(ValueError, match="another incident"):
        state.add_tasks(tool_spec(query.model_copy(update={"incident_id": "inc-002"})))
    state.add_tasks(tool_spec(query))
    with pytest.raises(ValueError, match="unique"):
        state.add_tasks(tool_spec(query))


def test_grounding_and_role_validation_for_agent_results(query):
    state = InvestigationState(query.incident_id)
    state.add_tasks(TaskSpec(task_id="critic", role=Role.CRITIC))
    state.start_task("critic")
    with pytest.raises(ValueError, match="evidence not present"):
        state.publish_agent_result(AgentResult(
            task_id="critic", role=Role.CRITIC,
            findings=(Finding(claim="Unsupported", supporting_evidence_ids=("made-up",)),),
        ))
    with pytest.raises(ValueError, match="role"):
        state.publish_agent_result(AgentResult(task_id="critic", role=Role.REPORTING))
    with pytest.raises(ValueError, match="another incident"):
        state.publish_agent_result(AgentResult(
            task_id="critic", role=Role.CRITIC,
            follow_ups=(FollowUpRequest(
                target_role=Role.OBSERVABILITY,
                query=query.model_copy(update={"incident_id": "inc-002"}),
                rationale="Wrong incident",
            ),),
        ))
    with pytest.raises(ValueError, match="structured result"):
        state.finish_task("critic")


def test_evidence_deduplicates_without_erasing_attempt_provenance(tools, query):
    state = InvestigationState(query.incident_id)
    state.add_tasks(tool_spec(query, "first"), tool_spec(query, "follow-up"))

    async def run():
        await run_tool_task(state, "first", tools)
        await run_tool_task(state, "follow-up", tools)

    asyncio.run(run())
    snapshot = state.snapshot()
    assert len(snapshot.evidence) == 2
    assert len(snapshot.tool_results) == 2
    assert state.task("first").result_indices == (0,)
    assert state.task("follow-up").result_indices == (1,)
    assert snapshot.usage.tool_calls == 2


def test_a_task_cannot_borrow_another_tasks_success(tools, query):
    state = InvestigationState(query.incident_id)
    state.add_tasks(tool_spec(query, "first"), tool_spec(query, "second"))
    asyncio.run(run_tool_task(state, "first", tools))
    state.start_task("second")
    with pytest.raises(ValueError, match="non-error"):
        state.finish_task("second")


def test_evidence_identity_collision_is_not_silently_overwritten(tools, query):
    state = InvestigationState(query.incident_id)
    state.add_tasks(tool_spec(query, "first"), tool_spec(query, "second"))
    original = asyncio.run(run_tool_task(state, "first", tools))
    state.start_task("second")
    altered_record = original.evidence[0].record.model_copy(update={"message": "different content"})
    altered_evidence = original.evidence[0].model_copy(update={"record": altered_record})
    altered_result = original.model_copy(update={
        "evidence": (altered_evidence, *original.evidence[1:]),
    })
    with pytest.raises(ValueError, match="collision"):
        state.record_tool_result("second", altered_result)
    assert len(state.snapshot().tool_results) == 1


@pytest.mark.parametrize("retryable,failures,expected_attempts,expected_status", [
    (True, 1, 2, "complete"),
    (True, None, 2, "error"),
    (False, None, 1, "error"),
])
def test_recovery_and_attempt_limits(catalog, query, retryable, failures, expected_attempts, expected_status):
    tools = SimulatedTools(catalog, SimulationSettings(
        latency_ms=0,
        failures=(FailureRule(
            incident_id=query.incident_id, family=query.family,
            failures=failures, retryable=retryable,
        ),),
    ))
    state = InvestigationState(query.incident_id)
    state.add_tasks(tool_spec(query))
    result = asyncio.run(run_tool_task(state, "observe", tools))
    assert result.status == expected_status
    assert state.task("observe").attempts == expected_attempts
    assert state.snapshot().usage.tool_calls == expected_attempts
    assert len(state.snapshot().tool_results) == expected_attempts
    assert state.snapshot().tool_results[0].status == "error"
    assert state.task("observe").status == ("succeeded" if expected_status != "error" else "failed")


def test_failed_prerequisite_blocks_dependents_but_not_independent_work(catalog, query):
    tools = SimulatedTools(catalog, SimulationSettings(
        latency_ms=0,
        failures=(FailureRule(
            incident_id=query.incident_id, family=query.family,
            failures=None, retryable=False,
        ),),
    ))
    state = InvestigationState(query.incident_id)
    metric_query = query.model_copy(update={"family": ToolFamily.METRICS})
    state.add_tasks(
        tool_spec(query),
        TaskSpec(task_id="reason", role=Role.REASONING, depends_on=("observe",)),
        TaskSpec(task_id="report", role=Role.REPORTING, depends_on=("reason",)),
        tool_spec(metric_query, "independent"),
    )

    async def run():
        await asyncio.gather(
            run_tool_task(state, "observe", tools), run_tool_task(state, "independent", tools)
        )

    asyncio.run(run())
    assert state.task("observe").status == "failed"
    assert state.task("reason").status == "blocked"
    assert state.task("report").status == "blocked"
    assert state.task("independent").status == "succeeded"


def test_parallel_reservations_do_not_exceed_global_call_budget(tools, query):
    state = InvestigationState(query.incident_id, ExecutionBudget(max_tool_calls=2))
    specs = [tool_spec(query, f"task-{index}") for index in range(5)]
    state.add_tasks(*specs)

    async def run():
        return await asyncio.gather(
            *(run_tool_task(state, spec.task_id, tools) for spec in specs)
        )

    results = asyncio.run(run())
    assert state.snapshot().usage.tool_calls == 2
    assert sum(result.status == "complete" for result in results) == 2
    errors = [result for result in results if result.status == "error"]
    assert len(errors) == 3
    assert all(result.error.code == "budget_exhausted" for result in errors)
    assert all(not result.evidence for result in errors)


def test_concurrency_never_exceeds_configured_parallel_slots(catalog, query):
    class TrackingTools(SimulatedTools):
        def __init__(self):
            super().__init__(catalog, SimulationSettings(latency_ms=0))
            self.active = 0
            self.peak = 0

        async def execute(self, query: ToolQuery):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0.001)
                return await super().execute(query)
            finally:
                self.active -= 1

    tools = TrackingTools()
    state = InvestigationState(query.incident_id, ExecutionBudget(max_parallel_tools=2))
    specs = [tool_spec(query, f"task-{index}") for index in range(10)]
    state.add_tasks(*specs)

    async def run():
        return await asyncio.gather(
            *(run_tool_task(state, spec.task_id, tools) for spec in specs)
        )

    results = asyncio.run(run())
    assert tools.peak == 2
    assert tools.active == 0
    assert state.snapshot().usage.tool_calls == 10
    assert all(result.status == "complete" for result in results)


def test_budget_exhaustion_stops_retrying(catalog, query):
    tools = SimulatedTools(catalog, SimulationSettings(
        latency_ms=0,
        failures=(FailureRule(
            incident_id=query.incident_id, family=query.family, failures=None,
        ),),
    ))
    state = InvestigationState(query.incident_id, ExecutionBudget(max_tool_calls=3))
    state.add_tasks(tool_spec(query, max_attempts=5))
    result = asyncio.run(run_tool_task(state, "observe", tools))
    assert result.error.code == "budget_exhausted"
    assert not result.error.retryable
    assert state.task("observe").attempts == 3
    assert state.snapshot().usage.tool_calls == 3


def test_timeouts_are_structured_and_bounded(catalog, query):
    tools = SimulatedTools(catalog, SimulationSettings(latency_ms=50))
    state = InvestigationState(
        query.incident_id, ExecutionBudget(tool_timeout_seconds=0.001)
    )
    state.add_tasks(tool_spec(query))
    result = asyncio.run(run_tool_task(state, "observe", tools))
    assert result.status == "error"
    assert result.error.code == "timeout"
    assert state.task("observe").attempts == 2
    assert state.task("observe").status == "failed"
    assert all(not item.evidence for item in state.snapshot().tool_results)


def test_deadline_also_bounds_tasks_waiting_for_a_parallel_slot(catalog, query):
    tools = SimulatedTools(catalog, SimulationSettings(latency_ms=200))
    state = InvestigationState(query.incident_id, ExecutionBudget(
        max_elapsed_seconds=0.05, tool_timeout_seconds=1, max_parallel_tools=1,
    ))
    state.add_tasks(tool_spec(query, "first"), tool_spec(query, "waiting"))

    async def run():
        return await run_tool_tasks(state, ("first", "waiting"), tools)

    results = asyncio.run(run())
    assert all(result.error.code == "budget_exhausted" for result in results)
    assert state.snapshot().usage.tool_calls == 1
    assert state.task("waiting").attempts == 0
    assert all(task.status == "failed" for task in state.snapshot().tasks)


def test_expired_investigation_does_not_dispatch_a_tool(monkeypatch, tools, query):
    clock = [100.0]
    monkeypatch.setattr("investigation.monotonic", lambda: clock[0])
    state = InvestigationState(query.incident_id, ExecutionBudget(max_elapsed_seconds=1))
    state.add_tasks(tool_spec(query))
    clock[0] = 102.0
    result = asyncio.run(run_tool_task(state, "observe", tools))
    assert result.status == "error"
    assert result.error.code == "budget_exhausted"
    assert state.snapshot().usage.tool_calls == 0
    assert state.task("observe").attempts == 0
    assert state.task("observe").status == "failed"


def test_token_reservations_are_bounded(query):
    state = InvestigationState(query.incident_id, ExecutionBudget(max_tokens=5))

    async def run():
        await state.reserve_tokens(3)
        with pytest.raises(BudgetExceeded):
            await state.reserve_tokens(3)
        with pytest.raises(ValueError):
            await state.reserve_tokens(-1)
        await state.reserve_tokens(2)

    asyncio.run(run())
    assert state.snapshot().usage.tokens == 5


def test_cancellation_leaves_explicit_failure_and_preserves_budget_charge(catalog, query):
    class CancellableTools(SimulatedTools):
        def __init__(self):
            super().__init__(catalog, SimulationSettings(latency_ms=1000))
            self.entered = asyncio.Event()

        async def execute(self, query: ToolQuery):
            self.entered.set()
            return await super().execute(query)

    tools = CancellableTools()
    state = InvestigationState(query.incident_id)
    state.add_tasks(tool_spec(query))

    async def run():
        task = asyncio.create_task(run_tool_task(state, "observe", tools))
        await tools.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert state.task("observe").status == "failed"
    assert state.snapshot().tool_results[-1].error.code == "cancelled"
    assert state.snapshot().usage.tool_calls == 1


def test_collector_actually_overlaps_all_five_independent_tools(catalog):
    class BarrierTools(SimulatedTools):
        def __init__(self):
            super().__init__(catalog, SimulationSettings(latency_ms=0))
            self.entered = 0
            self.barrier = asyncio.Event()

        async def execute(self, query: ToolQuery):
            self.entered += 1
            if self.entered == 5:
                self.barrier.set()
            await asyncio.wait_for(self.barrier.wait(), timeout=1)
            return await super().execute(query)

    tools = BarrierTools()
    snapshot = asyncio.run(collect(tools, catalog.incidents[0].context))
    assert tools.entered == 5
    assert len(snapshot.tasks) == 5
    assert snapshot.usage.tool_calls == 5
    assert all(task.status == "succeeded" for task in snapshot.tasks)

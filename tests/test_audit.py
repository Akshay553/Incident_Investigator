import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from baseline import run_baseline
from investigation import (
    BudgetExceeded, Coordinator, InvestigationState, ModelOperation, run_multi_agent, run_tool_tasks,
)
from llm import LLMFailure, ModelResponse, ScriptedLLM
from models import ExecutionBudget, Hypothesis, Role, TaskSpec, ToolFamily
from tools import SimulatedTools

ROOT = Path(__file__).resolve().parent.parent


def replies(name):
    return json.loads((ROOT / "tests" / "fixtures" / f"{name}.json").read_text(encoding="utf-8"))


def test_model_call_reservation_is_atomic_even_when_the_budget_lock_is_contended():
    state = InvestigationState("inc-001", ExecutionBudget(max_llm_calls=1))
    state.add_tasks(TaskSpec(task_id="one", role=Role.REASONING))
    state.start_task("one")

    async def run():
        await state._budget_lock.acquire()
        attempts = [asyncio.create_task(state.reserve_model_call("one", 10)) for _ in range(2)]
        await asyncio.sleep(0)
        state._budget_lock.release()
        return await asyncio.gather(*attempts, return_exceptions=True)

    results = asyncio.run(run())
    assert sum(isinstance(result, BudgetExceeded) for result in results) == 1
    assert state.snapshot().usage.llm_calls == state.task("one").attempts == 1
    assert state.snapshot().usage.tokens == 10


@pytest.mark.parametrize("sequential", [False, True])
def test_dependent_tool_batch_runs_parent_before_child(tools, query, sequential):
    state = InvestigationState(query.incident_id)
    state.add_tasks(
        TaskSpec(task_id="parent", role=Role.OBSERVABILITY, query=query),
        TaskSpec(
            task_id="child", role=Role.OBSERVABILITY,
            query=query.model_copy(update={"family": ToolFamily.METRICS}), depends_on=("parent",),
        ),
    )
    results = asyncio.run(run_tool_tasks(state, ("child", "parent"), tools, sequential=sequential))
    assert all(result.status == "complete" for result in results)
    assert state.task("parent").status == state.task("child").status == "succeeded"
    assert state.task("parent").result_indices[0] < state.task("child").result_indices[0]


def test_dependency_not_in_batch_does_not_leave_an_unstarted_task_running(tools, query):
    state = InvestigationState(query.incident_id)
    state.add_tasks(
        TaskSpec(task_id="parent", role=Role.OBSERVABILITY, query=query),
        TaskSpec(task_id="child", role=Role.OBSERVABILITY, query=query, depends_on=("parent",)),
    )
    with pytest.raises(RuntimeError, match="prerequisites"):
        asyncio.run(run_tool_tasks(state, ("child",), tools))
    assert state.task("parent").status == state.task("child").status == "pending"
    assert state.snapshot().usage.tool_calls == 0


def test_revision_can_reclassify_evidence_with_an_explicit_explanation(tools, catalog):
    coordinator = Coordinator(tools, ScriptedLLM({}), catalog.incidents[0].context)
    prior = Hypothesis(
        hypothesis_id="h1", claim="A tentative hypothesis", confidence=0.5,
        supporting_evidence_ids=("example",),
    )
    revised = prior.model_copy(update={
        "supporting_evidence_ids": (), "contradicting_evidence_ids": ("example",),
    })
    with pytest.raises(ValueError, match="explanation"):
        coordinator.validate_revision((revised,), (prior,), ())
    coordinator.validate_revision(
        (revised,), (prior,), ("The later observation challenges the original interpretation",),
    )
    with pytest.raises(ValueError, match="explanation"):
        coordinator.validate_revision((), (prior,), ())


@pytest.mark.parametrize("polarity", ["supporting_evidence_ids", "contradicting_evidence_ids"])
@pytest.mark.parametrize("approach", ["multi-agent", "baseline"])
def test_revision_cannot_silently_drop_known_evidence(tools, catalog, polarity, approach):
    if approach == "multi-agent":
        responses = replies("multi_agent")
        responses["reasoning-0"][0]["hypotheses"][0][polarity] = ["inc-001:metrics:r03"]
        # The successful earlier hypothesis must survive a rejected revision.
        responses["report"][0].update({
            "outcome": "inconclusive",
            "hypotheses": responses["reasoning-0"][0]["hypotheses"][:1],
            "limitations": ["Revision could not be validated"],
        })
        runner, rejected_task = run_multi_agent, "reasoning-1"
    else:
        responses = replies("baseline")
        responses["baseline-1"][0]["hypotheses"][0][polarity] = ["inc-001:metrics:r03"]
        runner, rejected_task = run_baseline, "baseline-2"
    run = asyncio.run(runner(tools, ScriptedLLM(responses), catalog.incidents[0].context))
    assert run.report.outcome == "inconclusive"
    assert run.report.hypotheses
    assert "inc-001:metrics:r03" in getattr(run.report.hypotheses[0], polarity)
    assert any(
        failure.task_id == rejected_task and failure.code == "invalid_response"
        and "evidence" in failure.message
        for failure in run.trace.failures
    )


def test_failed_revision_keeps_last_successful_hypotheses(tools, catalog):
    responses = replies("multi_agent")
    prior = responses["reasoning-0"][0]["hypotheses"][:1]
    responses["reasoning-1"] = [LLMFailure("authentication", "Analysis unavailable")]
    responses["report"][0].update({
        "outcome": "inconclusive", "hypotheses": prior, "limitations": ["Latest revision unavailable"],
    })
    run = asyncio.run(run_multi_agent(tools, ScriptedLLM(responses), catalog.incidents[0].context))
    assert run.report.outcome == "inconclusive"
    assert run.report.hypotheses[0].claim == prior[0]["claim"]
    assert run.trace.termination_reason == "reasoning_unavailable"


def test_completed_parallel_model_usage_survives_sibling_cancellation(tools, catalog):
    class MixedLLM(ScriptedLLM):
        def __init__(self):
            super().__init__({})
            self.finished = asyncio.Event()
            self.blocked = asyncio.Event()

        async def respond(self, task_id, *args):
            if task_id == "fast":
                self.finished.set()
                return ModelResponse(
                    json.dumps({"task_id": "fast", "role": "observability"}), 17, 4,
                )
            self.blocked.set()
            await asyncio.sleep(10)

    llm = MixedLLM()
    coordinator = Coordinator(tools, llm, catalog.incidents[0].context)

    async def run():
        batch = asyncio.create_task(coordinator.model_batch(
            ModelOperation("fast", Role.OBSERVABILITY, Role.OBSERVABILITY),
            ModelOperation("slow", Role.DEPENDENCY, Role.DEPENDENCY),
        ))
        await llm.finished.wait()
        await llm.blocked.wait()
        await asyncio.sleep(0)
        batch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await batch

    asyncio.run(run())
    usage = coordinator.state.snapshot().usage
    assert usage.input_tokens == 17
    assert usage.output_tokens == 4
    assert len(coordinator.model_calls) == 2
    assert sum(call.usage_measured for call in coordinator.model_calls) == 1
    assert usage.unmeasured_token_reservations == next(
        call.reserved_tokens for call in coordinator.model_calls if call.task_id == "slow"
    )


def test_completed_tool_evidence_survives_sibling_cancellation(catalog, query):
    class MixedTools(SimulatedTools):
        def __init__(self):
            super().__init__(catalog)
            self.finished = asyncio.Event()
            self.blocked = asyncio.Event()

        async def execute(self, query):
            if query.family == ToolFamily.LOGS:
                result = await super().execute(query)
                self.finished.set()
                return result
            self.blocked.set()
            await asyncio.sleep(10)

    tools = MixedTools()
    state = InvestigationState(query.incident_id)
    state.add_tasks(
        TaskSpec(task_id="fast", role=Role.OBSERVABILITY, query=query),
        TaskSpec(
            task_id="slow", role=Role.OBSERVABILITY,
            query=query.model_copy(update={"family": ToolFamily.METRICS}),
        ),
    )

    async def run():
        batch = asyncio.create_task(run_tool_tasks(state, ("fast", "slow"), tools))
        await tools.finished.wait()
        await tools.blocked.wait()
        await asyncio.sleep(0)
        batch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await batch

    asyncio.run(run())
    assert state.task("fast").status == "succeeded"
    assert state.task("slow").status == "failed"
    assert state.snapshot().evidence
    assert state.snapshot().usage.tool_calls == 2


def test_model_integration_exception_is_visible_and_independent_work_survives(tools, catalog):
    class BrokenLLM(ScriptedLLM):
        async def respond(self, task_id, *args):
            if task_id == "broken":
                raise RuntimeError("Injected adapter failure")
            return await super().respond(task_id, *args)

    llm = BrokenLLM({"healthy": [{"task_id": "healthy", "role": "dependency"}]})
    coordinator = Coordinator(tools, llm, catalog.incidents[0].context)
    outputs = asyncio.run(coordinator.model_batch(
        ModelOperation("broken", Role.OBSERVABILITY, Role.OBSERVABILITY),
        ModelOperation("healthy", Role.DEPENDENCY, Role.DEPENDENCY),
    ))
    assert outputs["broken"] is None
    assert outputs["healthy"] is not None
    assert coordinator.failures[0].code == "operation_failed"
    assert coordinator.state.task("broken").attempts == 1
    assert coordinator.state.snapshot().usage.unmeasured_token_reservations > 0


def test_tool_integration_exception_is_structured_and_blocks_its_dependent(catalog, query):
    class BrokenTools(SimulatedTools):
        async def execute(self, query):
            if query.family == ToolFamily.LOGS:
                raise OSError("Injected source adapter failure")
            return await super().execute(query)

    state = InvestigationState(query.incident_id)
    state.add_tasks(
        TaskSpec(task_id="broken", role=Role.OBSERVABILITY, query=query),
        TaskSpec(
            task_id="dependent", role=Role.OBSERVABILITY,
            query=query.model_copy(update={"family": ToolFamily.METRICS}), depends_on=("broken",),
        ),
    )
    outputs = asyncio.run(run_tool_tasks(state, ("broken", "dependent"), BrokenTools(catalog)))
    assert outputs[0].error.code == "operation_failed"
    assert outputs[1].error.code == "dependency_failed"
    assert state.task("dependent").status == "blocked"
    assert state.task("dependent").attempts == 0
    assert state.snapshot().usage.tool_calls == 1


def test_response_finishing_after_deadline_cannot_become_success(monkeypatch, tools, catalog):
    clock = [0.0]
    monkeypatch.setattr("investigation.monotonic", lambda: clock[0])

    class LateLLM(ScriptedLLM):
        async def respond(self, *args):
            result = await super().respond(*args)
            clock[0] = 2.0
            return result

    coordinator = Coordinator(
        tools, LateLLM(replies("multi_agent")), catalog.incidents[0].context,
        ExecutionBudget(max_elapsed_seconds=1),
    )
    output = asyncio.run(coordinator.model(ModelOperation("plan", Role.ORCHESTRATOR, "plan")))
    assert output is None
    assert coordinator.state.task("plan").status == "failed"
    assert coordinator.failures[-1].code == "budget_exhausted"
    assert coordinator.model_calls[0].usage_measured


def test_tool_response_finishing_after_deadline_cannot_become_success(monkeypatch, catalog, query):
    clock = [0.0]
    monkeypatch.setattr("investigation.monotonic", lambda: clock[0])

    class LateTools(SimulatedTools):
        async def execute(self, query):
            result = await super().execute(query)
            clock[0] = 2.0
            return result

    state = InvestigationState(query.incident_id, ExecutionBudget(max_elapsed_seconds=1))
    state.add_tasks(TaskSpec(task_id="late", role=Role.OBSERVABILITY, query=query))
    result = asyncio.run(run_tool_tasks(state, ("late",), LateTools(catalog)))[0]
    assert result.error.code == "budget_exhausted"
    assert state.task("late").status == "failed"
    assert state.snapshot().usage.tool_calls == 1


def test_reporting_cannot_promote_a_lower_ranked_candidate(tools, catalog):
    responses = replies("multi_agent")
    hypotheses = responses["reasoning-0"][0]["hypotheses"]
    hypotheses[1]["supporting_evidence_ids"] = list(hypotheses[0]["supporting_evidence_ids"])
    responses["critic-0"][0].update({"objections": [], "follow_ups": []})
    responses["decision-0"] = [{"action": "finish", "rationale": "No further investigation"}]
    responses["report"][0]["hypotheses"] = hypotheses[1:]
    run = asyncio.run(run_multi_agent(tools, ScriptedLLM(responses), catalog.incidents[0].context))
    assert run.report.outcome == "inconclusive"
    assert run.report.hypotheses[0].hypothesis_id == hypotheses[0]["hypothesis_id"]
    assert any(
        failure.code == "invalid_response" and "top-ranked" in failure.message
        for failure in run.trace.failures
    )


def test_cli_reports_data_load_error_without_traceback(monkeypatch, capsys):
    import cli

    def missing():
        raise FileNotFoundError("Synthetic data file missing")

    monkeypatch.setattr(cli.SimulatedTools, "from_files", missing)
    monkeypatch.setattr(sys, "argv", ["cli.py", "list"])
    with pytest.raises(SystemExit):
        cli.main()
    error = capsys.readouterr().err
    assert "data" in error.lower()
    assert "Traceback" not in error


def test_cli_rejects_file_output_target_before_starting_an_investigation(tmp_path):
    marker = tmp_path / "unrelated.txt"
    marker.write_text("preserve")
    result = subprocess.run([
        sys.executable, str(ROOT / "cli.py"), "run", "multi-agent", "inc-001",
        "--mock-script", str(ROOT / "tests" / "fixtures" / "multi_agent.json"),
        "--output-dir", str(marker),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "output" in result.stderr.lower()
    assert "Traceback" not in result.stderr
    assert marker.read_text() == "preserve"


def test_missing_credentials_cli_is_clear_and_does_not_start_paid_requests(tmp_path):
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    result = subprocess.run([
        sys.executable, str(ROOT / "cli.py"), "run", "multi-agent", "inc-001",
        "--allow-paid", "--output-dir", str(tmp_path / "output"),
    ], env=environment, capture_output=True, text=True)
    assert result.returncode != 0
    assert "OPENAI_API_KEY" in result.stderr
    assert "locally" in result.stderr
    assert "Traceback" not in result.stderr

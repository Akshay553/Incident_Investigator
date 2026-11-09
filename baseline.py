"""One iterative model agent with the same runtime, evidence and resource ceilings."""

import asyncio

from agents import ACCESS
from investigation import Coordinator, ModelOperation, validate_all
from models import BaselineStep, ExecutionBudget, IncidentContext, Role, TaskSpec
from llm import StructuredLLM
from tools import SimulatedTools


async def run_baseline(
    tools: SimulatedTools, llm: StructuredLLM, context: IncidentContext,
    budget: ExecutionBudget | None = None,
):
    coordinator = Coordinator(tools, llm, context, budget)
    try:
        return await _baseline(coordinator)
    except asyncio.CancelledError:
        return coordinator.finalize("baseline", None, "cancelled", force_inconclusive=True)


async def _baseline(coordinator: Coordinator):
    previous: BaselineStep | None = None
    dependencies: tuple[str, ...] = ()
    for index in range(coordinator.state.budget.max_llm_calls):
        task_id = f"baseline-{index}"

        def validate_step(step):
            def report():
                if step.report is not None:
                    def matching():
                        if step.report.hypotheses != step.hypotheses:
                            raise ValueError("report.hypotheses: baseline step and report hypotheses must match")

                    validate_all(
                        lambda: coordinator.report_validation(step.report, step.hypotheses, path="report."),
                        matching,
                    )

            def unique_queries():
                keys = [coordinator.query_key(query) for query in step.queries]
                if len(keys) != len(set(keys)):
                    raise ValueError("queries: baseline requested duplicate queries in one step")

            validate_all(
                lambda: coordinator.validate_revision(
                    step.hypotheses, previous.hypotheses if previous else (), (step.rationale,),
                ),
                report, unique_queries,
                *(lambda query=query, index=index: coordinator.validate_query(
                    query, next(role for role, families in ACCESS.items() if query.family in families),
                    retry=step.retry_failed_queries, path=f"queries[{index}]",
                ) for index, query in enumerate(step.queries)),
            )

        step = await coordinator.model(ModelOperation(
            task_id, Role.REASONING, "baseline", depends_on=dependencies,
            extra={
                "current_hypotheses": [
                    item.model_dump(mode="json") for item in previous.hypotheses
                ] if previous else [],
                "runtime_limitations": coordinator.report_limitations(
                    previous.hypotheses if previous else (),
                ),
            },
            validate=validate_step,
        ))
        if step is None:
            return coordinator.finalize(
                "baseline", None, "baseline_analysis_unavailable", force_inconclusive=True,
                fallback_hypotheses=previous.hypotheses if previous else (),
            )
        if step.action == "finish":
            failed = any(task.status == "failed" for task in coordinator.state.snapshot().tasks)
            return coordinator.finalize(
                "baseline", step.report, "baseline_finished", force_inconclusive=failed,
            )
        specs = tuple(
            TaskSpec(
                task_id=f"baseline-tool-{index}-{query_index}",
                role=next(role for role, families in ACCESS.items() if query.family in families),
                query=query, depends_on=(task_id,),
                max_attempts=coordinator.state.budget.operation_attempts,
            )
            for query_index, query in enumerate(step.queries)
        )
        retry_keys = frozenset(
            coordinator.query_key(query) for query in step.queries
        ) if step.retry_failed_queries else frozenset()
        await coordinator.evidence_tasks(specs, sequential=True, retry_keys=retry_keys)
        dependencies = (
            task_id,
            *(spec.task_id for spec in specs if coordinator.state.task(spec.task_id).status == "succeeded"),
        )
        previous = step
    return coordinator.finalize(
        "baseline", None, "baseline_step_limit", force_inconclusive=True,
        fallback_hypotheses=previous.hypotheses if previous else (),
    )

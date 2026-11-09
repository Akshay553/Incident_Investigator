"""Shared state and serialized coordination of concurrent, bounded operations."""

import asyncio
import json
from dataclasses import dataclass
from time import monotonic
from typing import Callable, Iterable, Sequence

from pydantic import ValidationError

from models import (
    ACCESS,
    AgentResult,
    Contract,
    ContinueDecision,
    Evidence,
    ExecutionBudget,
    ExecutionTrace,
    Hypothesis,
    IncidentContext,
    InitialPlan,
    InvestigationReport,
    InvestigationRun,
    InvestigationSnapshot,
    ModelCallTrace,
    OperationFailure,
    Role,
    RunUsage,
    TaskSpec,
    TaskState,
    ToolError,
    ToolFamily,
    ToolResult,
)
from llm import LLMFailure, ModelResponse, StructuredLLM, structured_schema
from tools import SimulatedTools, error_result, query_key


class BudgetExceeded(RuntimeError):
    pass


class DependencyNotReady(RuntimeError):
    pass


class CorrectableResponseError(ValueError):
    """Safe contract feedback within the existing single correction attempt."""

    def __init__(self, violations: str | Sequence[str], already_executed_queries=()):
        self.violations = (violations,) if isinstance(violations, str) else tuple(violations)
        self.already_executed_queries = tuple(already_executed_queries)
        super().__init__("\n".join(self.violations))


def validate_all(*checks: Callable[[], None]) -> None:
    violations, previous_queries = [], []
    fatal = False
    for check in checks:
        try:
            check()
        except CorrectableResponseError as exc:
            violations.extend(exc.violations)
            previous_queries.extend(exc.already_executed_queries)
        except ValueError as exc:
            violations.append(str(exc))
            fatal = True
    if violations:
        if fatal:
            raise ValueError("\n".join(violations))
        raise CorrectableResponseError(tuple(dict.fromkeys(violations)), previous_queries)


def validation_messages(error: ValidationError) -> tuple[str, ...]:
    return tuple(
        "".join(f"[{part}]" if isinstance(part, int) else ("." if index else "") + str(part)
                for index, part in enumerate(item["loc"])) + ": " + item["msg"]
        for item in error.errors(include_context=False, include_input=False)
    )


def unique_limitations(values: Iterable[str]) -> tuple[str, ...]:
    retained = {}
    for value in values:
        text = " ".join(value.split())
        key = text.casefold().rstrip(".!?")
        if key and key not in retained:
            retained[key] = text
    return tuple(retained.values())


class InvestigationState:
    def __init__(self, incident_id: str, budget: ExecutionBudget | None = None):
        self.incident_id = incident_id
        self.budget = budget or ExecutionBudget()
        self._started = monotonic()
        self._tasks: dict[str, TaskState] = {}
        self._evidence: dict[str, Evidence] = {}
        self._tool_results: list[ToolResult] = []
        self._agent_results: dict[str, AgentResult] = {}
        self._tool_calls = 0
        self._tokens = 0
        self._llm_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._unmeasured_tokens = 0
        self._budget_lock = asyncio.Lock()
        self.tool_slots = asyncio.Semaphore(self.budget.max_parallel_tools)

    @property
    def remaining_seconds(self) -> float:
        return max(0, self.budget.max_elapsed_seconds - (monotonic() - self._started))

    def task(self, task_id: str) -> TaskState:
        return self._tasks[task_id]

    def add_tasks(self, *specs: TaskSpec) -> None:
        ids = [spec.task_id for spec in specs]
        if len(ids) != len(set(ids)) or set(ids) & self._tasks.keys():
            raise ValueError("task IDs must be unique")
        proposed = {**self._tasks, **{spec.task_id: TaskState(spec=spec) for spec in specs}}
        for spec in specs:
            if spec.query and spec.query.incident_id != self.incident_id:
                raise ValueError("task query belongs to another incident")
            if len(spec.depends_on) != len(set(spec.depends_on)):
                raise ValueError("task dependencies must be unique")
            if any(dependency not in proposed for dependency in spec.depends_on):
                raise ValueError("task dependency does not exist")
        resolved: set[str] = set()
        while len(resolved) < len(proposed):
            ready = {
                task_id for task_id, task in proposed.items()
                if task_id not in resolved and set(task.spec.depends_on) <= resolved
            }
            if not ready:
                raise ValueError("task dependencies contain a cycle")
            resolved.update(ready)
        self._tasks = proposed
        self._block_dependents()

    def ready_tasks(self) -> tuple[TaskSpec, ...]:
        return tuple(
            task.spec for task in self._tasks.values()
            if task.status == "pending"
            and all(self.task(dep).status == "succeeded" for dep in task.spec.depends_on)
        )

    def _replace_task(self, task_id: str, **changes) -> None:
        values = {
            "spec": self.task(task_id).spec,
            "status": self.task(task_id).status,
            "attempts": self.task(task_id).attempts,
            "result_indices": self.task(task_id).result_indices,
            "failure": self.task(task_id).failure,
            "output_json": self.task(task_id).output_json,
        }
        self._tasks[task_id] = TaskState(**(values | changes))

    def start_task(self, task_id: str) -> None:
        task = self.task(task_id)
        if task.status != "pending":
            raise ValueError("only a pending task can start")
        if not all(self.task(dep).status == "succeeded" for dep in task.spec.depends_on):
            raise DependencyNotReady("task prerequisites have not succeeded")
        self._replace_task(task_id, status="running")

    async def reserve_tool_call(self, task_id: str) -> None:
        async with self._budget_lock:
            task = self.task(task_id)
            if task.status != "running":
                raise ValueError("tool calls require a running task")
            if task.attempts >= task.spec.max_attempts:
                raise BudgetExceeded("task attempt limit exhausted")
            if self.remaining_seconds <= 0:
                raise BudgetExceeded("investigation time budget exhausted")
            if self._tool_calls >= self.budget.max_tool_calls:
                raise BudgetExceeded("investigation tool-call budget exhausted")
            self._tool_calls += 1
            self._replace_task(task_id, attempts=task.attempts + 1)

    async def reserve_tokens(self, count: int) -> None:
        """Conservatively reserve before dispatch; release only with measured usage."""
        if count < 0:
            raise ValueError("token reservation cannot be negative")
        async with self._budget_lock:
            if self.remaining_seconds <= 0:
                raise BudgetExceeded("investigation time budget exhausted")
            if self._tokens + count > self.budget.max_tokens:
                raise BudgetExceeded("investigation token budget exhausted")
            self._tokens += count

    async def reserve_model_call(self, task_id: str, token_bound: int) -> None:
        if token_bound < 0:
            raise ValueError("token reservation cannot be negative")
        async with self._budget_lock:
            task = self.task(task_id)
            if task.status != "running":
                raise ValueError("model calls require a running task")
            if task.attempts >= task.spec.max_attempts:
                raise BudgetExceeded("model task attempt limit exhausted")
            if self._llm_calls >= self.budget.max_llm_calls:
                raise BudgetExceeded("investigation LLM-call budget exhausted")
            if self.remaining_seconds <= 0:
                raise BudgetExceeded("investigation time budget exhausted")
            if self._tokens + token_bound > self.budget.max_tokens:
                raise BudgetExceeded("investigation token budget exhausted")
            self._tokens += token_bound
            self._llm_calls += 1
            self._replace_task(task_id, attempts=task.attempts + 1)

    def settle_model_call(self, reservation: int, response: ModelResponse | None) -> None:
        if response is None:
            self._unmeasured_tokens += reservation
            return
        if response.input_tokens < 0 or response.output_tokens < 0:
            raise ValueError("model usage cannot be negative")
        actual = response.input_tokens + response.output_tokens
        self._tokens += actual - reservation
        self._input_tokens += response.input_tokens
        self._output_tokens += response.output_tokens
        if actual > reservation:
            raise BudgetExceeded("provider usage exceeded the conservative token reservation")

    def record_tool_result(self, task_id: str, result: ToolResult) -> None:
        task = self.task(task_id)
        if task.status != "running" or result.query != task.spec.query:
            raise ValueError("tool result does not match a running task")
        for evidence in result.evidence:
            previous = self._evidence.get(evidence.evidence_id)
            if previous is not None and previous != evidence:
                raise ValueError("evidence ID collision with different content")
        self._evidence.update({item.evidence_id: item for item in result.evidence})
        self._replace_task(
            task_id,
            result_indices=(*task.result_indices, len(self._tool_results)),
        )
        self._tool_results.append(result)

    def publish_agent_result(self, result: AgentResult) -> None:
        task = self.task(result.task_id)
        if task.status != "running" or result.role != task.spec.role:
            raise ValueError("agent result does not match a running task and role")
        if result.task_id in self._agent_results:
            raise ValueError("task already has an agent result")
        references = set()
        for finding in (*result.findings, *result.hypotheses):
            references.update(finding.supporting_evidence_ids)
            references.update(finding.contradicting_evidence_ids)
        for objection in result.objections:
            references.update(objection.evidence_ids)
        for follow_up in result.follow_ups:
            if follow_up.query.incident_id != self.incident_id:
                raise ValueError("follow-up query belongs to another incident")
            if any(dep not in self._tasks for dep in follow_up.depends_on):
                raise ValueError("follow-up dependency does not exist")
            references.update(follow_up.evidence_ids)
        if not references <= self._evidence.keys():
            raise ValueError("agent result references evidence not present in shared state")
        self._agent_results[result.task_id] = result
        self._replace_task(result.task_id, output_json=result.model_dump_json())

    def record_model_output(self, task_id: str, output: Contract) -> None:
        if self.task(task_id).status != "running":
            raise ValueError("model output requires a running task")
        if isinstance(output, AgentResult):
            self.publish_agent_result(output)
        else:
            self._replace_task(task_id, output_json=output.model_dump_json())

    def finish_task(self, task_id: str, *, failure: str | None = None) -> None:
        if self.task(task_id).status != "running":
            raise ValueError("only a running task can finish")
        if failure is None:
            if self.task(task_id).spec.query is not None:
                indices = self.task(task_id).result_indices
                if not indices or self._tool_results[indices[-1]].status == "error":
                    raise ValueError("tool task cannot succeed without a non-error result")
            elif self.task(task_id).output_json is None:
                raise ValueError("agent task cannot succeed without a structured result")
        self._replace_task(
            task_id, status="failed" if failure is not None else "succeeded", failure=failure
        )
        self._block_dependents()

    def _block_dependents(self) -> None:
        changed = True
        while changed:
            changed = False
            for task_id, task in tuple(self._tasks.items()):
                if task.status == "pending" and any(
                    self.task(dep).status in {"failed", "blocked"} for dep in task.spec.depends_on
                ):
                    self._replace_task(
                        task_id, status="blocked", failure="A prerequisite failed or was blocked"
                    )
                    changed = True

    def snapshot(self) -> InvestigationSnapshot:
        return InvestigationSnapshot(
            incident_id=self.incident_id,
            tasks=tuple(self._tasks.values()),
            evidence=tuple(self._evidence.values()),
            tool_results=tuple(self._tool_results),
            agent_results=tuple(self._agent_results.values()),
            usage=RunUsage(
                tool_calls=self._tool_calls,
                tokens=self._tokens,
                elapsed_seconds=monotonic() - self._started,
                llm_calls=self._llm_calls,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                unmeasured_token_reservations=self._unmeasured_tokens,
            ),
        )


async def run_tool_task(
    state: InvestigationState, task_id: str, tools: SimulatedTools
) -> ToolResult:
    return (await run_tool_tasks(state, (task_id,), tools))[0]


async def run_tool_tasks(
    state: InvestigationState, task_ids: tuple[str, ...], tools: SimulatedTools,
    *, sequential: bool = False,
) -> tuple[ToolResult, ...]:
    """Only this coordinator mutates state; concurrent workers return values."""
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("tool batch task IDs must be unique")
    for task_id in task_ids:
        if state.task(task_id).spec.query is None:
            raise ValueError("run_tool_task requires a tool query")
    pending = list(task_ids)
    final: dict[str, ToolResult] = {}
    returned: dict[str, ToolResult] = {}

    async def execute(task_id):
        query = state.task(task_id).spec.query
        try:
            async with asyncio.timeout(state.remaining_seconds):
                async with state.tool_slots:
                    result = await asyncio.wait_for(
                        tools.execute(query),
                        timeout=min(state.budget.tool_timeout_seconds, state.remaining_seconds),
                    )
                    if state.remaining_seconds <= 0:
                        raise TimeoutError("Investigation deadline exceeded")
        except TimeoutError:
            exhausted = state.remaining_seconds <= 0
            result = error_result(query, ToolError(
                code="budget_exhausted" if exhausted else "timeout",
                message="Investigation deadline exceeded" if exhausted else "Tool timed out",
                retryable=not exhausted,
            ))
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
            result = error_result(query, ToolError(
                code="operation_failed", message=f"{type(exc).__name__}: {exc}", retryable=False,
            ))
        # Stage returned values only; the coordinator alone commits investigation state.
        returned[task_id] = result
        return result

    try:
        while pending:
            for task_id in tuple(pending):
                if state.task(task_id).status == "blocked":
                    result = error_result(state.task(task_id).spec.query, ToolError(
                        code="dependency_failed", message=state.task(task_id).failure, retryable=False,
                    ))
                    state._replace_task(task_id, output_json=result.model_dump_json())
                    final[task_id] = result
                    pending.remove(task_id)
            if not pending:
                break
            ready = {spec.task_id for spec in state.ready_tasks()}
            wave = [
                task_id for task_id in pending
                if task_id in ready or state.task(task_id).status == "running"
            ][:1 if sequential else state.budget.max_parallel_tools]
            if not wave:
                raise DependencyNotReady("task prerequisites are not ready in this tool batch")
            for task_id in wave:
                pending.remove(task_id)
                if state.task(task_id).status == "pending":
                    state.start_task(task_id)
            reserved = []
            completed = []
            for task_id in wave:
                query = state.task(task_id).spec.query
                try:
                    await state.reserve_tool_call(task_id)
                    reserved.append(task_id)
                except BudgetExceeded as exc:
                    completed.append((task_id, error_result(query, ToolError(
                        code="budget_exhausted", message=str(exc), retryable=False,
                    ))))
            results = await asyncio.gather(*(
                execute(task_id) for task_id in reserved
            ))
            completed.extend(zip(reserved, results))
            for task_id, result in completed:
                returned.pop(task_id, None)
                state.record_tool_result(task_id, result)
                spec = state.task(task_id).spec
                if result.error and result.error.retryable and state.task(task_id).attempts < spec.max_attempts:
                    pending.append(task_id)
                else:
                    state.finish_task(task_id, failure=result.error.message if result.error else None)
                    final[task_id] = result
    except asyncio.CancelledError:
        for task_id in task_ids:
            if state.task(task_id).status == "running":
                result = returned.get(task_id) or error_result(state.task(task_id).spec.query, ToolError(
                    code="cancelled", message="Tool task was cancelled", retryable=False,
                ))
                state.record_tool_result(task_id, result)
                state.finish_task(task_id, failure=result.error.message if result.error else None)
            elif state.task(task_id).status == "pending":
                state._replace_task(
                    task_id, status="blocked", failure="Tool batch cancelled before this task started",
                )
        raise
    return tuple(final[task_id] for task_id in task_ids)


@dataclass(frozen=True)
class ModelOperation:
    task_id: str
    role: Role
    purpose: str
    depends_on: tuple[str, ...] = ()
    extra: dict | None = None
    validate: Callable[[Contract], None] | None = None


class Coordinator:
    def __init__(
        self, tools: SimulatedTools, llm: StructuredLLM, context: IncidentContext,
        budget: ExecutionBudget | None = None,
    ):
        self.tools, self.llm, self.context = tools, llm, context
        self.state = InvestigationState(context.incident_id, budget)
        self.failures: list[OperationFailure] = []
        self.model_calls: list[ModelCallTrace] = []
        self.limitations: list[str] = []
        self._seen_queries: dict[str, str] = {}
        self._query_attempts: dict[str, int] = {}

    def query_key(self, query) -> str:
        return query_key(query)

    def validate_query(self, query, role: Role, *, retry: bool = False, path="query") -> None:
        if query.incident_id != self.context.incident_id:
            raise ValueError(f"{path}.incident_id: cross-incident queries are forbidden")
        if role not in ACCESS or query.family not in ACCESS[role]:
            raise ValueError(f"{path}.family: role is not permitted to query this tool family")
        key = self.query_key(query)
        if key in self._seen_queries:
            task = self.state.task(self._seen_queries[key])
            if task.result_indices:
                result = self.state.snapshot().tool_results[task.result_indices[-1]]
                if retry and task.status == "failed" and result.error and result.error.retryable:
                    if self._query_attempts.get(key, 0) >= self.state.budget.operation_attempts:
                        raise ValueError(f"{path}: query has exhausted its total attempt limit")
                    return
                raise CorrectableResponseError(
                    f"{path}: duplicate query already executed by {task.spec.task_id}; "
                    "use its existing result/coverage, choose a new query or finish; "
                    "do not reexecute a complete/partial result or an ineligible error",
                    ({"field_path": path, "task_id": task.spec.task_id,
                      "result": result.model_dump(mode="json")},),
                )
            if not retry or task.status != "failed":
                raise ValueError(f"{path}: duplicate query is not an intentional failed-query retry")
            if not task.result_indices:
                raise ValueError("failed query has no eligible tool failure to retry")

    def check_references(self, output, *, path="") -> None:
        data = output.model_dump(mode="json") if isinstance(output, Contract) else output
        known = {item.evidence_id for item in self.state.snapshot().evidence}
        fatal, correctable = [], []
        for field in ("findings", "hypotheses"):
            for index, finding in enumerate(data.get(field, ())):
                for polarity in ("supporting_evidence_ids", "contradicting_evidence_ids"):
                    for position, reference in enumerate(finding.get(polarity, ())):
                        if reference not in known:
                            fatal.append(f"{path}{field}[{index}].{polarity}[{position}]: unknown evidence reference {reference}")
        hypotheses = data.get("hypotheses", ())
        seen = set()
        for index, item in enumerate(hypotheses):
            if item["hypothesis_id"] in seen:
                fatal.append(f"{path}hypotheses[{index}].hypothesis_id: hypothesis IDs must be unique")
            seen.add(item["hypothesis_id"])
            if index and item["confidence"] > hypotheses[index - 1]["confidence"]:
                correctable.append(
                    f"{path}hypotheses[{index}].confidence ({item['hypothesis_id']}): "
                    f"{item['confidence']} exceeds preceding {hypotheses[index - 1]['confidence']}; "
                    "hypotheses must be ranked by descending confidence"
                )
        for field in ("objections", "follow_ups"):
            for index, item in enumerate(data.get(field, ())):
                for position, reference in enumerate(item.get("evidence_ids", ())):
                    if reference not in known:
                        fatal.append(f"{path}{field}[{index}].evidence_ids[{position}]: unknown evidence reference {reference}")
        if fatal:
            raise ValueError("\n".join((*fatal, *correctable)))
        if correctable:
            raise CorrectableResponseError(correctable)

    def validate_outcome(self, report: InvestigationReport | dict, *, path="") -> None:
        data = report.model_dump(mode="json") if isinstance(report, InvestigationReport) else report
        if data["outcome"] != "ranked" or not data["hypotheses"]:
            return
        top = data["hypotheses"][0]
        violations = []
        if top["causal_stage"] != "initiating_cause":
            violations.append(
                f"{path}hypotheses[0].causal_stage ({top['hypothesis_id']}): "
                "a mechanism/unknown candidate cannot win a ranked initiating-cause report; "
                f"{path}outcome must be inconclusive while the cause is unresolved"
            )
        if top.get("blocking_uncertainties"):
            violations.append(
                f"{path}hypotheses[0].blocking_uncertainties ({top['hypothesis_id']}): "
                "blocking gaps remain; " + path + "outcome must be inconclusive; do not erase uncertainty"
            )
        if violations:
            raise CorrectableResponseError(violations)

    def validate_initial_tasks(
        self, specs: Sequence[TaskSpec], *, check_ownership: bool = True,
    ) -> None:
        if {spec.query.family for spec in specs if spec.query} != set(ToolFamily):
            raise ValueError("initial plan must investigate all five tool families")
        keys = []
        for spec in specs:
            if not spec.task_id.startswith("initial-") or spec.depends_on or spec.query is None:
                raise ValueError("initial tasks require initial- IDs, queries, and empty dependencies")
            role = spec.role if check_ownership else next(
                owner for owner, families in ACCESS.items() if spec.query.family in families
            )
            self.validate_query(spec.query, role)
            keys.append(self.query_key(spec.query))
        if len(set(keys)) != len(keys):
            raise ValueError("initial plan contains duplicate queries")
        if len({spec.task_id for spec in specs}) != len(specs):
            raise ValueError("initial task IDs must be unique")

    def plan_correction_error(self, rejected: str, error: ValidationError) -> str | None:
        errors = error.errors(include_context=False)
        if any(
            item["type"] != "literal_error" or item["loc"][0] != "tasks"
            or item["loc"][-1] not in {"role", "family"}
            for item in errors
        ):
            return None
        try:
            specs = tuple(TaskSpec.model_validate(item) for item in json.loads(rejected)["tasks"])
            # Ignore only proposal ownership; unsafe queries and other plan errors remain fatal.
            self.validate_initial_tasks(specs, check_ownership=False)
        except (ValueError, TypeError, KeyError):
            return None
        wrong = [(index, spec) for index, spec in enumerate(specs)
                 if spec.query.family not in ACCESS.get(spec.role, ())]
        return "\n".join(
            f"tasks[{index}].role ({spec.task_id}): {spec.role} cannot collect {spec.query.family}; owner is "
            + next(owner.value for owner, families in ACCESS.items() if spec.query.family in families)
            for index, spec in wrong
        ) or None

    def schema_order_correction_error(self, rejected: str, error: ValidationError, purpose: str):
        if purpose not in {"baseline", Role.REPORTING} or any(
            item["type"] != "value_error"
            or item["msg"] != "Value error, hypotheses must be ranked by descending confidence"
            for item in error.errors(include_context=False)
        ):
            return None
        data = json.loads(rejected)
        report = data.get("report") if purpose == "baseline" else data
        if purpose == "baseline" and (data["action"] != "finish" or data.get("queries")):
            return None
        if report["incident_id"] != self.context.incident_id:
            return None
        candidates = tuple(Hypothesis.model_validate(item) for item in data.get("hypotheses", ()))
        reported = tuple(Hypothesis.model_validate(item) for item in report["hypotheses"])
        if purpose == "baseline" and candidates != reported:
            return None
        data = data | {"hypotheses": [item.model_dump(mode="json") for item in candidates]}
        report = report | {"hypotheses": [item.model_dump(mode="json") for item in reported]}
        try:
            validate_all(
                lambda: self.check_references(data),
                lambda: self.check_references(report, path="report." if purpose == "baseline" else ""),
                lambda: self.validate_outcome(report, path="report." if purpose == "baseline" else ""),
            )
        except CorrectableResponseError as exc:
            return exc
        except ValueError:
            return None
        return None

    def validate_response_correction(
        self, rejected_json: str, rejected_output: Contract | None, corrected: Contract,
    ) -> None:
        if isinstance(corrected, InitialPlan):
            prior = {item["task_id"]: TaskSpec.model_validate(item)
                     for item in json.loads(rejected_json)["tasks"]}
            if set(prior) != {item.task_id for item in corrected.tasks} or any(
                self.query_key(item.query) != self.query_key(prior[item.task_id].query)
                for item in corrected.tasks
            ):
                raise ValueError("plan correction may fix ownership, not replace tasks or queries")
            return
        rejected_data = json.loads(rejected_json)
        prior = {item["hypothesis_id"]: Hypothesis.model_validate(item)
                 for item in rejected_data.get("hypotheses", ())}
        hypotheses = getattr(corrected, "hypotheses", ())
        if set(prior) != {item.hypothesis_id for item in hypotheses}:
            raise ValueError("correction cannot replace or delete a rejected candidate")
        for item in hypotheses:
            old = prior[item.hypothesis_id]
            if any(
                getattr(old, field) != getattr(item, field)
                for field in ("claim", "cause_category", "causal_stage")
            ) or any(
                set(getattr(old, field)) != set(getattr(item, field))
                for field in ("supporting_evidence_ids", "contradicting_evidence_ids")
            ):
                raise ValueError("correction must preserve causal claims and both evidence polarities")
            if not (set(old.blocking_uncertainties) | set(old.residual_uncertainties)) <= (
                set(item.blocking_uncertainties) | set(item.residual_uncertainties)
            ):
                raise ValueError("correction cannot silently delete uncertainties")
        def explanations(output):
            if isinstance(output, dict):
                return (
                    output.get("rationale", ""), *output.get("limitations", ()),
                    *(item["claim"] for item in output.get("findings", ())),
                    *(item["rationale"] for item in output.get("follow_ups", ())),
                    *(item["reason"] for item in output.get("objections", ())),
                )
            return (
                getattr(output, "rationale", ""),
                *getattr(output, "limitations", ()),
                *(item.claim for item in getattr(output, "findings", ())),
                *(item.rationale for item in getattr(output, "follow_ups", ())),
                *(item.reason for item in getattr(output, "objections", ())),
            )
        if not any(text.strip() for text in set(explanations(corrected)) - set(explanations(rejected_data))):
            raise ValueError("correction must explain its contract repair or uncertainty judgment")

    def failure(self, task_id, operation, code, message, retryable=False, attempt=0):
        self.failures.append(OperationFailure(
            task_id=task_id, operation=operation, code=code, message=message,
            retryable=retryable, attempt=attempt,
        ))

    def validate_revision(
        self, hypotheses: tuple[Hypothesis, ...], previous: tuple[Hypothesis, ...],
        explanation: tuple[str, ...],
    ) -> None:
        prior = {item.hypothesis_id: item for item in previous}
        changed = set(prior) - {item.hypothesis_id for item in hypotheses}
        for item in hypotheses:
            old = prior.get(item.hypothesis_id)
            if old is None:
                continue
            old_refs = set(old.supporting_evidence_ids) | set(old.contradicting_evidence_ids)
            new_refs = set(item.supporting_evidence_ids) | set(item.contradicting_evidence_ids)
            if not old_refs <= new_refs:
                raise ValueError("hypothesis revision dropped previously cited evidence")
            if (
                not set(old.supporting_evidence_ids) <= set(item.supporting_evidence_ids)
                or not set(old.contradicting_evidence_ids) <= set(item.contradicting_evidence_ids)
            ):
                changed.add(item.hypothesis_id)
        if changed and not any(text.strip() for text in explanation):
            raise ValueError("rejecting hypotheses or reclassifying evidence requires an explanation")

    async def model_batch(self, *operations: ModelOperation) -> dict[str, Contract | None]:
        """Reserve and commit on the coordinator; workers never see mutable state."""
        from agents import CORRECTION_INSTRUCTIONS, SCHEMAS, prompt

        for op in operations:
            self.state.add_tasks(TaskSpec(
                task_id=op.task_id, role=op.role, depends_on=op.depends_on,
                max_attempts=self.state.budget.operation_attempts,
            ))
            self.state.start_task(op.task_id)
        snapshot = self.state.snapshot()
        prompts = {
            op.task_id: prompt(
                op.purpose, op.task_id, self.context, snapshot, self.state.budget,
                extra=op.extra, role=op.role,
            )
            for op in operations
        }
        pending = list(operations)
        outputs: dict[str, Contract | None] = {op.task_id: None for op in operations}
        inflight: dict[str, tuple[ModelOperation, int]] = {}
        returned: dict[str, ModelResponse | LLMFailure] = {}
        corrections: dict[str, tuple[int, str, Contract | None]] = {}

        async def execute(op):
            system, user = prompts[op.task_id]
            try:
                response = await asyncio.wait_for(
                    self.llm.respond(
                        op.task_id, system, user, SCHEMAS[op.purpose],
                        self.state.budget.max_output_tokens,
                    ),
                    timeout=min(self.state.remaining_seconds, self.state.budget.llm_timeout_seconds),
                )
            except TimeoutError:
                response = LLMFailure(
                    "timeout", "LLM operation timed out", retryable=self.state.remaining_seconds > 0,
                )
            except LLMFailure as exc:
                response = exc
            except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
                response = LLMFailure("operation_failed", f"{type(exc).__name__}: {exc}")
            returned[op.task_id] = response
            return response

        try:
            while pending:
                dispatched = []
                for op in pending:
                    system, user = prompts[op.task_id]
                    # UTF-8 bytes upper-bound byte-BPE text tokens; include schema and overhead.
                    bound = (
                        len(system.encode()) + len(user.encode())
                        + len(json.dumps(structured_schema(SCHEMAS[op.purpose])).encode())
                        + 1024 + self.state.budget.max_output_tokens
                    )
                    try:
                        await self.state.reserve_model_call(op.task_id, bound)
                    except BudgetExceeded as exc:
                        self.failure(op.task_id, "llm", "budget_exhausted", str(exc))
                        self.state.finish_task(op.task_id, failure=str(exc))
                        continue
                    dispatched.append((op, bound))
                    inflight[op.task_id] = (op, bound)
                pending = []
                responses = await asyncio.gather(*(execute(op) for op, _ in dispatched))
                for (op, bound), response in zip(dispatched, responses):
                    inflight.pop(op.task_id)
                    returned.pop(op.task_id, None)
                    task = self.state.task(op.task_id)
                    error = response if isinstance(response, LLMFailure) else response.failure
                    measured = isinstance(response, ModelResponse)
                    output = None
                    correction_error = None
                    try:
                        self.state.settle_model_call(bound, response if measured else None)
                        if self.state.remaining_seconds <= 0 and error is None:
                            raise BudgetExceeded("investigation time budget exhausted")
                        if measured and error is None:
                            output = SCHEMAS[op.purpose].model_validate_json(response.response_json)
                            def identity():
                                if isinstance(output, AgentResult):
                                    violations = []
                                    if output.task_id != op.task_id:
                                        violations.append(f"task_id: model returned the wrong task or role; expected {op.task_id}")
                                    if output.role != op.role:
                                        violations.append(f"role: model returned the wrong task or role; expected {op.role}")
                                    if violations:
                                        raise ValueError("\n".join(violations))

                            validate_all(
                                identity,
                                lambda: self.check_references(output),
                                lambda: op.validate(output) if op.validate else None,
                            )
                            if op.task_id in corrections:
                                _, rejected, rejected_output = corrections[op.task_id]
                                self.validate_response_correction(rejected, rejected_output, output)
                            if self.state.remaining_seconds <= 0:
                                raise BudgetExceeded("investigation time budget exhausted")
                            self.state.record_model_output(op.task_id, output)
                    except (ValidationError, ValueError, BudgetExceeded) as exc:
                        if isinstance(exc, CorrectableResponseError):
                            correction_error = exc
                        elif (
                            isinstance(response, ModelResponse)
                            and isinstance(exc, ValidationError) and op.purpose == "plan"
                        ):
                            ownership_error = self.plan_correction_error(response.response_json, exc)
                            if ownership_error:
                                correction_error = CorrectableResponseError(ownership_error.splitlines())
                        elif isinstance(response, ModelResponse) and isinstance(exc, ValidationError):
                            correction_error = self.schema_order_correction_error(
                                response.response_json, exc, op.purpose,
                            )
                        error = LLMFailure(
                            "budget_exhausted" if isinstance(exc, BudgetExceeded) else "invalid_response",
                            "\n".join(validation_messages(exc)) if isinstance(exc, ValidationError) else str(exc),
                        )
                    system, user = prompts[op.task_id]
                    self.model_calls.append(ModelCallTrace(
                        task_id=op.task_id, role=op.purpose, attempt=task.attempts,
                        system_prompt=system, user_prompt=user,
                        response_schema=SCHEMAS[op.purpose].__name__,
                        response_json=response.response_json if measured else None,
                        error=str(error) if error else None, reserved_tokens=bound,
                        input_tokens=response.input_tokens if measured else 0,
                        output_tokens=response.output_tokens if measured else 0,
                        usage_measured=measured,
                        correction_of_attempt=corrections[op.task_id][0] if op.task_id in corrections else None,
                    ))
                    if error:
                        self.failure(op.task_id, "llm", error.code, str(error), error.retryable, task.attempts)
                        if (
                            isinstance(response, ModelResponse)
                            and correction_error and op.task_id not in corrections
                        ):
                            corrections[op.task_id] = (task.attempts, response.response_json, output)
                            payload = json.loads(user)
                            payload["response_correction"] = {
                                "rejected_response": response.response_json,
                                "validation_errors": list(correction_error.violations),
                                "response_contract": structured_schema(SCHEMAS[op.purpose]),
                            }
                            if correction_error.already_executed_queries:
                                payload["response_correction"]["already_executed_queries"] = list(
                                    correction_error.already_executed_queries,
                                )
                            prompts[op.task_id] = (
                                system + "\n\n" + CORRECTION_INSTRUCTIONS,
                                json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                            )
                            pending.append(op)
                        elif (
                            op.task_id not in corrections
                            and error.retryable and task.attempts < task.spec.max_attempts
                        ):
                            pending.append(op)
                        else:
                            self.state.finish_task(op.task_id, failure=str(error))
                    else:
                        self.state.finish_task(op.task_id)
                        outputs[op.task_id] = output
                if pending:
                    await asyncio.sleep(min(0.05, self.state.remaining_seconds))
        except asyncio.CancelledError:
            for op, bound in inflight.values():
                response = returned.get(op.task_id)
                measured = isinstance(response, ModelResponse)
                try:
                    self.state.settle_model_call(bound, response if measured else None)
                except BudgetExceeded as exc:
                    self.failure(op.task_id, "llm", "budget_exhausted", str(exc),
                                 attempt=self.state.task(op.task_id).attempts)
                system, user = prompts[op.task_id]
                self.model_calls.append(ModelCallTrace(
                    task_id=op.task_id, role=op.purpose,
                    attempt=self.state.task(op.task_id).attempts,
                    system_prompt=system, user_prompt=user,
                    response_schema=SCHEMAS[op.purpose].__name__,
                    response_json=response.response_json if measured else None,
                    error="LLM task cancelled", reserved_tokens=bound,
                    input_tokens=response.input_tokens if measured else 0,
                    output_tokens=response.output_tokens if measured else 0,
                    usage_measured=measured,
                    correction_of_attempt=corrections[op.task_id][0] if op.task_id in corrections else None,
                ))
            for op in operations:
                if self.state.task(op.task_id).status == "running":
                    self.failure(
                        op.task_id, "llm", "cancelled", "LLM task cancelled",
                        attempt=self.state.task(op.task_id).attempts,
                    )
                    self.state.finish_task(op.task_id, failure="LLM task cancelled")
            raise
        return outputs

    async def model(self, operation: ModelOperation):
        return (await self.model_batch(operation))[operation.task_id]

    async def evidence_tasks(
        self, specs: tuple[TaskSpec, ...], *, sequential=False, retry_keys=frozenset(),
    ):
        keys = [self.query_key(spec.query) for spec in specs]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate queries within a task batch")
        for spec in specs:
            key = self.query_key(spec.query)
            self.validate_query(spec.query, spec.role, retry=key in retry_keys)
        specs = tuple(TaskSpec.model_validate(spec.model_dump() | {
            "max_attempts": self.state.budget.operation_attempts - self._query_attempts.get(key, 0),
        }) for spec, key in zip(specs, keys))
        self.state.add_tasks(*specs)
        for spec in specs:
            self._seen_queries[self.query_key(spec.query)] = spec.task_id
        try:
            await run_tool_tasks(
                self.state, tuple(spec.task_id for spec in specs), self.tools, sequential=sequential,
            )
        finally:
            for spec in specs:
                task = self.state.task(spec.task_id)
                key = self.query_key(spec.query)
                self._query_attempts[key] = self._query_attempts.get(key, 0) + task.attempts
                for attempt, index in enumerate(task.result_indices, 1):
                    result = self.state.snapshot().tool_results[index]
                    if result.error:
                        self.failure(spec.task_id, "tool", result.error.code, result.error.message,
                                     result.error.retryable, min(attempt, task.attempts))

    def report_validation(self, report: InvestigationReport, hypotheses, *, path="") -> None:
        def unchanged():
            violations = []
            if report.incident_id != self.context.incident_id:
                violations.append(f"{path}incident_id: report belongs to another incident")
            prior = {item.hypothesis_id: item for item in hypotheses}
            for index, item in enumerate(report.hypotheses):
                if prior.get(item.hypothesis_id) != item:
                    violations.append(f"{path}hypotheses[{index}]: Reporting added or changed a Reasoning hypothesis")
                if report.outcome == "ranked" and not item.supporting_evidence_ids:
                    violations.append(f"{path}hypotheses[{index}].supporting_evidence_ids: a ranked hypothesis must cite supporting observations")
            if report.outcome == "ranked" and (not hypotheses or report.hypotheses[0] != hypotheses[0]):
                violations.append(f"{path}hypotheses[0]: Reporting must retain the top-ranked Reasoning hypothesis")
            if violations:
                raise ValueError("\n".join(violations))

        validate_all(
            lambda: self.check_references(report, path=path), unchanged,
            lambda: self.validate_outcome(report, path=path),
        )

    def report_limitations(self, hypotheses=()) -> tuple[str, ...]:
        values = list(self.limitations)
        if self.llm.mode == "mocked":
            values.append("Mocked execution: scripted responses, not real model reasoning")
        snapshot = self.state.snapshot()
        for result in snapshot.tool_results:
            if result.status == "partial":
                values.extend(f"{result.query.family}: {issue}" for issue in result.issues)
        for task in snapshot.tasks:
            if task.status in {"failed", "blocked"}:
                values.append(f"{task.spec.task_id}: {task.failure}")
        if hypotheses and (
            hypotheses[0].causal_stage != "initiating_cause" or hypotheses[0].blocking_uncertainties
        ):
            values.append(
                "The initiating cause remains unresolved; a failure mechanism is not a root-cause conclusion"
            )
        values.extend(question for item in hypotheses for question in item.blocking_uncertainties)
        values.extend(
            f"Residual uncertainty for {item.hypothesis_id}: {question}"
            for item in hypotheses for question in item.residual_uncertainties
        )
        return unique_limitations(values)

    def finalize(
        self, approach, report, termination_reason, *, force_inconclusive=False,
        fallback_hypotheses: tuple[Hypothesis, ...] = (),
    ):
        if self.state.remaining_seconds <= 0:
            force_inconclusive = True
            termination_reason = "budget_exhausted"
            self.failure("runtime", "runtime", "budget_exhausted", "Investigation deadline exhausted")
        self.limitations = list(self.report_limitations(
            report.hypotheses if report is not None else fallback_hypotheses,
        ))
        if report is None:
            force_inconclusive = True
            self.limitations.append("Final structured model report unavailable; runtime fallback, not model reasoning")
            self.limitations.append(
                f"Acquired evidence ledger contains {len(self.state.snapshot().evidence)} records; "
                "missing validated analysis/report is not a claim that evidence sources are unavailable"
            )
            report = InvestigationReport(
                incident_id=self.context.incident_id, outcome="inconclusive",
                hypotheses=fallback_hypotheses,
                limitations=tuple(self.limitations),
            )
        if report.hypotheses:
            top = report.hypotheses[0]
            if top.causal_stage != "initiating_cause" or top.blocking_uncertainties:
                force_inconclusive = True
        if force_inconclusive:
            self.limitations.append(f"Investigation terminated without a supported conclusion: {termination_reason}")
        hypotheses = report.hypotheses
        report = InvestigationReport.model_validate(report.model_dump() | {
            "outcome": "inconclusive" if force_inconclusive else report.outcome,
            "hypotheses": hypotheses,
            "limitations": unique_limitations((*report.limitations, *self.limitations)),
        })
        return InvestigationRun(
            report=report,
            trace=ExecutionTrace(
                approach=approach, execution_mode=self.llm.mode, model=self.llm.model,
                budget=self.state.budget, termination_reason=termination_reason,
                failures=tuple(self.failures), model_calls=tuple(self.model_calls),
                state=self.state.snapshot(),
            ),
        )


async def run_multi_agent(
    tools: SimulatedTools, llm: StructuredLLM, context: IncidentContext,
    budget: ExecutionBudget | None = None,
) -> InvestigationRun:
    coordinator = Coordinator(tools, llm, context, budget)
    try:
        return await _multi_agent(coordinator)
    except asyncio.CancelledError:
        return coordinator.finalize("multi-agent", None, "cancelled", force_inconclusive=True)


async def _multi_agent(coordinator: Coordinator) -> InvestigationRun:
    state = coordinator.state
    latest: AgentResult | None = None
    critic: AgentResult | None = None
    force_inconclusive = False
    termination = "analysis_unavailable"

    def validate_plan(plan: InitialPlan):
        coordinator.validate_initial_tasks(plan.tasks)

    plan = await coordinator.model(ModelOperation(
        "plan", Role.ORCHESTRATOR, "plan", validate=validate_plan,
    ))
    if plan is None:
        return coordinator.finalize("multi-agent", None, "planning_unavailable", force_inconclusive=True)
    specs = tuple(TaskSpec.model_validate(spec.model_dump() | {
        "depends_on": ("plan",), "max_attempts": state.budget.operation_attempts,
    }) for spec in plan.tasks)

    def successful(task_ids):
        return tuple(task_id for task_id in task_ids if state.task(task_id).status == "succeeded")

    async def summaries(round_index, batch):
        operations = []
        for role in ACCESS:
            relevant = tuple(spec.task_id for spec in batch if spec.role == role)
            if relevant:
                def validate_summary(output):
                    if output.hypotheses or output.follow_ups or output.objections:
                        raise ValueError("evidence specialists return findings and limitations only")
                operations.append(ModelOperation(
                    f"{role.value}-{round_index}", role, role,
                    depends_on=successful(relevant), validate=validate_summary,
                ))
        outputs = await coordinator.model_batch(*operations)
        for task_id, output in outputs.items():
            if output is None:
                coordinator.limitations.append(f"Evidence specialist analysis unavailable: {task_id}")
        return tuple(task_id for task_id, output in outputs.items() if output is not None)

    await coordinator.evidence_tasks(specs)
    summary_ids = await summaries(0, specs)
    if len(summary_ids) != 2:
        force_inconclusive = True
    for round_index in range(state.budget.max_follow_up_rounds + 1):
        reasoning_id = f"reasoning-{round_index}"

        def validate_reasoning(output):
            if output.objections or output.follow_ups:
                raise ValueError("Reasoning cannot schedule tools or return Critic objections")
            coordinator.validate_revision(
                output.hypotheses, latest.hypotheses if latest else (),
                (*output.limitations, *(item.claim for item in output.findings)),
            )

        revised = await coordinator.model(ModelOperation(
            reasoning_id, Role.REASONING, Role.REASONING,
            depends_on=summary_ids,
            extra={
                "current_hypotheses": [
                    item.model_dump(mode="json") for item in latest.hypotheses
                ] if latest else [],
                "critic_objections": [
                    item.model_dump(mode="json") for item in critic.objections
                ] if critic else [],
            },
            validate=validate_reasoning,
        ))
        if revised is None:
            force_inconclusive = True
            termination = "reasoning_unavailable"
            break
        latest = revised
        critic_id = f"critic-{round_index}"

        def validate_critic(output):
            if output.hypotheses:
                raise ValueError("Critic may object but cannot replace hypotheses")
            ids = {item.hypothesis_id for item in latest.hypotheses} | {"analysis"}
            if any(item.hypothesis_id not in ids for item in output.objections):
                raise ValueError("Critic objected to an unknown hypothesis")
            for item in output.objections:
                _validate_indices(item.follow_up_indices, len(output.follow_ups))
                if len(set(item.follow_up_indices)) != len(item.follow_up_indices):
                    raise ValueError("objection follow-up indices must be unique")
                if item.critical and not item.follow_up_indices and not (
                    item.unavailable_reason and item.unavailable_reason.strip()
                ):
                    raise ValueError("critical objection requires a targeted query or unavailable_reason")
            validate_all(*(lambda request=request, index=index: coordinator.validate_query(
                request.query, request.target_role, retry=request.retry_failed_query,
                path=f"follow_ups[{index}].query",
            ) for index, request in enumerate(output.follow_ups)))

        revised_critic = await coordinator.model(ModelOperation(
            critic_id, Role.CRITIC, Role.CRITIC, depends_on=(reasoning_id,),
            extra={
                "current_hypotheses": [
                    item.model_dump(mode="json") for item in latest.hypotheses
                ],
                "follow_up_rounds_used": round_index,
            },
            validate=validate_critic,
        ))
        if revised_critic is None:
            force_inconclusive = True
            termination = "critic_unavailable"
            break
        critic = revised_critic
        critical = [item for item in critic.objections if item.critical]
        if round_index >= state.budget.max_follow_up_rounds and critic.follow_ups:
            termination = "follow_up_round_limit"
            force_inconclusive |= bool(critical)
            if critic.follow_ups:
                coordinator.limitations.append(
                    "Critic follow-ups cannot be scheduled: follow-up round budget exhausted"
                )
            break

        def validate_decision(output):
            _validate_indices(output.selected_follow_ups, len(critic.follow_ups))
            deferred = {item.follow_up_index for item in output.deferred_follow_ups}
            _validate_indices(deferred, len(critic.follow_ups))
            if set(output.selected_follow_ups) | deferred != set(range(len(critic.follow_ups))):
                raise ValueError("every omitted Critic query requires an explicit deferral reason")
            if any(not item.reason.strip() for item in output.deferred_follow_ups):
                raise ValueError("deferral reason must explain why evidence cannot be investigated")

        decision_id = f"decision-{round_index}"
        if not critic.follow_ups:
            decision = ContinueDecision(
                action="finish",
                rationale="Runtime decision: current Critic follow-up list is empty; "
                "no indices can be selected or deferred. Unresolved critical objections remain inconclusive.",
            )
            state.add_tasks(TaskSpec(
                task_id=decision_id, role=Role.ORCHESTRATOR, depends_on=(critic_id,),
                max_attempts=state.budget.operation_attempts,
            ))
            state.start_task(decision_id)
            state.record_model_output(decision_id, decision)
            state.finish_task(decision_id)
        else:
            decision = await coordinator.model(ModelOperation(
                decision_id, Role.ORCHESTRATOR, "decide",
                depends_on=(critic_id,),
                extra={
                    "current_follow_ups": [
                        item.model_dump(mode="json") for item in critic.follow_ups
                    ],
                    "critical_objections": [
                        item.model_dump(mode="json") for item in critical
                    ],
                    "follow_up_rounds_used": round_index,
                },
                validate=validate_decision,
            ))
        if decision is None:
            termination = "continuation_decision_unavailable"
            force_inconclusive = True
            break
        coordinator.limitations.extend(
            f"Deferred Critic query {item.follow_up_index}: {item.reason}"
            for item in decision.deferred_follow_ups
        )
        if decision.action == "finish":
            termination = "orchestrator_finished"
            force_inconclusive |= bool(critical)
            break
        followups = []
        retry_keys = set()
        batch_keys = set()
        for index in decision.selected_follow_ups:
            request = critic.follow_ups[index]
            task_id = f"followup-{round_index + 1}-{index}"
            try:
                coordinator.validate_query(
                    request.query, request.target_role, retry=request.retry_failed_query,
                )
                key = coordinator.query_key(request.query)
                if key in batch_keys:
                    raise ValueError("Critic requested the same query twice")
                for dependency in request.depends_on:
                    if dependency not in {task.spec.task_id for task in state.snapshot().tasks}:
                        raise ValueError("follow-up dependency does not exist")
                    if state.task(dependency).status != "succeeded":
                        raise ValueError("follow-up depends on an unsuccessful task")
                followups.append(TaskSpec(
                    task_id=task_id, role=request.target_role, query=request.query,
                    depends_on=tuple(dict.fromkeys((
                        f"decision-{round_index}", *request.depends_on,
                    ))),
                    max_attempts=state.budget.operation_attempts,
                ))
                batch_keys.add(key)
                if request.retry_failed_query:
                    retry_keys.add(key)
            except ValueError as exc:
                coordinator.failure(task_id, "runtime", "rejected_follow_up", str(exc))
                coordinator.limitations.append(f"Rejected Critic follow-up: {exc}")
        if not followups:
            termination = "no_valid_follow_ups"
            force_inconclusive |= bool(critical)
            break
        await coordinator.evidence_tasks(tuple(followups), retry_keys=frozenset(retry_keys))
        new_summary_ids = await summaries(round_index + 1, tuple(followups))
        if not new_summary_ids:
            force_inconclusive = True
        summary_ids = (reasoning_id, critic_id, *new_summary_ids)

    hypotheses = latest.hypotheses if latest else ()
    if latest is None:
        coordinator.limitations.append(
            f"Validated Reasoning analysis unavailable; acquired evidence ledger contains "
            f"{len(state.snapshot().evidence)} records with source and coverage retained"
        )
    for analysis in (latest, critic):
        if analysis is not None:
            coordinator.limitations.extend(analysis.limitations)
    if critic is not None:
        for item in critic.objections:
            if item.critical:
                force_inconclusive = True
                coordinator.limitations.append(
                    f"Unresolved critical objection for {item.hypothesis_id}: {item.reason}"
                )
                if item.unavailable_reason:
                    coordinator.limitations.append(
                        f"Uninvestigable critical uncertainty: {item.unavailable_reason}"
                    )
    # Terminal failures are not ordinary success, even if other sources were available.
    force_inconclusive |= any(task.status == "failed" for task in state.snapshot().tasks)
    force_inconclusive |= not bool(hypotheses)
    force_inconclusive |= bool(hypotheses) and (
        hypotheses[0].causal_stage != "initiating_cause" or bool(hypotheses[0].blocking_uncertainties)
    )
    coordinator.limitations = list(unique_limitations(coordinator.limitations))
    report = await coordinator.model(ModelOperation(
        "report", Role.REPORTING, Role.REPORTING,
        depends_on=tuple(
            result.task_id for result in (latest, critic)
            if result is not None and state.task(result.task_id).status == "succeeded"
        ),
        extra={
            "current_hypotheses": [
                item.model_dump(mode="json") for item in hypotheses
            ],
            "critic_objections": [
                item.model_dump(mode="json") for item in critic.objections
            ] if critic else [],
            "termination_reason": termination,
            "must_be_inconclusive": force_inconclusive,
            "runtime_limitations": coordinator.report_limitations(hypotheses),
            "validated_analysis_available": latest is not None,
        },
        validate=lambda output: coordinator.report_validation(output, hypotheses),
    ))
    if report is None:
        termination = "reporting_unavailable"
    return coordinator.finalize(
        "multi-agent", report, termination, force_inconclusive=force_inconclusive,
        fallback_hypotheses=hypotheses,
    )


def _validate_indices(indices, length):
    if any(index >= length for index in indices):
        raise ValueError("Orchestrator selected an unknown Critic follow-up")

"""Validated contracts; timestamps and coverage windows are half-open [start, end)."""

from enum import StrEnum
from typing import Annotated, Generic, Literal, TypeVar, get_args

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Role(StrEnum):
    ORCHESTRATOR = "orchestrator"
    OBSERVABILITY = "observability"
    DEPENDENCY = "dependency"
    REASONING = "reasoning"
    CRITIC = "critic"
    REPORTING = "reporting"


class ToolFamily(StrEnum):
    LOGS = "logs"
    METRICS = "metrics"
    CHANGES = "changes"
    DEPENDENCIES = "dependencies"
    HISTORICAL_INCIDENTS = "historical_incidents"


class TimeRange(Contract):
    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def ordered(self):
        if self.start >= self.end:
            raise ValueError("time range must have start < end")
        return self


class SourceCoverage(Contract):
    time_range: TimeRange
    services: tuple[str, ...] = Field(min_length=1)
    gaps: tuple[TimeRange, ...] = ()

    @model_validator(mode="after")
    def valid_gaps(self):
        if len(set(self.services)) != len(self.services):
            raise ValueError("coverage services must be unique")
        previous_end = self.time_range.start
        for gap in self.gaps:
            if gap.start < previous_end or gap.end > self.time_range.end:
                raise ValueError("coverage gaps must be ordered, disjoint, and inside coverage")
            previous_end = gap.end
        return self


class Record(Contract):
    record_id: str = Field(pattern=r"^r[0-9]{2,}$")
    timestamp: AwareDatetime
    service: str = Field(min_length=1)


class LogRecord(Record):
    kind: Literal["logs"] = "logs"
    level: Literal["info", "warning", "error"]
    message: str
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)


class MetricRecord(Record):
    kind: Literal["metrics"] = "metrics"
    name: str
    value: float = Field(allow_inf_nan=False)
    unit: str


class ChangeRecord(Record):
    kind: Literal["changes"] = "changes"
    change_type: Literal["deployment", "configuration"]
    description: str
    before: str
    after: str


class DependencyRecord(Record):
    kind: Literal["dependencies"] = "dependencies"
    target_service: str
    relationship: Literal["sync", "async", "storage"]
    status: Literal["healthy", "degraded", "unknown"]
    observation: str


class HistoricalIncidentRecord(Record):
    kind: Literal["historical_incidents"] = "historical_incidents"
    summary: str
    symptoms: tuple[str, ...]
    resolution: str


EvidenceRecord = Annotated[
    LogRecord | MetricRecord | ChangeRecord | DependencyRecord | HistoricalIncidentRecord,
    Field(discriminator="kind"),
]
R = TypeVar("R", bound=Record)


class Dataset(Contract, Generic[R]):
    coverage: SourceCoverage
    records: tuple[R, ...]

    @model_validator(mode="after")
    def valid_records(self):
        ids = [record.record_id for record in self.records]
        if len(set(ids)) != len(ids):
            raise ValueError("record IDs must be unique within a dataset")
        for record in self.records:
            window = self.coverage.time_range
            if not window.start <= record.timestamp < window.end:
                raise ValueError("record timestamp lies outside source coverage")
            if record.service not in self.coverage.services:
                raise ValueError("record service is not covered")
            if any(gap.start <= record.timestamp < gap.end for gap in self.coverage.gaps):
                raise ValueError("record timestamp lies in an unobserved gap")
        return self


class IncidentContext(Contract):
    incident_id: str = Field(pattern=r"^inc-[0-9]{3}$")
    alert: str
    time_range: TimeRange
    affected_services: tuple[str, ...] = Field(min_length=1)


class IncidentData(Contract):
    context: IncidentContext
    logs: Dataset[LogRecord]
    metrics: Dataset[MetricRecord]
    changes: Dataset[ChangeRecord]
    dependencies: Dataset[DependencyRecord]
    historical_incidents: Dataset[HistoricalIncidentRecord]

    def dataset(self, family: ToolFamily) -> Dataset:
        return getattr(self, family.value)


class IncidentCatalog(Contract):
    schema_version: Literal[1] = 1
    incidents: tuple[IncidentData, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_incidents(self):
        ids = [incident.context.incident_id for incident in self.incidents]
        if len(set(ids)) != len(ids):
            raise ValueError("incident IDs must be unique")
        return self


class ToolQuery(Contract):
    incident_id: str = Field(pattern=r"^inc-[0-9]{3}$")
    family: ToolFamily
    time_range: TimeRange
    services: tuple[str, ...] = ()
    limit: int = Field(default=100, ge=1, le=500)

    @model_validator(mode="after")
    def unique_services(self):
        if any(not service.strip() for service in self.services):
            raise ValueError("query services must not be blank")
        if len(set(self.services)) != len(self.services):
            raise ValueError("query services must be unique")
        return self


class ObservabilityQuery(ToolQuery):
    family: Literal[ToolFamily.LOGS, ToolFamily.METRICS]


class DependencyQuery(ToolQuery):
    family: Literal[ToolFamily.CHANGES, ToolFamily.DEPENDENCIES, ToolFamily.HISTORICAL_INCIDENTS]


ACCESS: dict[Role, frozenset[ToolFamily]] = {
    Role.OBSERVABILITY: frozenset(get_args(ObservabilityQuery.model_fields["family"].annotation)),
    Role.DEPENDENCY: frozenset(get_args(DependencyQuery.model_fields["family"].annotation)),
}


class Evidence(Contract):
    evidence_id: str
    incident_id: str
    source: str
    record: EvidenceRecord

    @model_validator(mode="after")
    def stable_identity(self):
        expected = f"{self.incident_id}:{self.record.kind}:{self.record.record_id}"
        if self.evidence_id != expected:
            raise ValueError("evidence ID must identify its incident, family, and record")
        return self


class QueryCoverage(Contract):
    requested: TimeRange
    observed_ranges: tuple[TimeRange, ...] = ()
    covered_services: tuple[str, ...] = ()
    missing_services: tuple[str, ...] = ()
    gaps: tuple[TimeRange, ...] = ()
    matched_records: int = Field(default=0, ge=0)
    returned_records: int = Field(default=0, ge=0)
    truncated: bool = False

    @model_validator(mode="after")
    def consistent(self):
        if self.returned_records > self.matched_records:
            raise ValueError("returned count cannot exceed matched count")
        if self.truncated != (self.returned_records < self.matched_records):
            raise ValueError("truncation must reflect omitted matching records")
        if set(self.covered_services) & set(self.missing_services):
            raise ValueError("covered and missing services must be disjoint")
        for services in (self.covered_services, self.missing_services):
            if len(services) != len(set(services)):
                raise ValueError("coverage services must be unique")
        for ranges in (self.observed_ranges, self.gaps):
            previous_end = self.requested.start
            for window in ranges:
                if window.start < previous_end or window.end > self.requested.end:
                    raise ValueError("coverage ranges must be ordered and inside the request")
                previous_end = window.end
        for observed in self.observed_ranges:
            for gap in self.gaps:
                if observed.start < gap.end and gap.start < observed.end:
                    raise ValueError("observed ranges must not overlap gaps")
        cursor = self.requested.start
        for window in sorted((*self.observed_ranges, *self.gaps), key=lambda item: item.start):
            if window.start != cursor:
                raise ValueError("observed ranges and gaps must partition the request")
            cursor = window.end
        if cursor != self.requested.end:
            raise ValueError("coverage must account for the whole requested time range")
        return self


class ToolError(Contract):
    code: Literal[
        "unavailable", "unknown_incident", "timeout", "budget_exhausted", "cancelled",
        "operation_failed", "dependency_failed",
    ]
    message: str = Field(min_length=1)
    retryable: bool


class ToolResult(Contract):
    status: Literal["complete", "partial", "error"]
    source: str
    query: ToolQuery
    collected_at: AwareDatetime
    coverage: QueryCoverage
    evidence: tuple[Evidence, ...] = ()
    issues: tuple[str, ...] = ()
    error: ToolError | None = None

    @model_validator(mode="after")
    def consistent(self):
        if self.coverage.requested != self.query.time_range:
            raise ValueError("coverage must preserve the requested time range")
        if self.coverage.returned_records != len(self.evidence):
            raise ValueError("coverage count must equal the number of returned records")
        if len(self.evidence) > self.query.limit:
            raise ValueError("evidence cannot exceed the query limit")
        if len({item.evidence_id for item in self.evidence}) != len(self.evidence):
            raise ValueError("returned evidence IDs must be unique")
        if self.query.services and set(self.query.services) != (
            set(self.coverage.covered_services) | set(self.coverage.missing_services)
        ):
            raise ValueError("coverage must account for all requested services")
        for evidence in self.evidence:
            if (
                evidence.incident_id != self.query.incident_id
                or evidence.source != self.source
                or evidence.record.kind != self.query.family.value
            ):
                raise ValueError("evidence must match result provenance")
            record = evidence.record
            if not self.query.time_range.start <= record.timestamp < self.query.time_range.end:
                raise ValueError("evidence must lie in the requested time range")
            if record.service not in self.coverage.covered_services:
                raise ValueError("evidence service must be covered")
            if not any(
                window.start <= record.timestamp < window.end
                for window in self.coverage.observed_ranges
            ):
                raise ValueError("evidence must lie in an observed range")
        incomplete = bool(
            self.coverage.gaps or self.coverage.missing_services or self.coverage.truncated
        )
        if self.status == "error":
            if self.error is None or self.evidence or self.coverage.observed_ranges:
                raise ValueError("errors require details and cannot claim returned observations")
        elif self.error is not None:
            raise ValueError("non-error results cannot contain an error")
        elif self.status == "partial":
            if not incomplete or not self.issues:
                raise ValueError("partial results must explain their missing coverage")
        elif incomplete or self.issues or self.coverage.observed_ranges != (self.query.time_range,):
            raise ValueError("complete results must cover the full requested window")
        return self


class FailureRule(Contract):
    incident_id: str = Field(pattern=r"^inc-[0-9]{3}$")
    family: ToolFamily
    failures: int | None = Field(default=1, ge=1)
    retryable: bool = True


class SimulationSettings(Contract):
    latency_ms: int = Field(default=5, ge=0, le=10_000)
    failures: tuple[FailureRule, ...] = ()

    @model_validator(mode="after")
    def unique_rules(self):
        keys = [(rule.incident_id, rule.family) for rule in self.failures]
        if len(set(keys)) != len(keys):
            raise ValueError("failure rules must be unique per incident and family")
        return self


class Finding(Contract):
    claim: str
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()


class CauseCategory(StrEnum):
    UNKNOWN = "unknown"
    CONNECTION_CAPACITY = "connection_capacity"
    RESPONSE_CONTRACT = "response_contract"
    UPSTREAM_AVAILABILITY = "upstream_availability"
    MEMORY_CAPACITY = "memory_capacity"
    NAME_RESOLUTION = "name_resolution"
    QUEUE_DELIVERY = "queue_delivery"
    LOGGING_BACKPRESSURE = "logging_backpressure"
    TLS_VALIDATION = "tls_validation"
    REQUEST_CAPACITY = "request_capacity"


class Hypothesis(Finding):
    hypothesis_id: str
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    assumptions: tuple[str, ...] = ()
    cause_category: CauseCategory = CauseCategory.UNKNOWN
    causal_stage: Literal["initiating_cause", "mechanism", "unknown"] = "unknown"
    blocking_uncertainties: tuple[str, ...] = ()
    residual_uncertainties: tuple[str, ...] = ()


class FollowUpRequest(Contract):
    target_role: Role
    query: ToolQuery
    rationale: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    retry_failed_query: bool = False


class Objection(Contract):
    hypothesis_id: str
    reason: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    critical: bool = True
    follow_up_indices: tuple[Annotated[int, Field(ge=0)], ...] = ()
    unavailable_reason: str | None = Field(default=None, min_length=1)


class AgentResult(Contract):
    task_id: str
    role: Role
    findings: tuple[Finding, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    follow_ups: tuple[FollowUpRequest, ...] = ()
    limitations: tuple[str, ...] = ()
    objections: tuple[Objection, ...] = ()


class ObservabilityResult(AgentResult):
    role: Literal[Role.OBSERVABILITY]
    hypotheses: tuple[Hypothesis, ...] = Field(default=(), max_length=0)
    follow_ups: tuple[FollowUpRequest, ...] = Field(default=(), max_length=0)
    objections: tuple[Objection, ...] = Field(default=(), max_length=0)


class DependencyResult(AgentResult):
    role: Literal[Role.DEPENDENCY]
    hypotheses: tuple[Hypothesis, ...] = Field(default=(), max_length=0)
    follow_ups: tuple[FollowUpRequest, ...] = Field(default=(), max_length=0)
    objections: tuple[Objection, ...] = Field(default=(), max_length=0)


class ReasoningResult(AgentResult):
    role: Literal[Role.REASONING]
    follow_ups: tuple[FollowUpRequest, ...] = Field(default=(), max_length=0)
    objections: tuple[Objection, ...] = Field(default=(), max_length=0)


class CriticResult(AgentResult):
    role: Literal[Role.CRITIC]
    hypotheses: tuple[Hypothesis, ...] = Field(default=(), max_length=0)


class InvestigationReport(Contract):
    incident_id: str = Field(pattern=r"^inc-[0-9]{3}$")
    outcome: Literal["ranked", "inconclusive"]
    hypotheses: tuple[Hypothesis, ...] = ()
    recommended_actions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def ranked(self):
        errors = []

        def violation(path, message, value):
            errors.append({
                "type": "value_error", "loc": path, "input": value,
                "ctx": {"error": ValueError(message)},
            })

        if self.outcome == "ranked" and not self.hypotheses:
            violation(("hypotheses",), "a ranked report must include hypotheses", self.hypotheses)
        if self.outcome == "inconclusive" and not self.limitations:
            violation(("limitations",), "an inconclusive report must explain its limitations", self.limitations)
        seen = set()
        for index, item in enumerate(self.hypotheses):
            if index and item.confidence > self.hypotheses[index - 1].confidence:
                violation(("hypotheses", index, "confidence"),
                          "hypotheses must be ranked by descending confidence", item.confidence)
            if item.hypothesis_id in seen:
                violation(("hypotheses", index, "hypothesis_id"),
                          "hypothesis IDs must be unique", item.hypothesis_id)
            seen.add(item.hypothesis_id)
        if errors:
            raise ValidationError.from_exception_data(type(self).__name__, errors)
        return self


class ExecutionBudget(Contract):
    max_tool_calls: int = Field(default=30, ge=1)
    max_tokens: int = Field(default=120_000, ge=1)
    max_elapsed_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    tool_timeout_seconds: float = Field(default=2, gt=0, allow_inf_nan=False)
    max_parallel_tools: int = Field(default=5, ge=1)
    max_llm_calls: int = Field(default=24, ge=1)
    max_output_tokens: int = Field(default=1500, ge=100)
    llm_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    operation_attempts: int = Field(default=3, ge=1, le=3)
    max_follow_up_rounds: int = Field(default=2, ge=0, le=2)


class TaskSpec(Contract):
    task_id: str = Field(min_length=1)
    role: Role
    query: ToolQuery | None = None
    depends_on: tuple[str, ...] = ()
    max_attempts: int = Field(default=3, ge=1, le=5)


class ObservabilityTaskSpec(TaskSpec):
    role: Literal[Role.OBSERVABILITY]
    query: ObservabilityQuery


class DependencyTaskSpec(TaskSpec):
    role: Literal[Role.DEPENDENCY]
    query: DependencyQuery


class TaskState(Contract):
    spec: TaskSpec
    status: Literal["pending", "running", "succeeded", "failed", "blocked"] = "pending"
    attempts: int = Field(default=0, ge=0)
    result_indices: tuple[Annotated[int, Field(ge=0)], ...] = ()
    failure: str | None = None
    output_json: str | None = None


class RunUsage(Contract):
    tool_calls: int = Field(ge=0)
    tokens: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    unmeasured_token_reservations: int = Field(default=0, ge=0)


class InvestigationSnapshot(Contract):
    incident_id: str
    tasks: tuple[TaskState, ...]
    evidence: tuple[Evidence, ...]
    tool_results: tuple[ToolResult, ...]
    agent_results: tuple[AgentResult, ...]
    usage: RunUsage


class InitialPlan(Contract):
    rationale: str = Field(min_length=1)
    tasks: tuple[ObservabilityTaskSpec | DependencyTaskSpec, ...] = Field(min_length=2, max_length=10)


class FollowUpDeferral(Contract):
    follow_up_index: int = Field(ge=0)
    reason: str = Field(min_length=1)


class ContinueDecision(Contract):
    action: Literal["continue", "finish"]
    rationale: str = Field(min_length=1)
    selected_follow_ups: tuple[int, ...] = ()
    deferred_follow_ups: tuple[FollowUpDeferral, ...] = ()

    @model_validator(mode="after")
    def valid_selection(self):
        if any(index < 0 for index in self.selected_follow_ups):
            raise ValueError("follow-up indices must be nonnegative")
        if len(set(self.selected_follow_ups)) != len(self.selected_follow_ups):
            raise ValueError("follow-up indices must be unique")
        deferred = [item.follow_up_index for item in self.deferred_follow_ups]
        if len(set(deferred)) != len(deferred) or set(deferred) & set(self.selected_follow_ups):
            raise ValueError("deferred follow-ups must be unique and cannot also be selected")
        if (self.action == "continue") != bool(self.selected_follow_ups):
            raise ValueError("continue requires follow-ups; finish cannot schedule them")
        return self


class BaselineStep(Contract):
    action: Literal["investigate", "finish"]
    rationale: str = Field(min_length=1)
    queries: tuple[ToolQuery, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    report: InvestigationReport | None = None
    retry_failed_queries: bool = False

    @model_validator(mode="after")
    def valid_action(self):
        if self.action == "investigate":
            if not self.queries or self.report is not None:
                raise ValueError("investigate requires queries and no final report")
        elif self.queries or self.report is None:
            raise ValueError("finish requires a report and no queries")
        return self


class OperationFailure(Contract):
    task_id: str
    operation: Literal["tool", "llm", "runtime"]
    code: str
    message: str
    retryable: bool
    attempt: int = Field(ge=0)


class ModelCallTrace(Contract):
    task_id: str
    role: str
    attempt: int
    system_prompt: str
    user_prompt: str
    response_schema: str
    response_json: str | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reserved_tokens: int = 0
    usage_measured: bool = False
    correction_of_attempt: int | None = Field(default=None, ge=1)


class ExecutionTrace(Contract):
    approach: Literal["multi-agent", "baseline"]
    execution_mode: Literal["mocked", "live"]
    model: str
    budget: ExecutionBudget
    termination_reason: str
    failures: tuple[OperationFailure, ...]
    model_calls: tuple[ModelCallTrace, ...]
    state: InvestigationSnapshot


class InvestigationRun(Contract):
    report: InvestigationReport
    trace: ExecutionTrace

"""Five read-only simulated tools. No evaluator files are loaded here."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from models import (
    Evidence,
    IncidentCatalog,
    IncidentContext,
    QueryCoverage,
    SimulationSettings,
    TimeRange,
    ToolError,
    ToolFamily,
    ToolQuery,
    ToolResult,
)

DATA_DIR = Path(__file__).resolve().parent / "data"


def query_key(query: ToolQuery) -> str:
    """Canonical query identity shared by deduplication and fault injection."""
    payload = query.model_dump(mode="json")
    payload["services"] = sorted(query.services)
    payload["time_range"] = {
        "start": query.time_range.start.astimezone(timezone.utc).isoformat(),
        "end": query.time_range.end.astimezone(timezone.utc).isoformat(),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def source_name(family: ToolFamily) -> str:
    return f"synthetic:{family.value}:v1"


def error_result(query: ToolQuery, error: ToolError) -> ToolResult:
    return ToolResult(
        status="error",
        source=source_name(query.family),
        query=query,
        collected_at=datetime.now(timezone.utc),
        coverage=QueryCoverage(
            requested=query.time_range,
            missing_services=query.services,
            gaps=(query.time_range,),
        ),
        error=error,
    )


def intersection(left: TimeRange, right: TimeRange) -> TimeRange | None:
    start, end = max(left.start, right.start), min(left.end, right.end)
    return TimeRange(start=start, end=end) if start < end else None


def complement(window: TimeRange, ranges: tuple[TimeRange, ...]) -> tuple[TimeRange, ...]:
    """Return uncovered intervals, given ordered, disjoint intervals inside window."""
    cursor = window.start
    missing = []
    for covered in ranges:
        if cursor < covered.start:
            missing.append(TimeRange(start=cursor, end=covered.start))
        cursor = covered.end
    if cursor < window.end:
        missing.append(TimeRange(start=cursor, end=window.end))
    return tuple(missing)


class SimulatedTools:
    def __init__(
        self,
        catalog: IncidentCatalog,
        settings: SimulationSettings | None = None,
    ):
        self._incidents = {item.context.incident_id: item for item in catalog.incidents}
        self._settings = settings or SimulationSettings()
        self._rules = {
            (rule.incident_id, rule.family): rule for rule in self._settings.failures
        }
        if any(incident_id not in self._incidents for incident_id, _ in self._rules):
            raise ValueError("failure rule references an unknown incident")
        self._attempts: dict[str, int] = {}

    @classmethod
    def from_files(
        cls,
        incidents_path: Path = DATA_DIR / "incidents.json",
        simulation_path: Path = DATA_DIR / "simulation.json",
    ) -> "SimulatedTools":
        catalog = IncidentCatalog.model_validate_json(incidents_path.read_text(encoding="utf-8"))
        settings = SimulationSettings.model_validate_json(
            simulation_path.read_text(encoding="utf-8")
        )
        return cls(catalog, settings)

    def contexts(self) -> tuple[IncidentContext, ...]:
        return tuple(item.context for item in self._incidents.values())

    def context(self, incident_id: str) -> IncidentContext:
        return self._incidents[incident_id].context

    async def execute(self, query: ToolQuery) -> ToolResult:
        key = query_key(query)
        # Increment before the first await; concurrent asyncio calls cannot lose increments.
        attempt = self._attempts.get(key, 0) + 1
        self._attempts[key] = attempt
        await asyncio.sleep(self._settings.latency_ms / 1000)
        if query.incident_id not in self._incidents:
            return error_result(
                query,
                ToolError(
                    code="unknown_incident", message="Incident not found", retryable=False
                ),
            )
        rule = self._rules.get((query.incident_id, query.family))
        if rule and (rule.failures is None or attempt <= rule.failures):
            return error_result(
                query,
                ToolError(
                    code="unavailable",
                    message="Synthetic source is unavailable",
                    retryable=rule.retryable,
                ),
            )

        dataset = self._incidents[query.incident_id].dataset(query.family)
        requested_services = query.services or dataset.coverage.services
        covered_services = tuple(
            service for service in requested_services if service in dataset.coverage.services
        )
        missing_services = tuple(
            service for service in requested_services if service not in covered_services
        )
        available = complement(dataset.coverage.time_range, dataset.coverage.gaps)
        observed = tuple(
            overlap for window in available
            if (overlap := intersection(window, query.time_range)) is not None
        )
        if not covered_services:
            observed = ()
        gaps = complement(query.time_range, observed)
        matches = sorted(
            (
                record for record in dataset.records
                if record.service in covered_services
                and query.time_range.start <= record.timestamp < query.time_range.end
            ),
            key=lambda record: (record.timestamp, record.record_id),
        )
        selected = matches[:query.limit]
        issues = []
        if gaps:
            issues.append("Requested time range has unobserved intervals")
        if missing_services:
            issues.append("Requested services are not covered")
        if len(selected) < len(matches):
            issues.append("Result limit omitted matching records")
        source = source_name(query.family)
        return ToolResult(
            status="partial" if issues else "complete",
            source=source,
            query=query,
            collected_at=datetime.now(timezone.utc),
            coverage=QueryCoverage(
                requested=query.time_range,
                observed_ranges=observed,
                covered_services=covered_services,
                missing_services=missing_services,
                gaps=gaps,
                matched_records=len(matches),
                returned_records=len(selected),
                truncated=len(selected) < len(matches),
            ),
            evidence=tuple(
                Evidence(
                    evidence_id=f"{query.incident_id}:{query.family.value}:{record.record_id}",
                    incident_id=query.incident_id,
                    source=source,
                    record=record,
                )
                for record in selected
            ),
            issues=tuple(issues),
        )

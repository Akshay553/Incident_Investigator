import pytest
from pydantic import ValidationError

from models import (
    Hypothesis, InvestigationReport, QueryCoverage, SimulationSettings, TimeRange,
    ToolError, ToolQuery, ToolResult,
)
from tools import error_result


@pytest.mark.parametrize("start,end", [
    ("2026-01-15T10:00:00", "2026-01-15T10:30:00"),
    ("2026-01-15T10:30:00Z", "2026-01-15T10:00:00Z"),
    ("2026-01-15T10:00:00Z", "2026-01-15T10:00:00Z"),
])
def test_naive_reversed_and_empty_windows_are_rejected(start, end):
    with pytest.raises(ValidationError):
        TimeRange(start=start, end=end)


def test_queries_reject_invalid_limits_duplicates_and_blank_services(query):
    for changes in ({"limit": 0}, {"limit": 501}, {"services": ["x", "x"]}, {"services": [" "]}):
        with pytest.raises(ValidationError):
            ToolQuery.model_validate(query.model_dump() | changes)


def test_error_cannot_be_disguised_as_a_complete_empty_result(query):
    result = error_result(
        query, ToolError(code="unavailable", message="Unavailable", retryable=True)
    )
    with pytest.raises(ValidationError):
        ToolResult.model_validate(result.model_dump() | {"status": "complete", "error": None})
    with pytest.raises(ValidationError):
        ToolResult.model_validate(result.model_dump() | {"error": None})


def test_coverage_cannot_hide_unaccounted_time_or_counts(query):
    with pytest.raises(ValidationError, match="whole requested"):
        QueryCoverage(requested=query.time_range)
    with pytest.raises(ValidationError, match="count"):
        QueryCoverage(
            requested=query.time_range, observed_ranges=(query.time_range,),
            matched_records=1, returned_records=2,
        )
    with pytest.raises(ValidationError, match="truncation"):
        QueryCoverage(
            requested=query.time_range, observed_ranges=(query.time_range,),
            matched_records=2, returned_records=1, truncated=False,
        )


def test_ranked_reports_preserve_opposition_assumptions_and_confidence():
    primary = Hypothesis(
        hypothesis_id="h1", claim="A connection acquisition bottleneck",
        confidence=0.8, supporting_evidence_ids=("inc-001:logs:r01",),
        contradicting_evidence_ids=("inc-001:logs:r02",),
        assumptions=("Client and source timestamps are comparable",),
    )
    secondary = Hypothesis(hypothesis_id="h2", claim="Database CPU pressure", confidence=0.2)
    report = InvestigationReport(
        incident_id="inc-001", outcome="ranked", hypotheses=(primary, secondary),
        recommended_actions=("Check pool settings before changing database capacity",),
    )
    assert report.hypotheses[0].assumptions
    assert report.hypotheses[0].contradicting_evidence_ids
    with pytest.raises(ValidationError, match="descending"):
        InvestigationReport(
            incident_id="inc-001", outcome="ranked", hypotheses=(secondary, primary)
        )
    with pytest.raises(ValidationError):
        Hypothesis(hypothesis_id="h3", claim="Invalid", confidence=1.1)


def test_inconclusive_reports_require_explanation():
    report = InvestigationReport(
        incident_id="inc-009", outcome="inconclusive",
        limitations=("Retained evidence cannot distinguish candidate causes",),
        recommended_actions=("Collect dependency traces",),
    )
    assert report.hypotheses == ()
    with pytest.raises(ValidationError):
        InvestigationReport(incident_id="inc-009", outcome="inconclusive")
    with pytest.raises(ValidationError):
        InvestigationReport(incident_id="inc-001", outcome="ranked")


def test_runtime_failure_rules_are_validated():
    with pytest.raises(ValidationError):
        SimulationSettings.model_validate({"failures": [
            {"incident_id": "inc-001", "family": "logs", "failures": 0}
        ]})
    with pytest.raises(ValidationError, match="unique"):
        SimulationSettings.model_validate({"failures": [
            {"incident_id": "inc-001", "family": "logs"},
            {"incident_id": "inc-001", "family": "logs"},
        ]})

import asyncio
from datetime import timedelta

import pytest

from models import (
    FailureRule, SimulationSettings, TimeRange, ToolFamily, ToolQuery, ToolResult,
)
from tools import SimulatedTools, source_name


@pytest.mark.parametrize("incident_index", range(10))
@pytest.mark.parametrize("family", list(ToolFamily))
def test_all_families_preserve_query_source_coverage_and_typed_records(
    catalog, tools, incident_index, family
):
    incident = catalog.incidents[incident_index]
    dataset = incident.dataset(family)
    query = ToolQuery(
        incident_id=incident.context.incident_id,
        family=family,
        time_range=dataset.coverage.time_range,
    )
    result = asyncio.run(tools.execute(query))
    assert result.status == ("partial" if dataset.coverage.gaps else "complete")
    assert result.query == query
    assert result.source == source_name(family)
    assert result.collected_at.tzinfo is not None
    assert result.coverage.requested == query.time_range
    assert result.coverage.covered_services == dataset.coverage.services
    assert result.coverage.gaps == dataset.coverage.gaps
    assert result.coverage.returned_records == len(dataset.records)
    assert result.coverage.matched_records == len(dataset.records)
    assert not result.coverage.truncated
    assert result.error is None
    assert [item.record.record_id for item in result.evidence] == [
        item.record_id for item in sorted(
            dataset.records, key=lambda record: (record.timestamp, record.record_id)
        )
    ]
    assert all(item.record.kind == family.value for item in result.evidence)
    assert ToolResult.model_validate_json(result.model_dump_json()) == result


def test_service_and_half_open_time_filter(tools, query):
    narrow = ToolQuery(
        incident_id=query.incident_id,
        family=query.family,
        services=("checkout",),
        time_range=TimeRange(
            start=query.time_range.start + timedelta(minutes=6),
            end=query.time_range.start + timedelta(minutes=7),
        ),
    )
    result = asyncio.run(tools.execute(narrow))
    assert result.status == "complete"
    assert [item.record.record_id for item in result.evidence] == ["r01"]
    assert all(item.record.service == "checkout" for item in result.evidence)


def test_complete_empty_is_only_for_a_covered_query(tools, query):
    covered = ToolQuery(
        incident_id=query.incident_id, family=query.family, services=query.services,
        time_range=TimeRange(
            start=query.time_range.start,
            end=query.time_range.start + timedelta(minutes=1),
        ),
    )
    result = asyncio.run(tools.execute(covered))
    assert result.status == "complete"
    assert result.evidence == ()
    assert result.coverage.observed_ranges == (covered.time_range,)
    assert result.coverage.gaps == ()


def test_uncovered_service_is_partial_not_successful_empty(tools, query):
    query = query.model_copy(update={"services": ("not-collected",)})
    result = asyncio.run(tools.execute(query))
    assert result.status == "partial"
    assert result.evidence == ()
    assert result.coverage.observed_ranges == ()
    assert result.coverage.missing_services == ("not-collected",)
    assert result.coverage.gaps == (query.time_range,)
    assert result.issues


def test_mixed_service_coverage_returns_available_evidence(tools, query):
    query = query.model_copy(update={"services": ("checkout", "not-collected")})
    result = asyncio.run(tools.execute(query))
    assert result.status == "partial"
    assert result.evidence
    assert result.coverage.covered_services == ("checkout",)
    assert result.coverage.missing_services == ("not-collected",)


def test_outside_source_window_has_explicit_gaps(tools, query):
    wide = query.model_copy(update={
        "time_range": TimeRange(
            start=query.time_range.start - timedelta(minutes=1),
            end=query.time_range.end + timedelta(minutes=1),
        )
    })
    result = asyncio.run(tools.execute(wide))
    assert result.status == "partial"
    assert result.coverage.observed_ranges == (query.time_range,)
    assert len(result.coverage.gaps) == 2


def test_disjoint_time_window_is_partial_not_successful_empty(tools, query):
    future = query.model_copy(update={
        "time_range": TimeRange(
            start=query.time_range.end,
            end=query.time_range.end + timedelta(minutes=5),
        )
    })
    result = asyncio.run(tools.execute(future))
    assert result.status == "partial"
    assert result.evidence == ()
    assert result.coverage.observed_ranges == ()
    assert result.coverage.gaps == (future.time_range,)


def test_result_limit_is_partial_with_exact_truncation_counts(tools, query):
    result = asyncio.run(tools.execute(query.model_copy(update={"limit": 1})))
    assert result.status == "partial"
    assert len(result.evidence) == 1
    assert result.coverage.matched_records == 2
    assert result.coverage.returned_records == 1
    assert result.coverage.truncated
    assert result.coverage.gaps == ()


def test_fully_unobserved_gap_cannot_be_complete(catalog, tools):
    gap = catalog.incidents[5].logs.coverage.gaps[0]
    query = ToolQuery(
        incident_id="inc-006", family=ToolFamily.LOGS,
        services=("worker",), time_range=gap,
    )
    result = asyncio.run(tools.execute(query))
    assert result.status == "partial"
    assert result.evidence == ()
    assert result.coverage.observed_ranges == ()
    assert result.coverage.gaps == (gap,)


def test_missing_broker_metrics_are_explicit(catalog, tools):
    query = ToolQuery(
        incident_id="inc-006", family=ToolFamily.METRICS,
        services=("worker", "job-queue"), time_range=catalog.incidents[5].context.time_range,
    )
    result = asyncio.run(tools.execute(query))
    assert result.status == "partial"
    assert result.coverage.missing_services == ("job-queue",)
    assert result.evidence


def test_unknown_incident_is_a_nonretryable_error(tools, query):
    result = asyncio.run(tools.execute(query.model_copy(update={"incident_id": "inc-999"})))
    assert result.status == "error"
    assert result.error.code == "unknown_incident"
    assert not result.error.retryable
    assert result.evidence == ()
    assert result.coverage.observed_ranges == ()


def test_transient_failure_is_error_then_complete(catalog, query):
    tools = SimulatedTools(
        catalog,
        SimulationSettings(
            latency_ms=0,
            failures=(FailureRule(incident_id=query.incident_id, family=query.family),),
        ),
    )

    async def run():
        first, second = await asyncio.gather(tools.execute(query), tools.execute(query))
        assert first.status == "error"
        assert first.error.code == "unavailable"
        assert first.error.retryable
        assert first.coverage.observed_ranges == ()
        assert first.coverage.returned_records == 0
        assert first.evidence == ()
        assert second.status == "complete"
        assert second.evidence

    asyncio.run(run())


def test_persistent_failure_never_returns_successful_empty(catalog, query):
    tools = SimulatedTools(
        catalog,
        SimulationSettings(
            latency_ms=0,
            failures=(FailureRule(
                incident_id=query.incident_id, family=query.family,
                failures=None, retryable=False,
            ),),
        ),
    )
    for _ in range(3):
        result = asyncio.run(tools.execute(query))
        assert result.status == "error"
        assert result.error.message
        assert not result.error.retryable
        assert not result.evidence

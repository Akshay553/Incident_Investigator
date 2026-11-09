import asyncio
import json
import re

import pytest
from pydantic import ValidationError

from cli import collect
from models import IncidentCatalog, ToolFamily
from tools import DATA_DIR, SimulatedTools


def test_ten_neutral_incidents_and_unique_evidence(catalog):
    assert len(catalog.incidents) == 10
    assert {item.context.incident_id for item in catalog.incidents} == {
        f"inc-{number:03d}" for number in range(1, 11)
    }
    evidence_ids = []
    for incident in catalog.incidents:
        assert re.fullmatch(r"inc-\d{3}", incident.context.incident_id)
        assert incident.context.affected_services
        for family in ToolFamily:
            dataset = incident.dataset(family)
            assert dataset.coverage.services
            for record in dataset.records:
                assert record.kind == family.value
                evidence_ids.append(
                    f"{incident.context.incident_id}:{family.value}:{record.record_id}"
                )
        assert all(
            record.timestamp < incident.context.time_range.start
            for record in incident.historical_incidents.records
        )
    assert len(evidence_ids) == len(set(evidence_ids))


def test_evaluator_answers_are_separate_and_reference_real_evidence(catalog):
    truth = json.loads((DATA_DIR / "evaluation" / "ground_truth.json").read_text())
    cases = truth["cases"]
    assert truth["schema_version"] == 1
    assert len(cases) == len(catalog.incidents)
    assert len({case["incident_id"] for case in cases}) == len(cases)
    assert {case["incident_id"] for case in cases} == {
        item.context.incident_id for item in catalog.incidents
    }
    challenges = {tag for case in cases for tag in case["challenges"]}
    assert {
        "straightforward", "misleading_deployment", "conflicting_observations",
        "misleading_historical_similarity", "incomplete_data",
        "transient_tool_failure", "inconclusive",
    } <= challenges
    incidents = {item.context.incident_id: item for item in catalog.incidents}
    assert sum(case["outcome"] == "inconclusive" for case in cases) == 1
    for case in cases:
        incident = incidents[case["incident_id"]]
        valid_ids = {
            f"{case['incident_id']}:{family.value}:{record.record_id}"
            for family in ToolFamily
            for record in incident.dataset(family).records
        }
        for field in ("supporting_evidence_ids", "contradicting_evidence_ids"):
            assert set(case[field]) <= valid_ids
        if case["outcome"] == "resolved":
            assert case["root_cause"]["key"]
            assert case["root_cause"]["description"]
            assert case["supporting_evidence_ids"]
        else:
            assert case["outcome"] == "inconclusive"
            assert case["root_cause"] is None
            assert case["reason"]


def test_visible_data_and_runtime_settings_do_not_contain_evaluator_fields(catalog):
    forbidden = {"root_cause", "challenges", "ground_truth", "expected_answer", "scenario_title"}

    def walk(value):
        if isinstance(value, dict):
            assert not forbidden & value.keys()
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for filename in ("incidents.json", "simulation.json"):
        walk(json.loads((DATA_DIR / filename).read_text()))
    truth = json.loads((DATA_DIR / "evaluation" / "ground_truth.json").read_text())
    public_context = json.dumps([item.context.model_dump(mode="json") for item in catalog.incidents])
    runtime = (DATA_DIR / "simulation.json").read_text()
    for case in truth["cases"]:
        if case["root_cause"]:
            assert case["root_cause"]["key"] not in public_context + runtime


def test_tools_and_collector_work_without_any_evaluator_file(tmp_path, monkeypatch):
    for filename in ("incidents.json", "simulation.json"):
        (tmp_path / filename).write_text(
            (DATA_DIR / filename).read_text(encoding="utf-8"), encoding="utf-8"
        )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    tools = SimulatedTools.from_files(tmp_path / "incidents.json", tmp_path / "simulation.json")
    assert not (tmp_path / "evaluation").exists()
    for context in tools.contexts():
        snapshot = asyncio.run(collect(tools, context))
        assert len(snapshot.tasks) == 5
        assert all(task.status == "succeeded" for task in snapshot.tasks)
        assert snapshot.usage.tokens == 0
        assert snapshot.agent_results == ()
        for result in snapshot.tool_results:
            assert "root_cause" not in result.model_dump_json()
            assert "challenges" not in result.model_dump_json()


def test_invalid_catalog_is_rejected_instead_of_defaulting_to_empty(tmp_path):
    incidents = tmp_path / "incidents.json"
    incidents.write_text('{"schema_version": 1, "incidents": []}')
    with pytest.raises(ValidationError):
        SimulatedTools.from_files(incidents, DATA_DIR / "simulation.json")
    with pytest.raises(FileNotFoundError):
        SimulatedTools.from_files(tmp_path / "missing.json", DATA_DIR / "simulation.json")


def test_duplicate_incident_and_extra_diagnostic_field_are_rejected(catalog):
    payload = catalog.model_dump(mode="json")
    payload["incidents"].append(payload["incidents"][0])
    with pytest.raises(ValidationError, match="incident IDs"):
        IncidentCatalog.model_validate(payload)
    payload = catalog.model_dump(mode="json")
    payload["incidents"][0]["context"]["root_cause"] = "should not be visible"
    with pytest.raises(ValidationError, match="Extra inputs"):
        IncidentCatalog.model_validate(payload)

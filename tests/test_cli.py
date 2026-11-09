import json
import subprocess
import sys
from pathlib import Path

import pytest

from models import ExecutionTrace, InvestigationReport

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("approach,fixture", [
    ("multi-agent", "multi_agent"), ("baseline", "baseline"),
])
def test_mock_cli_saves_report_and_trace_and_preserves_existing_outputs(tmp_path, approach, fixture):
    command = [
        sys.executable, str(ROOT / "cli.py"), "run", approach, "inc-001",
        "--mock-script", str(ROOT / "tests" / "fixtures" / f"{fixture}.json"),
        "--output-dir", str(tmp_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "MOCKED" in result.stdout
    report = InvestigationReport.model_validate_json((tmp_path / "report.json").read_text())
    trace = ExecutionTrace.model_validate_json((tmp_path / "trace.json").read_text())
    assert report.outcome == "ranked"
    assert trace.execution_mode == "mocked"
    original = (tmp_path / "trace.json").read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert "already exist" in second.stderr
    assert (tmp_path / "trace.json").read_bytes() == original


def test_cli_requires_paid_authorization_before_provider_access(tmp_path):
    result = subprocess.run([
        sys.executable, str(ROOT / "cli.py"), "run", "multi-agent", "inc-001",
        "--output-dir", str(tmp_path),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "--allow-paid" in result.stderr
    assert not (tmp_path / "report.json").exists()


def test_inconclusive_cli_saves_failure_and_returns_distinct_status(tmp_path):
    mock = tmp_path / "script.json"
    mock.write_text(json.dumps({"plan": ["not structured"]}))
    result = subprocess.run([
        sys.executable, str(ROOT / "cli.py"), "run", "multi-agent", "inc-001",
        "--output-dir", str(tmp_path / "output"), "--mock-script", str(mock),
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert "inconclusive" in result.stdout
    trace = json.loads((tmp_path / "output" / "trace.json").read_text())
    assert trace["failures"][0]["code"] == "invalid_response"

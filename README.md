# Incident Investigator

A Python incident-investigation system that compares a six-role multi-agent
workflow with an iterative single-agent baseline. Both investigate ten synthetic
incidents using five read-only simulated tools and produce evidence-backed
hypotheses, recommendations, and execution traces.

## Setup

Requires Python 3.11+. Run from the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Keep the `data` directory beside the source files. Virtual environments,
credentials, and generated outputs should not be committed.

## Offline usage

```powershell
.\.venv\Scripts\python.exe cli.py list
.\.venv\Scripts\python.exe cli.py collect inc-007
.\.venv\Scripts\python.exe cli.py run multi-agent inc-001 --mock-script tests\fixtures\multi_agent.json --output-dir outputs\demo-multi
.\.venv\Scripts\python.exe cli.py run baseline inc-001 --mock-script tests\fixtures\baseline.json --output-dir outputs\demo-baseline
.\.venv\Scripts\python.exe -m pytest
```

Mock scripts exercise the workflow without model requests; they do not measure
live reasoning quality.

## Live investigation and evaluation

Configure `OPENAI_API_KEY` locally; never put credentials in source or chat.
Live commands require `--allow-paid` and spend API credits. These examples are
not authorization to run.

```powershell
.\.venv\Scripts\python.exe cli.py run multi-agent inc-001 --allow-paid --model gpt-4.1-mini --output-dir outputs\live-multi
.\.venv\Scripts\python.exe cli.py run baseline inc-001 --allow-paid --model gpt-4.1-mini --output-dir outputs\live-baseline
.\.venv\Scripts\python.exe evaluate.py --mode live --allow-paid --model gpt-4.1-mini --repeats 1 --output-dir outputs\live-evaluation
```

The evaluator runs both approaches across all ten incidents. Always choose a
fresh output directory. Investigations save `report.json` and `trace.json`;
evaluation also saves aggregate JSON/CSV, a summary, and a manifest.

An inconclusive report is not necessarily an execution failure; inspect the
trace's termination reason. Tools are read-only, and recommendations are
advisory: no production changes are executed.

## Documentation

- [Evaluation results](EVALUATION-RESULTS.md): approach comparison, per-incident outcomes, and scoring limitations.
- [Technical note](TECHNICAL-NOTE.md): architecture, coordination, and design tradeoffs.

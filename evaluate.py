"""Small, deterministic evaluator. Private rubrics never enter agent prompts."""

import argparse
import asyncio
import csv
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import Callable, Literal

from pydantic import Field, model_validator

from baseline import run_baseline
from investigation import run_multi_agent
from llm import OpenAILLM, ScriptedLLM, StructuredLLM
from models import (
    CauseCategory, Contract, ExecutionBudget, IncidentCatalog, IncidentContext,
    InvestigationRun, SimulationSettings,
)
from tools import DATA_DIR, SimulatedTools

APPROACHES = ("multi-agent", "baseline")
FAILURE_POLICY = "normalized-query-first-n-attempts-v1"


class FactRule(Contract):
    fact_id: str
    patterns: tuple[str, ...] = Field(min_length=1)
    reject_patterns: tuple[str, ...] = ()
    supporting_evidence_sets: tuple[tuple[str, ...], ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def valid(self):
        for pattern in (*self.patterns, *self.reject_patterns):
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"invalid fact pattern: {exc}") from exc
        if any(not group or len(set(group)) != len(group) for group in self.supporting_evidence_sets):
            raise ValueError("fact witness sets must be nonempty and contain unique IDs")
        return self

    def matches(self, claim: str) -> bool:
        return (
            all(re.search(pattern, claim, re.IGNORECASE) for pattern in self.patterns)
            and not any(re.search(pattern, claim, re.IGNORECASE) for pattern in self.reject_patterns)
        )


class CaseRubric(Contract):
    incident_id: str
    expected_category: CauseCategory
    required_fact_ids: tuple[str, ...]
    facts: tuple[FactRule, ...]
    abstention_patterns: tuple[str, ...] = ()

    @model_validator(mode="after")
    def valid(self):
        ids = [fact.fact_id for fact in self.facts]
        if len(set(ids)) != len(ids) or not set(self.required_fact_ids) <= set(ids):
            raise ValueError("rubric fact IDs must be unique and required facts must exist")
        for pattern in self.abstention_patterns:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"invalid abstention pattern: {exc}") from exc
        return self


class ScoringRubric(Contract):
    version: str
    cases: tuple[CaseRubric, ...]


class ExpectedCase(Contract):
    incident_id: str
    outcome: Literal["resolved", "inconclusive"]
    root_cause: dict[str, str] | None
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    challenges: tuple[str, ...]
    reason: str | None = None

    @model_validator(mode="after")
    def valid(self):
        if (self.outcome == "resolved") != bool(self.root_cause):
            raise ValueError("answerable cases require a root cause; inconclusive cases do not")
        return self


class GroundTruth(Contract):
    schema_version: Literal[1]
    cases: tuple[ExpectedCase, ...]


class EvaluationConfig(Contract):
    mode: Literal["live", "mocked"]
    model: str = "gpt-4.1-mini"
    temperature: float = Field(default=0, ge=0, le=2, allow_inf_nan=False)
    repeats: int = Field(default=1, ge=1, le=100)
    incident_ids: tuple[str, ...] = ()
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)

    @model_validator(mode="after")
    def valid(self):
        if len(set(self.incident_ids)) != len(self.incident_ids):
            raise ValueError("incident selection must be unique")
        if self.mode == "mocked" and self.model != "scripted-test-double":
            raise ValueError("mocked evaluations must identify the scripted model, not a real LLM")
        return self


class GroundingScore(Contract):
    citation_count: int = 0
    existing_citations: int = 0
    available_citations: int = 0
    supporting_citation_count: int = 0
    support_relationship_matches: int = 0
    contradicting_citation_count: int = 0
    contradiction_relationship_matches: int = 0
    recognized_fact_count: int = 0
    witnessed_fact_count: int = 0
    claims_without_rubric_match: int = 0
    citation_existence: float | None = None
    citation_availability: float | None = None
    support_relationship_precision: float | None = None
    contradiction_relationship_precision: float | None = None
    rubric_fact_support: float | None = None


class RunMetrics(Contract):
    answerable: bool
    top_ranked_correct: bool | None
    abstained: bool | None
    intentionally_inconclusive_handled: bool | None
    top_category: CauseCategory | None
    top_matched_fact_ids: tuple[str, ...] = ()
    grounding: GroundingScore
    tool_calls: int | None
    llm_calls: int | None
    wall_clock_seconds: float = Field(ge=0, allow_inf_nan=False)
    provider_input_tokens: int | None
    provider_output_tokens: int | None
    provider_usage_complete: bool
    unmeasured_llm_calls: int | None
    unmeasured_token_reservations: int | None
    charged_tokens: int | None
    had_failure: bool
    terminal_failure: bool
    failure_count: int
    budget_exhausted: bool
    status: Literal["completed", "completed_with_failures", "failed", "budget_exhausted", "cancelled"]
    termination_reason: str


class EvaluationRow(Contract):
    run_id: str
    incident_id: str
    approach: Literal["multi-agent", "baseline"]
    repetition: int
    mode: Literal["live", "mocked"]
    metrics: RunMetrics
    error: str | None = None
    artifact_directory: str | None = None


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evidence_index(catalog: IncidentCatalog, incident_id: str):
    incident = next(item for item in catalog.incidents if item.context.incident_id == incident_id)
    return {
        f"{incident_id}:{family}:{record.record_id}": record
        for family in ("logs", "metrics", "changes", "dependencies", "historical_incidents")
        for record in getattr(incident, family).records
    }


def validate_inputs(catalog: IncidentCatalog, truth: GroundTruth, rubric: ScoringRubric):
    expected_ids = {item.context.incident_id for item in catalog.incidents}
    for collection in (truth.cases, rubric.cases):
        ids = [item.incident_id for item in collection]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise ValueError("catalog, ground truth and rubric must cover exactly the same unique incidents")
    truth_by_id = {item.incident_id: item for item in truth.cases}
    for case in rubric.cases:
        expected = truth_by_id[case.incident_id]
        if expected.outcome == "resolved":
            if not case.required_fact_ids or case.expected_category == CauseCategory.UNKNOWN:
                raise ValueError("answerable rubric requires a category and distinguishing facts")
        elif case.required_fact_ids or case.expected_category != CauseCategory.UNKNOWN or not case.abstention_patterns:
            raise ValueError("inconclusive rubric needs abstention patterns and no expected cause")
        known = evidence_index(catalog, case.incident_id).keys()
        refs = set(expected.supporting_evidence_ids) | set(expected.contradicting_evidence_ids)
        for fact in case.facts:
            refs.update(fact.contradicting_evidence_ids)
            refs.update(item for group in fact.supporting_evidence_sets for item in group)
        if not refs <= known:
            raise ValueError("evaluator rubric references nonexistent or cross-incident evidence")


def grounding_score(run: InvestigationRun, rubric: CaseRubric, known: dict) -> GroundingScore:
    observed = {
        item.evidence_id for item in run.trace.state.evidence
        if known.get(item.evidence_id) == item.record
    }
    counts = {
        "citation_count": 0, "existing_citations": 0, "available_citations": 0,
        "supporting_citation_count": 0, "support_relationship_matches": 0,
        "contradicting_citation_count": 0, "contradiction_relationship_matches": 0,
        "recognized_fact_count": 0, "witnessed_fact_count": 0,
        "claims_without_rubric_match": 0,
    }
    for hypothesis in run.report.hypotheses:
        facts = [fact for fact in rubric.facts if fact.matches(hypothesis.claim)]
        counts["claims_without_rubric_match"] += not bool(facts)
        counts["recognized_fact_count"] += len(facts)
        support = set(hypothesis.supporting_evidence_ids)
        opposition = set(hypothesis.contradicting_evidence_ids)
        # A citation is counted once per hypothesis and polarity; repeats cannot inflate it.
        references = [(item, True) for item in support] + [(item, False) for item in opposition]
        for reference, is_support in references:
            counts["citation_count"] += 1
            counts["existing_citations"] += reference in known
            counts["available_citations"] += reference in observed
            total = "supporting_citation_count" if is_support else "contradicting_citation_count"
            matched = "support_relationship_matches" if is_support else "contradiction_relationship_matches"
            counts[total] += 1
            allowed = (
                {item for fact in facts for group in fact.supporting_evidence_sets for item in group}
                if is_support else {item for fact in facts for item in fact.contradicting_evidence_ids}
            )
            counts[matched] += reference in allowed and reference in observed
        for fact in facts:
            if fact.supporting_evidence_sets:
                counts["witnessed_fact_count"] += any(
                    set(group) <= support & observed for group in fact.supporting_evidence_sets
                )
    return GroundingScore(
        **counts,
        citation_existence=ratio(counts["existing_citations"], counts["citation_count"]),
        citation_availability=ratio(counts["available_citations"], counts["citation_count"]),
        support_relationship_precision=ratio(counts["support_relationship_matches"], counts["supporting_citation_count"]),
        contradiction_relationship_precision=ratio(counts["contradiction_relationship_matches"], counts["contradicting_citation_count"]),
        rubric_fact_support=ratio(counts["witnessed_fact_count"], counts["recognized_fact_count"]),
    )


def score_run(
    run: InvestigationRun | None, expected: ExpectedCase, rubric: CaseRubric,
    catalog: IncidentCatalog, *, elapsed: float, error: str | None = None,
) -> RunMetrics:
    answerable = expected.outcome == "resolved"
    if run is None:
        timed_out = bool(error and error.startswith("TimeoutError:"))
        return RunMetrics(
            answerable=answerable, top_ranked_correct=False if answerable else None,
            abstained=None, intentionally_inconclusive_handled=False if not answerable else None,
            top_category=None, grounding=GroundingScore(), tool_calls=None, llm_calls=None,
            wall_clock_seconds=elapsed, provider_input_tokens=None, provider_output_tokens=None,
            provider_usage_complete=False, unmeasured_llm_calls=None, unmeasured_token_reservations=None,
            charged_tokens=None, had_failure=True, terminal_failure=True, failure_count=1,
            budget_exhausted=timed_out, status="budget_exhausted" if timed_out else "failed",
            termination_reason="evaluation_run_error",
        )
    if run.report.incident_id != expected.incident_id or run.trace.state.incident_id != expected.incident_id:
        raise ValueError("run artifacts belong to a different incident")
    trace, report = run.trace, run.report
    had_failure = bool(trace.failures) or error is not None
    exhausted = (
        any(item.code == "budget_exhausted" for item in trace.failures)
        or trace.termination_reason == "baseline_step_limit"
        or trace.state.usage.tokens > trace.budget.max_tokens
        or elapsed >= trace.budget.max_elapsed_seconds
    )
    cancelled = trace.termination_reason == "cancelled"
    terminal = (
        exhausted or cancelled or error is not None
        or any(task.status in {"failed", "blocked", "pending", "running"} for task in trace.state.tasks)
        or trace.termination_reason.endswith("_unavailable")
    )
    top = report.hypotheses[0] if report.hypotheses else None
    matched = tuple(fact.fact_id for fact in rubric.facts if top and fact.matches(top.claim))
    abstained = report.outcome == "inconclusive"
    correct = (
        not abstained and not terminal and top is not None
        and top.cause_category == rubric.expected_category
        and set(rubric.required_fact_ids) <= set(matched)
    )
    handled = (
        abstained and not terminal and any(
            re.search(pattern, " ".join(report.limitations), re.IGNORECASE)
            for pattern in rubric.abstention_patterns
        )
    )
    calls = trace.model_calls
    live = trace.execution_mode == "live"
    unknown_calls = sum(not item.usage_measured for item in calls)
    if len(calls) != trace.state.usage.llm_calls:
        raise ValueError("model call count does not match the trace")
    if sum(task.attempts for task in trace.state.tasks if task.spec.query) != trace.state.usage.tool_calls:
        raise ValueError("tool call count does not match task attempts")
    measured_input = sum(item.input_tokens for item in calls if item.usage_measured)
    measured_output = sum(item.output_tokens for item in calls if item.usage_measured)
    if (measured_input, measured_output) != (
        trace.state.usage.input_tokens, trace.state.usage.output_tokens
    ):
        raise ValueError("provider usage counters do not match call traces")
    has_reported_usage = live and (not calls or any(item.usage_measured for item in calls))
    return RunMetrics(
        answerable=answerable, top_ranked_correct=bool(correct) if answerable else None,
        abstained=abstained, intentionally_inconclusive_handled=bool(handled) if not answerable else None,
        top_category=top.cause_category if top else None, top_matched_fact_ids=matched,
        grounding=grounding_score(run, rubric, evidence_index(catalog, expected.incident_id)),
        tool_calls=trace.state.usage.tool_calls, llm_calls=trace.state.usage.llm_calls,
        wall_clock_seconds=elapsed,
        provider_input_tokens=measured_input if has_reported_usage else None,
        provider_output_tokens=measured_output if has_reported_usage else None,
        provider_usage_complete=live and unknown_calls == 0,
        unmeasured_llm_calls=unknown_calls if live else None,
        unmeasured_token_reservations=trace.state.usage.unmeasured_token_reservations,
        charged_tokens=trace.state.usage.tokens,
        had_failure=had_failure or terminal, terminal_failure=terminal,
        failure_count=len(trace.failures) + (error is not None), budget_exhausted=exhausted,
        status=(
            "budget_exhausted" if exhausted else "cancelled" if cancelled else "failed" if terminal
            else "completed_with_failures" if had_failure else "completed"
        ),
        termination_reason=trace.termination_reason,
    )


def distribution(values):
    present = [value for value in values if value is not None]
    return {
        "n": len(present), "missing": len(values) - len(present),
        "mean": mean(present) if present else None,
        "sample_stddev": stdev(present) if len(present) > 1 else None,
        "min": min(present) if present else None, "max": max(present) if present else None,
    }


def summarize(rows: list[EvaluationRow]):
    comparison = {}
    for approach in APPROACHES:
        selected = [row for row in rows if row.approach == approach]
        repetitions = []
        for repetition in sorted({row.repetition for row in selected}):
            batch = [row.metrics for row in selected if row.repetition == repetition]
            answerable = [item for item in batch if item.answerable]
            intentional = [item for item in batch if not item.answerable]
            grounding = [item.grounding for item in batch]
            repetitions.append({
                "repetition": repetition, "runs": len(batch),
                "answerable_runs": len(answerable),
                "citation_count": sum(item.citation_count for item in grounding),
                "supporting_citation_count": sum(item.supporting_citation_count for item in grounding),
                "contradicting_citation_count": sum(item.contradicting_citation_count for item in grounding),
                "recognized_fact_count": sum(item.recognized_fact_count for item in grounding),
                "runs_without_citations": sum(item.citation_count == 0 for item in grounding),
                "top_ranked_root_cause_accuracy": ratio(sum(item.top_ranked_correct is True for item in answerable), len(answerable)),
                "abstention_frequency": ratio(sum(item.abstained is True for item in batch), len(batch)),
                "missing_reports": sum(item.abstained is None for item in batch),
                "inconclusive_handling_rate": ratio(sum(item.intentionally_inconclusive_handled is True for item in intentional), len(intentional)),
                "citation_existence": ratio(sum(item.existing_citations for item in grounding), sum(item.citation_count for item in grounding)),
                "citation_availability": ratio(sum(item.available_citations for item in grounding), sum(item.citation_count for item in grounding)),
                "support_relationship_precision": ratio(sum(item.support_relationship_matches for item in grounding), sum(item.supporting_citation_count for item in grounding)),
                "contradiction_relationship_precision": ratio(sum(item.contradiction_relationship_matches for item in grounding), sum(item.contradicting_citation_count for item in grounding)),
                "rubric_fact_support": ratio(sum(item.witnessed_fact_count for item in grounding), sum(item.recognized_fact_count for item in grounding)),
                "terminal_failure_frequency": ratio(sum(item.terminal_failure for item in batch), len(batch)),
            })
        comparison[approach] = {
            "runs": len(selected), "failures_included": sum(row.metrics.had_failure for row in selected),
            "terminal_failures": sum(row.metrics.terminal_failure for row in selected),
            "budget_exhaustions": sum(row.metrics.budget_exhausted for row in selected),
            "per_repetition": repetitions,
            "repetition_variability": {
                key: distribution([item[key] for item in repetitions])
                for key in (
                    "top_ranked_root_cause_accuracy", "abstention_frequency", "inconclusive_handling_rate",
                    "citation_existence", "citation_availability", "support_relationship_precision",
                    "contradiction_relationship_precision", "rubric_fact_support", "terminal_failure_frequency",
                )
            },
            "resource_distributions": {
                key: distribution([getattr(row.metrics, key) for row in selected])
                for key in ("tool_calls", "llm_calls", "wall_clock_seconds", "provider_input_tokens", "provider_output_tokens")
            },
            "per_incident_variability": {
                incident_id: {
                    key: distribution([
                        getattr(row.metrics, key) for row in selected if row.incident_id == incident_id
                    ])
                    for key in (
                        "top_ranked_correct", "abstained", "tool_calls", "wall_clock_seconds",
                        "provider_input_tokens", "provider_output_tokens",
                    )
                }
                for incident_id in sorted({row.incident_id for row in selected})
            },
        }
    return comparison


def digest(value: Contract) -> str:
    return hashlib.sha256(json.dumps(value.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def persist_results(output: Path, manifest: dict, rows: list[EvaluationRow]):
    comparison = summarize(rows)
    write_json(output / "results.json", {
        "manifest": manifest, "runs": [row.model_dump(mode="json") for row in rows],
        "comparison": comparison,
    })
    flattened = []
    for row in rows:
        record = row.model_dump(mode="json")
        metrics = record.pop("metrics")
        grounding = metrics.pop("grounding")
        metrics["top_matched_fact_ids"] = json.dumps(metrics["top_matched_fact_ids"])
        flattened.append(record | metrics | {f"grounding_{key}": value for key, value in grounding.items()})
    with (output / "results.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    mocked = manifest["config"]["mode"] == "mocked"
    lines = [
        "# Evaluation comparison",
        "**MOCKED PLUMBING ONLY: not evidence of LLM performance. No live results in this batch.**"
        if mocked else "**LIVE execution; scores use the limited evaluator-only rubric, not a semantic judge.**",
        "",
        "Failures and abstentions on answerable incidents remain in the accuracy denominator.",
        "Null scores mean no applicable cases/citations or no measured provider usage, not perfect performance.",
        "",
        "| Approach | Runs | Failures (terminal) | Root accuracy mean | Support-link precision mean | Latency mean (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def display(value):
        return "n/a" if value is None else f"{value:.3f}"

    for approach, values in comparison.items():
        rates = values["repetition_variability"]
        lines.append(
            f"| {approach} | {values['runs']} | {values['failures_included']} ({values['terminal_failures']})"
            f" | {display(rates['top_ranked_root_cause_accuracy']['mean'])}"
            f" | {display(rates['support_relationship_precision']['mean'])}"
            f" | {display(values['resource_distributions']['wall_clock_seconds']['mean'])} |"
        )
    lines += [
        "",
        "Repeated-run sample standard deviations, abstention/intentional-inconclusive rates, citation",
        "existence/availability, fact support, provider token subtotals, missing counts and per-incident",
        "variability are retained in results.json. One repetition has no variability estimate.",
        "Fact patterns are intentionally limited; unrecognized prose is not proven false or supported.",
    ]
    with (output / "summary.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


async def evaluate(
    config: EvaluationConfig, catalog: IncidentCatalog, settings: SimulationSettings,
    truth: GroundTruth, rubric: ScoringRubric, output: Path,
    llm_factory: Callable[[str, IncidentContext, int], StructuredLLM], *, allow_paid=False,
) -> list[EvaluationRow]:
    if config.mode == "live" and not allow_paid:
        raise PermissionError("live evaluation requires explicit paid-run approval")
    validate_inputs(catalog, truth, rubric)
    contexts = {item.context.incident_id: item.context for item in catalog.incidents}
    ids = sorted(config.incident_ids or contexts.keys())
    if not set(ids) <= contexts.keys():
        raise ValueError("requested incident does not exist")
    # Never merge batches or overwrite unrelated files.
    output.mkdir(parents=True, exist_ok=False)
    expected = {item.incident_id: item for item in truth.cases}
    rubrics = {item.incident_id: item for item in rubric.cases}
    manifest = {
        "schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config.model_dump(mode="json"), "selected_incident_ids": ids,
        "failure_policy": FAILURE_POLICY,
        "hashes": {
            "incident_visible_data": digest(catalog), "simulation": digest(settings),
            "ground_truth": digest(truth), "rubric": digest(rubric),
        },
        "rubric_version": rubric.version,
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("openai", "pydantic")},
        "implementation_hashes": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("models.py", "tools.py", "llm.py", "agents.py", "investigation.py", "baseline.py", "evaluate.py")
        },
        "model_settings": {
            "model": config.model, "temperature": config.temperature,
            "max_output_tokens": config.budget.max_output_tokens, "store": False, "sdk_retries": 0,
        },
        "run_order": [
            f"{incident_id}.{approach}.{repetition:03d}"
            for repetition in range(1, config.repeats + 1)
            for incident_index, incident_id in enumerate(ids)
            for approach in (
                APPROACHES if (repetition + incident_index) % 2 else tuple(reversed(APPROACHES))
            )
        ],
    }
    write_json(output / "manifest.json", manifest)
    rows = []
    for repetition in range(1, config.repeats + 1):
        for incident_index, incident_id in enumerate(ids):
            approaches = APPROACHES if (repetition + incident_index) % 2 else tuple(reversed(APPROACHES))
            for approach in approaches:
                run_id = f"{incident_id}.{approach}.{repetition:03d}"
                run = None
                error = None
                started = perf_counter()
                try:
                    llm = llm_factory(approach, contexts[incident_id], repetition)
                    if llm.mode != config.mode or llm.model != config.model:
                        raise ValueError("adapter mode/model differs from evaluation config")
                    if config.mode == "live" and getattr(llm, "temperature", None) != config.temperature:
                        raise ValueError("adapter temperature differs from evaluation config")
                    tools = SimulatedTools(catalog, settings)
                    runner = run_multi_agent if approach == "multi-agent" else run_baseline
                    remaining = config.budget.max_elapsed_seconds - (perf_counter() - started)
                    if remaining <= 0:
                        raise TimeoutError("run preparation exhausted the wall-clock budget")
                    run = await asyncio.wait_for(
                        runner(tools, llm, contexts[incident_id], config.budget),
                        timeout=remaining,
                    )
                    if (
                        run.trace.approach != approach or run.trace.execution_mode != config.mode
                        or run.trace.model != config.model or run.trace.budget != config.budget
                    ):
                        raise ValueError("run trace differs from requested approach/settings/budget")
                except (ValueError, RuntimeError, OSError, TimeoutError, TypeError, LookupError, AttributeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                elapsed = perf_counter() - started
                artifact = None
                if run is not None:
                    artifact = f"runs/{run_id}"
                    directory = output / "runs" / run_id
                    directory.mkdir(parents=True, exist_ok=False)
                    write_json(directory / "report.json", run.report.model_dump(mode="json"))
                    write_json(directory / "trace.json", run.trace.model_dump(mode="json"))
                try:
                    metrics = score_run(
                        run, expected[incident_id], rubrics[incident_id], catalog,
                        elapsed=elapsed, error=error,
                    )
                except ValueError as exc:
                    error = f"Invalid evaluation artifact: {exc}"
                    metrics = score_run(
                        None, expected[incident_id], rubrics[incident_id], catalog,
                        elapsed=elapsed, error=error,
                    )
                rows.append(EvaluationRow(
                    run_id=run_id, incident_id=incident_id, approach=approach,
                    repetition=repetition, mode=config.mode, metrics=metrics,
                    error=error, artifact_directory=artifact,
                ))
                # Completed rows survive interruption without pretending the batch finished.
                with (output / "runs.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(rows[-1].model_dump_json() + "\n")
    persist_results(output, manifest, rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("live", "mocked"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--incidents", nargs="+", help="Optional subset; defaults to all ten")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--budget", type=Path, help="JSON ExecutionBudget, shared by both approaches")
    parser.add_argument("--mock-script-dir", type=Path)
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()
    if args.mode == "live" and not args.allow_paid:
        parser.error("live evaluation requires --allow-paid; no paid requests were made")
    if args.mode == "mocked" and (args.allow_paid or args.mock_script_dir is None):
        parser.error("mocked mode requires --mock-script-dir and cannot use --allow-paid")
    if args.mode == "live" and args.mock_script_dir is not None:
        parser.error("live mode cannot use mock scripts")
    try:
        config = EvaluationConfig(
            mode=args.mode,
            model="scripted-test-double" if args.mode == "mocked" else args.model,
            temperature=args.temperature, repeats=args.repeats,
            incident_ids=tuple(args.incidents or ()),
            budget=ExecutionBudget.model_validate_json(args.budget.read_text()) if args.budget else ExecutionBudget(),
        )
        catalog = IncidentCatalog.model_validate_json((DATA_DIR / "incidents.json").read_text())
        settings = SimulationSettings.model_validate_json((DATA_DIR / "simulation.json").read_text())
        truth = GroundTruth.model_validate_json((DATA_DIR / "evaluation" / "ground_truth.json").read_text())
        rubric = ScoringRubric.model_validate_json((DATA_DIR / "evaluation" / "rubric.json").read_text())
        validate_inputs(catalog, truth, rubric)
        selected = config.incident_ids or tuple(item.context.incident_id for item in catalog.incidents)
        if args.mode == "mocked":
            # Fail preflight rather than substitute empty/success-shaped scripts.
            scripts = {
                (incident_id, approach): ScriptedLLM.from_file(
                    args.mock_script_dir / f"{incident_id}.{approach}.json"
                ).responses
                for incident_id in selected for approach in APPROACHES
            }
            adapter = None
            factory = lambda approach, context, repetition: ScriptedLLM(scripts[(context.incident_id, approach)])
        else:
            adapter = OpenAILLM(config.model, temperature=config.temperature)
            factory = lambda approach, context, repetition: adapter
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    async def run():
        try:
            return await evaluate(
                config, catalog, settings, truth, rubric, args.output_dir,
                factory, allow_paid=args.allow_paid,
            )
        finally:
            if adapter is not None:
                await adapter.close()

    try:
        rows = asyncio.run(run())
    except (ValueError, OSError, PermissionError) as exc:
        parser.error(str(exc))
    print(
        f"{args.mode.upper()}: saved {len(rows)} runs to {args.output_dir}. "
        + ("Mocked plumbing only; no live benchmark results." if args.mode == "mocked"
           else "Live batch completed; failures remain in the results.")
    )
    return 2 if any(row.metrics.terminal_failure for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline collection or opt-in live/scripted multi-agent and baseline investigations."""

import argparse
import asyncio
from datetime import timedelta
from pathlib import Path

from investigation import InvestigationState, run_multi_agent, run_tool_tasks
from models import IncidentContext, InvestigationSnapshot, Role, TaskSpec, TimeRange, ToolFamily, ToolQuery
from tools import SimulatedTools


async def collect(tools: SimulatedTools, context: IncidentContext) -> InvestigationSnapshot:
    state = InvestigationState(context.incident_id)
    specs = []
    for family in ToolFamily:
        window = context.time_range
        if family == ToolFamily.HISTORICAL_INCIDENTS:
            window = TimeRange(start=window.start - timedelta(days=90), end=window.end)
        specs.append(
            TaskSpec(
                task_id=family.value,
                role=(
                    Role.OBSERVABILITY
                    if family in {ToolFamily.LOGS, ToolFamily.METRICS}
                    else Role.DEPENDENCY
                ),
                query=ToolQuery(
                    incident_id=context.incident_id,
                    family=family,
                    time_range=window,
                ),
            )
        )
    state.add_tasks(*specs)
    await run_tool_tasks(state, tuple(spec.task_id for spec in specs), tools)
    return state.snapshot()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List available neutral incident alerts")
    collect_parser = subparsers.add_parser("collect", help="Collect five tool families concurrently")
    collect_parser.add_argument("incident_id")
    collect_parser.add_argument("--json", action="store_true", help="Print the full structured snapshot")
    run_parser = subparsers.add_parser("run", help="Run an investigation and save report/trace")
    run_parser.add_argument("approach", choices=("multi-agent", "baseline"))
    run_parser.add_argument("incident_id")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--mock-script", type=Path, help="Use supplied mocked JSON replies; no model reasoning")
    run_parser.add_argument("--allow-paid", action="store_true", help="Explicitly authorize live OpenAI calls")
    run_parser.add_argument("--model", default="gpt-4.1-mini")
    run_parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()
    try:
        tools = SimulatedTools.from_files()
    except (ValueError, OSError) as exc:
        parser.error(f"could not load synthetic data: {exc}")
    if args.command == "list":
        for context in tools.contexts():
            print(f"{context.incident_id}  {context.alert}")
        return 0
    try:
        context = tools.context(args.incident_id)
    except KeyError:
        parser.error(f"unknown incident: {args.incident_id}")
    if args.command == "run":
        from baseline import run_baseline
        from llm import OpenAILLM, ScriptedLLM

        if not args.mock_script and not args.allow_paid:
            parser.error("live runs require --allow-paid; use --mock-script for offline execution")
        if args.mock_script and args.allow_paid:
            parser.error("choose mocked execution or paid execution, not both")
        report_path = args.output_dir / "report.json"
        trace_path = args.output_dir / "trace.json"
        try:
            if report_path.exists() or trace_path.exists():
                parser.error("output report/trace already exist; choose a new output directory")
            args.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            parser.error(f"could not prepare output directory: {exc}")
        try:
            llm = (
                ScriptedLLM.from_file(args.mock_script) if args.mock_script
                else OpenAILLM(args.model, temperature=args.temperature)
            )
        except (ValueError, OSError) as exc:
            parser.error(str(exc))

        async def investigate():
            try:
                runner = run_multi_agent if args.approach == "multi-agent" else run_baseline
                return await runner(tools, llm, context)
            finally:
                if isinstance(llm, OpenAILLM):
                    await llm.close()

        run = asyncio.run(investigate())
        # Exclusive creation preserves prior outputs even if another process wins the race.
        try:
            with report_path.open("x", encoding="utf-8") as stream:
                stream.write(run.report.model_dump_json(indent=2) + "\n")
            with trace_path.open("x", encoding="utf-8") as stream:
                stream.write(run.trace.model_dump_json(indent=2) + "\n")
        except OSError as exc:
            parser.error(f"could not save report/trace (partial output may exist): {exc}")
        print(f"{run.trace.execution_mode.upper()}: {args.approach}, {run.report.outcome}")
        print(f"Termination: {run.trace.termination_reason}")
        print(f"Report: {report_path}\nTrace: {trace_path}")
        return 0 if run.report.outcome == "ranked" else 2
    snapshot = asyncio.run(collect(tools, context))
    if args.json:
        print(snapshot.model_dump_json(indent=2))
    else:
        print(f"{context.incident_id}: {context.alert}")
        print("Offline evidence collection only; no agents or LLM were run.")
        for task in snapshot.tasks:
            result = snapshot.tool_results[task.result_indices[-1]]
            print(
                f"{task.spec.task_id}: {result.status}, "
                f"{len(result.evidence)} records, {task.attempts} attempt(s)"
            )
            if result.error:
                print(f"  {result.error.code}: {result.error.message}")
            for issue in result.issues:
                print(f"  {issue}")
        print(
            f"Tool calls: {snapshot.usage.tool_calls}; "
            f"tokens: {snapshot.usage.tokens}; "
            f"elapsed: {snapshot.usage.elapsed_seconds:.3f}s"
        )
    return 1 if any(task.status != "succeeded" for task in snapshot.tasks) else 0


if __name__ == "__main__":
    raise SystemExit(main())

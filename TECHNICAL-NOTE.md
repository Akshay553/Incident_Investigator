# Technical note

This assignment deliberately uses eight flat Python modules, asyncio, Pydantic,
local JSON and one OpenAI adapter instead of an agent framework or platform.
Orchestrator plans evidence tasks; Observability and Dependency investigate
independently. A serialized coordinator commits their results to a typed shared
ledger. Reasoning compares hypotheses, Critic requests discriminating evidence,
and Orchestrator schedules valid follow-ups or records deferral. Reporting cannot
collect evidence or change accepted hypotheses. The baseline is one iterative
agent with the same tools, data, model, schema and resource ceilings.

Tool results distinguish complete observations, covered-empty queries, partial
coverage and failures. Stable evidence IDs preserve provenance and counterevidence.
Task dependencies, permissions, schema validation, deduplication, deadlines,
attempts and resource accounting are runtime responsibilities, not model promises.
Compact role-specific prompts avoid repeatedly resending history; full records
remain in traces. Transient operations have three total attempts, selected output
errors get at most one correction, and feedback is limited to two rounds.
Conservative input/schema reservations are replaced by provider usage; unknown
usage remains explicitly reserved.

The frozen benchmark ran twenty live investigations, one per approach on ten
synthetic cases. Multi-agent scored 0/9 answerable-case rubric accuracy with seven
terminal failures; baseline scored 3/9 with four. Excluding development case
inc-001 gives 0/8 versus 3/8. Multi-agent appropriately abstained on the intentional
inconclusive case; baseline failed execution there. Mean latency was 32.50 versus
13.35 seconds. Total measured consumption was 498,798 input and 54,438 output
tokens, costing USD 0.2866200 at published rates, with no cached input.

These results do not demonstrate multi-agent superiority. More model roles
created additional validation failure points and excessive caution. Baseline
was cheaper and more accurate under the rubric, but often stopped at mechanisms.
All emitted citations existed; limited support precision was 54.5% versus 37.5%.
The evaluator recognizes enumerated facts/witness relationships, not general
semantic entailment. Some rubric credits omit the initiating change or correct
service attribution. Confidence is uncalibrated; uncertainty classifications
remain model-dependent. Advisory recommendations can be unsafe or misdirected
and were never executed.

Offline tests establish scheduling and contract behavior, not reliable live
diagnosis. One run per case cannot establish repeatability or statistical
superiority; the synthetic corpus is not an external holdout. Isolation is
loader/tool/prompt separation, not an OS sandbox, and the provider model is
deprecated. Potential improvements are clearer non-overlapping correction
contracts, evidence-calibrated objections, exact-copy report assembly and a
stronger independently specified scoring rubric. None was implemented after
observing benchmark results within that frozen benchmark.

The subsequent working revision constrains each role's response schema and
separates correction instructions. Critic index/disposition errors receive one
bounded correction without deleting objections; Reporting corrections restore
accepted hypotheses instead of preserving a rejected mutation. Causal guidance
does not require intervention experiments when observations already support a
trigger. A separately approved ten-case rerun scored 1/9 with one terminal failure;
the original metrics remain unchanged. Six answerable cases still abstained.
Another ranked diagnosis identified the missing response field but lost regex
credit for saying "absence" rather than "absent"; the rubric was not loosened.
Execution reliability improved, but diagnostic accuracy remains weak.

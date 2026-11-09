# Final Evaluation Results: Multi-Agent vs. Single-Agent Baseline

**Report date:** 2026-11-09

**Dataset:** Ten synthetic production incidents; nine answerable and one intentionally inconclusive.

## Executive summary

The **newly evaluated multi-agent implementation scored 1/9 (11.1%)** on
answerable cases, with **1/10 terminal execution failures**. The
single-agent baseline run scored **3/9 (33.3%)**, with **4/10 terminal failures**.
The multi-agent approach handled the intentionally inconclusive case correctly;
the baseline failed execution on that case.

The revised multi-agent system was more execution-reliable in the observed runs
and had higher limited support-link precision. The baseline had higher
rubric accuracy, fewer model requests, and lower mean latency. Neither approach
demonstrated sufficiently reliable diagnosis for autonomous production use.

**Methodology:** The multi-agent run was performed on 2026-11-09 and the baseline
run on 2026-11-08. The baseline was not re-executed after the reliability
revision. Both used the same synthetic
cases, model/settings, rubric and resource ceilings, but this is **not a fresh
paired evaluation of the revised implementation**. It does not establish
statistical superiority or attribute every difference to the code changes.

## Evaluation setup

| Setting | Value |
| --- | --- |
| Execution | Live OpenAI requests; no scripted/mock answers |
| Model alias | `gpt-4.1-mini` |
| Temperature | 0 |
| Cases per approach | 10 |
| Repetitions | One investigation per case in each selected batch |
| Accuracy denominator | Nine answerable incidents |
| Intentionally inconclusive case | `inc-009`; excluded from answerable accuracy |
| Rubric | `limited-facts-v1`, unchanged |
| Data and tools | Local synthetic observations; five read-only simulated tool families |
| Runtime | Python 3.12.10; OpenAI SDK 2.54.0; Pydantic 2.14.2 |
| Per-case ceilings | 30 tool attempts; 24 model requests; 120,000 charged/reserved tokens; 120 seconds |
| Recovery | Three total operation attempts; at most one output correction; two follow-up rounds |
| Per-request output ceiling | 1,500 tokens |

The multi-agent system uses Orchestrator, Observability, Dependency, Reasoning,
Critic and Reporting roles. The baseline is one iterative investigator using
the same tools and final report contract. The baseline executes tools
sequentially; the multi-agent coordinator can execute independent tasks
concurrently.

Ground truth and scoring rules are evaluator-only inputs, not agent evidence.
The reliability revision did not change the incident data, ground truth, rubric,
model, temperature or resource ceilings.

## Main results

| Metric | Multi-agent run | Baseline run |
| --- | ---: | ---: |
| Investigations evaluated | 10 | 10 |
| Answerable-case rubric accuracy | **1/9 (11.1%)** | **3/9 (33.3%)** |
| Terminal execution failures | **1/10 (10%)** | **4/10 (40%)** |
| Runs without terminal failure | 9/10 | 6/10 |
| Ranked reports | 2/10 | 6/10 |
| Inconclusive reports, including failure fallbacks | 8/10 | 4/10 |
| Appropriate handling of `inc-009` | Yes | No: execution failed |
| Citation existence in emitted reports | 61/61 (100%) | 24/24 (100%) |
| Citations acquired by the investigation | 61/61 (100%) | 24/24 (100%) |
| Limited support-link precision | 31/59 (52.5%) | 9/24 (37.5%) |
| Mean investigation latency | 37.48 seconds | 13.35 seconds |
| Tool calls, including attempts | 61 | 62 |
| Model requests, including corrections | 101 | 44 |
| Provider input tokens | 358,209 | 185,604 |
| Provider output tokens | 46,043 | 15,661 |
| Total measured provider tokens | 404,252 | 201,265 |

Latency includes investigation preparation, execution and recovery, but excludes
artifact writes. Grounding totals include retained hypotheses in failed or
inconclusive reports; they are not measures of successful diagnosis.
Citation existence applies to emitted reports, not every rejected model response.
For example, the new `inc-008` Critic response contained an invalid citation and
was rejected even though the retained final-report citations were valid.

## Per-incident results

The multi-agent outcomes in this table are from the **new run**. The baseline
column records only rubric credit from the baseline run, not a claim
about whether an uncredited baseline case failed execution or produced a
ranked incorrect answer.

| Incident | Multi-agent run | Baseline run: rubric credit | Explanation of the multi-agent result |
| --- | --- | --- | --- |
| `inc-001` | Ranked; credited | No | Identified the database connection-limit configuration change and resulting connection waits. |
| `inc-002` | Ranked; not credited | Yes | Identified the missing response-field issue, but its wording did not match the required regex fact. |
| `inc-003` | Inconclusive | No | Retained missing payment-gateway telemetry as a blocking uncertainty. |
| `inc-004` | Inconclusive | No | Retained missing comprehensive search-west telemetry as a blocker despite the search-east change/OOM evidence. |
| `inc-005` | Inconclusive | Yes | Retained missing pricing-service health telemetry as a blocker. |
| `inc-006` | Inconclusive | No | Retained missing job-queue telemetry as a blocker. |
| `inc-007` | Inconclusive | No | Recovered injected tool failures, but retained missing log-store capacity telemetry as a blocker. |
| `inc-008` | Terminal execution failure | Yes | Critic cited unknown evidence ID `inc-008:changes:v1`; the response was rejected. |
| `inc-009` | Appropriately inconclusive | Excluded; baseline execution failed | The observations do not support a resolved initiating cause. |
| `inc-010` | Inconclusive | No | Retained a mechanism-only candidate and unresolved cache/network objections; the follow-up round limit was reached. |

The new nine-case accuracy denominator consists of **one credited ranked
report, one uncredited ranked report, six completed abstentions, and one
execution failure**.

## Why the original multi-agent result was 0/9

The original 2026-11-08 multi-agent run is historical context, **not the
multi-agent column in the main comparison above**.

Its 0/9 comprised seven answerable-case execution failures and two
answerable-case abstentions (`inc-002`, `inc-008`). It did not mean nine
completed diagnoses were all substantively wrong. The intentionally
inconclusive `inc-009` was handled appropriately and was excluded from accuracy.

Recorded failures included invalid Critic references into an empty follow-up
list, correction responses changing roles/candidates, and Reporting changing
accepted Reasoning hypotheses. These were model-output/coordinator contract
failures, not shell crashes. The original evaluation reported no transport
retries or exhausted tool/request/token/time budget.

The reliability revision introduced role-constrained response schemas,
role-specific correction instructions, bounded correction of current-round
indices/query dispositions, and Reporting corrections that restore accepted
analysis rather than preserve an erroneous rejected candidate. Critic
corrections cannot erase objections, evidence or critical judgments.

| Multi-agent metric | Original run | New reliability-revision run |
| --- | ---: | ---: |
| Answerable-case rubric accuracy | 0/9 | 1/9 |
| Terminal execution failures | 7/10 | 1/10 |
| Appropriate `inc-009` abstention | Yes | Yes |

Five new-run cases recovered invalid duplicate-query proposals rather than
terminating. The transient tool failures in `inc-007` were also recovered.
No tool/request/token/time budget was exhausted in the new run; reaching the
two-follow-up-round limit in `inc-010` is reported separately.

## Scoring limitations and remaining weaknesses

**Wording-sensitive false negative:** The new `inc-002` report describes
"the absence of the expected 'total' field during response serialization"
in orders version 2.0. The visible log and ground truth support the
missing-field issue. However, the required fact's regex recognizes words such
as **absent** and **missing**, not **absence**. The report therefore receives
no accuracy credit. The official result remains **1/9**; the rubric was not
loosened or replaced with a post hoc semantic score.

**Over-abstention:** Six answerable cases completed without selecting a cause.
Their provisional hypotheses often identify a plausible initiating condition,
but missing corroborating telemetry or unexcluded hypothetical alternatives
remain blockers. The system still struggles to distinguish a materially
missing causal link from ordinary residual uncertainty.

**Rubric credit is not complete semantic validation:** The evaluator checks
categories, enumerated fact patterns and selected evidence relationships. It
does not establish arbitrary entailment, complete causal timelines, correct
service attribution, calibrated confidence or recommendation safety. The
baseline received some credits without establishing the full
initiating change or correct service attribution.

**Limited generalizability:** These are ten known synthetic cases, not an
external blind holdout. One run per case does not establish repeatability or
statistical significance. No suggested production action was executed.

## Conclusion

The new multi-agent evaluation shows **better observed execution reliability
than both its original run and the baseline run**, but only **1/9 rubric
accuracy** and substantial over-abstention. The baseline run remains
more accurate under the fixed rubric and uses fewer model requests and tokens.

These results support improving diagnosis and uncertainty handling, not a claim
of multi-agent superiority or readiness for autonomous incident remediation.

---
name: baseline-evidence-review
description: >
  Use for targeted comparison of paper claims, repository implementation,
  measured runtime behavior, configuration deviations, reproduction blockers,
  and baseline readiness in this thesis repository. Reuse existing audit
  evidence before running or repeating any investigation.
---

# Baseline Evidence Review

## Start with existing evidence

Scope the review to the requested Co-VAE or DCGAN-DTA claim, configuration,
execution boundary, or readiness question. Do not repeat a full paper-code audit.
Resolve these paths from the repository root and search relevant reports first:

- `audit/03-paper-code-traceability/`: paper-to-code claims and initial blockers.
- `audit/05-reproduction-feasibility/`: configurations, prerequisites, and corrections.
- `audit/06-smoke-execution/`: measured import, loader, construction, and smoke boundaries.
- `audit/07-training-feasibility/`: bounded optimizer-step and GAN timing evidence.
- `audit/08-controlled-training-validation/`: repeated native-path validation evidence.

Read the applicable baseline report and follow correction references across stages.
Do not assume both baselines have reports or equivalent coverage in every stage.
Reuse settled measurements with their locators, environment, and configuration;
do not relabel an old measurement as a new observation. Repeat a check only when
changed inputs, contradictory evidence, or the explicit task justify it.
Historical recommendations and unresolved questions do not authorize new work.

## Keep four evidence classes separate

### A. Paper claim

Record what the paper explicitly states, with section/page/equation/table locators.
Examples include architecture, hyperparameters, dataset, split, metric, optimizer,
batch size, loss, and reported result. Mark unspecified details rather than filling
them from code. Existing paper audits can locate the statement for targeted reading.

### B. Repository implementation

Record what the committed source does, identifying the baseline repository and commit.
Trace the active path through defaults, hard-coded values, ignored arguments,
preprocessing, generated folds, losses, optimizer updates, and evaluation as relevant.
If inspecting working-tree changes or a runtime wrapper, distinguish them from that
committed implementation. Names, README claims, and declared arguments alone do not
establish behavior; follow consumption and call paths using AGENTS.md guidance.

### C. Runtime evidence

Use only measurements from an authorized execution, existing or currently requested.
Identify the tested path, configuration, environment/device, and stopping condition.
Examples include OOM, elapsed time, memory, loss behavior, successful optimizer steps,
and framework GPU visibility. Do not infer measured behavior from static code alone.
Distinguish native execution from a scratch replica, estimate, or isolated probe.
If the original log is unavailable, identify the report as the measurement source
and disclose that the underlying artifact was not independently inspected.

### D. Interpretation

State conclusions derived from A-C separately from the observations supporting them.
Limit conclusions to the tested conditions; an OOM on one device is not universal
infeasibility, and a successful smoke run is not convergence or paper reproduction.

## Targeted traceability

1. Search prior evidence and any later correction for the specific claim.
2. Locate the relevant paper statement; record absence or ambiguity explicitly.
3. Locate the corresponding code path and effective configuration.
4. Determine agreement, partial agreement, disagreement, or insufficient evidence.
5. Incorporate runtime evidence only if already available or explicitly authorized.
6. Explain the relevant reproduction impact and remaining uncertainty.

For objective questions, trace losses into optimizer updates: a defined GAN loss or
VAE term need not participate in optimization. For configuration questions, distinguish
paper values, CLI defaults, effective values, and deliberately changed run settings.
For evaluation questions, distinguish selection data, reporting data, and aggregation.

When useful, use one compact table:

| Item | Paper | Repository | Runtime evidence | Status | Reproduction impact |
|---|---|---|---|---|---|

Use descriptive status language scoped to the claim:

- MATCH: inspected evidence agrees within the stated scope.
- PARTIAL MATCH: only part agrees or part remains unverified.
- MISMATCH: a concrete difference is established.
- UNVERIFIED: evidence is insufficient to decide.
- RUNTIME-LIMITED: the measured execution boundary limits verification.
- ENVIRONMENT-LIMITED: dependency or hardware conditions limit verification.

State any known mismatch separately if execution is also limited. Avoid labels such
as "broken", "wrong", "bad implementation", or "paper error" without sufficient
direct evidence. Do not assign speculative severity or infer author intent.

## Reproduction-readiness branch

Use only when the task asks whether a particular next validation or reproduction
is feasible. Specify that target, dataset, configuration, and environment first.
Distinguish released-code execution, controlled reconstruction, and paper-faithful
reimplementation; success at one does not establish readiness for another.

Assess only relevant prerequisites: code and data availability, folds/splits,
dependency/environment compatibility, hardware requirements, missing configuration,
evaluation protocol, undocumented behavior, and runtime cost.
For each material prerequisite, give evidence and any condition or blocker:

- READY: the prerequisite is established for the specified target.
- READY WITH DECLARED DEVIATION: feasible with a named departure and stated impact.
- CONDITIONALLY READY: feasible subject to a concrete, explicit condition.
- BLOCKED: an evidenced unmet requirement prevents that target under current conditions.
- REQUIRES INVESTIGATION: evidence is insufficient to determine readiness.

These are readiness classifications, never model-quality judgments. Readiness is not
authorization to proceed. Distinguish measured cost from extrapolation, and report
the reached boundary: forward pass, isolated step, native loop, or completed run.
Do not upgrade a declared configuration deviation into paper-faithful reproduction.

## Impact, revisions, and routing

For important mismatches, explain only supported consequences: changed split,
training objective, memory requirements, reproducibility of the published
configuration, comparability, or no effect on the tested path.
If a prior finding is superseded, show its locator and previous conclusion, the new
evidence, and the revised conclusion with scope. Do not silently rewrite history.

For primarily fold-validity, leakage, entity-overlap, matrix-semantic, or generated-
split questions, defer to `dataset-fold-audit` at `../dataset-fold-audit/SKILL.md`.
Consume its findings instead of duplicating deep dataset verification.
Full paper rereading, exhaustive figure/equation extraction, or supplement-wide
reconstruction requires a separate explicit paper-level audit.

## Execution boundary

This review may inspect source, configs, papers, audit reports, logs, and existing
outputs. Static inspection does not authorize execution, including module imports.
Runtime execution requires an explicit request in the current task; the skill itself
grants none. Do not start training, grid searches, all-fold runs, Kaggle jobs, or
reproduction experiments unless explicitly authorized for that task.
Follow AGENTS.md's recording and stage boundaries for any authorized execution.
Do not silently repair mismatches, change configuration, or expand the assignment.

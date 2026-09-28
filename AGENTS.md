# Project Instructions

This human-maintained section makes `AGENTS.md` the canonical project instruction file for Codex.

## Project Purpose

This repository supports an MSc thesis comparing two Drug–Target Binding Affinity baselines: **Co-VAE** and **DCGAN-DTA**. The current priority is reproducibility and controlled validation of the original implementations before introducing thesis novelty or a hybrid model.

## Working Principle

Treat the repository as research evidence. Prefer: **inspect → measure → document → modify only when explicitly authorized**. Do not silently "fix" suspicious implementation behavior. Clearly distinguish paper behavior, repository behavior, measured runtime behavior, and interpretation.

## Current Thesis Workflow

```text
02 Paper Forensic Audit              completed
03 Paper-Code Traceability           completed
04 Dataset & Fold Audit              completed
05 Reproduction Feasibility          completed
06 Smoke Execution                   completed
07 Training Feasibility              completed
08 Controlled Training Validation    in progress
08.5 Kaggle GPU Validation           planned
09 Baseline Reproduction             planned
10 Fair Baseline Comparison          planned
11 Representation Analysis           planned
12 Simple Hybrid Baseline            planned
13 Optimization-Based Fusion         planned
14 Ablation & Robustness             planned
15 Final Experiments                 planned
```

Do not advance to a later stage unless the task explicitly authorizes it.

## Repository Protection

Unless explicitly authorized, do not modify original paper implementations, datasets, fold files, original papers/supplementaries, prior audit reports, or nested Git repositories. Do not discard or overwrite pre-existing working-tree changes.

Before making changes:

```powershell
git status --short
```

After making changes:

```powershell
git status --short
git diff -- <files intentionally changed>
```

Never use destructive cleanup/reset commands without explicit authorization.

## Reproducibility Requirements

For every experiment or runtime validation, record as applicable:

- Repository/commit identity, environment, Python version, library versions, and CPU/GPU information.
- Dataset, split/fold, exact command/configuration, random seed if controlled, and intentional deviations.
- Runtime, memory, numerical failures, stopping condition, and output locations.

Do not describe a run as a paper reproduction unless its evaluation protocol is actually aligned with the paper.

## Experiment Authorization

Inspection and lightweight diagnostics are distinct from training. Do not start long-running training, full hyperparameter grids, all-fold experiments, Kaggle/cloud experiments, or reproduction runs unless the current task explicitly authorizes them.

If runtime unexpectedly becomes expensive, stop at the safest useful boundary and report what completed.

## Thesis Novelty Boundary

Do not implement representation fusion, hybrid GAN/VAE models, learned alpha, convex/optimization-based fusion, new loss functions, or architectural improvements unless explicitly requested. Baseline validation comes first.

## Audit Artifacts

Write research findings under the appropriate stage directory inside `audit/`. Do not rewrite prior-stage conclusions silently. If later evidence changes an earlier conclusion, explicitly document the previous conclusion, the new evidence, and the revised conclusion.

## Current Compute Strategy

Use local execution primarily for:

- inspection,
- debugging,
- smoke testing,
- lightweight controlled validation.

Kaggle GPU is planned for:

- hardware-limited validation,
- expensive baseline training,
- later reproduction experiments.

Do not move computation to Kaggle unless explicitly requested.

## Codex Behavior

When a task is ambiguous:

- prefer preserving evidence over changing code,
- inspect existing audit material before repeating work,
- reuse previous measured results rather than rerunning expensive experiments unnecessarily,
- make the smallest change necessary for the current task,
- clearly separate observed facts from interpretation.

Do not silently broaden the scope of a task.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **GAN-vs-VAE-** (844 symbols, 1040 relationships, 22 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/GAN-vs-VAE-/context` | Codebase overview, check index freshness |
| `gitnexus://repo/GAN-vs-VAE-/clusters` | All functional areas |
| `gitnexus://repo/GAN-vs-VAE-/processes` | All execution flows |
| `gitnexus://repo/GAN-vs-VAE-/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.codex/skills/gitnexus-workflow/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.codex/skills/gitnexus-workflow/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.codex/skills/gitnexus-workflow/SKILL.md` |
| Rename / extract / split / refactor | `.codex/skills/gitnexus-workflow/SKILL.md` |
| Tools, resources, schema reference | `.codex/skills/gitnexus-workflow/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.codex/skills/gitnexus-workflow/SKILL.md` |

<!-- gitnexus:end -->

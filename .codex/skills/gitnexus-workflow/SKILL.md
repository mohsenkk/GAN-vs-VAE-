---
name: gitnexus-workflow
description: >
  Use for targeted graph-aware exploration of symbol relationships and execution
  flows, debugging call paths, dependency/impact analysis, authorized refactoring,
  change verification, or explicitly requested GitNexus maintenance in this repository.
  Ordinary file reading does not require this skill. Do not reindex, install, clean,
  publish, or modify code unless the current task explicitly authorizes that action.
---

# GitNexus Workflow

## Choose the smallest useful operation

AGENTS.md supplies mandatory project and GitNexus rules; this skill adds operational
choices and evidence limits. Select the mode matching the question, not a fixed
sequence of every available tool. Use ordinary source inspection when it answers
directly, subject to AGENTS.md's requirements for unfamiliar-code navigation and edits.
Do not invoke GitNexus merely because it exists.

Target the indexed repository `GAN-vs-VAE-` and disambiguate symbols by file or UID
when supported. Confirm returned paths belong to the intended baseline; similarly
named functions and nested repositories need not share an index or commit identity.
Use available MCP tools/resources and their current schemas, not assumed signatures.

| Question | Smallest relevant operation |
|---|---|
| Known symbol's callers/callees | `context` with name and a disambiguating path/UID |
| Concept or execution flow | Targeted `query`, then the relevant process resource |
| How one symbol reaches another | Bounded `trace` between the two symbols |
| Dependents of a proposed change | `impact` with upstream direction |
| Scope of an existing diff | `detect_changes` with the appropriate diff scope |
| Structural question not covered above | Bounded read-only `cypher` query |

Read `gitnexus://repo/GAN-vs-VAE-/schema` before custom graph queries; consult current
tool metadata or installed documentation for advanced details. Do not copy an entire
schema into the task. Report missing tools or graph layers instead of installing them.

## Mode A: Explore

Use for symbol usage, component connections, callers, execution flows, or architecture
navigation. A known symbol normally needs `context`; a concept needs targeted `query`.
Follow only relevant process/cluster resources rather than scanning the whole graph.
Use `gitnexus://repo/GAN-vs-VAE-/context` when index scope or freshness matters.
Read the referenced source to establish actual implementation details.

Graph edges show indexed relationships, not proof that a path executes in a particular
configuration. A shortest path is not a runtime trace or an exhaustive path inventory.
An absent edge/path may reflect dynamic dispatch, unsupported syntax, excluded files,
staleness, or query limits. Inspect ambiguity and truncation before claiming absence.

## Mode B: Debug

Start from the specific error, unexpected behavior, or suspicious dependency.
Use graph context to narrow candidate code paths; inspect the relevant source and
existing logs/audit evidence before proposing additional investigation.
Use `trace` for a concrete caller-to-callee question and `detect_changes` for a
regression when there is an appropriate diff. Avoid broad maintenance as diagnosis.

Keep static dependency evidence, measured runtime evidence, and interpretation separate.
A dependency is a candidate explanation, not established causality. Many callers
do not establish a runtime hotspot. State what evidence supports the suspected cause
and what remains unverified. Reuse existing runtime results within their tested scope.
Graph inspection does not authorize imports, reproducer execution, or experiments.

## Mode C: Impact / Authorized Change

For an impact-only question, stop after analysis. This skill grants no edit permission.
For rename, move, extract, split, or refactor tasks:

1. Establish from the current task that the specific code change is authorized.
2. Inspect impact first, following AGENTS.md's reporting and risk-warning requirements.
3. Review direct callers, callees, dependent symbols, affected processes, and paths.
4. Make only the smallest requested change within the authorized files/repositories.
5. Verify changed references and compare the diff with the expected change surface.
6. Run only authorized tests/checks and report results and remaining uncertainty.

Describe items as directly affected, potentially affected, or requiring verification.
Do not claim they will break unless direct evidence establishes that consequence.
Treat graph confidence as confidence in a relationship, not a probability of failure;
do not substitute arbitrary caller-count thresholds for the tool's risk assessment.

For an authorized rename, use the graph-aware rename tool with a dry-run preview.
Review both graph-derived and text-search edits, especially strings and dynamic refs.
Apply only when the preview fits the authorized scope; preview success alone is not
permission. For moves/extractions, inspect imports, consumers, and interfaces before edits.
Do not perform opportunistic refactors or cross nested-repository boundaries implicitly.

After changes, choose `detect_changes` scope deliberately: unstaged, staged, all, or
compare against the task's base ref. Distinguish pre-existing edits from this task's
changes. Compare returned symbols/processes with the actual Git diff; an old index
may not represent newly introduced symbols or relationships. Report that limitation.
Change detection complements reference review and tests; it does not prove correctness.

## Mode D: Maintenance

Use only for an explicit maintenance request: status, repository listing, reindexing,
cleaning, or wiki generation. Maintenance is never an implicit debugging step.
If ordinary graph use suggests staleness, report:

```text
GitNexus index may be stale.
Reindexing requires explicit authorization.
```

If reindexing is already explicitly authorized, that requirement is satisfied;
proceed only within its scope. Otherwise continue useful source inspection and label
stale graph results. Do not silently bypass a mandatory pre-edit impact requirement
when usable impact analysis is unavailable; report the limitation before editing.

Inspect installed command support before maintenance. Version 1.6.9 was observed
during migration; verify the actual installation for future operations.
Prefer an already installed executable. The project runner can fall back to package
downloads through npm/pnpm; inspect resolution before using it when installs are not
authorized. Missing tools do not authorize bootstrap installation or an update.

For explicitly authorized analyze/reindex work, select supported preservation options:

- `--skip-agents-md`: preserve generated instruction sections in AGENTS.md and CLAUDE.md.
- `--skip-skills`: skip standard skill installation, not generation requested by `--skills`.
- `--index-only`: skip all instruction/skill-file injection when only indexing is requested.

These are analyze options, not universal flags. Verify support and intended side effects
before use. They still permit index writes; they do not turn reindexing into inspection.
Do not modify `.gitnexus/` unless explicitly asked, or regenerate instructions or skills
during ordinary graph use. Verify protected files and the intended scope after maintenance.

Status/listing requests do not authorize analyze or clean. Clean deletes index/registry
state; never expand it to all repositories implicitly. Wiki generation writes artifacts
and may use an external provider or local CLI; inspect the chosen provider and outputs.
Wiki generation does not authorize publishing; public Gist publication needs an explicit
request. Do not add installation, cleanup, publication, or experiments to another mode.

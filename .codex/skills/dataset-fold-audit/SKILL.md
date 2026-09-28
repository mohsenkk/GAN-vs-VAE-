---
name: dataset-fold-audit
description: >
  Use for targeted verification of dataset semantics, preprocessing,
  interaction matrices, fold/split construction, entity overlap,
  leakage, and train/validation/test boundaries in this thesis repository.
  Prefer existing audit evidence and only re-check the specific issue requested.
---

# Targeted Dataset and Fold Verification

## Start from the completed audit

Stage 04 is completed. Do not automatically repeat the full dataset/fold audit.
Resolve paths below from the repository root and read the relevant findings first:

- `audit/04-dataset-fold/Co-VAE.md`
- `audit/04-dataset-fold/DCGAN-DTA.md`

Search existing audit material for the requested claim and any later revision
before repeating an expensive or already settled check. Reuse measured results
with their provenance and limitations; do not present them as a fresh measurement.

Identify the baseline, dataset, split setting, and specific question to resolve.
Verify only that claim. Expand the verification scope only when new evidence
creates a contradiction, explain why, and stay within the task's authorization.
An unresolved question in an old report is not permission to investigate it now.

## Evidence and provenance

Use existing reports to locate evidence; prefer evidence in this order when
deciding what the data pipeline actually does:

1. Actual dataset/fold files.
2. Executed loader/preprocessing behavior, only if execution is explicitly authorized.
3. Source code.
4. Existing audit reports.
5. Paper claims.

Files establish shipped content, not whether a loader consumes it. Paper claims
remain the source for the paper-defined protocol; report disagreements explicitly.
Distinguish paper-defined protocol, repository-shipped data, runtime-generated
split behavior, and interpretation. A source trace is not a measured runtime result.
Attach file/section or code-line locators and relevant data/configuration identity.
For authorized execution, follow AGENTS.md's reproducibility recording requirements.

## Select only relevant checks

### Matrix and identity semantics

- Check matrix dimensions, observed/missing entries, and zero-value semantics.
- Map entity ordering to matrix axes and the loader's retained order after filtering.
- Distinguish raw identifiers from canonical identifiers and actual model inputs.
- Check duplicated entities, exact sequences, or input strings when relevant.
- Do not equate distinct target IDs with distinct protein sequences.

### Preprocessing and labels

- Trace filtering, normalization, truncation/padding, and label/affinity transforms.
- Establish units and transform flags; do not treat KIBA scores as pKd by default.
- Check which entries survive preprocessing and participate in the loss or features.
- Determine whether transformations are fixed or fitted using train/test information.

### Folds and boundaries

- Establish whether the active loader reads shipped folds or generates runtime splits.
- Identify what each index addresses: an entity, matrix cell, or observed-pair array.
- Check index bounds against the applicable post-filter ordering, not just raw shape.
- Check coverage, duplicate indices, and train/validation/test overlap as needed.
- Compare pair overlap, drug overlap, and target overlap separately.
- Check sequence-level target leakage and duplicate-input overlap across boundaries.
- State cold-start semantics at the identifier and actual-input levels separately.
- Distinguish pretraining exposure, label leakage, and model-selection contamination.
- Do not infer runtime validity from a shipped fold file's internal consistency alone.

The Stage 04 reports discuss Co-VAE's generated splits and DCGAN-DTA's file-based
splits, along with filtering, duplicated identities, and pretraining overlap.
Use those sections as starting points, not as immutable claims about changed code.
Do not force both baselines into a shared split or data protocol during verification.

## Preservation and execution boundary

Unless explicitly authorized, do not modify datasets, rewrite or regenerate folds,
clean corrupted-looking data, normalize/canonicalize inputs, or patch loaders.
Document suspicious behavior and its consequences instead of silently correcting it.

Static inspection does not authorize training or loader execution. Small loader-level
or split-level execution requires explicit authorization in the current task.
Before authorized execution, inspect the entry point for training and write side effects;
if it cannot stay within the authorized boundary, report the limitation and stop.
Do not start model training from this skill.

## Report the bounded result

State the requested claim, evidence inspected, result, and remaining uncertainty.
Label observations, prior reported measurements, runtime measurements, and inference
separately. For leakage, use NONE, POSSIBLE, LIKELY, or CONFIRMED with supporting
evidence and the boundary checked; NONE is limited to the checked channel and scope.
Do not claim measured performance impact from overlap counts alone.

If new evidence contradicts an earlier finding, report the old finding with its
locator, show the new evidence, and explain the revised conclusion and its scope.
Do not silently overwrite audit history. Follow the current task and AGENTS.md
for any authorized report artifact; an inspection-only task returns findings only.

---
name: reproduction-feasibility
description: Assess whether a published ML method can be reproduced from its paper, released code, datasets, dependencies, and experimental protocol.
---

# Reproduction Feasibility Audit

Determine whether the published experiment can be reproduced from the available evidence and implementation.

Do not modify the implementation.

Do not begin full training unless explicitly requested.

---

# Evidence tags

Use:

[PAPER]
[CODE]
[VERIFIED]
[INFERRED]
[OPEN]
[PLANNED]

---

# Evaluate separately

For each project evaluate:

1. Repository completeness
2. Dependencies
3. Dataset availability
4. Preprocessing availability
5. Fold availability
6. Model implementation
7. Hyperparameter completeness
8. Training entry point
9. Evaluation entry point
10. Result generation
11. Paper-code consistency

---

# Status

For every requirement classify:

READY
RECOVERABLE
REQUIRES RECONSTRUCTION
BLOCKED

Definitions:

READY:
Available and sufficiently specified.

RECOVERABLE:
Missing but recoverable from public/repository evidence with limited effort.

REQUIRES RECONSTRUCTION:
Requires implementing or reconstructing unspecified methodology.

BLOCKED:
A necessary component cannot reasonably be recovered.

---

# Reproduction levels

Evaluate separately:

## Level A — Released-Code Reproduction

Can the released implementation be executed as-is?

## Level B — Controlled Reconstruction

Can missing pieces be reconstructed while preserving the intended methodology?

## Level C — Paper-Faithful Reimplementation

Can the methodology described in the paper be implemented independently?

Do not treat these three levels as equivalent.

---

# Feasibility matrix

Produce:

| Component | DCGAN-DTA | Co-VAE | Evidence | Status |
|---|---|---|---|---|

Components:

- Environment
- Dependencies
- Dataset
- Preprocessing
- Fold files
- Model
- Loss
- Training
- Evaluation
- Results

---

# Effort assessment

Estimate effort only after identifying concrete blockers.

Use:

LOW
MEDIUM
HIGH
VERY HIGH

Do not provide false precision such as "3.5 hours".

---

# Scientific risk

Identify:

- paper-code mismatch
- undocumented preprocessing
- split ambiguity
- leakage
- unavailable data
- unavailable dependencies
- missing hyperparameters
- ambiguous evaluation

---

# Final verdict

For each paper:

GREEN
YELLOW
RED

GREEN:
Released implementation is sufficiently complete for reproduction.

YELLOW:
Reproduction is possible but requires controlled reconstruction.

RED:
Substantial information or implementation is missing.

Then separately assess:

## Head-to-Head Comparability

Determine whether the two implementations can currently be compared fairly.

Do not answer this merely from model architecture.

Consider:

- datasets
- folds
- preprocessing
- evaluation
- available features
- training protocol
- reporting protocol

---

# Final sections

## Reproduction Status

## Blocking Issues

## Recoverable Issues

## Reconstruction Requirements

## Scientific Risks

## Head-to-Head Comparability

## Minimum Next Step

Do not propose model modifications.
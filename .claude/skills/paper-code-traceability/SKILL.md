---
name: paper-code-traceability
description: Trace published methodological claims to concrete implementation behavior and identify paper-code mismatches without modifying the repository.
---

# Paper-Code Traceability Audit

You are auditing whether a released implementation actually implements the methodology described in its paper.

The paper audit is the expected specification.

The repository is the implementation evidence.

Do not modify code.

Do not silently repair discrepancies.

---

# Evidence tags

Use:

[PAPER]
[CODE]
[VERIFIED]
[INFERRED]
[OPEN]

---

# Core principle

A matching variable name, class name, or README statement is NOT evidence of matching behavior.

Trace actual execution.

Prefer:

- execution flow
- function calls
- tensor operations
- loss construction
- optimizer steps
- data transformations
- fold loading

over naming conventions.

---

# Audit procedure

For each major paper claim:

1. Identify the paper claim.
2. Identify paper evidence.
3. Find the implementation.
4. Identify file path.
5. Identify class/function/method.
6. Trace callers and callees using GitNexus when useful.
7. Inspect actual behavior.
8. Classify the correspondence.

---

# Audit dimensions

## Dataset

Check:

- source
- files
- filtering
- preprocessing
- normalization
- missing values
- entity ordering
- interaction representation

## Data split

Check:

- train/test folds
- validation
- fold loading
- fold generation
- random seeds
- pair/entity split

## Representation

Check:

- drug representation
- protein representation
- feature dimensions
- transformations

## Architecture

Check:

- layer structure
- dimensions
- activation
- latent representation
- generator
- discriminator
- encoder
- decoder

## Objective

Check:

- reconstruction loss
- prediction loss
- adversarial loss
- KL divergence
- regularization
- co-regularization
- coefficients

## Training

Check:

- optimizer
- learning rate
- batch size
- epochs
- initialization
- update schedule
- checkpointing

## Evaluation

Check:

- metrics
- fold aggregation
- checkpoint selection
- result aggregation
- reported outputs

---

# Match status

Use exactly one:

MATCH
PARTIAL
MISMATCH
MISSING
UNCLEAR

Definitions:

MATCH:
The implementation behavior agrees with the paper.

PARTIAL:
Only part of the paper-described behavior is implemented or evidence is incomplete.

MISMATCH:
The implementation behavior differs materially.

MISSING:
The paper describes a component but no corresponding implementation was found.

UNCLEAR:
The implementation exists but its behavior cannot yet be established confidently.

---

# Traceability table

Produce:

| ID | Paper claim | Paper evidence | Code evidence | Status | Severity | Impact |
|---|---|---|---|---|---|---|

Severity:

CRITICAL
HIGH
MEDIUM
LOW

---

# Special DCGAN checks

Explicitly verify:

1. Is there a generator?
2. Is there a discriminator?
3. What enters the generator?
4. What enters the discriminator?
5. Is adversarial loss calculated?
6. Is adversarial loss optimized?
7. Are generator and discriminator updated separately?
8. Is there alternating optimization?
9. Does the actual optimization correspond to the paper's GAN objective?
10. Are any GAN components present in code but disconnected from training?

---

# Special Co-VAE checks

Explicitly verify:

1. Encoder
2. Latent mean
3. Latent variance/log-variance
4. Reparameterization
5. Decoder
6. Reconstruction loss
7. KL divergence
8. Co-regularization
9. Prediction objective
10. All loss coefficients
11. Whether all described losses participate in optimization

---

# Discrepancy protocol

For every discrepancy report:

### Paper behavior

### Code behavior

### Evidence

### Why they differ

### Reproducibility impact

### Scientific-comparison impact

Do not decide which version is "correct".

---

# Code behavior not described by paper

Also report important behavior that exists in code but is not described by the paper.

---

# Paper claims unsupported by code

Report paper claims for which no implementation evidence exists.

---

# Final sections

## Confirmed Matches

## Partial Matches

## Major Mismatches

## Missing Components

## Code Behavior Not Described in Paper

## Paper Claims Not Supported by Code

## Critical Issues for Reproduction

## Issues Affecting Head-to-Head Comparison
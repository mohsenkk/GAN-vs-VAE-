---
name: dataset-fold-audit
description: Audit datasets, preprocessing, interaction matrices, train-test folds, validation protocols, and information leakage in machine-learning research repositories.
---

# Dataset and Fold Audit

You are auditing the experimental data pipeline of a machine-learning research implementation.

The goal is to determine exactly what information enters training, representation learning, validation, and testing.

Do not modify data or source code.

---

# Evidence tags

Use:

[PAPER]
[CODE]
[VERIFIED]
[INFERRED]
[OPEN]

---

# Audit principles

Do not assume:

- zeros mean negative interactions;
- missing values are negatives;
- a fold file represents the same split described in the paper;
- preprocessing is train-only;
- similarity matrices are independent of test information;
- entity IDs have the same ordering across files.

Verify these explicitly.

---

# Dataset inventory

For every dataset identify:

- dataset name
- source
- files
- number of drugs
- number of targets
- number of observed interactions
- affinity representation
- missing values
- preprocessing
- normalization
- filtering
- entity ordering

---

# Interaction matrix audit

Determine:

- matrix dimensions
- observed cells
- missing cells
- zero-valued cells
- positive/negative semantics
- affinity transformation
- whether missing cells participate in loss
- whether missing cells participate in representation learning

---

# Fold audit

For every fold file determine:

- train indices
- test indices
- validation indices if any
- number of samples
- drug overlap
- target overlap
- pair overlap
- fold construction strategy

Classify the split:

PAIR-WISE
DRUG-WISE
TARGET-WISE
COLD-DRUG
COLD-TARGET
COLD-PAIR
OTHER
UNCLEAR

---

# Information-flow audit

Trace information flow from:

dataset
→ preprocessing
→ feature construction
→ similarity calculation
→ representation learning
→ model training
→ validation
→ test evaluation

Determine whether test information can influence:

- feature construction
- normalization
- similarity matrices
- graph construction
- vocabulary
- embeddings
- model selection
- hyperparameter selection

---

# Leakage classification

For each potential leakage:

NONE
POSSIBLE
LIKELY
CONFIRMED

Do not label something CONFIRMED without direct evidence.

---

# Cross-project comparison

For DCGAN-DTA and Co-VAE separately report:

- datasets
- split protocols
- fold files
- entity overlap
- preprocessing
- leakage risks

Do not force the two methods into identical protocols at this stage.

---

# Output

## Dataset Inventory

## Interaction Matrix Specification

## Fold Specification

## Information Flow

## Leakage Assessment

## Paper-Code Split Discrepancies

## Unknowns

## Reproduction-Critical Data Issues
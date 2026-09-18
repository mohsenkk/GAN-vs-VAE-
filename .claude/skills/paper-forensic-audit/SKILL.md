---
name: paper-forensic-audit
description: Forensic audit of scientific ML papers for reproducibility. Extracts methodology from text, figures, tables, equations, captions, appendices, and supplementary material without inventing unspecified details.
---

# Paper Forensic Audit

You are a scientific reproducibility auditor.

Your task is to reconstruct exactly what a published paper specifies about its methodology.

You are NOT performing a general paper summary.

You are NOT judging whether the method is good.

You are NOT comparing papers.

You are NOT checking the code yet.

Your primary objective is to produce a reproducibility-oriented specification of the paper.

## Core principle

Distinguish strictly between:

1. What the paper explicitly states.
2. What is visually represented in figures/tables.
3. What can be mathematically reconstructed from equations.
4. What is inferred.
5. What the paper leaves unspecified.

Never silently fill missing information.

---

# Evidence tags

Every substantive claim must use one of:

[PAPER]
[FIGURE]
[TABLE]
[EQUATION]
[SUPPLEMENT]
[INFERRED]
[OPEN]

Do not use [VERIFIED] in this skill.

[VERIFIED] is reserved for facts confirmed by executing or independently inspecting the implementation.

---

# Source hierarchy

Prefer evidence in this order:

1. Explicit equations
2. Explicit methodological text
3. Tables containing experimental configuration
4. Architecture figures
5. Figure/table captions
6. Supplementary material
7. Inference

If two sources disagree:

- report both;
- do not silently resolve the disagreement;
- mark the issue as an ambiguity.

---

# Mandatory PDF workflow

For each paper:

## Step 1 — Identify document structure

Determine:

- title
- authors
- venue
- year
- DOI
- sections
- subsections
- appendix
- supplementary files

## Step 2 — Extract text

Read the full searchable PDF.

Do not rely only on the abstract, introduction, or conclusion.

## Step 3 — Identify visual evidence

Locate every:

- architecture figure
- data-flow figure
- workflow diagram
- result plot
- ablation plot
- dataset table
- hyperparameter table
- performance table

For every methodology-relevant figure, inspect the actual figure, not only its caption.

If visual inspection is necessary, render the relevant PDF page or figure.

## Step 4 — Inspect equations

Identify every equation related to:

- model definition
- latent representation
- loss
- regularization
- prediction
- optimization

Record equation numbers.

## Step 5 — Inspect supplementary material

If supplementary files exist, inspect them.

Determine whether they contain:

- preprocessing
- dataset details
- architecture details
- hyperparameters
- experimental settings
- additional experiments
- ablations
- implementation details

---

# Required methodology extraction

## A. Problem formulation

Extract:

- task
- input
- output
- prediction target
- regression/classification formulation
- mathematical formulation

---

## B. Dataset

For every dataset extract:

- name
- source
- version if specified
- number of drugs
- number of targets
- number of interactions
- affinity measurements
- affinity units
- missing values
- filtering
- duplicates
- preprocessing
- normalization
- transformation

Do not assume missing values are zeros unless explicitly stated.

---

## C. Data split

Determine:

- train/test split
- validation split
- cross-validation
- number of folds
- fold generation procedure
- predefined vs generated folds
- random seed
- pair-wise split
- drug-wise split
- target-wise split
- cold-start/warm-start behavior if applicable

If the paper does not specify the exact procedure:

[OPEN]

---

## D. Drug representation

Determine exactly how drugs are represented.

Possible representations include:

- SMILES
- molecular fingerprints
- molecular descriptors
- graphs
- CNN inputs
- embeddings
- similarity matrices
- latent vectors

Record dimensionality whenever specified.

---

## E. Protein/target representation

Determine:

- sequence representation
- PSSM
- BLOSUM
- descriptors
- embeddings
- CNN input
- similarity matrix
- graph representation
- dimensionality

---

# F. Architecture reconstruction

Reconstruct the model layer by layer.

For every component record:

- component name
- input
- output
- dimensions
- number of layers
- activation
- kernel
- stride
- pooling
- dropout
- latent dimension

Use architecture figures as evidence.

Do not infer missing dimensions.

---

# G. GAN-specific audit

If the method is described as GAN-based, explicitly identify:

- generator
- discriminator
- generator input
- discriminator input
- discriminator output
- adversarial objective
- generator loss
- discriminator loss
- auxiliary prediction loss
- optimization schedule
- alternating updates
- update frequency

Do not assume that a discriminator class or GAN terminology implies actual adversarial optimization.

---

# H. VAE-specific audit

If the method is described as VAE-based, identify:

- encoder
- latent mean
- latent variance/log-variance
- reparameterization
- decoder
- reconstruction loss
- KL divergence
- auxiliary prediction loss
- co-regularization
- regularization coefficients

Do not assume that the presence of "VAE" in the title proves that all VAE components are implemented.

---

# I. Mathematical objective

For every important equation record:

- equation number
- variables
- terms
- coefficients
- optimization direction
- relationship to architecture

Then reconstruct:

L_total = ...

Only include terms that are supported by evidence.

If the total objective cannot be reconstructed:

[OPEN]

---

# J. Training configuration

Extract:

- optimizer
- learning rate
- scheduler
- batch size
- epochs
- early stopping
- initialization
- dropout
- weight decay
- latent dimensions
- hidden dimensions
- activation
- regularization coefficients
- noise parameters
- random seed
- hardware/software environment

Separate:

### Explicitly specified

from:

### Not specified

---

# K. Evaluation

Extract:

- evaluation datasets
- metrics
- metric definitions
- cross-validation aggregation
- mean/std
- repeated runs
- best-fold reporting
- best-run reporting
- checkpoint selection
- baseline models
- statistical testing

Pay special attention to whether results are:

- mean across folds
- best fold
- final epoch
- best run
- average across repeated experiments

---

# L. Figures

For every relevant figure produce:

| Figure | What it shows | Methodological information | Evidence |
|---|---|---|---|

Do not merely repeat the caption.

For architecture diagrams describe the actual information flow.

For plots identify:

- axes
- datasets
- metrics
- experimental groups
- error bars
- aggregation if shown

---

# M. Tables

For every relevant table produce:

| Table | Purpose | Important values | Experimental implication |
|---|---|---|---|

---

# N. Reproducibility specification

Produce these canonical sections:

## Dataset Specification

## Preprocessing Specification

## Drug Representation Specification

## Target Representation Specification

## Architecture Specification

## Mathematical Objective

## Training Specification

## Evaluation Specification

## Experimental Protocol

Every important value must include an evidence locator.

---

# Evidence locator

Use:

- Section
- Page
- Figure
- Table
- Equation
- Supplement

Example:

[PAPER] Section 3.2, p. 5

[FIGURE] Figure 2, p. 6

[TABLE] Table 3, p. 8

[EQUATION] Eq. (7), p. 7

[SUPPLEMENT] Supplementary Section S2

---

# Mandatory final sections

## Reproduction-Critical Facts

Only facts necessary for reproducing the published experiment.

## Paper-Level Unknowns

Only details that cannot be determined from the paper and supplementary material.

## Internal Ambiguities

Cases where different parts of the paper appear inconsistent.

## Claims Requiring Code Verification

List methodological claims that must later be checked against the repository.

## Paper Reproduction Checklist

A concise checklist of all information required before attempting reproduction.

---

# Restrictions

Do not:

- inspect source code in this phase;
- modify repository files;
- propose implementation fixes;
- judge model quality;
- compare against another paper;
- infer undocumented preprocessing;
- infer undocumented hyperparameters.

The output is an evidence-grounded paper specification.
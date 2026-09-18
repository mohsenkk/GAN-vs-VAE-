# Phase 2 — Paper Forensic Audit: Co-VAE

**Source of truth:** `Co-VAE/papaer.pdf` (13 pages, read in full: all text, all 10 figures, all 5 tables, all 10 numbered equations).
**Supplementary material:** `Co-VAE/supplementaries/` exists but is **empty**; the paper itself declares no supplementary material. [PAPER] (no "Supplementary Information" section exists) / [OPEN]
**Scope:** paper-only. No source code was inspected, no git history consulted, nothing executed, no file modified.

**Evidence tags:** `[PAPER]` `[FIGURE]` `[TABLE]` `[EQUATION]` `[SUPPLEMENT]` `[INFERRED]` `[OPEN]`.
`[SUPPLEMENT]` is unused in this document — there is no supplementary material to cite.

---

## 1. Executive Summary

**Problem.** Co-VAE predicts continuous **drug–target binding affinity** from a drug's SMILES string and a target's amino-acid sequence, specifically for **new drugs** and **new targets** not seen during training. [PAPER] Abstract p. 8861; §2.1 p. 8863

**Architecture.** Two variational autoencoders — one over drug SMILES, one over target sequences — coupled by a third "co-regularized block" (Reg block) that regresses the affinity from the two latent codes. Encoders are label-encoding → embedding → three GatedCNN layers → max-pooling → FC → (μ, σ); decoders are FC → three deconvolutional layers → FC. [PAPER] §2.5 p. 8865; [FIGURE] Fig. 4 p. 8866

**Datasets.** Davis (68 drugs × 442 targets, Kd → pKd) and KIBA (filtered to 2111 drugs × 229 targets, density 24.4%). [PAPER] §3.1 p. 8866

**Protocol.** Two settings — *new-drug* and *new-target*. Entities (drugs or targets) are randomly divided into **six equal folds**; five folds train, one fold tests. Each setting is repeated with **ten independent random splits**, and means ± standard deviations are reported. Hyperparameters (two filter lengths and λ) are chosen by **five-fold cross-validation on the training data only**. [PAPER] §3.3 pp. 8867–8868

**Sufficiently specified.** The variational formulation is derived rigorously (Theorem 2.1, Eqs. 5–10) with an explicit prior, posterior family, reparameterization, KL term and reconstruction term; the dataset identities, entity counts, sequence-length caps, embedding dimensions, vocabulary sizes, filter counts, dropout rate, epoch count, batch size, optimizer and learning rate are all stated numerically; the split granularity (entity-wise, 6 folds, 10 repetitions) is stated; five metrics are defined; four baselines plus four generative baselines are named.

**Important information missing.** No latent dimensionality (J_d, J_t); no Monte-Carlo sample counts (L_d, L_t); no random seed; no hardware; no software versions; no early stopping or checkpoint-selection rule; no weight initialization; no statement of how the ~75% missing entries of the KIBA affinity matrix are handled in the loss or the metrics; no numeric sizes for the validation partition that Fig. 6 depicts but the prose never mentions.

**Top reproducibility risks.**
1. **The sign and scale of λ are not reconstructable.** Eq. (9) defines the co-regularization weight as `−λ(y−ŷ)²` inside a quantity that is maximized, while Table 1 and §3.3 give λ ∈ {−3, −5}. Taken literally, a negative λ rewards prediction error. The paper never writes λ as an exponent. [EQUATION] Eq. (9) p. 8865; [TABLE] Table 1 p. 8868
2. **Validation data appears only in a figure caption.** Fig. 6's caption describes a three-way split (training / test / validation); §3.3's prose describes only training and test. Whether a validation partition exists at all, and how it is sized, is undetermined. [FIGURE] Fig. 6 p. 8867 vs [PAPER] §3.3 p. 8867
3. **The KIBA filtering rule as written cannot produce the stated result.** "We removed the drug-target pairs which have known affinities less than 10" cannot reduce 52,498 drugs to 2,111. [PAPER] §3.1 p. 8866
4. **Davis is ~72% censored at a single value.** Fig. 5 shows ≈21,500 of 30,056 Davis pairs at exactly pKd = 5. The paper acknowledges the distribution is "very uneven" but specifies no handling. [FIGURE] Fig. 5 p. 8866
5. **Latent dimensionality is never given**, so the model cannot be instantiated from the paper alone.

---

## 2. Bibliographic Information

| Field | Value | Evidence |
|---|---|---|
| Full title | *Co-VAE: Drug-Target Binding Affinity Prediction by Co-Regularized Variational Autoencoders* | [PAPER] p. 8861 |
| Authors | Tianjiao Li; Xing-Ming Zhao (Senior Member, IEEE); Limin Li | [PAPER] p. 8861 |
| Affiliation — Li, T. & Li, L. | School of Mathematics and Statistics, Xi'an Jiaotong University, Xi'an 710049, China | [PAPER] footnote p. 8861 |
| Affiliation — Zhao, X.-M. | Institute of Science and Technology for Brain-Inspired Intelligence, Fudan University, China; MOE Key Laboratory of Computational Neuroscience and Brain-Inspired Intelligence; MOE Frontiers Center for Brain Science, Shanghai 200433, China | [PAPER] footnote p. 8861 |
| Corresponding authors | Xing-Ming Zhao and Limin Li | [PAPER] footnote p. 8861 |
| Contact e-mails | litj468@stu.xjtu.edu.cn; xmzhao@fudan.edu.cn; liminli@mail.xjtu.edu.cn | [PAPER] footnote p. 8861 |
| Journal | IEEE Transactions on Pattern Analysis and Machine Intelligence | [PAPER] running head p. 8861 |
| Volume / issue | Vol. 44, No. 12 | [PAPER] p. 8861 |
| Pages | 8861–8873 | [PAPER] pagination |
| Issue date | December 2022 | [PAPER] p. 8861 |
| DOI | 10.1109/TPAMI.2021.3120428 | [PAPER] footnote p. 8861 |
| Manuscript received | 29 Oct. 2020 | [PAPER] footnote p. 8861 |
| Revised | 16 July 2021 | [PAPER] footnote p. 8861 |
| Accepted | 9 Oct. 2021 | [PAPER] footnote p. 8861 |
| Date of publication | 15 Oct. 2021 | [PAPER] footnote p. 8861 |
| Date of current version | 3 Nov. 2022 | [PAPER] footnote p. 8861 |
| Recommended for acceptance by | C. Liu | [PAPER] footnote p. 8861 |
| ISSN / rights | 0162-8828 © 2021 IEEE | [PAPER] p. 8861 |
| Index terms | Variational autoencoders; co-regularized VAE; drug-target binding affinity | [PAPER] p. 8861 |
| Funding | National Key R&D Program of China 2020YFA0712403; NSFC 11631102, 61932008, 61772308; Shanghai Municipal Science and Technology Major Project 2018SHZDZX01 | [PAPER] footnote p. 8861 |
| Repository URL | `https://github.com/LiminLi-xjtu/CoVAE` — stated twice: "Our codes are all based on Pytorch and available on …" (§3.3, p. 8868) and for the SARS-CoV-2 prediction results (§3.4, p. 8870) | [PAPER] |
| Supplementary material | **None declared.** The paper has no appendix, no supplementary-information section, and no supplementary file list. | [PAPER] / [OPEN] |

**Document structure** [PAPER]: §1 Introduction (§1.1 Related Work → §1.1.1 KronRLS, §1.1.2 DeepDTA, §1.1.3 DeepAffinity, §1.1.4 GraphDTA) → §2 Methods (§2.1 Problem Definition, §2.2 Variational Autoencoder, §2.3 The Co-VAE Model, §2.4 GatedCNN, §2.5 Network Structure in Co-VAE) → §3 Experiments and Results (§3.1 Datasets, §3.2 Evaluation Metrics, §3.3 Experimental Setting, §3.4 Experimental Results) → §4 Conclusion → References [1]–[38] → author biographies.

**No "Limitations" section exists.** [PAPER]

---

## 3. Problem Formulation

### 3.1 Task

> "Suppose we are given a set of drugs `D`, where each drug is represented by its SMILES string, and a set of protein targets `T`, where each target is represented by its amino-acid sequences. The affinity matrix `I` contains the affinity values among the above drugs and targets. For a given new drug `d` or a new target `t`, our goal in this work is to predict the affinity values between the drug `d` and the targets in `T`, or the values between the target `t` and the drugs in `D`."
> [PAPER] §2.1, p. 8863

| Aspect | Extraction | Evidence |
|---|---|---|
| Task type | **Regression** on a continuous affinity value. The paper explicitly contrasts this with "the first class … to predict binary DTIs" and describes the affinity class as "often formulate the affinity prediction problem as a regression problem". | [PAPER] §1 p. 8861; §1.1 p. 8862 |
| Input | A pair (drug SMILES string, target amino-acid sequence) | [PAPER] §2.1 p. 8863; [FIGURE] Fig. 4 p. 8866 |
| Output | Three outputs from one forward pass: (i) a scalar predicted affinity ŷ; (ii) a reconstructed drug SMILES string; (iii) a reconstructed target sequence | [FIGURE] Fig. 4 p. 8866 ("Drug SMILES", "affinity", "Target Sequence" output boxes); [PAPER] §2.5 p. 8865 |
| Predictive model | `p(y | x_d, x_t)` learned from training triplets `{x_d^i, x_t^i, y^i}_{i=1}^N` | [PAPER] §2.3 p. 8864 |
| Latent variables | `z_d` for the drug, `z_t` for the target | [PAPER] §2.3 p. 8864; [FIGURE] Fig. 2 p. 8864 |
| Prediction function | `ŷ^i = f_θy(z_d, z_t)` | [EQUATION] Eq. (9), p. 8865 |
| Secondary task | Generation of new drug SMILES sharing similar targets with an input drug | [PAPER] Abstract p. 8861; §3.4 p. 8870 |

### 3.2 Drug and target representation in the formulation

`x_d` and `x_t` are described as "the random vectors for a drug and a target"; `y` is "the random variable for the affinity values". [PAPER] §2.3 p. 8864

### 3.3 Treatment of unobserved / unknown interactions

This is explicitly addressed **only** for the derived classification metric:

> "By fixing a cutoff value of the ground truth affinity value, the drug-target pairs were labelled as positive and negative."
> [PAPER] §3.4, p. 8870

Therefore, for AUC, **negatives are observed pairs whose measured affinity falls below the cutoff** — not unobserved pairs. The paper does **not** treat unknown interactions as negatives anywhere. [PAPER]

The paper also criticises exactly that practice in prior work: "the pairs with missing values are often taken as pseudo negative pairs in the training process" is listed as one of the "two major limitations" of binary DTI methods. [PAPER] §1, p. 8861

**How missing entries of the affinity matrix are handled during training and evaluation is never stated.** Davis (68 × 442 = 30,056) is described without mention of missing values; KIBA is explicitly sparse ("affinity quantity density of 24.4%"), so ~75.6% of the 2111 × 229 matrix is unobserved, and no sentence in the paper says whether those cells are excluded from the reconstruction loss, the co-regularization loss, or the metrics. **[OPEN]**

---

## 4. Dataset Audit

### 4.1 Dataset table

| Property | Davis | Evidence | KIBA | Evidence |
|---|---|---|---|---|
| Name | Davis dataset | [PAPER] §3.1 p. 8866 | KIBA dataset | [PAPER] §3.1 p. 8866 |
| Reference | [20] = M. I. Davis et al., "Comprehensive analysis of kinase inhibitor selectivity", *Nature Biotechnol.*, vol. 29, no. 11, pp. 1046–1051, 2011 | [PAPER] refs p. 8873 | [21] = J. Tang et al., "Making sense of large-scale kinase inhibitor bioactivity data sets: a comparative and integrative analysis", *J. Chem. Inf. Model.*, vol. 54, no. 3, pp. 735–743, 2014 | [PAPER] refs p. 8873 |
| Prior use | "which were used in [11] previously" (DeepDTA) | [PAPER] §3.1 p. 8866 | same | [PAPER] §3.1 p. 8866 |
| SMILES source | PubChem database, by PubChem ID | [PAPER] §3.1 p. 8866 | same | [PAPER] §3.1 p. 8866 |
| Sequence source | UniProt, by gene name or RefSeq accession number | [PAPER] §3.1 p. 8866 | same | [PAPER] §3.1 p. 8866 |
| **Drugs (used)** | **68** | [PAPER] §3.1 p. 8866 | **2111** | [PAPER] §3.1 p. 8866 |
| **Targets (used)** | **442** | [PAPER] §3.1 p. 8866 | **229** | [PAPER] §3.1 p. 8866 |
| Matrix dimensions | 68 × 442 | [INFERRED] from the two counts | 2111 × 229 | [INFERRED] from the two counts |
| Total cells | 30,056 | [INFERRED] 68 × 442 | 483,419 | [INFERRED] 2111 × 229 |
| **Interaction count** | **Not stated.** "the pairwise affinities measured by Kd value" — no count given. | [OPEN] | **Not stated directly**; density 24.4% of 483,419 ⇒ ≈117,954 observed | [PAPER] density §3.1; count [INFERRED] |
| Density / sparsity | Not stated. Fig. 5's Davis histogram totals ≈30,000, consistent with a fully observed matrix. | [FIGURE] Fig. 5 p. 8866; [INFERRED] | "affinity quantity density of **24.4%**" | [PAPER] §3.1 p. 8866 |
| Raw affinity measure | Kd (kinase dissociation constant) | [PAPER] §3.1 p. 8866 | KIBA score, "obtained by combining the kinase inhibitor bioactivities from different sources such as Kd, Ki and IC50 through the method of KIBA [21]" | [PAPER] §3.1 p. 8866 |
| Raw affinity range | **[0.016, 10000]** | [PAPER] §3.1 p. 8866 | **[0, 17.2]** | [PAPER] §3.1 p. 8866 |
| Raw units | Not stated. The transform `pKd = −log₁₀(Kd/10⁹)` implies nM. | [INFERRED] |  Dimensionless KIBA score | [INFERRED] |
| Transformation | `pKd = −log₁₀(Kd / 10⁹)`, "same with [11]" (DeepDTA). Motivation given: "The affinities in Davis dataset are Kd, which are too large and affect accuracy sometimes." | [EQUATION] unnumbered, §3.1 p. 8866 | **None stated.** KIBA scores are used as-is. | [PAPER] §3.1; [OPEN] |
| Transformed range | ⇒ [5, 10.796] | [INFERRED] from the formula and [0.016, 10000] |  n/a | — |
| Observed distribution | Fig. 5 (left): x-axis 5–11; a single dominant bar at pKd ≈ 5 of height ≈21,500, i.e. **≈72% of all Davis pairs sit at the minimum value**; long thin right tail to ≈9.5 | [FIGURE] Fig. 5 p. 8866 | Fig. 5 (right): x-axis 8–18; mass spans ≈9.5–15.5; mode bar at ≈11.5–12 of height ≈72,000; histogram total ≈117,000 | [FIGURE] Fig. 5 p. 8866 |
| Paper's own comment on the distribution | "the affinity distribution in Davis dataset is very uneven" — given as the reason AUC is not computed on Davis | [PAPER] §3.2 p. 8867 | — | — |
| Pre-filter size | Not stated | [OPEN] | "The dataset contains **52498 drugs and 467 targets**, with a total of **246088 KIBA scores**" | [PAPER] §3.1 p. 8866 |
| Filtering rule | **None stated** | [OPEN] | "We removed the drug-target pairs which have known affinities less than 10 in the KIBA dataset" | [PAPER] §3.1 p. 8866 |
| Duplicate handling | Not stated | [OPEN] | Not stated | [OPEN] |
| Missing-value representation | Not stated | [OPEN] | Not stated | [OPEN] |
| Normalization (beyond the log transform) | Not stated | [OPEN] | Not stated | [OPEN] |
| Dataset version / access date | Not stated | [OPEN] | Not stated | [OPEN] |

### 4.2 Length-based selection (applies to both datasets)

> "According to [11], drug SMILES with the length less than 85 and protein sequences with length less than 1200 in Davis dataset, and drug SMILES with the length less than 100 and protein sequences with length less than 1000 in KIBA dataset cover at least 90% of the compound SMILES strings and 80% of the protein sequences in each dataset, respectively. So in our experiment, we chose the above length as the maximum length in the training and test data."
> [PAPER] §3.1, p. 8866

| Dataset | Max SMILES length | Max sequence length | Evidence |
|---|---|---|---|
| Davis | **85** | **1200** | [PAPER] §3.1 p. 8866; [PAPER] §3.3 p. 8868 (embedding dims (85,128) / (1200,128)) |
| KIBA | **100** | **1000** | [PAPER] §3.1 p. 8866; [PAPER] §3.3 p. 8868 (embedding dims (100,128) / (1000,128)) |

The paper states these are "the maximum length in the training and test data". Whether over-length entities are **truncated** or **excluded** is not stated; the coverage figures (90% / 80%) suggest exclusion but the paper does not say so. **[OPEN]**

---

## 5. Input Representation

### 5.1 Drug representation

| Aspect | Extraction | Evidence |
|---|---|---|
| Raw form | SMILES string | [PAPER] §2.1 p. 8863; §3.1 p. 8866 |
| Encoding step 1 | **Label encoding** — "Label encoding simply transforms each character in the input string to an integer based on an established character dictionary" | [PAPER] §2.5 p. 8865; [FIGURE] Fig. 4 p. 8866 |
| Dictionary construction | "we extract all the different characters in all the strings and create a dictionary where the key is a character and the value is an integer" | [PAPER] §3.1 p. 8866 |
| Vocabulary size | **64 different characters**, stated to hold "in both datasets" | [PAPER] §3.1 p. 8866 |
| Encoding step 2 | **Embedding layer** — "the embedding layer converts each integer into a vector with the same size" | [PAPER] §2.5 p. 8865 |
| Embedding dimension | **128** | [PAPER] §3.3 p. 8868 |
| Resulting input matrix | Davis **(85, 128)**; KIBA **(100, 128)** | [PAPER] §3.3 p. 8868 |
| Fingerprints / descriptors | **Not used** | [PAPER] — the paper contrasts itself with fingerprint methods; no fingerprint appears in Fig. 4 |
| Similarity features | **Not used.** Similarity matrices are described only as a property of the KronRLS/SimBoost baselines. | [PAPER] §1.1.1 p. 8862 |
| Padding | Not stated | [OPEN] |
| Normalization | Not stated | [OPEN] |

### 5.2 Target / protein representation

| Aspect | Extraction | Evidence |
|---|---|---|
| Raw form | Amino-acid sequence | [PAPER] §2.1 p. 8863 |
| Encoding | Identical pipeline: label encoding → embedding layer | [PAPER] §2.5 p. 8865; [FIGURE] Fig. 4 p. 8866 |
| Vocabulary size | **25 different characters**, "in both datasets" | [PAPER] §3.1 p. 8866 |
| Embedding dimension | **128** | [PAPER] §3.3 p. 8868 |
| Resulting input matrix | Davis **(1200, 128)**; KIBA **(1000, 128)** | [PAPER] §3.3 p. 8868 |
| PSSM / BLOSUM / descriptors | **Not used** | [PAPER] §2.5 — only label encoding + embedding is described |
| Similarity computation | **Not used** by Co-VAE | [PAPER] §1.1.1 p. 8862 (described only for KronRLS) |
| Alignment algorithm | **Not used** | [PAPER] |
| Padding | Not stated | [OPEN] |
| Normalization | Not stated | [OPEN] |

### 5.3 Interaction representation

| Aspect | Extraction | Evidence |
|---|---|---|
| Structure | Affinity matrix `I` over drugs × targets; training consumes triplets `(x_d^i, x_t^i, y^i)` | [PAPER] §2.1 p. 8863; §2.3 p. 8864 |
| Label type | Continuous scalar affinity (pKd for Davis, KIBA score for KIBA) | [PAPER] §3.1 p. 8866 |
| Masking | Not stated | [OPEN] |
| Missing values | Not stated for either dataset; KIBA is 24.4% dense so ~75.6% of cells are unobserved | [PAPER] density §3.1; handling [OPEN] |
| Positive / negative definition | Applies only to the AUC metric: pairs are labelled positive/negative by thresholding the **ground-truth affinity** at a cutoff in [10.5, 12.5] | [PAPER] §3.2 p. 8867; §3.4 p. 8870 |
| Unobserved pairs as negatives | **Not done.** No statement anywhere assigns a label to unobserved pairs. | [PAPER] |

---

## 6. Model Architecture

### 6.1 Data flow

```text
Drug SMILES                              Target Sequence
     ↓                                          ↓
Label encoding                           Label encoding
     ↓                                          ↓
Embedding (dim 128)                      Embedding (dim 128)
     ↓                                          ↓
gatedconv1 → gatedconv2 → gatedconv3     gatedconv1 → gatedconv2 → gatedconv3
     ↓                                          ↓
max-pooling                              max-pooling
     ↓                                          ↓
FC                                       FC
     ↓                                          ↓
(μ₁, σ₁) + ε  →  z_d = μ₁ + σ₁⊙ε         (μ₂, σ₂) + ε  →  z_t = μ₂ + σ₂⊙ε
  "Drug feature"                           "Target feature"
     │                    ╲              ╱                    │
     ↓                     ╲            ╱                     ↓
Drug decoder                ╲          ╱               Target decoder
FC→deconv1→deconv2           Reg block                 FC→deconv1→deconv2
  →deconv3→FC        (FC,FC per branch → FC,FC → FC)     →deconv3→FC
     ↓                          ↓                             ↓
Drug SMILES                  affinity                  Target Sequence
```
[FIGURE] Fig. 4, p. 8866 (structure); [PAPER] §2.5, p. 8865 (prose)

### 6.2 Component specification

#### Encoder (drug and target, identical structure)

| Property | Value | Evidence |
|---|---|---|
| Composition (prose) | "label encoding, embedding layer and GatedCNN blocks" | [PAPER] §2.5 p. 8865 |
| Composition (figure) | Label encoding → Embedding → gatedconv1 → gatedconv2 → gatedconv3 → max-pooling → **FC** | [FIGURE] Fig. 4 p. 8866 |
| GatedCNN block contents | "three 1D-gated CNN layers, two Rectified Linear Unit (ReLU) [23] activation functions and a max-pooling layer" | [PAPER] §2.5 p. 8865 |
| Activation placement | "There is a ReLU activation function after each 1D-convolutional layer." | [PAPER] §2.5 p. 8865 |
| Number of conv layers | 3 | [PAPER] §2.5; [FIGURE] Fig. 4 |
| Filter counts | **32×1; 32×2; 32×3** (i.e. 32, 64, 96) — "The numbers of filters in the second layer and third layer are twice and three times that of the first layer, respectively." | [TABLE] Table 1 p. 8868; [PAPER] §2.5 p. 8865 |
| Filter length (drug) | chosen from **{5, 7}** | [TABLE] Table 1 p. 8868; [PAPER] §3.3 p. 8868 |
| Filter length (target) | chosen from **{7, 11}** | [TABLE] Table 1 p. 8868; [PAPER] §3.3 p. 8868 |
| Stride | Not stated | [OPEN] |
| Padding | Not stated | [OPEN] |
| Pooling | Max-pooling, "at the end of encoder to compress the extracted features" | [PAPER] §2.5 p. 8865; [FIGURE] Fig. 4 |
| Pooling window / stride | Not stated | [OPEN] |
| Encoder FC output dim | Not stated | [OPEN] |
| Dropout in encoder | Not stated (dropout is described only in the Reg block) | [OPEN] |
| Normalization (BatchNorm etc.) | Not mentioned anywhere in the paper | [OPEN] |

#### GatedCNN gating mechanism

> "The input of GatedCNN layer is word embedding matrix. Different from ordinary convolution, the output of gated convolution is divided into two parts. One part is the convolution activation value, denoted by unit A, and the other part is the gate value, denoted by unit B, which functions as a gate unit to control the output of A. The final output is the multiplication of A and sigmoid B."
> [PAPER] §2.4, p. 8865; [FIGURE] Fig. 3, p. 8865

Formally: `output = A ⊙ σ(B)` where `[A; B]` is the channel-split convolution output. [FIGURE] Fig. 3 p. 8865. Reference [18] (Dauphin et al., "Language modeling with gated convolutional networks", ICML 2017). Motivation stated: "GatedCNN has been shown to outperform other language model such as LSTMs [19]". [PAPER] §2.4

#### Latent layer

| Property | Value | Evidence |
|---|---|---|
| Outputs | μ₁, σ₁ (drug); μ₂, σ₂ (target) | [FIGURE] Fig. 4 p. 8866 |
| Sampling | ε node feeding element-wise operations into "Drug feature" / "Target feature" | [FIGURE] Fig. 4 p. 8866 |
| **Latent dimension J_d, J_t** | **Never given a numeric value** anywhere in the paper or Table 1 | [OPEN] |

#### Decoder (drug and target, identical structure)

| Property | Value | Evidence |
|---|---|---|
| Composition | "a FC layer, three deconvolutional layers and another FC layer" | [PAPER] §2.5 p. 8865; [FIGURE] Fig. 4 p. 8866 |
| Role of first FC | "converts drug or target features into new features with right size for deconvolutional layers" | [PAPER] §2.5 p. 8865 |
| Role of final FC | "transforms deconvolutional layers output to drug or target which has the same size as the drug input or target input" | [PAPER] §2.5 p. 8865 |
| Filter counts | **32×3; 32×2; 32×1** (i.e. 96, 64, 32) | [TABLE] Table 1 p. 8868 |
| Deconvolution reference | [24] = Zeiler et al., "Deconvolutional networks", CVPR 2010 | [PAPER] §2.5 p. 8865 |
| Filter lengths in decoder | Not stated separately from the encoder | [OPEN] |
| Activations in decoder | Not stated | [OPEN] |
| Output parameterization | Categorical distribution over the character vocabulary — `log p_θd(x_d|z_d) = x_d^T log x̂_d` | [EQUATION] unnumbered, §2.3 p. 8865 |

#### Reg block (co-regularization / prediction head)

| Property | Value | Evidence |
|---|---|---|
| Composition (prose) | "we first use two FC layers for the features of drug and target, and then connect them by another FC layer" | [PAPER] §2.5 p. 8865 |
| Composition (figure) | **Five FC layers**: two in the drug branch (stacked), two in the target branch (stacked), then one wide merged FC producing the output | [FIGURE] Fig. 4 p. 8866 |
| Activation | "There is an activation function after each FC layer" — **type not named** | [PAPER] §2.5 p. 8865; [OPEN] |
| Dropout | "a dropout layer between every two FC layers" | [PAPER] §2.5 p. 8865 |
| Dropout rate | **0.2** | [TABLE] Table 1 p. 8868 |
| Output | "The final FC layer outputs a scalar which is a predicted affinity corresponding to the drug-target pair." | [PAPER] §2.5 p. 8865 |
| FC widths | **Never stated** for any FC layer in the model | [OPEN] |

### 6.3 Figure ↔ prose contradictions in the architecture

Reported here, **not resolved** (see §16 for the numbered audit):

- Fig. 4 shows an **FC layer inside the encoder** after max-pooling; §2.5's description of the encoder ("label encoding, embedding layer and GatedCNN blocks", with the GatedCNN block containing only conv layers, ReLUs and max-pooling) does not include it. [FIGURE] Fig. 4 vs [PAPER] §2.5 → **A5**
- §2.5 states the GatedCNN block has "three 1D-gated CNN layers, **two** ReLU activation functions" and, in the next sentence, "a ReLU activation function **after each** 1D-convolutional layer" (which implies three). [PAPER] §2.5 → **A4**
- Fig. 4's Reg block contains five FC layers; §2.5's wording most naturally reads as three. [FIGURE] Fig. 4 vs [PAPER] §2.5 → **A6**

---

## 7. Variational Formulation

### 7.1 Background VAE (§2.2, for reference)

Standard ELBO, Eq. (3), p. 8863:
```
L(θ, Φ; x^i) = E_{q_φ(z|x^i)} log p_θ(x^i|z) − D_KL( q_φ(z|x^i) ‖ p_θ(z) )
```
Monte-Carlo form, Eq. (4), p. 8864:
```
L(θ, Φ; x^i) ≈ (1/L) Σ_{l=1}^{L} log p_θ(x^i | z^{i,l})
             + (1/2) Σ_{j=1}^{J} ( 1 + log((σ_j^i)²) − (μ_j^i)² − (σ_j^i)² )
```
where `J` is the latent dimension. [EQUATION] Eqs. (3)–(4), pp. 8863–8864

### 7.2 Co-VAE specification

| Component | Specification | Evidence |
|---|---|---|
| **Prior — drug** | `p_θd(z_d) = N(0, I)` | [PAPER] §2.3 p. 8865 |
| **Prior — target** | `p_θt(z_t) = N(0, I)` | [PAPER] §2.3 p. 8865 |
| Prior independence | Assumed: "the independence between the priors `p_θd(z_d)` and `p_θt(z_t)`" — a stated assumption of Theorem 2.1 | [PAPER] Theorem 2.1, p. 8864; Eq. (8) p. 8864 |
| **Posterior — drug** | `log q_φd(z_d | x_d^i) = log N(z_d; μ_d^i, σ_d^{i,2} I)` — multivariate Gaussian with **diagonal** covariance | [PAPER] §2.3 p. 8865 |
| **Posterior — target** | `log q_φt(z_t | x_t^i) = log N(z_t; μ_t^i, σ_t^{i,2} I)` | [PAPER] §2.3 p. 8865 |
| Posterior factorization | `q_φ(z_d, z_t | x_d^i, x_t^i, y^i) = q_φd(z_d|x_d^i) q_φt(z_t|x_t^i)` — the joint posterior factorizes and **does not depend on y** | [PAPER] proof of Thm 2.1, p. 8864 |
| **μ, σ** | "μ_d^i and σ_d^i are outputs of encoder for x_d^i, and μ_t^i and σ_t^i are outputs of encoder for x_t^i" | [PAPER] §2.3 p. 8865 |
| Parameterization of σ | The paper writes **σ** (standard deviation), not log-variance. Eq. (10) uses `log((σ_{d,j}^i)²)`, i.e. the network output is treated as σ and squared. | [PAPER] §2.3; [EQUATION] Eq. (10) p. 8865 |
| **Reparameterization** | `z_d^{i,l} = μ_d^i + σ_d^i ⊙ ε_d^l`, `ε_d^l ~ N(0, I)`; `z_t^{i,l} = μ_t^i + σ_t^i ⊙ ε_t^l`, `ε_t^l ~ N(0, I)` | [PAPER] §2.3 p. 8865 |
| **KL divergence** | Analytic Gaussian form, appearing in Eq. (10) as `(1/2) Σ_{j=1}^{J_d} (1 + log((σ_{d,j}^i)²) − (μ_{d,j}^i)² − (σ_{d,j}^i)²)` and the analogous target term | [EQUATION] Eq. (10) p. 8865 |
| **Reconstruction likelihood** | Categorical: `log p_θd(x_d^i|z_d) = x_d^{iT} log x̂_d^i` and `log p_θt(x_t^i|z_t) = x_t^{iT} log x̂_t^i`, where `x̂_d^i = f_θd(z_d)` and `x̂_t^i = f_θt(z_t)`. The paper states: "let `p_θd(x_d|z_d)` and `p_θt(x_t|z_t)` be categorical distributions, whose probabilities are computed from `z_d` and `z_t` through the neural network". | [EQUATION] unnumbered, §2.3 p. 8865 |
| **Affinity likelihood** | Normal: "we further assume `p_θy(y|z_d, z_t)` to be a normal distribution, whose parameters are computed from `z_d` and `z_t` through the network" ⇒ **Eq. (9)**: `log p_θy(y^i | z_d, z_t) = −λ (y^i − ŷ^i)² + C`, with `ŷ^i = f_θy(z_d, z_t)`, `C` a constant, `λ` "the co-regularization parameter" | [EQUATION] Eq. (9), p. 8865 |

### 7.3 Theorem 2.1 and the objective

> **Theorem 2.1.** With the assumption of the independence between the priors `p_θd(z_d)` and `p_θt(z_t)`, the graphical model shown in Fig. 2 generates a lower bound of log likelihood of the joint distribution of `(x_d^i, x_t^i, y^i)` as
> ```
> log p_θ(x_d^i, x_t^i, y^i) ≥ L(θ, φ; x_d^i, x_t^i, y^i) =
>     E_{q_φd(z_d|x_d^i)} log p_θd(x_d^i|z_d) − D_KL( q_φd(z_d|x_d^i) ‖ p_θd(z_d) )     ← L_DrugVAE
>   + E_{q_φt(z_t|x_t^i)} log p_θt(x_t^i|z_t) − D_KL( q_φt(z_t|x_t^i) ‖ p_θt(z_t) )     ← L_TargetVAE
>   + E_{q_φ(z_d,z_t|x_d^i,x_t^i)} log p_θy(y^i | z_d, z_t)                              ← L_CoREG
> ```
> where `θ = {θ_d, θ_t, θ_y}` and `φ = {φ_d, φ_t}`.
> [EQUATION] Eq. (5), p. 8864

Compactly: **`L = L_DrugVAE + L_TargetVAE + L_CoREG`**, to be **maximized**. [PAPER] §2.3 p. 8865

Proof chain: Eq. (6) (joint ELBO) → Eq. (7) (factorization of the first term using the Fig. 2 graphical model `p_θ(x_d,x_t,y|z_d,z_t) = p_θd(x_d|z_d) p_θt(x_t|z_t) p_θy(y|z_d,z_t)`) → Eq. (8) (KL splits into two terms by prior independence) → Eq. (5). [EQUATION] Eqs. (6)–(8), p. 8864

### 7.4 The instantiated objective — Eq. (10)

At a single point `(x_d^i, x_t^i, y^i)`, the objective to maximize is:

```
L(θ, φ; x_d^i, x_t^i, y^i) =
      (1/L_d) Σ_{l_d=1}^{L_d}  x_d^{iT} log f_θd( z_d^{i,l_d} )
    + (1/2)   Σ_{j=1}^{J_d}   ( 1 + log((σ_{d,j}^i)²) − (μ_{d,j}^i)² − (σ_{d,j}^i)² )
    + (1/L_t) Σ_{l_t=1}^{L_t}  x_t^{iT} log f_θt( z_t^{i,l_t} )
    + (1/2)   Σ_{j=1}^{J_t}   ( 1 + log((σ_{t,j}^i)²) − (μ_{t,j}^i)² − (σ_{t,j}^i)² )
    − λ Σ_{l_d=1}^{L_d} Σ_{l_t=1}^{L_t} ( y^i − f_θy( z_d^{i,l_d}, z_t^{i,l_t} ) )²
```
[EQUATION] Eq. (10), p. 8865

Symbol glossary, as defined by the paper:
- `μ_{d,j}^i, μ_{t,j}^i, σ_{d,j}^i, σ_{t,j}^i` — the *j*th elements of `μ_d^i, μ_t^i, σ_d^i, σ_t^i`. [PAPER] §2.3 p. 8865
- `J_d, J_t` — "the dimensions for the latent variables `z_d` and `z_t`, respectively". **Numeric values never given.** [PAPER] §2.3 p. 8865; [OPEN]
- `L_d, L_t` — Monte-Carlo sample counts. **Never defined or valued in the paper**; introduced only implicitly in Eq. (10). [OPEN]
- `λ` — "the co-regularization parameter". [PAPER] §2.3 p. 8865

### 7.5 Unusual / ambiguous aspects of the parameterization

Recorded, **not resolved**:

1. **Sign and scale of λ.** Eq. (9) gives `log p_θy = −λ(y−ŷ)² + C`. For a Normal likelihood this requires `λ = 1/(2σ_y²) > 0`. Table 1 lists λ's "range" as `[-3,-5]` and §3.3 states "We choose **-3 or -5** for the regularization parameter λ". With λ negative and Eq. (10) maximized, the final term becomes `+3 Σ(y−ŷ)²` or `+5 Σ(y−ŷ)²` — i.e. it rewards squared error. The paper **never writes λ as an exponent** (e.g. `10^λ`), and never flags a sign convention. [EQUATION] Eq. (9); [TABLE] Table 1; [PAPER] §3.3 → **A1, CRITICAL**
2. **Asymmetric Monte-Carlo normalization.** The two reconstruction terms are averaged (`1/L_d`, `1/L_t`) but the co-regularization term is an **unnormalized double sum** over `L_d × L_t` samples. The effective weight on the affinity term therefore scales with `L_d·L_t`, whose values are unstated. [EQUATION] Eq. (10) → **A13**
3. **`σ` vs log-variance.** The encoder output is written as `σ` and squared inside the KL term; no softplus, exp, or positivity constraint is described. [PAPER] §2.3; [OPEN]
4. **No β or KL-annealing coefficient.** The KL terms carry coefficient 1; λ is the only stated weighting coefficient in the objective. [EQUATION] Eq. (10)

### 7.6 Reconstructed total objective

```
L_total  =  L_DrugVAE + L_TargetVAE + L_CoREG         (maximized)
```
with all three terms fully instantiated by Eq. (10), **except** that `λ`'s sign/scale (A1), `J_d`/`J_t`, and `L_d`/`L_t` are undetermined. The paper does not state whether the reported training loss is the negative of Eq. (10), nor how it is summed or averaged across a minibatch of 256. **[OPEN]**

---

## 8. Co-Regularization / Co-Training Mechanism

| Aspect | Extraction | Evidence |
|---|---|---|
| **The two branches** | A **drug VAE** (drug encoder + drug decoder) and a **target VAE** (target encoder + target decoder) | [PAPER] §2.3 p. 8864; §4 p. 8871; [FIGURE] Fig. 2 p. 8864, Fig. 4 p. 8866 |
| What each branch learns | Drug VAE: a latent `z_d` from which the drug SMILES string is reconstructed. Target VAE: a latent `z_t` from which the target sequence is reconstructed. | [PAPER] §4 p. 8871 |
| **How the branches interact** | Only through the **Reg block**: `ŷ = f_θy(z_d, z_t)`. The two encoders and decoders share no weights and exchange no information except via the shared affinity loss. | [FIGURE] Fig. 2 p. 8864, Fig. 4 p. 8866; [EQUATION] Eq. (5) p. 8864 |
| **Co-regularization loss** | `L_CoREG = E_{q_φ(z_d,z_t|·)} log p_θy(y^i|z_d,z_t)`, instantiated as `−λ Σ_{l_d}Σ_{l_t}(y^i − f_θy(z_d^{i,l_d}, z_t^{i,l_t}))²` | [EQUATION] Eq. (5) p. 8864; Eq. (10) p. 8865 |
| Paper's description of its role | "`L_CoREG` is a co-regularized term which represents the regression bound responsible for the affinity reconstruction penalty for a pair of drug and target" | [PAPER] §2.3 p. 8865 |
| **Weighting coefficient** | **λ**, the sole coefficient; candidate values −3 or −5 | [EQUATION] Eq. (9); [TABLE] Table 1; [PAPER] §3.3 |
| **Training schedule** | **Not described.** The paper states a single joint objective (Eq. 5 / Eq. 10) and gives one epoch count, one batch size, one optimizer and one learning rate. | [PAPER] §3.3 p. 8868 |
| Alternating optimization | **Not described anywhere.** [INFERRED]: the single summed objective implies simultaneous joint optimization of `θ_d, θ_t, θ_y, φ_d, φ_t`. | [INFERRED] from Eq. (5); **not an explicit author claim** |
| E-step / M-step | **Not present.** The paper does not use EM. | [PAPER] |
| Which variables update when | Not stated separately; `θ = {θ_d, θ_t, θ_y}` and `φ = {φ_d, φ_t}` are presented as jointly optimized | [PAPER] §2.3 p. 8864 |
| **Stopping criterion** | **Not stated.** Only a fixed epoch budget of 100 is given; no early stopping, no patience, no convergence test. | [PAPER] §3.3 p. 8868; [OPEN] |
| Theoretical claim | "We theoretically show that the Co-VAE is to maximize the lower bound of log likelihood of the joint distribution of triplets (drug, target, affinity) under certain assumptions." | [PAPER] §1 contribution 3, p. 8862; Theorem 2.1 p. 8864 |

---

## 9. Training Procedure

| Parameter | Value | Evidence |
|---|---|---|
| **Optimizer** | **Adam** [14] | [PAPER] §3.3 p. 8868 |
| **Learning rate** | **0.001** | [PAPER] §3.3 p. 8868; [TABLE] Table 1 p. 8868 |
| Adam β₁, β₂, ε | Not stated | [OPEN] |
| LR schedule | Not stated | [OPEN] |
| Weight decay | Not stated | [OPEN] |
| **Batch size** | **256** | [PAPER] §3.3 p. 8868; [TABLE] Table 1 p. 8868 |
| **Epochs** | **100** | [PAPER] §3.3 p. 8868; [TABLE] Table 1 p. 8868 |
| **Dropout rate** | **0.2** (Reg block; "between every two FC layers") | [TABLE] Table 1 p. 8868; [PAPER] §2.5 p. 8865 |
| Weight initialization | Not stated | [OPEN] |
| **Random seed** | **Not stated anywhere in the paper** | [OPEN] |
| Early stopping | Not stated | [OPEN] |
| Patience | Not stated | [OPEN] |
| Checkpointing / model selection | Not stated | [OPEN] |
| Gradient clipping | Not stated | [OPEN] |
| **Number of repetitions** | **10** random splits per setting per dataset | [PAPER] §3.3 p. 8867 |
| **Number of folds (split)** | **6** (5 train + 1 test) | [PAPER] §3.3 p. 8867 |
| **Number of folds (hyperparameter search)** | **5**, "based on only training data" | [PAPER] §3.3 p. 8868 |
| Hyperparameter search strategy | Five-fold CV over a small discrete grid (see §13) | [PAPER] §3.3 p. 8868 |
| **Framework** | **PyTorch** — "Our codes are all based on Pytorch" | [PAPER] §3.3 p. 8868 |
| Framework version | Not stated | [OPEN] |
| **Hardware / GPU / CPU** | **Not stated anywhere** (full-text search for GPU, CUDA, hardware returns nothing) | [OPEN] |
| Training wall-clock time | Not stated | [OPEN] |
| Parameter count | Not stated | [OPEN] |
| Third-party tools | **RDKit** [30], used to validate generated SMILES | [PAPER] §3.2 p. 8867; §3.4 p. 8870 |

---

## 10. Data Splitting and Cross-Validation

### 10.1 What the paper states verbatim

> "To evaluate the performance of affinity prediction, we compare the Co-VAE method with DeepDTA [11], KronRLS [9], DeepAffinity [12] and GraphDTA [13] for two settings, a new-drug setting and a new-target setting. **In the new-drug setting, drugs were randomly divided into six equal folds, five of which were considered as training drugs and the remaining fold was considered as the test drugs.** Once the Co-VAE model is learnt on the training data, which consists of the training drugs and all the targets, it is used to predict the pairwise affinities between the test drugs and all the targets. **In the new-target setting, we randomly divided the targets to six folds.** Similarly to the new-target setting, the Co-VAE model is learnt and used to predict the affinities between new targets and all the drugs. The two settings are shown in Fig. 6. **For each setting in each dataset, we randomly split the drugs or targets for ten times, and the means and standard deviations for each evaluation metrics were reported. For a fair comparison, the same settings were used for all the comparison methods.**"
> [PAPER] §3.3, p. 8867

> "We then determined the remaining hyperparameters by the strategy of **five-fold cross-validation based on only training data**."
> [PAPER] §3.3, p. 8868

> Fig. 6 caption: "The two experimental settings illustrated in affinities matrix from Davis dataset [20], where the rows and columns correspond to the drugs and targets, respectively. **The affinities matrix is split into training data, test data (white part) and validation data (black part).**"
> [FIGURE] Fig. 6 caption, p. 8867

### 10.2 Extraction

| Question | Answer | Evidence |
|---|---|---|
| Number of folds (split) | **6** — five training folds, one test fold | [PAPER] §3.3 p. 8867 |
| Fold construction | **Entity-wise**, over drugs (new-drug) or targets (new-target) | [PAPER] §3.3 p. 8867 |
| Random vs deterministic | **Random** — "randomly divided", "randomly split … for ten times" | [PAPER] §3.3 p. 8867 |
| Drug-wise split | Yes — the *new-drug* setting. Fig. 6 top panel shows whole **rows** held out across all targets. | [PAPER] §3.3; [FIGURE] Fig. 6 p. 8867 |
| Target-wise split | Yes — the *new-target* setting. Fig. 6 bottom panel shows whole **columns** held out across all drugs. | [PAPER] §3.3; [FIGURE] Fig. 6 p. 8867 |
| Pair-wise (warm-start) split | **Not performed.** No random-pair setting appears anywhere. | [PAPER] |
| Is this leave-one-fold-out CV over all 6 folds? | **Not stated.** The text describes one held-out fold per split, then says the split is repeated ten times randomly. It does not say the six folds are rotated. | [OPEN] |
| Training data composition | "the training drugs and all the targets" (new-drug); symmetric for new-target | [PAPER] §3.3 p. 8867 |
| Test data composition | Pairs between test entities and all entities of the other type | [PAPER] §3.3 p. 8867 |
| **Does a validation partition exist?** | **Contradictory.** Fig. 6's caption names three partitions including validation (black); §3.3's prose names only training and test. | [FIGURE] Fig. 6 vs [PAPER] §3.3 → **A2** |
| Validation partition size | Never stated | [OPEN] |
| **Hyperparameter selection** | Five-fold CV "based on only training data" over {drug filter length} × {target filter length} × {λ} | [PAPER] §3.3 p. 8868 |
| **Is test data used during model selection?** | The paper states the five-fold CV uses **only training data**, which asserts it is not. Whether the Fig. 6 "validation" partition plays a role, and how it relates to that five-fold CV, is undetermined. | [PAPER] §3.3; [OPEN] |
| Do folds overlap? | Not stated; "six equal folds" implies disjointness | [INFERRED] |
| **Random seed** | **Not reported** | [OPEN] |
| Same split reused across experiments? | Yes across **methods** — "For a fair comparison, the same settings were used for all the comparison methods." Whether the same ten splits are reused across the two settings or across datasets is not stated. | [PAPER] §3.3; partial [OPEN] |
| Repetitions | **10** per (dataset × setting) | [PAPER] §3.3 p. 8867 |
| Aggregation | Mean and standard deviation across the ten repetitions | [PAPER] §3.3 p. 8867 |

### 10.3 Arithmetic note

Davis has 68 drugs. 68 / 6 = 11.33, so "six **equal** folds" cannot be literally satisfied for the Davis new-drug setting. Similarly 442 / 6 = 73.67 (Davis targets), 2111 / 6 = 351.83 (KIBA drugs), 229 / 6 = 38.17 (KIBA targets). The paper does not describe a remainder-handling rule. [INFERRED] from [PAPER] §3.1, §3.3 → **A10**

---

## 11. Evaluation Metrics

Five metrics are used: "we used five evaluation metrics in our experiments to compare our method and the existing methods. To evaluate the Co-VAE's performance on generating new drugs, we use two metrics validity and uniqueness." [PAPER] §3.2, p. 8867

### 11.1 Concordance Index (CI)

```
CI = (1/Z) Σ_{δ_i > δ_j} Φ( b_i − b_j )

Φ(x) = 1    if x > 0
       0.5  if x = 0
       0    if x < 0
```
[EQUATION] unnumbered, §3.2, p. 8867

- `b_i` = prediction for the larger true value `δ_i`; `b_j` = prediction for the smaller true value `δ_j`; `Z` = a normalization constant. [PAPER] §3.2
- Interpretation given: "CI measures the probability that the predicted interactions are in the same order as the observed interactions, and its value ranges from 0 to 1. The closer the value is to 1, the better the model fits." [PAPER] §3.2
- Cited to [9] (KronRLS) and [29] (Gönen & Heller). [PAPER] §3.2
- Positive class / threshold: n/a (ranking metric).
- Per-fold vs global computation: **not stated** [OPEN]

### 11.2 MSE and MAE

As printed in §3.2, p. 8867:
```
MSE = Σ_i ( y_i − ŷ_i )²          MAE = Σ_i | y_i − ŷ_i |
```
[EQUATION] unnumbered, §3.2, p. 8867

**Note:** both are printed **without a 1/n normalization**, whereas the DeepDTA loss quoted in §1.1.2 (p. 8863) is written `(1/n) Σ (P_i − Y_i)²`. The reported Davis/KIBA MSE values (0.42–0.96, Table 2) are of the magnitude of a mean, not a sum over tens of thousands of pairs. → **A3**

- Positive class / threshold: n/a.
- Aggregation: mean ± std across the ten repetitions. [PAPER] §3.3
- Per-fold vs global: **not stated** [OPEN]

### 11.3 r²m index

```
r²m = r² * ( 1 − √( r² − r₀² ) )
```
[EQUATION] unnumbered, §3.2, p. 8867

- "`r²` and `r₀²` are the squared correlation coefficients with and without intercept, respectively. Larger `r²m` values for the test set indicate a more acceptable model." [PAPER] §3.2
- Attributed to DeepDTA: "The `r²m` index, which was also used in DeepDTA, can be used to evaluate the external predictive performance of quantitative structure–activity relationship (QSAR) models." [PAPER] §3.2
- No threshold for acceptability is stated in this paper.
- **No standard deviation is reported for `r²m` in Table 2**, unlike the other three metrics. → **A8**

### 11.4 AUC (AUROC)

- Definition given verbally only: "The area under the ROC curve (AUC) could evaluate the performance of a classification model. The AUC can be considered as the probability that the classifier will rank a randomly chosen positive instance higher than a randomly chosen negative instance [31]. The range of AUC is 0 to 1, and a larger AUC value indicates better performance of a classifier." [PAPER] §3.2, p. 8867. **No formula.**
- **Positive class definition:** "By fixing a cutoff value of the ground truth affinity value, the drug-target pairs were labelled as positive and negative." [PAPER] §3.4, p. 8870 ⇒ positives are observed pairs with ground-truth affinity above the cutoff. [INFERRED] for the direction (the paper does not state which side is positive).
- **Threshold:** "we take thresholds in the interval **[10.5, 12.5]** with increment **0.1**" [PAPER] §3.2, p. 8867. §3.4 refers to the same range as "(10.5,12.5)" [PAPER] p. 8870 → **A12**
- **Dataset restriction:** "We only use AUC for the KIBA dataset, since the affinity distribution in Davis dataset is very uneven." [PAPER] §3.2, p. 8867
- Reported as a curve of AUC versus threshold, not a single number. [FIGURE] Fig. 10, p. 8870
- Mean/std across the ten repetitions: **not indicated** in Fig. 10 (no error bands visible). [FIGURE] Fig. 10; [OPEN]

### 11.5 Validity and Uniqueness (generation metrics)

- **Validity** = "the proportion of valid drug SMILES sequences in all the generated drug SMILES sequences, where valid drug SMILES are determined through the tool Rdkit [30]" [PAPER] §3.2, p. 8867
- **Uniqueness** = "the proportion of new valid drugs among the overall valid drugs, where new valid drug means the valid drug SMILE sequences that never appear in the training data set" [PAPER] §3.2, p. 8867
- Number of sequences generated, sampling procedure, and which split's training set defines "new": **not stated** [OPEN]

### 11.6 Aggregation summary

| Question | Answer | Evidence |
|---|---|---|
| Aggregation unit | The **ten random splits** | [PAPER] §3.3 p. 8867 |
| Mean reported | Yes, for CI, MSE, MAE | [TABLE] Table 2 p. 8869 |
| Std reported | Yes for CI, MSE, MAE; **no** for `r²m` | [TABLE] Table 2 p. 8869 |
| Confidence intervals | **Not reported** | [TABLE] Table 2; [OPEN] |
| Statistical significance testing | **None performed anywhere in the paper** | [PAPER] |
| Best-fold vs mean-fold reporting | Table 2's caption says "The **Average** CI, MSE, MAE and `r²m`" | [TABLE] Table 2 p. 8869 |
| Epoch / checkpoint used for evaluation | **Not stated** | [OPEN] |

---

## 12. Baselines and Comparisons

### 12.1 Affinity-prediction baselines

| Name | Architecture (as described by this paper) | Reference | Reimplemented / rerun? | Datasets | Metrics |
|---|---|---|---|---|---|
| **KronRLS** | Kernel regularized least squares over Kronecker-product drug and target kernels built from similarity matrices; solves `(K + λI)a = y`; "can only capture linear dependencies in the training data" | [9] Pahikkala et al., *Brief. Bioinform.* 2015 | **Rerun.** "For a fair comparison, the same settings were used for all the comparison methods." [PAPER] §3.3 p. 8867 | Davis, KIBA | CI, MSE, MAE, r²m, AUC |
| **DeepDTA** | Two CNN blocks (three 1D-conv layers each) over label-encoded drug and target, then FC layers; MSE loss | [11] Öztürk et al., *Bioinformatics* 2018 | **Rerun**, sharing Co-VAE's hyperparameter protocol: "The settings of the hyperparameters are almost the same for the two methods, except that the Co-VAE model has an extra regularization parameter λ" [PAPER] §3.3 p. 8868 | Davis, KIBA | CI, MSE, MAE, r²m, AUC |
| **DeepAffinity** | Seq2Seq [37] pretrained unsupervised autoencoder producing deep features for drugs and proteins, then a regression model | [12] Karimi et al., *Bioinformatics* 2019 | **Rerun with default parameters** — "DeepAffinity [12] and GraphDTA [13] were performed using the default parameters." [PAPER] §3.3 p. 8868 | Davis, KIBA | CI, MSE, MAE, r²m, AUC |
| **GraphDTA** | RDKit converts SMILES to a molecular graph; GCN extracts drug features; CNN extracts target features; FC layers produce affinity | [13] Nguyen et al., *Bioinformatics* 2021 | **Rerun with default parameters** [PAPER] §3.3 p. 8868 | Davis, KIBA | CI, MSE, MAE, r²m, AUC |

**Key determination:** the paper states that the same splits were applied to all comparison methods and that two of the four were run "using the default parameters". Taken together these are statements of **independent execution by the authors**, not transcription from prior publications. The paper **never** says any baseline number was taken from another publication. [PAPER] §3.3, p. 8867–8868

Also discussed in §1.1 but **not** included in the experiments: SimBoost [10]. [PAPER] §1.1.1

### 12.2 Generative baselines (KIBA only)

| Name | Reference | Evidence |
|---|---|---|
| GAN | [32] Goodfellow et al., NeurIPS 2014 | [PAPER] §3.4 p. 8870; [TABLE] Table 3 p. 8870 |
| SeqGAN | [34] Yu et al., AAAI 2017 | [PAPER] §3.4; [TABLE] Table 3 |
| AAE | [33] Kadurin et al., *Mol. Pharmaceutics* 2017 | [PAPER] §3.4; [TABLE] Table 3 |
| VAE | [14] Kingma & Welling | [PAPER] §3.4; [TABLE] Table 3 |

Paper's remark on two of them: "In our experiments GAN and seqGAN could not generate effective drugs possibly due to the small training data." [PAPER] §3.4, p. 8870

Paper's own caveat on this comparison: "we note that these methods are all less effective than other computational methods in drug discovery such as REINVENT [35] and ORGAN [36]. The reason might be that the Co-VAE and the comparison methods only use a relatively small dataset including drug and target structures, while other computational methods such as REINVENT or ORGAN use much larger datasets which include structures and other characteristics like solubility." [PAPER] §3.4, p. 8870

---

## 13. Hyperparameter Search

Table 1's header column is literally labelled "range", though several rows contain single fixed values. [TABLE] Table 1, p. 8868 → **A15**

| Parameter | Value / Range | Dataset | Purpose | Evidence |
|---|---|---|---|---|
| Number of filters in encoder | `32*1; 32*2; 32*3` (= 32, 64, 96) | both | **Fixed** — "We fixed some of the hyperparameters …, including the number of filters in encoder, the number of filters in decoder and … the rate of dropout" | [TABLE] Table 1; [PAPER] §3.3 p. 8868 |
| Number of filters in decoder | `32*3; 32*2; 32*1` (= 96, 64, 32) | both | **Fixed** | [TABLE] Table 1; [PAPER] §3.3 |
| Filter length (drug SMILES) | **{5, 7}** — "The filter size for drug SMILES strings was chosen from 5 and 7" | both | **Searched** by 5-fold CV on training data | [TABLE] Table 1; [PAPER] §3.3 p. 8868 |
| Filter length (target sequence) | **{7, 11}** — "we selected the corresponding filter size from the numbers 7 and 11" | both | **Searched** by 5-fold CV | [TABLE] Table 1; [PAPER] §3.3 |
| Rate of dropout | **0.2** | both | **Fixed** | [TABLE] Table 1; [PAPER] §3.3 |
| λ (co-regularization) | **{−3, −5}** — "We choose -3 or -5 for the regularization parameter λ through cross-validation" | both | **Searched** by 5-fold CV | [TABLE] Table 1; [PAPER] §3.3 |
| Epoch | **100** | both | Fixed | [TABLE] Table 1; [PAPER] §3.3 |
| Batch size | **256** | both | Fixed | [TABLE] Table 1; [PAPER] §3.3 |
| Learning rate | **0.001** | both | Fixed | [TABLE] Table 1; [PAPER] §3.3 |
| Embedding dimension | **128** | both | Fixed (stated in prose, absent from Table 1) | [PAPER] §3.3 p. 8868 |
| Max SMILES length | 85 (Davis) / 100 (KIBA) | per dataset | Fixed | [PAPER] §3.1, §3.3 |
| Max sequence length | 1200 (Davis) / 1000 (KIBA) | per dataset | Fixed | [PAPER] §3.1, §3.3 |
| **Latent dimension (J_d, J_t)** | **Not given** | — | — | [OPEN] |
| **MC sample counts (L_d, L_t)** | **Not given** | — | — | [OPEN] |
| **FC layer widths** | **Not given** for any FC layer | — | — | [OPEN] |
| Pooling window / stride | **Not given** | — | — | [OPEN] |
| Conv stride / padding | **Not given** | — | — | [OPEN] |
| Weight decay | **Not given** | — | — | [OPEN] |
| Selected final values | **Not reported.** The paper gives the search grid but never states which filter lengths or which λ were actually selected for Davis or for KIBA. | — | — | [OPEN] |

**Search procedure:** five-fold cross-validation on training data only, over a grid of 2 × 2 × 2 = 8 configurations. [PAPER] §3.3, p. 8868. Selection criterion (which metric was optimized): **[OPEN]**.

---

## 14. Ablation / Sensitivity Experiments

**No component-removal ablation is reported anywhere in the paper.** There is no experiment that removes the drug VAE, the target VAE, the co-regularization term, the GatedCNN gating, or the KL term. [PAPER]

The paper contains no sensitivity analysis over λ, over the latent dimension, or over the filter lengths — the hyperparameter search (§13) is described as a selection procedure, and its per-configuration results are not reported. [PAPER] §3.3; [OPEN]

Experiments that are present but are **comparisons rather than ablations**:

| Experiment | Control | Metric | Dataset | Paper's conclusion | Evidence |
|---|---|---|---|---|---|
| Affinity prediction vs 4 baselines | KronRLS, DeepDTA, DeepAffinity, GraphDTA | CI, MSE, MAE, r²m | Davis, KIBA (both settings) | "our Co-VAE performed the best overall" | [TABLE] Table 2 p. 8869; [PAPER] §3.4 |
| Scatter of predicted vs true | same four baselines | visual | Davis, KIBA | "the Co-VAE tend to generate the most concentrated and less outlying points" | [FIGURE] Figs. 7–8 pp. 8868–8869 |
| Per-drug MSE | same four baselines | MSE per test drug | Davis | "the Co-VAE could obtain the best affinities or slightly worse affinities than the best prediction for almost all the drugs" | [FIGURE] Fig. 9 p. 8870 |
| AUC vs threshold sweep | same four baselines | AUC over cutoffs 10.5–12.5 | KIBA | "our method Co-VAE performs stably better than the other four methods for both new-drug setting and new-target setting", with two noted exceptions | [FIGURE] Fig. 10 p. 8870; [PAPER] §3.4 |
| Drug generation quality | GAN, SeqGAN, AAE, VAE | Validity %, Uniqueness % | KIBA | "The Co-VAE could obtain the highest validity and uniqueness among all the models" | [TABLE] Table 3 p. 8870 |
| SARS-CoV-2 case study | none | top-5 predicted drugs per gene | KIBA-trained model applied to NCBI SARS-CoV-2 proteins | "The predicted top drugs all have affinities larger than 12, which implies that the drugs are highly likely to have interactions" | [PAPER] §3.4 p. 8870 |

---

## 15. Reported Results

### 15.1 Table 2 — Average CI, MSE, MAE and r²m (five methods, two datasets, two settings)

[TABLE] Table 2, p. 8869. Caption: "The Average CI, MSE, MAE and `r²m` Obtained by Five Methods for the Davis Dataset and KIBA Dataset With Two Settings". Values are mean (std) across the ten random splits; `r²m` carries **no std**. Bold in the source marks the best value per column-block.

**Davis — new-drug setting**

| Method | CI (std) | MSE (std) | MAE (std) | r²m |
|---|---|---|---|---|
| KronRLS | 0.656 (0.033) | 0.796 (0.231) | 0.606 (0.024) | 0.143 |
| DeepDTA | 0.671 (0.023) | 0.726 (0.091) | **0.527 (0.033)** | 0.122 |
| DeepAffinity | 0.673 (0.057) | 0.955 (0.140) | 0.639 (0.040) | 0.114 |
| GraphDTA | **0.732 (0.067)** | 0.840 (0.058) | 0.570 (0.020) | **0.176** |
| **Co-VAE** | 0.712 (0.063) | **0.724 (0.096)** | 0.550 (0.048) | 0.107 |

**Davis — new-target setting**

| Method | CI (std) | MSE (std) | MAE (std) | r²m |
|---|---|---|---|---|
| KronRLS | **0.836 (0.014)** | 0.429 (0.055) | 0.428 (0.025) | 0.448 |
| DeepDTA | 0.780 (0.071) | 0.490 (0.095) | 0.456 (0.047) | 0.433 |
| DeepAffinity | 0.803 (0.006) | 0.477 (0.019) | 0.434 (0.050) | 0.444 |
| GraphDTA | 0.778 (0.016) | 0.457 (0.009) | 0.397 (0.011) | 0.429 |
| **Co-VAE** | 0.816 (0.011) | **0.419 (0.016)** | **0.380 (0.013)** | **0.477** |

**KIBA — new-drug setting**

| Method | CI (std) | MSE (std) | MAE (std) | r²m |
|---|---|---|---|---|
| KronRLS | 0.729 (0.005) | 0.449 (0.005) | 0.457 (0.009) | 0.355 |
| DeepDTA | 0.728 (0.007) | 0.466 (0.022) | 0.419 (0.014) | 0.342 |
| DeepAffinity | 0.707 (0.016) | 0.526 (0.024) | 0.491 (0.023) | 0.283 |
| GraphDTA | 0.728 (0.008) | 0.442 (0.012) | 0.447 (0.019) | **0.379** |
| **Co-VAE** | **0.742 (0.011)** | **0.416 (0.004)** | **0.401 (0.007)** | 0.358 |

**KIBA — new-target setting**

| Method | CI (std) | MSE (std) | MAE (std) | r²m |
|---|---|---|---|---|
| KronRLS | 0.591 (0.025) | 0.825 (0.042) | 0.603 (0.010) | **0.399** |
| DeepDTA | 0.722 (0.017) | 0.430 (0.016) | 0.421 (0.013) | 0.114 |
| DeepAffinity | 0.700 (0.007) | 0.490 (0.016) | 0.470 (0.011) | 0.324 |
| GraphDTA | 0.658 (0.050) | 0.519 (0.045) | 0.504 (0.027) | 0.314 |
| **Co-VAE** | **0.741 (0.010)** | 0.421 (0.010) | **0.396 (0.005)** | 0.356 |

Paper's narrative reading of Davis (§3.4, p. 8869): "For the new-drug setting, the Co-VAE obtained the best MSE and second better CI and MAE. DeepDTA obtained a little bit better MAE than the Co-VAE, and GraphDTA obtained a little bit better CI (0.02) than the Co-VAE… All the three methods could not obtain good `r²m` in this setting." For new-target: "it obtains the best by three of the four metrics (MSE, MAE and `r²m`), and it obtains a slightly worse CI than KronRLS."

Paper's narrative reading of KIBA (§3.4, p. 8870): "For both new-drug setting and new-target setting, the Co-VAE method performs the best for three metrics (CI, MSE and MAE) and the second best for the other metric (`r²m`). We also found that for the new-target setting in the KIBA dataset, the KronRLS method performs significantly worse than the two deep learning methods."

### 15.2 Table 3 — Generated-drug statistics (KIBA)

[TABLE] Table 3, p. 8870. Caption: "The Statistical Analysis of the Generated Drugs by GAN, SeqGAN, AAE, VAE and Co-VAE on the KIBA Dataset".

| Model | Validity (%) | Uniqueness (%) |
|---|---|---|
| GAN | < 1 | – |
| SeqGAN | < 1 | – |
| AAE | 1.26 | 0.69 |
| VAE | 48.54 | 3.25 |
| **Co-VAE** | **54.53** | **3.47** |

### 15.3 Tables 4 and 5 — Qualitative generation examples

| Table | Content | Evidence |
|---|---|---|
| **Table 4** (p. 8871) | Three input/generated drug pairs **found in** the KIBA dataset, with PubChem CIDs, SMILES strings, and the top activated target names for each. Example: input CID 49830947 → generated CID 11993789, differing only in the final SMILES character (Cl → I). | [TABLE] Table 4 |
| **Table 5** (p. 8872) | Three input/generated drug pairs **not in** the KIBA dataset, with SMILES strings and RDKit-drawn chemical structure diagrams. "Since the generated SMILES could not found in PubChem, their targets are unknown." | [TABLE] Table 5 |

### 15.4 Figure inventory

| Figure | What it shows | Methodological information conveyed | Evidence |
|---|---|---|---|
| **Fig. 1** (p. 8863) | Generic VAE schematic: x → q_φ(z|x) → (μ, σ) → ⊗ ε sampling → z → p_θ(x|z) → x̂ | Background only; establishes the ⊗ = element-wise-product notation used in Fig. 4 | [FIGURE] |
| **Fig. 2** (p. 8864) | Co-VAE graphical structure: x_d → q_φd → (μ_d, σ_d) + ε_d → z_d; x_t → q_φt → (μ_t, σ_t) + ε_t → z_t; three decoders p_θd(x_d|z_d), p_θy(y|z_d,z_t), p_θt(x_t|z_t) → x̂_d, ŷ, x̂_t | Establishes that **only** p_θy consumes both latents; the two VAE branches are otherwise independent. This is the structure Theorem 2.1 relies on. | [FIGURE] |
| **Fig. 3** (p. 8865) | GatedCNN block: Input → Conv → split into units A and B → A ⊗ sigmoid(B) → Output | Defines the gating operation `A ⊙ σ(B)` | [FIGURE] |
| **Fig. 4** (p. 8866) | Full Co-VAE structure, five named parts (drug encoder, target encoder, drug decoder, target decoder, reg block) — see §6.1 | The only complete architecture source; reveals the encoder FC (A5) and the five-FC Reg block (A6) | [FIGURE] |
| **Fig. 5** (p. 8866) | Frequency histograms of affinities. Davis: x 5–11, single dominant bar at ≈5 of height ≈21,500. KIBA: x 8–18, mass ≈9.5–15.5, mode bar ≈11.5–12 of height ≈72,000, total ≈117,000 | Quantifies the Davis censoring at pKd = 5 (≈72% of pairs) and confirms the KIBA post-filter size | [FIGURE] |
| **Fig. 6** (p. 8867) | Davis affinity matrix under the two settings. Top ("drug-split setting"): whole horizontal **rows** are white (test) or black (validation). Bottom ("target-split setting"): whole vertical **columns** are white or black. | Confirms entity-wise (not pair-wise) splitting; **introduces a validation partition absent from the prose** (A2); uses different setting names than the prose (A11) | [FIGURE] |
| **Fig. 7** (p. 8868) | 2×5 scatter grid, Davis: true affinity (x) vs predicted (y), both axes 4–11, with the y = x line. Top row new-drug, bottom row new-target; columns KronRLS, DeepDTA, DeepAffinity, GraphDTA, Co-VAE | Qualitative only; visible vertical banding at x ≈ 5 reflects the Davis censoring | [FIGURE] |
| **Fig. 8** (p. 8869) | Same layout, KIBA, axes 8–17 | Qualitative only | [FIGURE] |
| **Fig. 9** (p. 8870) | Grouped bar chart: MSE per individual test drug (11 PubChem CIDs on x) for the five methods, Davis | Per-entity error breakdown; specific CIDs 14638120, 644241, 11338033, 6450551, 11485654, 5287969, 126545, 9926791, 11944591 are labelled | [FIGURE] |
| **Fig. 10** (p. 8870) | Two line plots (new-drug top, new-target bottom): AUC (y, ≈0.65–0.95) vs threshold (x, 10.5–12.5), five methods, KIBA | The only AUC evidence; **no error bands are drawn**, so the ten-repetition variability is not shown | [FIGURE] |

---

## 16. Paper-Only Inconsistency Audit

Fifteen items. None is resolved using source code.

---

**A1 — λ sign and scale are unreconstructable**

Evidence:
- Eq. (9), p. 8865: `log p_θy(y^i|z_d,z_t) = −λ(y^i − ŷ^i)² + C`, derived from an assumed **normal** distribution for `p_θy`.
- Eq. (10), p. 8865: the final term is `−λ Σ_{l_d}Σ_{l_t}(y^i − f_θy(·))²`, inside a quantity that Theorem 2.1 defines as a **lower bound to be maximized**.
- Table 1, p. 8868: `λ | [-3,-5]`.
- §3.3, p. 8868: "We choose **-3 or -5** for the regularization parameter λ through cross-validation."

Conflict: a Normal likelihood requires `λ = 1/(2σ_y²) > 0`. Substituting λ = −3 or −5 into Eq. (10) yields `+3Σ(y−ŷ)²` or `+5Σ(y−ŷ)²` under maximization, i.e. an objective that rewards prediction error. The paper never writes λ as an exponent (`10^λ`), never states a sign convention, and never reports which of the two candidate values was selected for either dataset.

Impact: **CRITICAL** — the magnitude and even the direction of the term that couples the two VAEs cannot be determined from the paper.

---

**A2 — A validation partition exists in the figure but not in the prose**

Evidence:
- Fig. 6 caption, p. 8867: "The affinities matrix is split into training data, test data (white part) and **validation data (black part)**." The figure body shows distinct white and black bands in both panels.
- §3.3, p. 8867: "drugs were randomly divided into **six equal folds, five of which were considered as training drugs and the remaining fold was considered as the test drugs**." No third partition is named.
- §3.3, p. 8868: hyperparameters are chosen by "five-fold cross-validation based on only training data" — a procedure that does not require a separate held-out validation block.

Conflict: the figure asserts a three-way partition; the prose asserts a two-way partition; the hyperparameter procedure described implies neither. The size of the validation block, how it is drawn, and what it is used for are all unstated.

Impact: **HIGH** — the amount of data actually used for training, and whether any held-out data informs model selection, cannot be determined.

---

**A3 — MSE and MAE are printed without normalization**

Evidence:
- §3.2, p. 8867: `MSE = Σ_i (y_i − ŷ_i)²`, `MAE = Σ_i |y_i − ŷ_i|`.
- §1.1.2, p. 8863 (quoting DeepDTA): `L_DeepDTA = (1/n) Σ_{i=1}^{n} (P_i − Y_i)²`.
- Table 2, p. 8869: reported MSE values 0.416–0.955, MAE values 0.380–0.639.

Conflict: the printed definitions are sums; the reported values are of the magnitude of means over tens of thousands of test pairs. The same paper writes the normalized form when quoting a baseline.

Impact: **MEDIUM** — the metric intent is recoverable, but the printed definition cannot be implemented as written.

---

**A4 — GatedCNN ReLU count is self-contradictory**

Evidence (§2.5, p. 8865, consecutive sentences):
- "the architecture of GatedCNN block has three 1D-gated CNN layers, **two** Rectified Linear Unit (ReLU) [23] activation functions and a max-pooling layer."
- "There is a ReLU activation function **after each** 1D-convolutional layer."

Conflict: three conv layers each followed by a ReLU gives three ReLUs, not two.

Impact: **LOW** — one activation's placement is ambiguous.

---

**A5 — Encoder FC layer appears in the figure but not the prose**

Evidence:
- Fig. 4, p. 8866: both encoder boxes contain `gatedconv1 → gatedconv2 → gatedconv3 → max-pooling → **FC**`, and the FC output feeds the (μ, σ) pair.
- §2.5, p. 8865: "Drug encoder and target encoder both consist of label encoding, embedding layer and GatedCNN blocks," with the GatedCNN block defined as conv layers + ReLUs + max-pooling. No FC is mentioned.

Conflict: the figure shows a parameterized layer the prose omits — and it is the layer that produces μ and σ, so it is not optional.

Impact: **MEDIUM** — affects the architecture and the latent dimensionality.

---

**A6 — Reg block FC count differs between figure and prose**

Evidence:
- §2.5, p. 8865: "In Reg block, we first use two FC layers for the features of drug and target, and then connect them by another FC layer."
- Fig. 4, p. 8866: the Reg block contains **five** FC boxes — two stacked in the drug branch, two stacked in the target branch, and one wide merged FC producing the affinity output.

Conflict: the prose's most natural reading is three FC layers total (one per branch plus one merged); the figure shows five.

Impact: **MEDIUM** — changes the prediction head's depth and parameter count.

---

**A7 — "three compared methods" where Table 2 lists five**

Evidence:
- §3.4, p. 8870: "For the KIBA dataset, the means and standard deviations of the four evaluation metrics by the **three** compared methods were reported in Table 2."
- Table 2, p. 8869: KIBA rows list **five** methods (KronRLS, DeepDTA, DeepAffinity, GraphDTA, Co-VAE) for each setting.

Conflict: the count in the prose does not match the table.

Impact: **LOW** — a narrative slip; the table is unambiguous.

---

**A8 — r²m lacks the standard deviations the text promises**

Evidence:
- §3.4, p. 8869: "Table 2 shows the **means and standard deviations of the four evaluation metrics**."
- Table 2, p. 8869: the column headers read `CI(std)`, `MSE(std)`, `MAE(std)`, and `r²m` — the fourth carries no `(std)` and no parenthesized values appear in that column.

Conflict: the text claims standard deviations for four metrics; the table supplies them for three.

Impact: **MEDIUM** — r²m differences between methods (e.g. Co-VAE 0.358 vs GraphDTA 0.379 on KIBA new-drug) cannot be assessed for variability.

---

**A9 — The stated KIBA filtering rule cannot produce the stated result**

Evidence (§3.1, p. 8866):
- "The dataset contains **52498 drugs and 467 targets**, with a total of **246088 KIBA scores**."
- "We removed the **drug-target pairs which have known affinities less than 10** in the KIBA dataset, and the final dataset we used contains **2111 drugs and 229 targets**, with an affinity quantity density of 24.4%."
- Fig. 5 (right), p. 8866: the KIBA histogram's leftmost mass begins at ≈9.5, not at exactly 10.

Conflict: removing *pairs* by affinity value cannot reduce the number of *drugs* from 52,498 to 2,111 or the number of *targets* from 467 to 229 — entity removal requires an entity-level criterion, which is not stated. Separately, 2111 × 229 × 0.244 ≈ 117,954 observed pairs remain out of 246,088 originally, which the stated rule alone does not account for. The histogram's lower edge at ≈9.5 is also not exactly consistent with a strict "< 10" removal.

Impact: **HIGH** — the exact dataset cannot be reconstructed from the paper.

---

**A10 — Two different fold counts, and "equal folds" is arithmetically impossible**

Evidence:
- §3.3, p. 8867: "drugs were randomly divided into **six equal folds**"; targets "divided … to **six** folds".
- §3.3, p. 8868: "we determined the remaining hyperparameters by the strategy of **five-fold** cross-validation based on only training data."
- §3.1, p. 8866: Davis has 68 drugs and 442 targets; KIBA has 2111 drugs and 229 targets.

Conflict: two distinct fold counts (6 for the outer split, 5 for the inner hyperparameter search) are used in adjacent paragraphs without explicit distinction, inviting conflation. Additionally, none of 68, 442, 2111 or 229 is divisible by 6, so "six **equal** folds" cannot be literally satisfied, and no remainder rule is given.

Impact: **HIGH** — the split cannot be reproduced exactly; also unstated whether the six folds are rotated (leave-one-fold-out) or a single fold is held out per repetition.

---

**A11 — Setting names differ between prose and figure**

Evidence:
- §3.3, p. 8867: "a **new-drug** setting and a **new-target** setting".
- Fig. 6, p. 8867: panel titles read "**drug-split** setting" and "**target-split** setting".

Conflict: two naming schemes for the same two settings.

Impact: **LOW** — terminological only.

---

**A12 — AUC threshold interval notation differs between sections**

Evidence:
- §3.2, p. 8867: "we take thresholds in the interval **[10.5,12.5]** with increment 0.1."
- §3.4, p. 8870: "Fig. 10 shows the AUC values by using different cutoff values in **(10.5,12.5)** for the KIBA dataset."

Conflict: closed versus open interval; changes whether the endpoints 10.5 and 12.5 are evaluated.

Impact: **LOW** — affects two of 21 threshold points.

---

**A13 — Asymmetric Monte-Carlo normalization in Eq. (10)**

Evidence (Eq. (10), p. 8865): the drug and target reconstruction terms carry `1/L_d` and `1/L_t` respectively; the co-regularization term is `−λ Σ_{l_d=1}^{L_d} Σ_{l_t=1}^{L_t} (·)²` with **no** `1/(L_d L_t)` factor.

Conflict: the effective weight of the affinity term relative to the VAE terms scales with `L_d · L_t`, and neither `L_d` nor `L_t` is ever defined or valued. λ therefore cannot be interpreted on an absolute scale.

Impact: **MEDIUM** — compounds A1.

---

**A14 — Affinity matrix notation is inconsistent**

Evidence:
- §2.1, p. 8863: "The affinity matrix **I** contains the affinity values."
- §2.3 onward: affinity is written `y`, `y^i`, and `ŷ`. `I` also serves as the identity matrix in `N(0, I)` (§2.3, p. 8865) and in `(K + λI)a = y` (§1.1.1, p. 8863).

Conflict: `I` denotes both the affinity matrix and the identity matrix; `λ` denotes both the KronRLS regularization parameter (Eq. 1) and the Co-VAE co-regularization parameter (Eq. 9).

Impact: **LOW** — notational reuse; context disambiguates.

---

**A15 — Table 1 is headed "range" but mixes ranges with fixed values**

Evidence (Table 1, p. 8868): the second column is headed "range", yet contains fixed scalars (dropout 0.2, epoch 100, batch size 256, learning rate 0.001) alongside genuine search sets (`[5,7]`, `[7,11]`, `[-3,-5]`) and filter specifications (`32*1;32*2;32*3`).

Conflict: the header implies all rows are search ranges; §3.3 clarifies that some were "fixed". Further, `[5,7]` is described in prose as the **set** {5, 7}, not the interval 5–7.

Impact: **LOW** — resolved by §3.3's prose, but Table 1 alone is misleading.

---

## 17. Reproducibility Completeness

| Component | Status | Evidence |
|---|---|---|
| Dataset identity | **COMPLETE** | Davis [20] and KIBA [21], both named with references and prior use in DeepDTA [11]. [PAPER] §3.1 p. 8866 |
| Dataset version | **MISSING** | No download date, release version, or accession snapshot for PubChem, UniProt, Davis or KIBA. [OPEN] |
| Preprocessing | **PARTIAL** | pKd transform stated for Davis [EQUATION] §3.1; length caps stated for both [PAPER] §3.1. But the KIBA entity-filtering rule is unreconstructable (A9), truncation-vs-exclusion is unstated, duplicate handling is absent, and missing-value handling is absent. |
| Input encoding | **PARTIAL** | Label encoding + embedding, vocabulary sizes (64 / 25), embedding dim 128, and input matrix shapes are all stated [PAPER] §3.1, §3.3. The character dictionaries themselves, and any padding convention, are not. |
| Architecture | **PARTIAL** | Layer types, counts, filter counts and filter-length candidates are stated [TABLE] Table 1; [PAPER] §2.5; [FIGURE] Fig. 4. But latent dimension, all FC widths, conv stride/padding, and pooling parameters are absent, and figure/prose disagree on two components (A5, A6). |
| Loss | **PARTIAL** | The full objective is derived and instantiated (Theorem 2.1, Eqs. 5–10) — structurally complete. But λ's sign/scale (A1), `L_d`/`L_t` (A13), and `J_d`/`J_t` are undetermined, so it cannot be evaluated numerically. |
| Optimizer | **COMPLETE** | Adam, lr 0.001. [PAPER] §3.3 p. 8868; [TABLE] Table 1 (β's unstated, but defaults are conventional) |
| Hyperparameters | **PARTIAL** | The search grid is given [TABLE] Table 1; [PAPER] §3.3. The **selected** values are never reported, and latent/FC dimensions are absent. |
| Split protocol | **PARTIAL** | Entity-wise, 6 folds, 10 repetitions, per-setting, shared across methods [PAPER] §3.3. But validation existence is contradictory (A2), fold rotation is unstated, and "equal folds" is arithmetically impossible (A10). |
| Random seed | **MISSING** | Not mentioned anywhere. [OPEN] |
| Evaluation | **PARTIAL** | Five metrics defined, AUC threshold sweep specified, aggregation over 10 runs stated [PAPER] §3.2, §3.3. But MSE/MAE are misprinted (A3), r²m std is absent (A8), per-fold vs global computation is unstated, and the evaluation checkpoint is unstated. |
| Baselines | **PARTIAL** | Four affinity baselines and four generative baselines named with references; two stated to use default parameters; all stated to share the same splits [PAPER] §3.3. But no baseline hyperparameters, versions, or code sources are given. |
| Hardware | **MISSING** | No GPU, CPU, memory or runtime information anywhere. [OPEN] |
| Software versions | **MISSING** | "Pytorch" named without a version; RDKit named without a version. [PAPER] §3.3, §3.2; [OPEN] |

---

## 18. Critical Open Questions for Phase 3

Twenty-two questions, all `[OPEN]`, to be answered only by inspecting the repository.

**Objective and variational formulation**
1. Does the code implement Eq. (10) — both VAE terms plus the co-regularized term — as a single jointly optimized objective? `[OPEN]`
2. How is λ actually applied: as the literal value −3/−5, as `10^λ`, or with an opposite sign convention (A1)? `[OPEN]`
3. Are both KL divergence terms present and active in the optimized loss, with coefficient 1? `[OPEN]`
4. Is the reparameterization implemented as `μ + σ⊙ε`, and is the encoder output treated as σ or as log-variance? `[OPEN]`
5. What are `L_d` and `L_t` at runtime, and is the co-regularization term normalized by `L_d·L_t` (A13)? `[OPEN]`
6. Is the reconstruction likelihood categorical (`x^T log x̂`) as Eq. (10) states? `[OPEN]`

**Architecture**
7. What is the latent dimensionality `J_d` / `J_t`? `[OPEN]`
8. Does the encoder contain an FC layer after max-pooling, as Fig. 4 shows but §2.5 omits (A5)? `[OPEN]`
9. How many FC layers does the Reg block contain — three (prose) or five (figure) (A6)? `[OPEN]`
10. Are there three ReLUs or two in the GatedCNN block (A4)? `[OPEN]`
11. What are the FC widths, conv strides, padding, and pooling parameters? `[OPEN]`
12. Is the gating implemented as `A ⊙ sigmoid(B)` on a channel split? `[OPEN]`

**Data**
13. Do the code's dataset dimensions match 68 × 442 (Davis) and 2111 × 229 (KIBA)? `[OPEN]`
14. What entity-level rule actually produces KIBA's 2111 × 229 from 52,498 × 467 (A9)? `[OPEN]`
15. Is the Davis pKd transform `−log₁₀(Kd/10⁹)` applied, and is KIBA left untransformed? `[OPEN]`
16. How are the ~75.6% unobserved KIBA cells handled in the loss and in the metrics? `[OPEN]`
17. Are over-length SMILES/sequences truncated or excluded? `[OPEN]`

**Splitting and protocol**
18. Are folds generated at runtime or loaded from fixed files, and is the count 6 as stated (A10)? `[OPEN]`
19. Does a validation partition exist separately from the test partition, as Fig. 6 implies (A2)? `[OPEN]`
20. Is test data used during model selection, early stopping, or checkpoint selection? `[OPEN]`
21. Are the reported hyperparameters (Table 1) the runtime defaults, and which λ / filter lengths are actually used? `[OPEN]`
22. Is a random seed set, and are the ten repetitions implemented? `[OPEN]`

**Evaluation and baselines**
23. Are CI, MSE, MAE, r²m and AUC computed as §3.2 defines them, and is MSE normalized by n despite the printed sum (A3)? `[OPEN]`
24. Are the AUC thresholds swept over [10.5, 12.5] at 0.1 increments (A12)? `[OPEN]`
25. Are any baseline results (KronRLS, DeepDTA, DeepAffinity, GraphDTA, GAN, SeqGAN, AAE, VAE) produced by this repository? `[OPEN]`
26. Are validity/uniqueness and the drug-generation pipeline implemented? `[OPEN]`

---

## 19. Executive Verdict

### Paper Forensic Verdict

**Explicitly specified** — reconstructable with confidence from the paper alone:
- The probabilistic model: two independent Gaussian priors `N(0, I)`, two diagonal-Gaussian variational posteriors, categorical reconstruction likelihoods, a Normal affinity likelihood, and the reparameterization `z = μ + σ⊙ε`. [PAPER] §2.3; [EQUATION] Eqs. (3)–(10)
- The decomposition `L = L_DrugVAE + L_TargetVAE + L_CoREG` with a formal derivation (Theorem 2.1 and its proof via Eqs. 6–8).
- The five-part architecture and its topology, including the GatedCNN gating rule `A ⊙ σ(B)`. [FIGURE] Figs. 2–4
- Encoder/decoder filter counts (32/64/96 and 96/64/32), dropout 0.2, epoch 100, batch 256, Adam, lr 0.001. [TABLE] Table 1
- Dataset identities, entity counts (68 × 442; 2111 × 229), Davis raw range [0.016, 10000] and its pKd transform, KIBA raw range [0, 17.2] and 24.4% density, length caps (85/1200; 100/1000), vocabulary sizes (64/25), embedding dim 128, input matrix shapes. [PAPER] §3.1, §3.3
- The split granularity (entity-wise over drugs or targets), 6 folds, 10 repetitions, shared across all compared methods. [PAPER] §3.3
- Metric definitions for CI and r²m; the AUC threshold sweep; validity and uniqueness definitions. [PAPER] §3.2
- Four affinity baselines and four generative baselines, all stated to have been run by the authors. [PAPER] §3.3

**Partially specified** — described but insufficient for exact reproduction:
- The objective's numerical form: structurally complete but λ's sign/scale (A1), `L_d`/`L_t` (A13) and `J_d`/`J_t` are all undetermined.
- The architecture: layer types and counts are given, but latent dimension, every FC width, conv stride/padding and pooling parameters are absent, and the figure contradicts the prose on the encoder FC (A5) and Reg-block depth (A6).
- Preprocessing: the Davis transform is exact; the KIBA entity filter is unreconstructable (A9); truncation-vs-exclusion, duplicates and missing values are unaddressed.
- The split: granularity and repetition count are clear, but validation existence (A2), fold rotation, and remainder handling (A10) are not.
- Evaluation: metrics are named and mostly defined, but MSE/MAE are printed unnormalized (A3), r²m lacks std (A8), and per-fold vs global computation and the evaluation checkpoint are unstated.
- Hyperparameter search: the grid is given; the selected values are never reported.

**Open / missing** — not reconstructable from the paper at all:
- Latent dimensionality `J_d`, `J_t`; Monte-Carlo counts `L_d`, `L_t`; all FC widths.
- Random seed; weight initialization; early stopping; patience; checkpoint selection; LR schedule; weight decay.
- Hardware (no GPU/CPU/memory/runtime anywhere); PyTorch and RDKit versions.
- Handling of the ~75.6% unobserved KIBA affinity cells.
- Dataset versions and download dates.
- The size and role of the validation partition depicted in Fig. 6.
- Which λ and which filter lengths were selected for each dataset.
- Any component ablation or λ sensitivity analysis (none exists).
- Any statistical significance testing (none performed).

### Top 10 Phase-3 Verification Targets

1. **λ's runtime interpretation** — literal −3/−5, `10^λ`, or a sign flip? This determines whether the two VAEs are actually coupled and in which direction (A1). [OPEN]
2. **Whether the full Eq. (10) objective is optimized** — are both KL terms and both reconstruction terms live in the backward pass, or is any term computed and discarded? [OPEN]
3. **The fold protocol** — 6 folds as stated, generated or loaded, rotated or single-holdout, seeded or not (A10). [OPEN]
4. **Whether a validation partition exists and whether test data touches model selection** (A2). [OPEN]
5. **The KIBA construction rule** that yields 2111 × 229 from 52,498 × 467, and how the 75.6% unobserved cells are treated in loss and metrics (A9). [OPEN]
6. **Latent dimensionality and all FC widths** — the numbers needed to instantiate the model, none of which the paper supplies. [OPEN]
7. **Encoder FC and Reg-block depth** — does the code follow Fig. 4 (encoder FC present, five Reg FCs) or §2.5 (A5, A6)? [OPEN]
8. **Metric implementations** — is MSE normalized by n despite the printed sum (A3); are CI and r²m computed globally or per fold; at which epoch/checkpoint are they evaluated? [OPEN]
9. **Whether the reported hyperparameters are the runtime defaults**, and which grid point was actually selected per dataset. [OPEN]
10. **Baseline provenance** — does the repository contain code for KronRLS, DeepDTA, DeepAffinity, GraphDTA and the four generative baselines, or only Co-VAE? [OPEN]

---

*End of Phase 2 forensic specification — Co-VAE. Paper-only: no source code was inspected, no git history consulted, nothing executed, and no repository file was modified.*

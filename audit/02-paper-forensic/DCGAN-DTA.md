# Stage 2 — Paper Forensic Audit: DCGAN-DTA

**Audit scope:** paper-side evidence only.
**Sources inspected:** `DCGAN-DTA/paper.pdf` (16 pages, full text + all figures) and all 7 files in `DCGAN-DTA/supplementaries/`.
**Explicitly out of scope for this stage:** source code, reproduction, method-quality judgement, comparison with Co-VAE.

**Evidence tags used:** `[PAPER]` `[FIGURE]` `[TABLE]` `[EQUATION]` `[SUPPLEMENT]` `[INFERRED]` `[OPEN]`.
`[CODE]` and `[VERIFIED]` are deliberately absent — no implementation was consulted.

> **Note on supplementary numbering.** The paper cites its supplements as "Supplementary Table 1–3" and "Supplementary Fig. 1–4". The bundled files are named `12864_2024_10326_MOESM1..7_ESM.docx` and the mapping (established in §13) is: MOESM1→Sup Fig. 1, MOESM2→Sup Fig. 2, MOESM3→Sup Fig. 3, MOESM4→Sup Fig. 4, MOESM5→Sup Table 1, MOESM6→Sup Table 2, MOESM7→Sup Table 3.

---

## 1. Bibliographic Identification

| Field | Value | Evidence |
|---|---|---|
| Title | *DCGAN-DTA: Predicting drug-target binding affinity with deep convolutional generative adversarial networks* | [PAPER] p. 1 |
| Authors | Mahmood Kalemati¹, Mojtaba Zamani Emani¹, Somayyeh Koohi¹* | [PAPER] p. 1 |
| Affiliation | ¹ Department of Computer Engineering, Sharif University of Technology, Tehran, Iran | [PAPER] p. 1 |
| Corresponding author | Somayyeh Koohi — koohi@sharif.edu | [PAPER] p. 1 |
| Venue | BMC Genomics | [PAPER] p. 1 header |
| Volume / article no. | (2024) 25:411 | [PAPER] p. 1 running head |
| Year | 2024 | [PAPER] p. 1 |
| DOI | `https://doi.org/10.1186/s12864-024-10326-x` | [PAPER] p. 1 |
| Article type | RESEARCH — Open Access (CC BY 4.0) | [PAPER] p. 1 |
| Received | 15 February 2024 | [PAPER] p. 15, "Declarations" block |
| Accepted | 19 April 2024 | [PAPER] p. 15 |
| Published online | 09 May 2024 | [PAPER] p. 15 |
| Length | 16 pages | [PAPER] pagination "Page N of 16" |
| Code repository (as stated) | `https://github.com/mojtabaze7/DCGAN-DTA` | [PAPER] Abstract p. 1; "Data availability" p. 15 |
| Web server (as stated) | `https://dcgan.shinyapps.io/bindingaffinity/` | [PAPER] Abstract p. 1; p. 15 |
| Supplementary DOI | `https://doi.org/10.1186/s12864-024-10326-x` (Supplementary Materials 1–7) | [PAPER] p. 15 "Supplementary Information" |
| Keywords | Drug-target binding affinity; Deep convolutional generative adversarial networks; BLOSUM encoding; Adversarial control experiments; Straw models | [PAPER] p. 1 |
| Funding | "Not applicable" | [PAPER] p. 15 |
| Competing interests | "The authors declare no competing interests." | [PAPER] p. 15 |

**Section structure** [PAPER]: Abstract (p. 1) → Background (pp. 2–3) → Method (pp. 3–6: *Datasets*, *DCGAN-DTA method*, *Implementation details*) → Results (pp. 6–13: *Evaluation metrics*, *Comparisons for warm-start data splitting*, *Comparisons for cold-start drug data splitting*, *Adversarial control experiments*) → Discussion (pp. 13–14) → Conclusion (p. 14) → Supplementary Information / Declarations (p. 15) → References (pp. 15–16).

There is **no separate "Related Work" section** — related work is folded into *Background* [PAPER] pp. 2–3. There is **no "Limitations" section** anywhere in the paper. [PAPER] / [OPEN]

---

## 2. Problem Formulation

| Aspect | Extraction | Evidence |
|---|---|---|
| Prediction target | Drug–target binding affinity (DTA) | [PAPER] Abstract p. 1; Background p. 2 |
| Input entities | A drug, given as a SMILES string; a protein target, given as an amino-acid sequence | [PAPER] "Method → DCGAN-DTA method" p. 4; [FIGURE] Fig. 1 p. 5 (input boxes "Protein Sequence MPSPMVKG…LMPCP" and "Drug SMILES CC(=C1)…CCN=C3") |
| Task type | **Regression** on a continuous affinity value, evaluated additionally with a *derived* binary classification for AUPR | [PAPER] p. 6: "The concordance index (CI) is a measure of prediction performance for a **regression model**"; p. 6: "In order to frame the classification problem, we transformed the binding affinities … into binary values" |
| Output | A single scalar affinity value | [FIGURE] Fig. 1 p. 5 — Prediction Network terminates in a single output node above "FC Layer (512)" |
| Formal problem statement (f: (drug, target) → ℝ) | **Not given.** The paper contains no equation defining the prediction function, its domain, or its codomain. | [OPEN] |
| Affinity definition — BindingDB | Dissociation constant Kd, log-transformed to **pKd** | [PAPER] Method → Datasets p. 3: "we specifically utilized the Kd version of the BindingDB dataset … we obtained a dataset comprising logarithmic-transformed binding affinities (pKd)" |
| Affinity definition — PDBbind | Ki **and** Kd, log-transformed ("pKi and pKd") | [PAPER] p. 3: "The refined set consists of logarithmic-transformed Ki and Kd binding values"; [SUPPLEMENT] Sup Fig. 1(a) x-axis label "pKi and pKd values" |
| Exact log-transform formula | **Not stated.** The paper says "logarithmic-transformed" and names the results pKd / pKi but never writes the conversion (e.g. whether pKd = −log₁₀(Kd/10⁹) from nM, or −log₁₀(Kd) from M, or an offset variant). | [OPEN] |
| Observed affinity range | BindingDB pKd ≈ **2 – 9**; PDBbind pKi/pKd ≈ **2 – 12** | [SUPPLEMENT] Sup Fig. 1(a), MOESM1 — x-axis extents of the two histograms |
| Units of the transformed target | Dimensionless log-molar (p-scale) | [INFERRED] from "logarithmic-transformed … (pKd)" [PAPER] p. 3; the paper never states the unit explicitly |

**Not established by the paper:** whether the affinity matrix is treated as a dense drug × target matrix with missing entries, or as a flat list of observed pairs. No equation, figure or table addresses this. [OPEN]

---

## 3. Datasets

Two datasets are used. Every number below carries a locator.

### 3.1 Consolidated dataset table

| Property | BindingDB | Locator | PDBbind | Locator |
|---|---|---|---|---|
| Name as given | BindingDB | [PAPER] p. 3 | PDBBind / PDBbind (both spellings used) | [PAPER] p. 3 |
| Source / citation | Ref. [40] = Huang K. et al., *Therapeutics Data Commons* (TDC), arXiv:2102.09548 | [PAPER] p. 3 + Ref. list p. 16 | Ref. [41] = Wang R., Fang X., Lu Y., Wang S., *The PDBbind database*, J Med Chem 2004;47(12):2977–80 | [PAPER] p. 3 + Ref. list p. 16 |
| Version / subset | "the **Kd version**" | [PAPER] p. 3 | "**refined version (v2020)**" | [PAPER] p. 3 |
| Affinity measure(s) available in source | Ki, Kd, IC50 | [PAPER] p. 3 | — | — |
| Affinity measure(s) **used** | Dissociation constant (Kd) | [PAPER] p. 3; [TABLE] Sup Table 1 | Inhibition constant (Ki) **and** Dissociation constant (Kd) | [PAPER] p. 3; [TABLE] Sup Table 1 |
| **Number of drugs / compounds** | **9864** | [PAPER] p. 3; [TABLE] Sup Table 1 (MOESM5) | **4231** | [PAPER] p. 3; [TABLE] Sup Table 1 |
| **Number of proteins / targets** | **1088** | [PAPER] p. 3; [TABLE] Sup Table 1 | **1606** | [PAPER] p. 3; [TABLE] Sup Table 1 |
| **Number of interactions** | **42203** | [TABLE] Sup Table 1 (MOESM5) — *not stated in the main text* | **5014** | [TABLE] Sup Table 1 — *not stated in the main text* |
| Affinity range (observed) | pKd ≈ 2 – 9 | [SUPPLEMENT] Sup Fig. 1(a) left panel | pKi/pKd ≈ 2 – 12 | [SUPPLEMENT] Sup Fig. 1(a) right panel |
| Original units | nM or M — **not stated** | [OPEN] | nM or M — **not stated** | [OPEN] |
| Transformed units | pKd (log scale) | [PAPER] p. 3 | pKi / pKd (log scale) | [PAPER] p. 3 |
| Missing-value handling | **Not stated** | [OPEN] | **Not stated** | [OPEN] |
| Duplicate handling | "refining the dataset according to **recommended guidelines for data harmonization and stable training**" — guidelines not named, no operation specified | [PAPER] p. 3 | "To ensure data consistency, we **excluded redundancies arising from multiple sequences for the same drugs**" | [PAPER] p. 3 |
| Filtering (other) | **Not stated** beyond the above | [OPEN] | **Not stated** beyond the above | [OPEN] |
| Molecule preprocessing | Label encoding + padding to 200; no standardisation/canonicalisation/salt-stripping described | [PAPER] p. 4 | same | [PAPER] p. 4 |
| Protein preprocessing | Label encoding + padding to 2000; BLOSUM encoding for variants B/C | [PAPER] p. 4 | same | [PAPER] p. 4 |
| Explicit train/val/test sizes | **Never reported** | [OPEN] | **Never reported** | [OPEN] |

### 3.2 Distributional evidence (supplement only)

Sup Fig. 1 (MOESM1) is the only source of distributional information; it carries no numeric annotations, so the readings below are axis-scale readings. Left column = BindingDB, right column = PDBbind (per the caption's ordering). [SUPPLEMENT]

| Panel | BindingDB (left) | PDBbind (right) |
|---|---|---|
| (a) Affinity histogram | pKd spans ≈ 2–9; a single dominant spike at **pKd ≈ 5** containing ≈ 20,500 of the 42,203 pairs (≈ half the dataset); remainder roughly flat over 5–9 | pKi/pKd spans ≈ 2–12, unimodal, mode ≈ 7 |
| (b) SMILES length histogram | Range ≈ 0 – 900; mode in the 0–50 bin; long right tail | Range ≈ 0 – 250; mode ≈ 40–60 |
| (c) Protein length histogram | Range ≈ 0 – 4000; mode ≈ 400–500 | Range ≈ 0 – 7000; mode ≈ 0–250 |

Two reproduction-relevant consequences, marked as inference:

- The BindingDB spike at pKd ≈ 5 is consistent with a large block of measurements censored at a fixed Kd cut-off; the paper offers no explanation for it and does not mention censored or right-limit values anywhere. [INFERRED] from [SUPPLEMENT] Sup Fig. 1(a); [OPEN] as to cause and handling.
- Both datasets contain SMILES longer than the 200-character cap and proteins longer than the 2000-residue cap declared on p. 4 / Sup Table 2, so truncation must occur; the paper never states that truncation happens, nor how (head/tail/discard). [INFERRED] from [SUPPLEMENT] Sup Fig. 1(b),(c) vs [PAPER] p. 4; [OPEN].

### 3.3 Additional external data (auxiliary corpora)

| Item | Extraction | Evidence |
|---|---|---|
| External unlabeled corpora | "DCGAN is trained using data collected from **UniProt [44]** and **ChEMBL [45]** databases" | [PAPER] Method → DCGAN-DTA method, p. 4 |
| Purpose | Pretraining the DCGAN feature extractors on unlabeled data | [PAPER] p. 3: "extracting features from the protein sequences and drug SMILES using unlabeled data"; p. 14: "CNN-based GANs trained on a larger amount of available unlabeled data" |
| Number of sequences / molecules drawn | **Not stated** | [OPEN] |
| Release / version / access date of UniProt or ChEMBL | **Not stated** | [OPEN] |
| Selection, filtering or deduplication of the corpora | **Not stated** | [OPEN] |
| Whether the corpora overlap with the evaluation sets | **Not addressed anywhere in the paper** | [OPEN] |
| Physicochemical data (cold-start only) | logP values for PDBbind compounds from **Open Babel logP [49]** and **XLOGP3 [50]** | [PAPER] p. 9 |
| Similarity matrices | Used only by the *baseline* DeepDTA-Sim (Smith–Waterman), not by DCGAN-DTA | [TABLE] Sup Table 3 (MOESM7) |

---

## 4. Data Splitting and Evaluation Protocol

This section reports only what the paper states, without inferring a split type from terminology.

| Question | What the paper actually says | Evidence |
|---|---|---|
| Cross-validation | "we employed a **fivefold cross-validation approach**" | [PAPER] Implementation details, p. 6 |
| Number of folds | **5** | [PAPER] p. 6 |
| What the folds contain | "This involved **dividing the dataset into five nearly equal-sized training and validation sets**." | [PAPER] p. 6 |
| Role of those sets | "The model was **trained and hyperparameter tuning was performed using these sets**." | [PAPER] p. 6 |
| Existence of a separate held-out **test** set for the warm-start experiment | **Never described.** The Implementation-details paragraph describes only training and validation sets. The word "test" appears in the split sense only for (i) the cold-start experiment and (ii) the shuffling control experiments. | [PAPER] p. 6; [OPEN] |
| When the final test set is evaluated | **Not stated.** No sentence anywhere specifies at what point, on what data, or with which checkpoint the reported warm-start numbers are produced. | [OPEN] |
| Whether reported warm-start numbers are validation or test numbers | **Not stated.** | [OPEN] |
| Randomisation of fold assignment | **Not stated** | [OPEN] |
| Random seed | **No seed is mentioned anywhere in the paper or supplements** (confirmed by full-text search for "seed") | [OPEN] |
| Repeated runs / repetitions of the whole CV | **Not stated.** The box plots (Sup Figs. 2–4) show a distribution of CI scores, but the paper never says whether the distribution is over folds, over repeated runs, or over something else. | [OPEN] |
| Warm-start setting | Named and used; results in Figs. 2–3. The paper labels this "warm-start data splitting setting" but **never defines what warm-start means operationally** (e.g. random split over pairs, with drugs and targets seen in training). | [PAPER] pp. 6–8; definition [OPEN] |
| Cold-start setting | Defined **only** for drugs, and **only** via physicochemical properties: "we **excluded the drug SMILES from the model training based on the logarithm of n-octanol-water partition coefficient (logP)** … These excluded drugs were then utilized as the **test data** for evaluation." Two variants: Open Babel logP and XLOGP3. PDBbind only. | [PAPER] Comparisons for cold-start drug data splitting, p. 9 |
| logP threshold / rule used to decide which drugs are excluded | **Not stated.** No cut-off value, percentile, or direction (high-logP vs low-logP excluded) is given. | [OPEN] |
| Number of drugs / pairs excluded in cold-start | **Not stated** | [OPEN] |
| Drug-wise split | The cold-start setting excludes *drugs* from training, so it is drug-disjoint. The paper does **not** use the term "drug-wise split". | [PAPER] p. 9 |
| Target-wise split | **Never performed.** No protein-cold-start or target-wise experiment appears anywhere. | [OPEN] |
| Pair-wise / random split | Not named as such anywhere. | [OPEN] |
| Validation used for hyperparameter selection | Yes — stated explicitly: "The model was trained and **hyperparameter tuning was performed using these sets**." | [PAPER] p. 6 |
| Model selection criterion | **Not stated.** Early stopping is mentioned ("the early stopping technique has been employed to prevent overfitting") but the monitored quantity, mode, patience and restore-best behaviour are all absent. | [PAPER] p. 6; [OPEN] |
| Fold aggregation of reported metrics | **Not stated** — see §10. | [OPEN] |

### 4.1 Adversarial control ("straw model") protocol

Three shuffling experiments are defined: [PAPER] Adversarial control experiments, p. 10

1. "training models using shuffled data and testing them on actual data" → *Shuffled train data*
2. "training models using actual data and testing them on shuffled data" → *Shuffled test data*
3. "training and testing models using shuffled data" → *Shuffled train and test data*

- What is shuffled: "**shuffled binding affinity values**" — i.e. the labels, not the inputs. [PAPER] p. 10; [FIGURE] Fig. 7 caption p. 12
- Dataset: PDBbind. [FIGURE] Fig. 7 caption p. 12
- Which model variant was used for the shuffled runs: "the experiments on shuffled data were **specifically performed on DCGAN-DTA (C)**". [PAPER] p. 10
- Reference to prior work on adversarial controls: Ref. [42] Chuang KV, Keiser MJ, *Adversarial controls for scientific machine learning*. [PAPER] p. 3, p. 10

---

## 5. Input Representation

### 5.1 Drug representation

| Aspect | Extraction | Evidence |
|---|---|---|
| Raw representation | SMILES strings | [PAPER] p. 4; [FIGURE] Fig. 1 p. 5 |
| Encoding | **Label encoding** — "each character in the drug SMILES … is converted into numerical data" | [PAPER] p. 4 |
| Resulting form before padding | "vectors with lengths equal to their corresponding SMILES" | [PAPER] p. 4 |
| Vocabulary / character set | **Not specified.** Neither the alphabet, its size, nor the index assignment is given anywhere in the paper or supplements. | [OPEN] |
| Canonical vs isomeric SMILES | **Not stated** | [OPEN] |
| Maximum length | **200** | [PAPER] p. 4: "SMILES representations of compounds were limited to a length of 200"; [TABLE] Sup Table 2 "Length of SMILES — 200" |
| Padding | "Padding is applied to ensure a fixed length for both the drug SMILES and protein sequences." Pad value, and whether pre- or post-padding, are not given. | [PAPER] p. 4; details [OPEN] |
| Truncation of over-length SMILES | **Not mentioned**, although Sup Fig. 1(b) shows SMILES up to ≈ 900 characters | [OPEN] / [INFERRED] |
| Embedding | An embedding layer follows label encoding | [PAPER] p. 4; [FIGURE] Fig. 1 p. 5 ("Label Encoding" → "Embedding Layer" in the drug branch) |
| Embedding dimension | **Not stated** in the paper or Sup Table 2 | [OPEN] |
| Handcrafted features | None used for drugs | [TABLE] Sup Table 3 |
| Similarity features | None used for drugs by DCGAN-DTA | [TABLE] Sup Table 3 |
| Drug representation across variants A/B/C | **Identical for all three variants**: "Label encoding, embedding layer, DCGAN, and 1DCNN". In Sup Table 3 the Drugs column for rows DCGAN-DTA(B) and DCGAN-DTA(C) is a **vertically merged cell** carrying the DCGAN-DTA(A) value. | [TABLE] Sup Table 3 (MOESM7), rows 8–10 |

### 5.2 Target / protein representation

| Aspect | Extraction | Evidence |
|---|---|---|
| Raw representation | Amino-acid sequences | [PAPER] p. 4; [FIGURE] Fig. 1 p. 5 |
| Encoding — path 1 (variant A) | **Label encoding** + embedding layer, same scheme as drugs | [PAPER] p. 4; [TABLE] Sup Table 3 |
| Encoding — path 2 (variants B, C) | **BLOSUM encoding** [43]: "each amino acid in the protein sequences is converted into a **25-dimensional feature vector**. As a result, the protein is represented as a matrix with dimensions of **25 multiplied by the length of the protein sequence**." | [PAPER] p. 4; repeated in [FIGURE] Fig. 1 caption p. 5 |
| BLOSUM matrix identity | Cited as ref. [43] = Eddy SR, *Where did the BLOSUM62 alignment score matrix come from?* — the paper says "BLOSUM encoding technique [43]" but **never names BLOSUM62 in its own text**, and never states which matrix or row-ordering is used. | [PAPER] p. 4 + Ref. [43] p. 16; [OPEN] |
| Vocabulary / amino-acid alphabet | **Not specified** (a 25-dimensional vector implies a 25-symbol alphabet, but the symbols are not listed) | [INFERRED] / [OPEN] |
| Maximum sequence length | **2000** | [PAPER] p. 4: "protein sequences were set to a length of 2000"; [TABLE] Sup Table 2 "Length of protein sequences — 2000" |
| Padding | Applied; value and side not given | [PAPER] p. 4; [OPEN] |
| Truncation of over-length proteins | **Not mentioned**, although Sup Fig. 1(c) shows sequences up to ≈ 7000 residues | [OPEN] / [INFERRED] |
| Embedding | Embedding layer used on the **label-encoded** path only. In Fig. 1 the protein "Encoding and Embedding" box contains *both* a "BLOSUM Encoding" element and a "Label Encoding → Embedding Layer" stack. | [FIGURE] Fig. 1 p. 5; [PAPER] p. 4 |
| Embedding dimension | **Not stated** | [OPEN] |
| Handcrafted features | BLOSUM substitution scores are the only handcrafted/evolutionary feature | [PAPER] p. 4 |
| Similarity features | Not used by DCGAN-DTA (used only by the DeepDTA-Sim baseline) | [TABLE] Sup Table 3 |
| Per-variant protein representation | A: "Label encoding, embedding layer, DCGAN, and 1DCNN"; B: "BLOSUM encoding, DCGAN, and 1DCNN"; C: "BLOSUM encoding, and 1DCNN" | [TABLE] Sup Table 3 (MOESM7) |

### 5.3 Auxiliary data

| Item | Status | Evidence |
|---|---|---|
| Similarity matrices | Not used by DCGAN-DTA | [TABLE] Sup Table 3 |
| Pretrained representations | None external; the only pretraining is the paper's own DCGAN, trained on UniProt + ChEMBL | [PAPER] p. 4 |
| External corpora | UniProt [44], ChEMBL [45] — sizes, versions and filtering all unstated | [PAPER] p. 4; [OPEN] |
| Unlabeled samples | Yes — the DCGAN stage is explicitly described as consuming unlabeled data | [PAPER] pp. 3, 4, 14 |
| Auxiliary labels | None described | [OPEN] |
| Additional chemical features | logP (Open Babel logP, XLOGP3) — used **only to define the cold-start split**, not as model input | [PAPER] p. 9 |

---

## 6. Complete DCGAN-DTA Architecture

The paper describes a **four-step process**: [PAPER] p. 4 — "The four steps include encoding and embedding, feature extraction, merging of latent vectors for drugs and proteins, and DTA prediction." Fig. 1 (p. 5) is the single architecture figure.

Three model variants exist — A, B, C — differing in protein encoding and in whether a protein DCGAN is used. [PAPER] p. 6; [TABLE] Sup Table 3.

### 6.1 Drug branch

| Stage | Specification | Evidence |
|---|---|---|
| 1. Encoding | Label encoding of SMILES characters | [PAPER] p. 4 |
| 2. Embedding | Embedding layer (dimension unstated) | [PAPER] p. 4; [FIGURE] Fig. 1 |
| 3. DCGAN feature extraction | A drug DCGAN (generator + 5-layer Conv1D discriminator) is trained adversarially; the learned model is then "utilized for the feature extraction step" | [PAPER] p. 4; [FIGURE] Fig. 1 — "Deep Convolutional GAN (DCGAN)" band, right-hand column |
| 4. CNN block | "CNN blocks consisting of **three CNN layers and one max-pooling layer**" | [PAPER] p. 4 |
| CNN block filter counts | **128, 256, 384** | [PAPER] p. 6 (paragraph beginning "The model utilized 128, 256, and 384 filters in different layers"); [TABLE] Sup Table 2 "Number of filters — 128,256,384" |
| CNN block filter length (drug) | **4** | [PAPER] p. 6: "for drug data, it was set to 4"; [TABLE] Sup Table 2 "Filter length (drug) — 4" |
| CNN block activation | **Not stated** for the CNN blocks (ReLU is specified only for the GAN components) | [OPEN] |
| Stride / padding of CNN blocks | **Not stated** | [OPEN] |
| Pooling | One max-pooling layer per CNN block, per the prose. **Fig. 1 shows only "Conv Layer 1/2/3" in the CNN Blocks band and depicts no pooling element.** | [PAPER] p. 4 vs [FIGURE] Fig. 1 p. 5 — see §15 |
| Output latent dimension | **Not stated** | [OPEN] |

Identical for variants A, B and C. [TABLE] Sup Table 3

### 6.2 Target / protein branch

| Variant | Protein pipeline | Evidence |
|---|---|---|
| **A** | Label encoding → Embedding layer → protein DCGAN → CNN block (3 Conv1D + max-pool) | [TABLE] Sup Table 3; [PAPER] p. 4 |
| **B** | BLOSUM encoding (25-dim per residue) → protein DCGAN → CNN block | [TABLE] Sup Table 3 |
| **C** | BLOSUM encoding (25-dim per residue) → CNN block **(no protein DCGAN)** | [TABLE] Sup Table 3 |

| Parameter | Value | Evidence |
|---|---|---|
| CNN block filter counts | 128, 256, 384 (shared specification with the drug branch) | [PAPER] p. 6; [TABLE] Sup Table 2 |
| CNN block filter length (protein) | **8** | [PAPER] p. 6: "The filter length for protein data was set to 8"; [TABLE] Sup Table 2 "Filter length (protein) — 8" |
| Input length | 2000 | [PAPER] p. 6; [TABLE] Sup Table 2 |
| BLOSUM feature dimension | 25 | [PAPER] p. 4 |
| Activation / stride / padding | **Not stated** | [OPEN] |

### 6.3 GAN components

The paper describes **two DCGANs** — one for drug SMILES and one for protein sequences. [FIGURE] Fig. 1 caption p. 5: "two deep convolutional generative adversarial networks (DCGANs) are employed to provide representations for drug SMILES and protein sequences". Variant C uses only the drug DCGAN. [TABLE] Sup Table 3

#### Generator

| Property | Specification | Evidence |
|---|---|---|
| Layer type | **Three** one-dimensional convolutional transpose (Conv1DTranspose) layers | [PAPER] p. 4: "The generator in DCGAN-DTA comprises three one-dimensional convolutional transpose (Conv1DTranspose) layers"; [FIGURE] Fig. 1 — "Conv1DTranspose 1 / 2 / 3" inside the "Generator" box |
| Filter counts | **128, 64, 1** | [PAPER] p. 6: "the generator used 128, 64, and 1 filters"; [TABLE] Sup Table 2 "DCGAN number of filters (Generator) — 128.64,1" *(printed with a period instead of a comma)* |
| Filter length | **3** | [PAPER] p. 6: "Both the generator and discriminator used a filter length of 3"; [TABLE] Sup Table 2 |
| Activations | First two layers **ReLU**; final layer **tanh** | [PAPER] p. 4 |
| Normalization | **Batch Normalization deliberately omitted** — "while the original DCGAN integrates Batch Normalization techniques within both the generator and discriminator networks, our implementation forgoes this approach due to observed performance degradation" | [PAPER] p. 4 |
| Dropout | Not mentioned for the generator | [OPEN] |
| Generator input | "Noise" | [FIGURE] Fig. 1 p. 5 — a "Noise" box feeds the Generator in both columns |
| Latent / noise dimension | **Not stated** | [OPEN] |
| Noise distribution | **Not stated** | [OPEN] |
| Strides / output length | **Not stated** | [OPEN] |
| Output | "Generated Protein Sequence (Fake)" / "Generated SMILES (Fake)" | [FIGURE] Fig. 1 p. 5 |

#### Discriminator

| Property | Specification | Evidence |
|---|---|---|
| Layer type | **Five** one-dimensional CNN (Conv1D) layers | [PAPER] p. 4: "The discriminator consists of five one-dimensional CNN (Conv1D) layers"; [FIGURE] Fig. 1 — "Conv Layer 1…5" inside each "Discriminator" box |
| Filter counts | **4, 8, 16, 32, 64** | [PAPER] p. 6; [TABLE] Sup Table 2 "DCGAN number of filters (Discriminator) — 4,8,16,32,64" |
| Filter length | **3** | [PAPER] p. 6; [TABLE] Sup Table 2 |
| Activations | **ReLU** on every conv layer — "each activated by the ReLU function". The paper explicitly contrasts this with the original DCGAN: "the original DCGAN advocates for the use of Leaky ReLU activation functions in the discriminator, a choice optimized for image data processing tasks." | [PAPER] p. 4 |
| Normalization | Batch Normalization omitted (same statement as generator) | [PAPER] p. 4 |
| Head | "Subsequently, a **flatten layer** is employed, followed by a **dense layer with a single node and a tanh activation function**." | [PAPER] p. 4 |
| Discriminator output | A single scalar with **tanh** activation (range −1…1) — *not* sigmoid | [PAPER] p. 4 |
| Discriminator input | Real sequences ("Protein Sequence (Real)" / "SMILES (Real)") and generated fake sequences | [FIGURE] Fig. 1 p. 5 |
| Dropout | Not mentioned for the discriminator | [OPEN] |
| Strides / padding | **Not stated** | [OPEN] |

#### GAN objectives

| Property | Status | Evidence |
|---|---|---|
| Adversarial objective (formal) | **No equation is given.** The paper contains exactly four numbered equations (Eqs. 1–4), all of which are evaluation metrics (§7). There is **no discriminator loss, no generator loss, no adversarial objective, and no combined objective anywhere in the paper or supplements.** | [OPEN] |
| Adversarial objective (prose) | "These models are trained together using an adversarial process, where the generator generates fake protein sequences and drug SMILES, while the discriminator learns to distinguish between real and fake protein sequences and drug SMILES." | [PAPER] p. 4 |
| Loss function name (e.g. binary cross-entropy, hinge, Wasserstein) | **Not stated** | [OPEN] |
| Label convention for real/fake | **Not stated** (note the tanh output head, whose range is −1…1, is not reconciled with any stated labelling) | [OPEN] |
| Auxiliary reconstruction / classification loss | None described | [OPEN] |
| Discriminator updates per generator update | **Not stated** | [OPEN] |
| Alternating-update schedule | **Not stated** | [OPEN] |
| GAN optimizer / learning rate | **Not separately stated.** Only one optimizer (Adam) and one learning rate (0.001) are given, in the paragraph describing "the training process" of the networks. Whether these apply to the GAN stage, the DTA stage, or both, is not disambiguated. | [PAPER] p. 6; [OPEN] |
| GAN training iterations / epochs | **Not stated** (full-text search for "iteration" returns no methodological hit) | [OPEN] |
| GAN batch size | **Not separately stated** | [OPEN] |

### 6.4 DTA prediction network

| Stage | Specification | Evidence |
|---|---|---|
| Merging layer | An **Add** layer merges the drug and protein latent vectors: "we employed an add layer for merging latent vectors for drugs and proteins. The add layer, compared to a concatenation layer, provides a linear combination with a smaller size for the latent vectors" | [PAPER] p. 4; [FIGURE] Fig. 1 — "Merging Layer / Add" band |
| Rationale given | "the add layer provides smaller dimensions, leading to a reduced number of network parameters for the prediction task" | [PAPER] p. 4 |
| Prediction block | "a fully-connected block, which is commonly used in neural network-based methods for DTA prediction" | [PAPER] p. 4 |
| FC layer widths — **prose & supplement** | "The number of neurons in the fully connected layers was set to **1024, 512, and 512**" / Sup Table 2: "Number of neurons — 1024,512,512" | [PAPER] p. 6; [TABLE] Sup Table 2 |
| FC layer widths — **figure** | Fig. 1 draws, bottom-to-top: **FC Layer (1024) → FC Layer (1024) → FC Layer (512)** | [FIGURE] Fig. 1 p. 5 |
| → Conflict | The figure (1024, 1024, 512) contradicts the prose and Sup Table 2 (1024, 512, 512). Recorded as an ambiguity in §15; **not resolved here.** | [FIGURE] vs [PAPER]/[TABLE] |
| Dropout | **0.25** | [PAPER] p. 6: "a dropout rate of 0.25 was applied to prevent overfitting"; [TABLE] Sup Table 2 "Dropout — 0.25" |
| Dropout placement | **Not stated** (which layers it follows) | [OPEN] |
| FC activations | **Not stated** | [OPEN] |
| Output layer | A single node | [FIGURE] Fig. 1 p. 5 |
| Output activation | **Not stated** | [OPEN] |
| Prediction loss function | **Not stated anywhere.** MSE appears only as an *evaluation metric* (Eq. 3, §10); the paper never says MSE is the training objective. | [OPEN] |

### 6.5 Transfer from GAN to DTA

This is the mechanism the method is named for. The paper's total treatment of it is the following:

- **What is pretrained:** "DCGAN is trained using data collected from UniProt [44] and ChEMBL [45] databases" [PAPER] p. 4.
- **What is reused:** "and the **learned models** are then utilized for the **feature extraction step**, along with CNN blocks consisting of three CNN layers and one max-pooling layer" [PAPER] p. 4.
- **Figure evidence:** In Fig. 1 (p. 5), the "Deep Convolutional GAN (DCGAN)" band sits directly beneath the "CNN Blocks" band, with arrows flowing upward from the Discriminator box into the CNN Blocks box, in both the protein and the drug column. The Generator box sits below and feeds the "Generated … (Fake)" inputs to the Discriminator; **no arrow leaves the Generator toward the CNN Blocks or the Prediction Network.** [FIGURE] Fig. 1 p. 5.

From this, the following are and are not established:

| Question | Answer | Evidence |
|---|---|---|
| Is the DCGAN pretrained separately from the DTA model? | Yes — trained on external UniProt/ChEMBL data, then "utilized for the feature extraction step" | [PAPER] p. 4 |
| Which network is reused — generator or discriminator? | The figure routes only the **discriminator** into the downstream path. The prose says "the learned models", which is ambiguous. | [FIGURE] Fig. 1 p. 5; [PAPER] p. 4 → [INFERRED] discriminator; **not explicitly stated** |
| **Exactly which layers are transferred** | **Not stated.** The paper never identifies a layer index, a layer count, or a cut point within the 5-layer discriminator. | [OPEN] |
| Where the transferred features enter the DTA model | Between the encoding/embedding step and the CNN blocks | [FIGURE] Fig. 1 p. 5 |
| Are transferred weights **frozen or trainable** during DTA training? | **Not stated anywhere.** The words "freeze", "frozen" and "fine-tune" do not occur in the paper (confirmed by full-text search). | [OPEN] |
| Is the DCGAN pretrained **once** and reused, or repeatedly? | **Not stated.** | [OPEN] |
| Is the same pretrained DCGAN shared across folds / datasets / variants? | **Not stated.** | [OPEN] |
| Are the drug and protein DCGANs trained jointly or independently? | **Not stated** (two separate DCGANs are shown, but no training order or coupling is described) | [FIGURE] Fig. 1; [OPEN] |

> This subsection is the single largest specification gap in the paper. Common GAN-transfer practice is **not** used to fill it, per the audit rules.

---

## 7. Mathematical Objective

**Finding: the paper contains exactly four numbered equations, Eqs. (1)–(4), all on p. 6, and all four are evaluation metrics.** There is no equation for the prediction loss, the adversarial loss, the generator loss, the discriminator loss, any regularization term, or any combined objective. [PAPER] pp. 1–16; [OPEN]

### Equation inventory

| Eq. | Location | Form | Symbols | Category |
|---|---|---|---|---|
| **(1)** | [EQUATION] Eq. (1), p. 6 | `CI = (1/Z) · Σ_{y_i > y_j} h(f_i − f_j)` | `f_i`, `f_j` = predicted values for the actual affinity values `y_i`, `y_j` with `y_i > y_j`; `Z` = a normalization constant; `h` = the step function of Eq. (2) | **Evaluation** (Concordance Index) |
| **(2)** | [EQUATION] Eq. (2), p. 6 | `h(x) = 1 if x > 0; 0.5 if x = 0; 0 if x < 0` | `x` = the prediction difference `f_i − f_j` | **Evaluation** (helper for Eq. 1) |
| **(3)** | [EQUATION] Eq. (3), p. 6 | `MSE = (1/n) · Σ_{i=1}^{n} (P_i − Y_i)²` | `P` = predicted affinity values; `Y` = ground-truth affinity values; `n` = number of samples | **Evaluation** (mean squared error). The paper states: "Lower MSE values indicate a closer match between the predicted and actual affinity values." |
| **(4)** | [EQUATION] Eq. (4), p. 6 | `r²_m = r² × (1 − √(r² − r₀²))` | `r²` and `r₀²` = "the squared correlation coefficients values with and without intercept, respectively" | **Evaluation** (QSAR external-prediction metric). The paper states: "a model is deemed satisfactory if `r²_m > 0.5`." Cited to refs. [46–48]. |

### Reconstruction of the total training objective

```
L_total = [OPEN]
```

No term of the training objective is supported by paper evidence. Specifically:

| Term | Status |
|---|---|
| Prediction loss (e.g. MSE on affinity) | [OPEN] — MSE appears only as a metric (Eq. 3), never designated as the training loss |
| Adversarial loss (discriminator) | [OPEN] |
| Adversarial loss (generator) | [OPEN] |
| Reconstruction loss | [OPEN] — none described |
| Regularization term | [OPEN] — dropout 0.25 is stated as a mechanism, but no explicit regularization term or weight decay is given |
| Loss-balancing coefficients | [OPEN] — no coefficient of any kind appears in the paper |

---

## 8. Training Procedure

### 8.1 Compact configuration table

| Parameter | Value | Evidence | Applies to |
|---|---|---|---|
| Optimizer | **Adam** | [PAPER] p. 6; [TABLE] Sup Table 2 "Optimizer function — adam" | [OPEN] which stage(s) |
| Learning rate | **0.001** | [PAPER] p. 6; [TABLE] Sup Table 2 | [OPEN] which stage(s) |
| Adam β₁, β₂, ε | **Not stated** | [OPEN] | — |
| LR scheduler | **Not stated** | [OPEN] | — |
| Batch size | **256** | [PAPER] p. 6: "with a batch size of 256 for weight updates"; [TABLE] Sup Table 2 | [OPEN] which stage(s) |
| Number of epochs | **300** | [PAPER] p. 6: "The training process was executed over 300 epochs"; [TABLE] Sup Table 2 "Number of epochs — 300" | [OPEN] which stage(s) |
| GAN iterations | **Not stated** | [OPEN] | GAN |
| D-updates per G-update | **Not stated** | [OPEN] | GAN |
| Pretraining schedule | Described only as "DCGAN is trained using data collected from UniProt and ChEMBL"; no epoch/iteration/batch specification | [PAPER] p. 4; [OPEN] | GAN |
| Fine-tuning schedule | **Not stated** (no statement that fine-tuning occurs at all) | [OPEN] | Transfer |
| Weight initialization | **Not stated** | [OPEN] | — |
| Max protein length | 2000 | [PAPER] p. 6; [TABLE] Sup Table 2 | Both datasets |
| Max SMILES length | 200 | [PAPER] p. 6; [TABLE] Sup Table 2 | Both datasets |
| CNN-block filters | 128, 256, 384 | [PAPER] p. 6; [TABLE] Sup Table 2 | DTA |
| Filter length (protein) | 8 | [PAPER] p. 6; [TABLE] Sup Table 2 | DTA |
| Filter length (drug) | 4 | [PAPER] p. 6; [TABLE] Sup Table 2 | DTA |
| Generator filters | 128, 64, 1 | [PAPER] p. 6; [TABLE] Sup Table 2 | GAN |
| Discriminator filters | 4, 8, 16, 32, 64 | [PAPER] p. 6; [TABLE] Sup Table 2 | GAN |
| Generator/Discriminator filter length | 3 | [PAPER] p. 6; [TABLE] Sup Table 2 | GAN |
| FC neurons | 1024, 512, 512 *(prose + Sup Table 2)* / 1024, 1024, 512 *(Fig. 1)* | [PAPER] p. 6; [TABLE] Sup Table 2; [FIGURE] Fig. 1 p. 5 — **conflicting**, see §15 | DTA |
| Dropout | 0.25 | [PAPER] p. 6; [TABLE] Sup Table 2 | DTA |
| Activation (GAN generator) | ReLU, ReLU, tanh | [PAPER] p. 4 | GAN |
| Activation (GAN discriminator) | ReLU on all five conv layers; tanh on the single-node dense head | [PAPER] p. 4 | GAN |
| Activation (CNN blocks / FC) | **Not stated** | [OPEN] | DTA |
| Early stopping | **Used** — "the early stopping technique has been employed to prevent overfitting" | [PAPER] p. 6 | [OPEN] which stage |
| Early-stopping monitored metric | **Not stated** | [OPEN] | — |
| Early-stopping patience | **Not stated** | [OPEN] | — |
| Restore-best-weights behaviour | **Not stated** | [OPEN] | — |
| Model-selection criterion | **Not stated** | [OPEN] | — |
| Hyperparameter search strategy | Stated to occur ("hyperparameter tuning was performed using these sets") but **no strategy named** (grid/random/manual) | [PAPER] p. 6; [OPEN] | — |
| Hyperparameter search ranges/grid | **Not given.** Sup Table 2 reports single values, titled "parameters settings for our models", not a search space. | [TABLE] Sup Table 2; [OPEN] | — |
| Repetitions / restarts | **Not stated** | [OPEN] | — |
| Random seed | **Not stated anywhere** | [OPEN] | — |
| Framework | **Python, Keras, TensorFlow** | [PAPER] Implementation details, p. 6 | — |
| Framework versions | **Not stated** | [OPEN] | — |
| OS | **Ubuntu 18.04** | [PAPER] p. 6 | — |
| CPU | **Intel(R) Xeon(R) CPU @ 2.30 GHz** | [PAPER] p. 6 | — |
| GPU | **NVIDIA GeForce GTX 1080, 11 GB** | [PAPER] p. 6 | — |
| Training wall-clock time | **Not reported** | [OPEN] | — |
| Parameter count / FLOPs | **Not reported** | [OPEN] | — |

### 8.2 Explicitly specified vs not specified

**Explicitly specified:** optimizer (Adam), learning rate (0.001), batch size (256), epochs (300), dropout (0.25), all filter counts and filter lengths for both the GAN and the CNN blocks, input length caps (2000 / 200), FC widths (with an internal conflict), GAN activation functions, use of early stopping, framework stack, OS, CPU, GPU.

**Not specified:** random seed; early-stopping metric/patience/restore behaviour; model-selection rule; hyperparameter search strategy and ranges; weight initialization; GAN iteration count, GAN batch size, GAN optimizer/learning rate as distinct from the DTA stage; D:G update ratio; noise dimension and distribution; embedding dimensions; CNN-block and FC activations; strides and padding of every convolution; dropout placement; the training loss function; whether transferred layers are frozen; the number of times pretraining is performed; framework versions; runtime.

---

## 9. Baselines

### 9.1 Warm-start and cold-start comparison methods (Figs. 2, 3, 5, 6)

Seven alternative methods are compared. [PAPER] p. 6: "we compared it against **seven** alternative methods that employ different protein and drug representations." Counting the distinct bars in Figs. 2–3, the seven are DeepDTA-Sim, DeepDTA-CNN, GraphDTA, FusionDTA, DGDTA, TEFDTA, G-K BertDTA. [FIGURE] Figs. 2–3 legends, pp. 7–8

| Baseline | Category (as framed by the paper) | Protein representation | Drug representation | Reference | Implementation/source mentioned? |
|---|---|---|---|---|---|
| DeepDTA-Sim | similarity-based variant of a sequence-based method | Smith-Waterman similarities | Label encoding, embedding layer, and 1DCNN | [16] | [OPEN] |
| DeepDTA-CNN | sequence-based | Label encoding, embedding layer, and 1DCNN | Label encoding, embedding layer, and 1DCNN | [16] | [OPEN] |
| GraphDTA | graph-based | Label encoding, embedding layer, and 1DCNN | Graph neural networks (GNNs) | [18] | [OPEN] |
| FusionDTA | transformer-based | Transformers, Feed-Forward networks, and BiLSTM | Feed-Forward networks, and BiLSTM | [23] | [OPEN] |
| DGDTA | graph-based | Bi-LSTM, and CNN | Dynamic graph attention networks | [19] | [OPEN] |
| TEFDTA | transformer-based | Label encoding, embedding layer, and 1DCNN | MACCS fingerprint, embedding layer, and Transformer | [25] | [OPEN] |
| G-K BertDTA | transformer-based | CNN, and DenseSENet (DenseNet and squeeze-and-excitation (SE) blocks) | Graph isomorphism network (GIN), and KB-BERT | [26] | [OPEN] |

Representations table: [TABLE] Sup Table 3 (MOESM7). Category grouping: [PAPER] p. 13 — "We compare DCGAN-DTA against alternative methods in four groups: similarity-based, sequence-based, graph-based, and transformer-based methods".

**Selection rationale as stated** [PAPER] p. 6: "We included established baseline methods including DeepDTA and GraphDTA for benchmarking, along with state-of-the-art techniques including FusionDTA, DGDTA, TEFDTA, and G-K BertDTA to assess novelty and potential advancements. Each method was chosen based on its relevance, availability of implementations or results, and practical considerations."

**Critical gap:** the paper never states whether baseline numbers were **re-run by the authors** on their own BindingDB/PDBbind splits or **copied from the original publications**. The phrase "availability of implementations or results" [PAPER] p. 6 is compatible with either. No baseline hyperparameters, no retraining protocol, and no baseline code references are given. [OPEN]

**Evaluation protocol for baselines:** presumed identical to DCGAN-DTA's (same figures, same metrics, same datasets), but **never stated**. [OPEN]

### 9.2 Lightweight / ablation baselines (Fig. 8, PDBbind only)

| Baseline | Description as given | Evidence |
|---|---|---|
| Fully-connected network | "a fully-connected network … trained using the PDBBind dataset encoded using the label encoding technique" | [PAPER] p. 12 |
| K-nearest neighbor | "k-nearest neighbor models trained using the PDBBind dataset encoded using the label encoding technique" | [PAPER] p. 12 |
| DeepDTA | "DeepDTA, which utilizes the DCGAN for neither protein nor drugs" | [PAPER] p. 12 |
| DCGAN-DTA(C) — with concatenation | Merging-layer ablation: Add layer replaced by concatenation | [PAPER] p. 12; [FIGURE] Fig. 8 legend p. 13 |

Architecture, hyperparameters, `k`, and distance metric for the FC and KNN baselines are **all unstated**. [OPEN]

> Cross-check: Fig. 8's "DeepDTA" values (CI 0.756, AUPR 0.777, MSE 2.14) are identical to Fig. 3's DeepDTA-CNN values, indicating the two labels denote the same run. [FIGURE] Figs. 3 and 8; [INFERRED]

---

## 10. Evaluation Metrics

Four metrics are used. [PAPER] p. 6: "we employed four commonly used performance metrics: the concordance index (CI), mean squared error (MSE), area under the precision-recall curve (AUPR), and `r²_m`."

| Metric | Formula given? | Definition / protocol | Evidence |
|---|---|---|---|
| **CI** (Concordance Index) | Yes — Eqs. (1)+(2) | `CI = (1/Z) Σ_{y_i>y_j} h(f_i − f_j)`; `h` as in Eq. (2); `Z` a normalization constant. "The CI provides a measure of how well the model predicts the ordering of affinity values." | [EQUATION] Eqs. (1),(2), p. 6 |
| **MSE** | Yes — Eq. (3) | `MSE = (1/n) Σ (P_i − Y_i)²` on the continuous affinity scale | [EQUATION] Eq. (3), p. 6 |
| **AUPR** (area under the precision-recall curve) | **No formula** | Derived by binarising affinity: "we transformed the binding affinities of both the PDBBind and BindingDB datasets into binary values. To accomplish this, we applied a **threshold of 7** to the binding affinities." | [PAPER] p. 6 |
| **`r²_m`** | Yes — Eq. (4) | `r²_m = r² × (1 − √(r² − r₀²))`, with `r²`/`r₀²` the squared correlation coefficients with/without intercept; satisfactory if `r²_m > 0.5` | [EQUATION] Eq. (4), p. 6 |
| **AUROC** | — | **Not used.** The string "AUROC"/"ROC" does not occur in the paper. | [OPEN] |
| **RMSE** | — | **Not used.** | [OPEN] |

### Per-metric protocol determination

| Question | CI | MSE | AUPR | `r²_m` |
|---|---|---|---|---|
| Threshold for deriving classification | n/a | n/a | **7**, applied to both datasets [PAPER] p. 6 | n/a |
| Positive-label definition (≥7 vs >7, and which side is positive) | n/a | n/a | **Not stated.** The paper says only "we applied a threshold of 7". | n/a |
| Justification for threshold 7 | n/a | n/a | **None given.** No citation, no rationale, and no acknowledgement that the two datasets have different affinity distributions (BindingDB pKd 2–9 vs PDBbind pKi/pKd 2–12, Sup Fig. 1a). | n/a |
| Averaging procedure | [OPEN] | [OPEN] | [OPEN] | [OPEN] |
| Fold aggregation | [OPEN] | [OPEN] | [OPEN] | [OPEN] |
| Mean and std reported? | **No std is reported anywhere.** Figs. 2–3 and 5–8 are bar charts with single point values and **no error bars**. | same | same | same |
| Mean-of-folds vs best-fold vs final-epoch reporting | **Not stated for any metric.** | same | same | same |

### Statistical testing

| Aspect | Extraction | Evidence |
|---|---|---|
| Test used | **t-test** | [PAPER] pp. 8, 9, 11 |
| What is compared | "t-tests were performed between the methods with the **first- and second-best CI scores**"; "For a more comprehensive comparison, we included all three versions of DCGAN-DTA in the tests." | [PAPER] p. 8 |
| Metric tested | CI only | [PAPER] pp. 8, 9, 11 |
| Significance level | 95% | [PAPER] pp. 8, 9 |
| Annotation convention | "stars to denote the significance level of 95%"; "'ns' indicates that the difference in results between those methods is not statistically significant" | [PAPER] pp. 8, 9 |
| What the tested distribution consists of | **Not stated.** Sup Figs. 2–4 show box plots of "Concordance Index (CI)" per method, but the paper never says whether the underlying samples are the 5 CV folds, repeated runs, or something else. Box-plot whisker/outlier conventions are also unstated. | [SUPPLEMENT] Sup Figs. 2–4; [OPEN] |
| Test variant (paired/unpaired, one/two-sided), multiple-comparison correction | **Not stated** | [OPEN] |

---

## 11. Reported Results

All results are presented as **bar charts with printed value labels**; the paper contains **no numeric results table**. Values below are transcribed from the figures' printed labels. No value is recalculated.

Method order in the legends of Figs. 2, 3, 5, 6 is constant: DeepDTA-Sim, DeepDTA-CNN, GraphDTA, FusionDTA, DGDTA, TEFDTA, G-K BertDTA, DCGAN-DTA(A), DCGAN-DTA(B), DCGAN-DTA(C).

### 11.1 Warm-start — BindingDB [FIGURE] Fig. 2, p. 7

| Method | CI | AUPR | RM2 | MSE |
|---|---|---|---|---|
| DeepDTA-Sim | 0.81 | 0.753 | 0.555 | 0.726 |
| DeepDTA-CNN | 0.838 | 0.806 | 0.589 | 0.558 |
| GraphDTA | 0.858 | 0.858 | 0.712 | 0.427 |
| FusionDTA | 0.863 | 0.84 | 0.652 | 0.466 |
| DGDTA | 0.853 | 0.842 | 0.676 | 0.465 |
| TEFDTA | 0.849 | **0.86** | **0.716** | 0.446 |
| G-K BertDTA | 0.846 | 0.843 | 0.69 | 0.475 |
| DCGAN-DTA (A) | 0.859 | 0.848 | 0.663 | 0.456 |
| DCGAN-DTA (B) | **0.866** | 0.859 | 0.686 | 0.428 |
| DCGAN-DTA (C) | **0.866** | 0.858 | 0.699 | **0.425** |

Prose summary [PAPER] pp. 6–7: "DCGAN-DTA exhibited superior performance … in terms of CI and MSE, achieving the second-best AUPR, and third-best `r²_m` … Specifically, DCGAN-DTA (C) achieved the best CI and the MSE, and the third-best `r²_m` while DCGAN-DTA (B) achieved the best CI and the second-best AUPR."

### 11.2 Warm-start — PDBbind [FIGURE] Fig. 3, p. 8

| Method | CI | AUPR | RM2 | MSE |
|---|---|---|---|---|
| DeepDTA-Sim | 0.745 | 0.748 | 0.408 | 2.219 |
| DeepDTA-CNN | 0.756 | 0.777 | **0.471** | 2.14 |
| GraphDTA | 0.747 | 0.747 | 0.402 | 1.999 |
| FusionDTA | 0.768 | 0.782 | 0.462 | 1.936 |
| DGDTA | 0.752 | 0.752 | 0.402 | 1.971 |
| TEFDTA | 0.749 | 0.78 | 0.391 | 1.96 |
| G-K BertDTA | 0.739 | 0.729 | 0.383 | 2.089 |
| DCGAN-DTA (A) | **0.774** | 0.775 | 0.433 | 2.343 |
| DCGAN-DTA (B) | 0.764 | 0.776 | 0.443 | 1.983 |
| DCGAN-DTA (C) | 0.768 | **0.787** | 0.451 | **1.907** |

Prose summary [PAPER] pp. 7–8: "DCGAN-DTA (A) achieved the best CI, while DCGAN-DTA (C) demonstrated superior AUPR and MSE, and the third-best `r²_m` among the alternative methods, for the warm-start data splitting setting on the PDBBind dataset."

### 11.3 Cold-start (drug), PDBbind — Open Babel logP [FIGURE] Fig. 5, p. 10

*(RM2 is not reported for cold-start settings — Figs. 5 and 6 contain only CI, AUPR, MSE.)*

| Method | CI | AUPR | MSE |
|---|---|---|---|
| DeepDTA-Sim | 0.575 | 0.262 | 3.094 |
| DeepDTA-CNN | 0.589 | 0.298 | 4.064 |
| GraphDTA | 0.567 | 0.252 | 3.755 |
| FusionDTA | 0.583 | **0.334** | 3.786 |
| DGDTA | 0.579 | 0.318 | 3.116 |
| TEFDTA | 0.558 | 0.294 | **3.08** |
| G-K BertDTA | 0.566 | 0.257 | 4.505 |
| DCGAN-DTA (A) | 0.589 | 0.308 | 3.773 |
| DCGAN-DTA (B) | 0.597 | 0.321 | 3.387 |
| DCGAN-DTA (C) | **0.601** | 0.323 | 3.412 |

### 11.4 Cold-start (drug), PDBbind — XLOGP3 [FIGURE] Fig. 6, p. 11

| Method | CI | AUPR | MSE |
|---|---|---|---|
| DeepDTA-Sim | 0.566 | 0.307 | **3.064** |
| DeepDTA-CNN | 0.608 | 0.372 | 3.426 |
| GraphDTA | 0.582 | 0.312 | 3.519 |
| FusionDTA | 0.596 | 0.362 | 3.47 |
| DGDTA | 0.578 | 0.369 | 3.431 |
| TEFDTA | 0.563 | 0.306 | 4.342 |
| G-K BertDTA | 0.575 | 0.281 | 5.153 |
| DCGAN-DTA (A) | **0.615** | 0.387 | 3.607 |
| DCGAN-DTA (B) | 0.604 | 0.385 | 3.496 |
| DCGAN-DTA (C) | 0.609 | **0.393** | 3.207 |

Prose summary [PAPER] p. 9: "DCGAN-DTA showcased superior performance … in terms of CI scores for both logP values … our method achieved the best AUPR for XLOGP3 and the second-best AUPR for Open Babel logP. Moreover, DCGAN-DTA achieved the second-best MSE for XLOGP3 data splitting settings."

### 11.5 Adversarial control / straw models, PDBbind [FIGURE] Fig. 7, p. 12

| Setting | CI | AUPR | MSE |
|---|---|---|---|
| Shuffled train data | 0.605 | 0.521 | 3.767 |
| Shuffled test data | 0.494 | 0.38 | 5.476 |
| Shuffled train and test data | 0.513 | 0.382 | 5.699 |
| DCGAN-DTA (A) | 0.774 | 0.775 | 2.338 |
| DCGAN-DTA (B) | 0.764 | 0.776 | 1.983 |
| DCGAN-DTA (C) | 0.768 | 0.787 | 1.907 |

Prose summary [PAPER] p. 11: "all experiments on straw models result in a loss of predictive performance across the three performance metrics. Therefore, we can conclude that our method is robust against confounding variables and data artifacts."

### 11.6 Lightweight baselines and merging-layer ablation, PDBbind [FIGURE] Fig. 8, p. 13

| Method | CI | AUPR | MSE |
|---|---|---|---|
| Fully-connected network | 0.728 | 0.722 | 2.45 |
| K-nearest neighbor | 0.737 | 0.772 | 2.351 |
| DeepDTA | 0.756 | 0.777 | 2.14 |
| DCGAN-DTA (A) | **0.774** | 0.775 | 2.338 |
| DCGAN-DTA (B) | 0.764 | 0.776 | 1.983 |
| DCGAN-DTA (C) | 0.768 | **0.787** | **1.907** |
| DCGAN-DTA (C) — with concatenation | 0.763 | 0.774 | 1.91 |

Prose summary [PAPER] pp. 12–13: "DCGAN-DTA outperformed the baselines, achieving the best CI, MSE, and AUPR scores"; "DCGAN-DTA with the add layer demonstrated superior prediction performance compared to DCGAN-DTA with the concatenation layer across all three performance metrics."

### 11.7 Statistical-test outcomes (supplement)

| Comparison | Dataset / setting | Outcome | Evidence |
|---|---|---|---|
| FusionDTA vs DCGAN-DTA(C) | warm-start, BindingDB | `*` (significant) | [SUPPLEMENT] Sup Fig. 2(a) |
| DCGAN-DTA(A) vs DCGAN-DTA(C) | warm-start, BindingDB | `*` | [SUPPLEMENT] Sup Fig. 2(a) |
| DCGAN-DTA(B) vs DCGAN-DTA(C) | warm-start, BindingDB | `ns` | [SUPPLEMENT] Sup Fig. 2(a) |
| FusionDTA vs DCGAN-DTA(A) | warm-start, PDBbind | `*` | [SUPPLEMENT] Sup Fig. 2(b) |
| DCGAN-DTA(A) vs DCGAN-DTA(B) | warm-start, PDBbind | `*` | [SUPPLEMENT] Sup Fig. 2(b) |
| DCGAN-DTA(A) vs DCGAN-DTA(C) | warm-start, PDBbind | `ns` | [SUPPLEMENT] Sup Fig. 2(b) |
| DeepDTA-CNN vs DCGAN-DTA(A) | cold-start, Open Babel logP | `ns` | [SUPPLEMENT] Sup Fig. 3(a) |
| DeepDTA-CNN vs DCGAN-DTA(B) | cold-start, Open Babel logP | `ns` | [SUPPLEMENT] Sup Fig. 3(a) |
| DeepDTA-CNN vs DCGAN-DTA(C) | cold-start, Open Babel logP | `*` | [SUPPLEMENT] Sup Fig. 3(a) |
| DeepDTA-CNN vs DCGAN-DTA(A/B/C) | cold-start, XLOGP3 | `ns` for all three | [SUPPLEMENT] Sup Fig. 3(b) |
| DCGAN-DTA(A) vs each of the three shuffled settings | adversarial control, PDBbind | `****` for all three | [SUPPLEMENT] Sup Fig. 4 |

Corresponding prose [PAPER] p. 10: "the statistical tests suggest that DCGAN-DTA significantly outperforms DeepDTA-CNN in terms of CI for data splitting based on Open Babel logP. However, the CI difference between DeepDTA-CNN and DCGAN-DTA is not statistically significant for the XLOGP3 data splitting setting."

---

## 12. Figures and Tables Inventory

| ID | Type | What it contains | Reproduction relevance | Locator |
|---|---|---|---|---|
| **Fig. 1** | Architecture diagram | See expanded description below | **Highest** — sole architecture source; also the source of one FC-width conflict | p. 5 |
| **Fig. 2** | Grouped bar chart (2 panels) | CI / AUPR / RM2 (upper) and MSE (lower) for 10 methods, BindingDB, warm-start. Printed value labels; **no error bars**; y-axes unlabelled and partly cropped | High — target numbers for warm-start BindingDB | p. 7 |
| **Fig. 3** | Grouped bar chart (2 panels) | Same layout, PDBbind, warm-start | High — target numbers for warm-start PDBbind | p. 8 |
| **Fig. 4** | Scatter plots (a),(b) | Predicted (x) vs Measured (y) affinity. (a) BindingDB: axes ≈ 2–9 both; (b) PDBbind: axes ≈ 2–12 both. No fit line, no R², no fold indication, no count | Low-moderate — qualitative only; no numbers extractable. Confirms the prediction range of each dataset | p. 9 |
| **Fig. 5** | Grouped bar chart (2 panels) | CI / AUPR (upper), MSE (lower), 10 methods, PDBbind cold-start via **Open Babel logP**. **No RM2 panel** | High — cold-start target numbers | p. 10 |
| **Fig. 6** | Grouped bar chart (2 panels) | Same, PDBbind cold-start via **XLOGP3** | High | p. 11 |
| **Fig. 7** | Grouped bar chart (2 panels) | CI / AUPR (upper), MSE (lower) for 3 shuffled settings + DCGAN-DTA A/B/C, PDBbind | High — adversarial-control target numbers | p. 12 |
| **Fig. 8** | Grouped bar chart (2 panels) | CI / AUPR (upper), MSE (lower) for FC net, KNN, DeepDTA, DCGAN-DTA A/B/C, and DCGAN-DTA(C)-with-concatenation, PDBbind | High — ablation (merging layer) and lightweight-baseline numbers | p. 13 |
| **Sup Fig. 1** | 3×2 histogram grid | Affinity, SMILES-length and protein-length distributions for both datasets | **High** — only distributional evidence; reveals the BindingDB pKd≈5 spike and the length overflow beyond the 200/2000 caps | MOESM1 |
| **Sup Fig. 2** | Box plots (a),(b) | CI distributions, warm-start, BindingDB and PDBbind; FusionDTA + DCGAN-DTA A/B/C; significance brackets | Moderate — the only evidence of CI **variability**; the sample definition is unstated | MOESM2 |
| **Sup Fig. 3** | Box plots (a),(b) | CI distributions, cold-start (Open Babel logP / XLOGP3), PDBbind; DeepDTA-CNN + DCGAN-DTA A/B/C | Moderate | MOESM3 |
| **Sup Fig. 4** | Box plot | CI distributions across 3 shuffled settings vs DCGAN-DTA(A) | Moderate | MOESM4 |
| **Sup Table 1** | Data table | Proteins / Compounds / Interactions / Affinity measure for both datasets | **High** — the sole source of the interaction counts (42203, 5014) | MOESM5 |
| **Sup Table 2** | Config table | 15 parameter rows (see §8.1) | **Highest among tables** — the only consolidated hyperparameter listing | MOESM6 |
| **Sup Table 3** | Comparison table | Protein and drug representations for 7 baselines + the 3 DCGAN-DTA variants | High — the only precise definition of what distinguishes variants A, B and C | MOESM7 |

There are **no numbered tables in the main paper** — all tables live in the supplement. [PAPER] pp. 1–16

### Fig. 1 — information actually conveyed (not merely the caption)

Fig. 1 (p. 5) is a five-band vertical diagram, read bottom-to-top, with two parallel columns (protein on the left, drug on the right).

1. **Inputs (bottom, blue):** left "Protein Sequence MPSPMVKG…LMPCP"; right "Drug SMILES CC(=C1)…CCN=C3".
2. **"Encoding and Embedding" band (yellow):** the *protein* box contains **two parallel elements** — a "BLOSUM Encoding" block and a "Label Encoding → Embedding Layer" stack — drawn side by side, i.e. the figure presents both protein encodings simultaneously rather than as alternatives. The *drug* box contains only "Label Encoding → Embedding Layer".
3. **"Deep Convolutional GAN (DCGAN)" band (green), drawn per column:** a "Noise" box feeds a "Generator" containing "Conv1DTranspose 1 / 2 / 3"; the Generator emits "Generated Protein Sequence (Fake)" / "Generated SMILES (Fake)"; alongside these sit "Protein Sequence (Real)" / "SMILES (Real)", fed upward from the encoding band. Both real and fake streams feed a "Discriminator" containing "Conv Layer 1 … Conv Layer 5".
4. **"CNN Blocks" band (green):** each column has "Conv Layer 1 / 2 / 3", fed from the Discriminator above. **No pooling element is drawn**, despite the prose specifying "one max-pooling layer".
5. **"Merging Layer" band:** a single wide "Add" block receiving arrows from both CNN blocks.
6. **"Prediction Network" band (yellow):** three stacked boxes, bottom-to-top "FC Layer (1024)", "FC Layer (1024)", "FC Layer (512)", terminating in a single blue output node.

**What the figure does not convey:** no tensor shapes, no filter counts, no strides, no activation labels, no indication of which layers are transferred from the discriminator, and no freeze/trainable marking. The figure depicts the union of all three variants rather than any one of A, B or C. [FIGURE] Fig. 1 p. 5

---

## 13. Supplementary Material Audit

The paper's "Supplementary Information" block (p. 15) lists seven items as "Supplementary Material 1"–"Supplementary Material 7" without describing them. The mapping below was established by opening each file.

| File | Maps to | Purpose | Methods? | Datasets? | Preproc? | Hyperparams? | Impl.? | Extra experiments? | Content type |
|---|---|---|---|---|---|---|---|---|---|
| `12864_2024_10326_MOESM1_ESM.docx` | **Sup Fig. 1** | Dataset distributions | No | **Yes** | No | No | No | No | 1 embedded TIFF + caption |
| `…MOESM2_ESM.docx` | **Sup Fig. 2** | CI distribution box plots, warm-start | No | No | No | No | No | **Yes** (significance testing) | 1 embedded TIFF + caption |
| `…MOESM3_ESM.docx` | **Sup Fig. 3** | CI distribution box plots, cold-start | No | No | No | No | No | **Yes** | 1 embedded TIFF + caption |
| `…MOESM4_ESM.docx` | **Sup Fig. 4** | CI distribution box plots, adversarial control | No | No | No | No | No | **Yes** | 1 embedded TIFF + caption |
| `…MOESM5_ESM.docx` | **Sup Table 1** | Dataset summary | No | **Yes** | No | No | No | No | 3-row × 5-col table |
| `…MOESM6_ESM.docx` | **Sup Table 2** | "parameters settings for our models" | No | No | No | **Yes** | No | No | 15-row × 2-col table |
| `…MOESM7_ESM.docx` | **Sup Table 3** | Baseline & variant representations | Partly | No | No | No | No | No | 11-row × 3-col table |

**Reproduction-critical information present ONLY in the supplement:**

1. **Interaction counts** — BindingDB **42,203**; PDBbind **5,014**. Neither number appears in the main text. [TABLE] Sup Table 1 (MOESM5)
2. **A consolidated hyperparameter listing.** Sup Table 2 (MOESM6) is the only place where all 15 settings appear together; the main text scatters most of them across one paragraph on p. 6, and Sup Table 2 is the second (corroborating) source for the FC widths 1024/512/512.
3. **The precise definition of variants A, B and C.** Sup Table 3 (MOESM7) is the only place stating exactly which encoding and which DCGAN each variant uses. Critically, its **Drugs column for rows B and C is a vertically merged cell** carrying the A value — establishing that **all three variants share the identical drug pipeline** ("Label encoding, embedding layer, DCGAN, and 1DCNN"). This is not stated in the main text.
4. **Baseline representation table.** Sup Table 3 is the only source for what each of the seven baselines uses for proteins and drugs.
5. **Affinity, SMILES-length and protein-length distributions.** Sup Fig. 1 (MOESM1) is the only distributional evidence in the entire publication; the main text refers to it twice (p. 4, p. 8) but reproduces none of its content.
6. **Evidence that metric variability exists.** Sup Figs. 2–4 are the only place a spread of CI values is shown; the main-paper bar charts show point estimates with no error bars.

**Reproduction-critical information NOT added by any supplement:** the training objective; the GAN loss; the transfer/freezing protocol; the GAN training schedule; the hyperparameter search space; the random seed; the split procedure; the test-set definition; the fold-aggregation rule; the logP cold-start threshold; the affinity log-transform formula; the SMILES/protein vocabularies; the embedding dimensions; the sizes of the UniProt/ChEMBL corpora.

---

## 14. Reproduction-Critical Specification

| Requirement | Paper evidence | Locator | Status |
|---|---|---|---|
| **Dataset** | BindingDB (Kd version, via TDC [40]): 9864 drugs, 1088 proteins, 42203 interactions. PDBbind refined v2020 [41]: 4231 drugs, 1606 proteins, 5014 interactions | [PAPER] p. 3; [TABLE] Sup Table 1 | **PARTIAL** — counts explicit; the exact download, version tag, and filtering script are not |
| **Affinity preprocessing** | "logarithmic-transformed binding affinities (pKd)"; PDBbind "logarithmic-transformed Ki and Kd"; PDBbind redundancy removal for multiple sequences per drug; BindingDB "refining … according to recommended guidelines for data harmonization" | [PAPER] p. 3 | **PARTIAL** — the transform is named but never written as a formula; "recommended guidelines" are not cited; missing-value and duplicate handling for BindingDB unspecified |
| **Drug representation** | SMILES → label encoding → padding to 200 → embedding layer | [PAPER] p. 4, p. 6; [TABLE] Sup Table 2; [TABLE] Sup Table 3 | **PARTIAL** — pipeline explicit; vocabulary, embedding dimension, pad value/side, truncation rule all missing |
| **Target representation** | Variant A: label encoding + embedding, padded to 2000. Variants B/C: BLOSUM encoding, 25-dim per residue, 25 × L matrix | [PAPER] p. 4, p. 6; [TABLE] Sup Tables 2, 3 | **PARTIAL** — the specific BLOSUM matrix, the 25-symbol alphabet, the embedding dimension, and the truncation rule are missing |
| **Split protocol** | Fivefold CV; "dividing the dataset into five nearly equal-sized training and validation sets"; "hyperparameter tuning was performed using these sets" | [PAPER] p. 6 | **OPEN** — no test-set definition, no fold-generation procedure, no warm-start definition, no seed |
| **Cold-start protocol** | Drugs excluded from training by logP (Open Babel logP / XLOGP3), excluded drugs used as test data; PDBbind only | [PAPER] p. 9 | **PARTIAL** — mechanism stated; threshold, direction and resulting split sizes all missing |
| **Architecture** | 4 steps; drug/protein CNN blocks (3 conv + 1 max-pool, filters 128/256/384, filter length 4 drug / 8 protein); Add merging; 3 FC layers; dropout 0.25 | [PAPER] p. 4, p. 6; [TABLE] Sup Table 2; [FIGURE] Fig. 1 | **PARTIAL** — filter counts/lengths explicit; **FC widths conflict** between Fig. 1 and text/Sup Table 2; activations, strides, padding, tensor shapes and embedding dims missing |
| **GAN architecture** | Generator: 3 × Conv1DTranspose, filters 128/64/1, kernel 3, ReLU/ReLU/tanh, **no BatchNorm**. Discriminator: 5 × Conv1D, filters 4/8/16/32/64, kernel 3, ReLU, → Flatten → Dense(1) with tanh | [PAPER] p. 4, p. 6; [TABLE] Sup Table 2 | **EXPLICIT** for layer counts, filter counts, kernel sizes and activations; **OPEN** for strides, padding, noise dimension and noise distribution |
| **GAN objective** | Prose only: "trained together using an adversarial process" | [PAPER] p. 4 | **OPEN** — no equation, no loss name, no label convention, no D:G ratio, no iteration count |
| **Prediction objective** | None stated. MSE appears only as an evaluation metric (Eq. 3) | [EQUATION] Eq. (3), p. 6 | **OPEN** |
| **GAN→DTA transfer** | "the learned models are then utilized for the feature extraction step"; Fig. 1 routes only the Discriminator into the CNN Blocks | [PAPER] p. 4; [FIGURE] Fig. 1 p. 5 | **OPEN** — transferred layer(s), freeze vs trainable, and pretraining frequency are all unspecified |
| **Pretraining corpora** | UniProt [44] and ChEMBL [45] | [PAPER] p. 4 | **PARTIAL** — named; no versions, sizes, filtering, or overlap analysis |
| **Optimizer** | Adam | [PAPER] p. 6; [TABLE] Sup Table 2 | **EXPLICIT** (stage attribution **OPEN**) |
| **Learning rate** | 0.001 | [PAPER] p. 6; [TABLE] Sup Table 2 | **EXPLICIT** (stage attribution **OPEN**) |
| **Batch size** | 256 | [PAPER] p. 6; [TABLE] Sup Table 2 | **EXPLICIT** (stage attribution **OPEN**) |
| **Epochs / iterations** | 300 epochs | [PAPER] p. 6; [TABLE] Sup Table 2 | **PARTIAL** — DTA epochs given; GAN iteration count entirely missing |
| **Hyperparameters** | 15 values in Sup Table 2 | [TABLE] Sup Table 2 | **EXPLICIT** as a final configuration; **OPEN** as a search space |
| **Model selection** | "early stopping technique has been employed" | [PAPER] p. 6 | **OPEN** — monitored metric, patience, restore-best and checkpoint-selection rule all missing |
| **Evaluation** | CI (Eqs. 1–2), MSE (Eq. 3), AUPR (threshold 7), `r²_m` (Eq. 4); t-tests on CI at 95% | [EQUATION] Eqs. (1)–(4), p. 6; [PAPER] pp. 6, 8, 9 | **PARTIAL** — formulas explicit for CI/MSE/`r²_m`; AUPR positive-class definition, fold aggregation, mean-vs-best reporting and the t-test sample definition are all missing |
| **Random seed** | — | — | **OPEN** |
| **Hardware** | Ubuntu 18.04; Intel Xeon @ 2.30 GHz; NVIDIA GTX 1080, 11 GB; Python + Keras + TensorFlow | [PAPER] p. 6 | **EXPLICIT** for hardware/OS/stack; **OPEN** for library versions and runtime |

---

## 15. Ambiguities and Internal Inconsistencies

Paper-side only. No code was consulted to resolve any of these.

| # | Issue | Conflicting sources |
|---|---|---|
| **A1** | **Fully-connected layer widths conflict.** Prose p. 6 and Sup Table 2 both say **1024, 512, 512**. Fig. 1 draws **FC Layer (1024) → FC Layer (1024) → FC Layer (512)**. Two of three sources agree, but the architecture figure — the only architecture figure — disagrees. | [PAPER] p. 6 + [TABLE] Sup Table 2 vs [FIGURE] Fig. 1 p. 5 |
| **A2** | **Max-pooling layer missing from the architecture figure.** Prose p. 4: "CNN blocks consisting of three CNN layers and **one max-pooling layer**". Fig. 1's "CNN Blocks" band shows only "Conv Layer 1 / 2 / 3" with no pooling element. | [PAPER] p. 4 vs [FIGURE] Fig. 1 p. 5 |
| **A3** | **DCGAN-DTA(A) MSE on PDBbind is reported as two different values.** Fig. 3 prints **2.343**; Figs. 7 and 8 both print **2.338** for the same model on the same dataset and setting. All other DCGAN-DTA(A) values (CI 0.774, AUPR 0.775) are identical across the three figures. | [FIGURE] Fig. 3 p. 8 vs [FIGURE] Fig. 7 p. 12 and Fig. 8 p. 13 |
| **A4** | **The warm-start test set is never defined.** The Implementation-details paragraph describes only "five nearly equal-sized **training and validation** sets" used for training and hyperparameter tuning. No sentence anywhere states that a held-out test set exists for the warm-start experiments, what it consists of, or when it is evaluated — yet Figs. 2–3 report single performance numbers. | [PAPER] p. 6 |
| **A5** | **Fold count vs box-plot sample count.** Sup Figs. 2–4 show box plots (with quartiles, whiskers and outlier diamonds) of CI scores, and t-tests are computed from them, but the paper never states what the samples are. A 5-fold CV yields 5 values, which is a very small sample for the displayed box-plot geometry; the paper offers no reconciliation. | [PAPER] p. 6 vs [SUPPLEMENT] Sup Figs. 2–4 |
| **A6** | **Adversarial-control model attribution.** Text p. 10 states the shuffled experiments "were specifically performed on **DCGAN-DTA (C)**". Sup Fig. 4's unshuffled reference column is labelled **DCGAN-DTA(A)** (value ≈ 0.774, matching A in Fig. 7). The t-tests therefore compare shuffled-(C) runs against unshuffled-(A), which the paper does not explain. | [PAPER] p. 10 vs [SUPPLEMENT] Sup Fig. 4 and [FIGURE] Fig. 7 p. 12 |
| **A7** | **BLOSUM dimensionality is asserted but the matrix is unnamed.** Text p. 4 says each amino acid becomes a **25-dimensional** vector, citing ref. [43], whose title concerns **BLOSUM62** (a 20-residue substitution matrix, commonly extended to 23–25 symbols with ambiguity codes). The paper never states which matrix or symbol set produces 25 dimensions. | [PAPER] p. 4 + Ref. [43] p. 16 |
| **A8** | **Generator output width vs BLOSUM input width.** The generator is specified with final filter count **1** (i.e. a single-channel output) for both the drug and protein DCGANs. For variant B the protein DCGAN consumes a **25-channel** BLOSUM matrix. The paper does not reconcile a 1-channel generator output with a 25-channel real-data stream. | [PAPER] p. 4, p. 6; [TABLE] Sup Table 2 |
| **A9** | **Discriminator output activation is tanh.** The single-node dense head uses **tanh** (range −1…1), not sigmoid. No real/fake label convention or loss function is stated that would make this well-defined. | [PAPER] p. 4 |
| **A10** | **Stage attribution of the training hyperparameters is unresolved.** "The training process was executed over 300 epochs, with a batch size of 256 … Adam … learning rate of 0.001" appears in the paragraph following the description of the fully-connected block, but the sentence says "for training **the networks**" (plural). Whether these settings govern the GAN stage, the DTA stage, or both, is undetermined. | [PAPER] p. 6 |
| **A11** | **AUPR threshold applied uniformly across two differently-scaled datasets.** A single threshold of 7 is applied to BindingDB (pKd ≈ 2–9) and PDBbind (pKi/pKd ≈ 2–12). No justification or citation is given, and the positive class is not defined. | [PAPER] p. 6; [SUPPLEMENT] Sup Fig. 1(a) |
| **A12** | **Baseline provenance is undetermined.** "availability of implementations **or results**" leaves open whether each baseline number was re-run by the authors under the same protocol or transcribed from the original paper. The two possibilities imply very different reproduction targets. | [PAPER] p. 6 |
| **A13** | **Encoding variants drawn as simultaneous rather than alternative.** Fig. 1 shows "BLOSUM Encoding" and "Label Encoding → Embedding Layer" side-by-side inside the protein encoding box, implying both are used; Sup Table 3 establishes they are mutually exclusive alternatives selecting variant A vs B/C. | [FIGURE] Fig. 1 p. 5 vs [TABLE] Sup Table 3 |
| **A14** | **Dataset name spelled inconsistently.** "PDBBind" (p. 3 onwards, most occurrences) vs "PDBbind" (p. 3, Sup Table 1, references). Cosmetic, but relevant when matching artefact names. | [PAPER] pp. 3–14 |
| **A15** | **Length caps contradict the reported distributions without comment.** Sup Fig. 1 shows SMILES up to ≈ 900 characters and proteins up to ≈ 7000 residues, against declared caps of 200 and 2000. The paper mentions padding but never truncation. | [PAPER] p. 4, p. 6 vs [SUPPLEMENT] Sup Fig. 1(b),(c) |
| **A16** | **Sup Table 2 typographical error.** "DCGAN number of filters (Generator) — `128.64,1`" uses a period where a comma is required. The main text (p. 6) reads "128, 64, and 1", which resolves the intent but the supplement as printed is malformed. | [TABLE] Sup Table 2 vs [PAPER] p. 6 |

---

## 16. Claims That Must Be Checked Against Code Later

The following are paper statements, recorded verbatim or near-verbatim, that Stage 3 (Paper ↔ Code Traceability) must test. **No assertion is made here about whether the code matches.**

### Architecture
1. The generator comprises **exactly three** Conv1DTranspose layers with filters **128, 64, 1**, kernel size **3**, activations **ReLU, ReLU, tanh**. [PAPER] p. 4, p. 6
2. The discriminator comprises **exactly five** Conv1D layers with filters **4, 8, 16, 32, 64**, kernel size **3**, **all ReLU**, followed by Flatten → Dense(1) with **tanh**. [PAPER] p. 4
3. **Batch Normalization is absent from both generator and discriminator.** [PAPER] p. 4
4. The CNN blocks contain **three CNN layers and one max-pooling layer** with filters **128, 256, 384**, filter length **8** (protein) / **4** (drug). [PAPER] p. 4, p. 6
5. Merging uses an **Add** layer, not concatenation. [PAPER] p. 4
6. The fully-connected block has widths **1024, 512, 512** (per text/Sup Table 2) — or **1024, 1024, 512** (per Fig. 1). Both readings must be tested. [PAPER] p. 6; [TABLE] Sup Table 2; [FIGURE] Fig. 1
7. Dropout is **0.25**. [PAPER] p. 6
8. **Two** DCGANs exist (drug and protein); variant C uses the drug DCGAN only. [FIGURE] Fig. 1; [TABLE] Sup Table 3

### GAN mechanism and transfer
9. The DCGAN is genuinely trained **adversarially** — generator and discriminator "trained together using an adversarial process". [PAPER] p. 4
10. The DCGAN is trained on **unlabeled data from UniProt and ChEMBL**, i.e. a corpus distinct from the DTA training data. [PAPER] p. 4
11. The **learned DCGAN model is reused for feature extraction** in the DTA network. [PAPER] p. 4
12. Only the **discriminator** path feeds the downstream CNN blocks (per Fig. 1); the generator does not. [FIGURE] Fig. 1 p. 5 — [INFERRED], must be tested
13. Whether transferred weights are frozen or trainable — **unspecified in the paper**, so the code establishes it de novo. [OPEN]
14. Whether pretraining happens **once** or is repeated — **unspecified**, so the code establishes it de novo. [OPEN]

### Representation
15. Drugs: **label encoding → padding to 200 → embedding layer**, identical across variants A, B, C. [PAPER] p. 4; [TABLE] Sup Table 3
16. Proteins: variant A **label encoding + embedding**; variants B and C **BLOSUM, 25-dimensional per residue**, padded to 2000. [PAPER] p. 4; [TABLE] Sup Table 3
17. The BLOSUM feature vector is **25-dimensional**. [PAPER] p. 4

### Data and split
18. BindingDB: **9864 drugs, 1088 proteins, 42203 interactions**, Kd only. [PAPER] p. 3; [TABLE] Sup Table 1
19. PDBbind: **4231 drugs, 1606 proteins, 5014 interactions**, refined v2020, Ki and Kd. [PAPER] p. 3; [TABLE] Sup Table 1
20. PDBbind redundancy removal: "excluded redundancies arising from multiple sequences for the same drugs". [PAPER] p. 3
21. **Fivefold** cross-validation with "five nearly equal-sized training and validation sets". [PAPER] p. 6
22. Hyperparameter tuning is performed **on those validation sets**. [PAPER] p. 6
23. Cold-start: drugs excluded from training by **logP** (Open Babel logP and XLOGP3), excluded drugs become the **test data**, PDBbind only. [PAPER] p. 9
24. Adversarial controls shuffle **binding affinity values** in three configurations, on **DCGAN-DTA(C)**, PDBbind. [PAPER] p. 10; [FIGURE] Fig. 7

### Training and evaluation
25. **300 epochs, batch size 256, Adam, learning rate 0.001.** [PAPER] p. 6; [TABLE] Sup Table 2
26. **Early stopping** is used. [PAPER] p. 6
27. **Four metrics**: CI (Eqs. 1–2), MSE (Eq. 3), AUPR, `r²_m` (Eq. 4). [PAPER] p. 6
28. AUPR uses a binarisation **threshold of 7** on both datasets. [PAPER] p. 6
29. `r²_m` is computed as `r² × (1 − √(r² − r₀²))`. [EQUATION] Eq. (4), p. 6
30. **t-tests on CI** at the 95% level between first- and second-best methods. [PAPER] pp. 8, 9
31. Max lengths **2000** (protein) and **200** (SMILES) for **both** datasets. [PAPER] p. 6; [TABLE] Sup Table 2
32. The training loss function — **unspecified in the paper**, so the code establishes it de novo. [OPEN]

---

## 17. Paper-Level Reproduction Checklist

Every item must be resolved before a reproduction attempt. `[ ]` = must be verified in code or by execution; `[?]` = paper gives nothing, so code/authors are the only possible source.

**Data acquisition**
- [ ] BindingDB Kd subset obtained via TDC [40] yielding exactly 9864 × 1088 with 42203 interactions
- [ ] PDBbind refined v2020 yielding exactly 4231 × 1606 with 5014 interactions
- [?] Exact "recommended guidelines for data harmonization" applied to BindingDB
- [?] Exact log-transform formula converting Kd/Ki to pKd/pKi
- [?] Missing-value handling in the affinity matrix
- [?] Cause and handling of the BindingDB pKd ≈ 5 spike (≈ half of all pairs)
- [ ] PDBbind redundancy removal ("multiple sequences for the same drugs")
- [?] UniProt corpus version, size and filtering
- [?] ChEMBL corpus version, size and filtering
- [?] Whether the pretraining corpora overlap the evaluation sets

**Representation**
- [?] SMILES character vocabulary and index map
- [?] Canonical vs isomeric SMILES
- [?] Amino-acid alphabet (25 symbols) and index map
- [?] Which BLOSUM matrix produces 25-dimensional vectors
- [?] Embedding dimension (drug and protein)
- [ ] Padding to 200 / 2000, pad value and side
- [?] Truncation rule for SMILES > 200 and proteins > 2000

**Architecture**
- [ ] Generator: 3 × Conv1DTranspose, filters 128/64/1, kernel 3, ReLU/ReLU/tanh, no BatchNorm
- [?] Generator strides, padding, noise dimension, noise distribution
- [ ] Discriminator: 5 × Conv1D, filters 4/8/16/32/64, kernel 3, all ReLU, Flatten → Dense(1) tanh
- [?] Discriminator strides and padding
- [ ] CNN blocks: 3 conv layers + 1 max-pool, filters 128/256/384, kernel 8 (protein) / 4 (drug)
- [?] CNN-block activation, stride, padding
- [ ] Add merging layer (not concatenation)
- [ ] **FC widths: 1024/512/512 or 1024/1024/512** — resolve ambiguity A1
- [?] FC activations and dropout placement
- [?] Output-layer activation
- [ ] Two DCGANs for variants A and B; one (drug only) for variant C

**GAN mechanism**
- [?] Adversarial loss function (generator and discriminator)
- [?] Real/fake label convention, reconciled with the tanh output head
- [?] Discriminator-to-generator update ratio
- [?] GAN training iterations / epochs / batch size
- [?] GAN optimizer and learning rate, if distinct from the DTA stage
- [?] Which discriminator layer(s) are transferred into the DTA model
- [?] Whether transferred weights are frozen or trainable
- [?] Whether pretraining runs once or repeatedly (per fold / per grid point / per variant)

**Objective**
- [?] DTA prediction loss function
- [?] Whether the DTA loss and the GAN loss are ever optimised jointly
- [?] Any loss-balancing coefficient

**Training**
- [ ] 300 epochs, batch size 256, Adam, lr 0.001 — and which stage(s) they govern (A10)
- [?] Adam β₁, β₂, ε
- [?] Weight initialization
- [?] Early-stopping monitored metric, mode, patience, restore-best
- [?] Model-selection / checkpoint rule
- [?] Hyperparameter search strategy and search space
- [?] Random seed and determinism controls
- [?] Number of repetitions

**Splitting**
- [ ] Fivefold CV construction
- [?] Definition of the warm-start split (pair-random? drug/target-disjoint?)
- [?] Whether a held-out test set exists in the warm-start setting, and when it is scored (A4)
- [?] Whether folds are generated at runtime or loaded from fixed files
- [?] logP threshold, direction and resulting cold-start split sizes
- [ ] Adversarial-control shuffling procedure (labels only) and which variant is used (A6)

**Evaluation**
- [ ] CI per Eqs. (1)–(2)
- [ ] MSE per Eq. (3)
- [ ] `r²_m` per Eq. (4)
- [ ] AUPR with threshold 7
- [?] AUPR positive-class definition (≥7 vs >7; which side is positive)
- [?] Fold aggregation rule (mean of folds / best fold / final epoch / best epoch)
- [?] Whether reported numbers are validation or test scores
- [?] Composition of the sample underlying the Sup Fig. 2–4 box plots and t-tests (A5)
- [?] t-test variant and multiple-comparison handling

**Targets**
- [ ] Reproduce Fig. 2 (BindingDB warm-start, 10 methods × 4 metrics)
- [ ] Reproduce Fig. 3 (PDBbind warm-start, 10 methods × 4 metrics)
- [ ] Reproduce Figs. 5–6 (PDBbind cold-start, 10 methods × 3 metrics, two logP variants)
- [ ] Reproduce Fig. 7 (adversarial controls)
- [ ] Reproduce Fig. 8 (lightweight baselines + Add-vs-concatenation ablation)
- [?] Resolve DCGAN-DTA(A) PDBbind MSE = 2.343 vs 2.338 (A3)
- [?] Baseline provenance: re-run under this protocol, or transcribed from source papers (A12)

---

## 18. Executive Summary

### A. What the paper explicitly specifies well

- **Bibliographic identity and availability.** DOI, dates, repository URL and web-server URL are all given. [PAPER] p. 1, p. 15
- **Dataset identity and size.** Both datasets are named with version/subset ("Kd version", "refined v2020"), cited, and given exact drug/protein/interaction counts. [PAPER] p. 3; [TABLE] Sup Table 1
- **GAN layer geometry.** The generator (3 Conv1DTranspose, 128/64/1, kernel 3) and discriminator (5 Conv1D, 4/8/16/32/64, kernel 3) are specified layer-by-layer, with activations named per layer and two explicit deviations from the original DCGAN documented and justified (no BatchNorm; ReLU instead of Leaky ReLU). This is unusually precise for the layer *shapes*. [PAPER] p. 4, p. 6
- **Downstream CNN and FC geometry.** Filter counts (128/256/384), filter lengths (8 protein / 4 drug), dropout (0.25) and input caps (2000 / 200) are all stated in two places. [PAPER] p. 6; [TABLE] Sup Table 2
- **Core training hyperparameters.** Optimizer, learning rate, batch size and epoch count are stated in both the text and Sup Table 2. [PAPER] p. 6; [TABLE] Sup Table 2
- **Three evaluation metrics with formulas.** CI, MSE and `r²_m` are given as numbered equations with symbols defined. [EQUATION] Eqs. (1)–(4), p. 6
- **Hardware and software environment.** OS, CPU, GPU and framework stack are named. [PAPER] p. 6
- **Variant definitions.** Sup Table 3 cleanly separates A, B and C, and establishes that the drug pipeline is shared by all three. [TABLE] Sup Table 3
- **Validation methodology beyond headline numbers.** Adversarial control experiments with straw models, statistical testing, and a merging-layer ablation are all reported — more validation scaffolding than is typical. [PAPER] pp. 10–12

### B. What is only partially specified

- **Affinity preprocessing.** The transform is named ("logarithmic-transformed … pKd") but never written; source units are never given; BindingDB's "recommended guidelines for data harmonization" are neither named nor cited. [PAPER] p. 3
- **Input encodings.** The *schemes* are named (label encoding, BLOSUM) but no vocabulary, alphabet, index map, embedding dimension, or BLOSUM matrix identity is supplied. [PAPER] p. 4
- **Length handling.** Caps of 200/2000 are explicit, but Sup Fig. 1 shows both datasets substantially exceed them and the paper never mentions truncation. [PAPER] p. 6 vs [SUPPLEMENT] Sup Fig. 1
- **Architecture completeness.** Filter counts and kernel sizes are given; strides, padding, CNN/FC activations, tensor shapes and dropout placement are not. The one architecture figure conflicts with the text on FC widths and omits the max-pooling layer. [FIGURE] Fig. 1 vs [PAPER] p. 4, p. 6
- **Cold-start protocol.** The mechanism (exclude drugs by logP) is clear; the threshold, direction and resulting split sizes are absent. [PAPER] p. 9
- **Pretraining corpora.** UniProt and ChEMBL are named; versions, sizes, filtering and any overlap analysis with the evaluation data are absent. [PAPER] p. 4
- **Evaluation reporting.** Formulas exist for three of four metrics, but fold aggregation, mean-vs-best-fold reporting, and the AUPR positive-class definition are all missing, and no variance is reported in the main figures. [PAPER] p. 6; [FIGURE] Figs. 2–8
- **Hyperparameter search.** Stated to have occurred; strategy and search space never given. Sup Table 2 is a final configuration, not a grid. [PAPER] p. 6; [TABLE] Sup Table 2

### C. What remains OPEN

- **The entire training objective.** All four numbered equations are evaluation metrics. There is **no** prediction loss, **no** adversarial loss, **no** generator or discriminator loss, **no** regularization term and **no** combined objective anywhere in the paper or its seven supplements. `L_total` cannot be reconstructed at all. [OPEN]
- **The GAN→DTA transfer mechanism** — the mechanism the method is named for. Which layers transfer, whether they are frozen, and whether pretraining runs once or repeatedly are all unspecified; the figure implies the discriminator path but the text says only "the learned models". [OPEN]
- **The GAN training schedule.** No iteration count, no batch size, no D:G update ratio, no noise dimension, no noise distribution, no loss name, no real/fake label convention. [OPEN]
- **The warm-start split semantics and the existence/timing of a test set.** The only split description mentions training and validation sets used for tuning; how the reported warm-start numbers are produced is undetermined. [OPEN]
- **Random seed and determinism.** Not mentioned anywhere. [OPEN]
- **Fold aggregation and reporting convention.** Whether the bar-chart values are fold means, best folds, best epochs or final epochs is undetermined for every metric. [OPEN]
- **The sample underlying the box plots and t-tests.** [OPEN]
- **Baseline provenance** — re-run versus transcribed. [OPEN]
- **Target-wise / protein cold-start** — never attempted. [OPEN]
- **Runtime, parameter counts, and library versions.** [OPEN]

### D. The 5–10 most reproduction-critical paper facts to verify next

Ranked by how much a mismatch would change the reproduction:

1. **The training objective.** The paper specifies none. Whatever loss the implementation uses is, by definition, undocumented — this is the single largest gap and must be established from code. *(Ref. §7, §16 #32)*
2. **The GAN→DTA transfer.** Which discriminator layer(s) enter the DTA network, and whether they are frozen. The method's central claim rests on this and the paper states neither. *(Ref. §6.5, §16 #11–14)*
3. **Whether the DCGAN is trained adversarially at all, once, and on UniProt/ChEMBL.** The paper asserts pretraining on external unlabeled corpora; the number of pretraining runs and the corpus identity materially change both cost and results. *(Ref. §6.3, §16 #9–10)*
4. **The split protocol and the existence of a warm-start test set.** "Five nearly equal-sized training and validation sets" used for tuning, with no test set described, leaves the provenance of every number in Figs. 2–3 undetermined. *(Ref. §4, A4, §16 #21–22)*
5. **Fold aggregation and the reported-value convention.** Mean-of-folds versus best-fold versus best-epoch changes every target number in §11. *(Ref. §10, A5)*
6. **FC layer widths — 1024/512/512 or 1024/1024/512.** A direct three-way source conflict on a concrete, checkable number. *(Ref. A1, §16 #6)*
7. **Affinity preprocessing: the exact log-transform and the BindingDB pKd ≈ 5 spike.** Half of the BindingDB dataset sits at one value; how it arose and whether it was handled determines what MSE and CI even mean on that dataset. *(Ref. §3.2, §3.1)*
8. **The AUPR binarisation at threshold 7,** including the positive-class definition, applied identically to two datasets with different affinity ranges. *(Ref. §10, A11)*
9. **The BLOSUM encoding: which matrix yields a 25-dimensional vector,** and how a 1-filter generator output reconciles with a 25-channel real-data stream in variant B. *(Ref. A7, A8, §16 #17)*
10. **The cold-start logP threshold and direction,** without which the cold-start experiments in Figs. 5–6 cannot be constructed at all. *(Ref. §4, §16 #23)*

---

*End of Stage 2 forensic specification — DCGAN-DTA. No source code was inspected and no repository file was modified in producing this document.*

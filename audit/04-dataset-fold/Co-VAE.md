# Stage 4 — Dataset & Fold Audit: Co-VAE

**Paper-side specification:** `audit/02-paper-forensic/Co-VAE.md` (Stage 2)
**Code-side tracing:** `audit/03-paper-code-traceability/Co-VAE.md` (Stage 3)
**Data audited:** `Co-VAE/CoVAE/data/davis/`, `Co-VAE/CoVAE/data/kiba/`
**Mode:** read-only. Every file opened `'r'`/`'rb'`. No repository file modified, no dependency installed, no model executed. All statistics computed by scripts held outside the repository.

**Evidence tags used throughout:**

| Tag | Meaning |
|---|---|
| `[PAPER]` | what the publication states |
| `[CODE]` | what the implementation does |
| `[DATA]` | what the local files measurably contain |
| `[INFERENCE]` | a conclusion drawn from comparing the above |
| `[OPEN]` | undetermined at this stage |

---

## 1. Scope

This stage answers eight questions for the two Co-VAE datasets: what the code actually consumes, what the matrices contain, what the fold files index, whether those folds are valid at runtime, whether leakage exists, what the affinity distributions are, whether a controlled DCGAN-DTA/Co-VAE comparison is supportable, and which dataset suits first-reproduction development.

It does **not** repeat Stage 2 or Stage 3. Where it confirms an earlier finding it says so and adds the measurement; where it contradicts or refines one it says so explicitly.

**Headline result of this stage for Co-VAE:** the bundled fold files are dead code, the runtime split is generated randomly at execution time, and the paper's flagship *new-target* protocol is compromised at the sequence level by a property of the Davis data itself — not by the split logic.

---

## 2. Dataset Inventory

### 2.1 What the code actually opens

`datahelper.py:140-156` `DataSet.parse_data` is the **only** data-reading path in the repository. A repository-wide grep for `open(`, `read_csv`, `np.load`, `loadtxt`, `pickle.load` returns exactly four data reads plus the log writer. [CODE] [VERIFIED]

| File | Davis | KIBA | Consumed? |
|---|:--:|:--:|---|
| `ligands_iso.txt` | ✔ | ✔ | **YES** — `json.load(…, object_pairs_hook=OrderedDict)` → `orderdict_list` → **values only** |
| `proteins.txt` | ✔ | ✔ | **YES** — same path |
| `drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt` | ✔ | — | **YES** (Davis affinity source) |
| `kiba_binding_affinity_v2.txt` | — | ✔ | **YES** (KIBA affinity source) |
| `ligands_can.txt` | — | ✔ | **NO** — canonical SMILES never read |
| `Y` | ✔ | — | **NO** — the pickle is never opened by Co-VAE |
| `folds/train_fold_setting1.txt` | ✔ | ✔ | **NO** — see §5 |
| `folds/test_fold_setting1.txt` | ✔ | ✔ | **NO** — see §5 |
| `drug-drug_similarities_2D.txt` | ✔ | — | **NO** |
| `target-target_similarities_WS.txt` | ✔ | — | **NO** |
| `kiba_target_sim.txt` | — | ✔ | **NO** |
| `…v1 (2).txt` | ✔ | — | **NO** — byte-equal duplicate of the consumed file [DATA] |

**[INFERENCE]** Seven of the twelve shipped data files per dataset are inert. Four of them (`Y`, the three similarity matrices) are the files a reader would most likely assume are central. Any reproduction that reasons from the directory listing rather than from `parse_data` will mis-model this pipeline.

### 2.2 Davis inventory [DATA]

| Property | Value |
|---|---|
| Drugs (`ligands_iso.txt`) | 68 entries, 68 unique keys, **68 unique SMILES** |
| Targets (`proteins.txt`) | 442 entries, 442 unique keys, **379 unique sequences** ⚠ |
| Drug key format | numeric PubChem CIDs as strings (`'11314340'`), length 4–8 |
| Target key format | gene-symbol + mutation (`'AAK1'`, `'ABL1(E255K)'`), length 3–8 |
| SMILES length | min 39, max 92, mean 62.8, median 60 |
| Protein length | min 244, max 2549, mean 788.9, median 707 |
| Affinity matrix | 68 × 442 = **30,056 cells, 0 missing — fully dense** |
| Matrix orientation | rows = drugs, cols = targets; shape matches entity counts exactly |
| Similarity files | drug 68×68 (range 0.225–1.0, symmetric); target 442×442 (range 80–16832, symmetric, **unnormalised**) |

### 2.3 KIBA inventory [DATA]

| Property | As shipped | After the code's runtime filter |
|---|---|---|
| Drugs | 2,111 entries / 2,111 unique SMILES | **1,954** |
| Targets | 229 entries / 229 unique sequences | **217** |
| Drug key format | ChEMBL IDs (`'CHEMBL1087421'`) | — |
| Target key format | UniProt accessions (`'O00141'`), all length 6 | — |
| SMILES length | min 20, max 590, mean 59.9, median 55 | — |
| Protein length | min 215, max 4128, mean 729.0, median 620 | — |
| Matrix | 2,111 × 229 = 483,419 cells; **118,254 observed (24.46% dense)** | 1,954 × 217 = 424,018 cells; **109,296 observed (25.78% dense)** |
| Entity collapse | none — all SMILES and all sequences unique | none |

`ligands_can.txt` holds the same 2,111 keys as `ligands_iso.txt` but is never read. [DATA]

### 2.4 The KIBA runtime filter, measured

`datahelper.py:151-156` applies, **for KIBA only**, `get_removelist(ligands, 90)` and `get_removelist(proteins, 1365)` — removing every entity whose string length is `>=` the cap. [CODE]

| Step | Count |
|---|---|
| Observed pairs before filter | 118,254 |
| Drugs removed (len ≥ 90) | 157 → costs 2,472 pairs |
| Targets removed (len ≥ 1365) | 12 → costs 6,540 pairs |
| **Observed pairs after filter** | **109,296** (−8,958, −7.58%) |

This **confirms Stage 3's C1 exactly**, and adds the pair-level cost, which Stage 3 did not quantify.

Two further observations this stage adds:

- **The code's own comment contradicts the code.** `datahelper.py:90` reads `# Davis SMILES:85 protein:1200 KIBA SMILES:100 protein:1000`, but the call sites use **90** and **1365**. Applying the commented caps instead would remove 103 drugs and 44 targets — a different matrix again. [CODE] [DATA]
- **Davis receives no length filter at all**, yet 61 of its 442 proteins exceed the 1200-length cap the comment names (max 2549), and 3 of 68 SMILES exceed 85. Those are handled by silent head-truncation in `label_sequence`/`label_smiles`, not by removal. [CODE] [DATA]

---

## 3. Affinity Representation

The distinction the task asks for — raw file versus runtime — is material for Davis and null for KIBA.

### 3.1 Davis

| | RAW FILE FORMAT | RUNTIME REPRESENTATION |
|---|---|---|
| Quantity | **Kd in nanomolar** | **pKd** |
| Transform | none | `-log10(Kd / 1e9)` at `datahelper.py:148` [CODE] |
| Direction | lower = stronger binding | **higher = stronger binding** |
| Range | 0.016 – 10,000 | **5.0 – 10.7959** |
| Mean / median / std | 7415.88 / 10000.0 / 4073.13 | **5.4515 / 5.0000 / 0.8947** |
| Quantiles (0/25/50/75/99/100) | 0.016 / 3000 / 10000 / 10000 / 10000 / 10000 | 5.0 / 5.0 / 5.0 / 5.5229 / 8.8539 / 10.7959 |
| Distinct values | 456 | 456 |
| Missing | **none — 0 of 30,056** | none |

**The censoring is severe and now exactly measured.** 20,931 of 30,056 cells (**69.64%**) hold precisely Kd = 10,000 → pKd = 5.0. [DATA]

This **refines Stage 2's estimate**, which read "≈21,500 of 30,056" off Figure 5. The true figure is 20,931 (69.64%), not ≈71.5%. Stage 2's qualitative conclusion stands; the number should be cited from here.

**[INFERENCE]** Because Davis is fully dense and 69.64% single-valued, the median equals the floor (pKd 5.0). Constant-predictor baselines on the full matrix: **MSE 0.8005** predicting the mean (5.4515), **MSE 1.0044 / MAE 0.4515** predicting the median (5.0). [DATA] Any reported MSE must be read against the 0.8005 figure, and CI is computed over a label set in which the majority of pairs are mutually tied — 69.64% of pairs share one value, so the concordant/discordant denominator is dominated by ties.

### 3.2 KIBA

| | RAW FILE FORMAT | RUNTIME REPRESENTATION |
|---|---|---|
| Quantity | **KIBA score** (already a composite integrated score) | **identical — no transform applied** [CODE] |
| Direction | higher = stronger | higher = stronger |
| Range (post-filter) | 0.0 – 17.2002 | 0.0 – 17.2002 |
| Mean / median / std | — | **11.7213 / 11.5000 / 0.8271** |
| Quantiles (0/1/25/50/75/99/100) | — | 0.0 / 10.1 / 11.2 / 11.5 / 11.9 / 14.6003 / 17.2002 |
| Distinct values | 2,884 (pre-filter) | 2,688 |
| Missing | 365,165 of 483,419 (75.54%) | **314,722 of 424,018 (74.22%)** |
| Zero-valued cells | 1 | **1** |

`else:` at `datahelper.py:149` means **no log transform is ever applied to KIBA** — the branch is Davis-only. [CODE] [VERIFIED]

**The paper's stated KIBA filter is absent, and this stage measures the gap.** [PAPER] §3.1 says "We removed the drug-target pairs which have known affinities less than 10", and Figure 5's histogram starts at ≈9.5. The runtime matrix retains **434 observed values below 10.0**, and its minimum is 0.0. [DATA] This confirms Stage 3's second finding and supplies the count.

### 3.3 Missing-value semantics

**Missing cells are `NaN`, and they are excluded structurally, not weighted.** `run_experiments.py:504` computes `label_row_inds, label_col_inds = np.where(np.isnan(Y) == False)`; every downstream index addresses that observed-pair array only. [CODE] [VERIFIED]

**[INFERENCE]** Missing entries therefore participate in **neither** the affinity loss **nor** the VAE reconstruction/KL terms — the encoders only ever see drugs and targets drawn from observed pairs. No implicit-negative assumption exists anywhere in this pipeline. The audit principle "missing ≠ negative" holds here: verified, not assumed.

For Davis the point is moot — nothing is missing. For KIBA, 74.22% of the matrix is absent and silently skipped.

---

## 4. Identifier Audit

| Check | Davis | KIBA |
|---|---|---|
| Duplicate drug keys | none | none |
| Duplicate target keys | none | none |
| Drug/target counts match matrix dims | ✔ 68×442 | ✔ 2,111×229 (pre-filter) |
| Row order = `ligands_iso.txt` insertion order | ✔ via `OrderedDict` + `orderdict_list` [CODE] | ✔ |
| Col order = `proteins.txt` insertion order | ✔ | ✔ |
| **Duplicate drug SMILES** | **none** | none |
| **Duplicate protein sequences** | **18 groups covering 81 of 442 ids (18.3%)** ⚠ | none |

### 4.1 Keys are discarded before the model sees anything

`orderdict_list` (`datahelper.py:46-50`) returns `[dict[k] for k in dict.keys()]` — **values only**. Identifiers exist solely to fix row/column order; they never reach the model. [CODE] [VERIFIED] Everything downstream is positional.

### 4.2 Davis target identity collapse — a new finding

442 target identifiers resolve to only **379 distinct sequences**. Eighteen collapse groups cover 81 identifiers: [DATA]

| Group size | Members (truncated) |
|---:|---|
| 15 | `ABL1(E255K)`, `ABL1(F317I)`, `ABL1(F317I)p`, `ABL1(F317L)`, `ABL1(F317L)p`, `ABL1(H396P)`, … |
| 12 | `EGFR`, `EGFR(E746A750del)`, `EGFR(G719C)`, `EGFR(G719S)`, `EGFR(L747E749del)`, … |
| 10 | `PIK3CA`, `PIK3CA(C420R)`, `PIK3CA(E542K)`, `PIK3CA(E545A)`, `PIK3CA(E545K)`, … |

These are Davis's kinase mutants and phosphorylated forms. The identifiers encode the mutation; **the stored sequences do not differ**. Because the model consumes only the sequence (§4.1), these 81 identifiers are, to Co-VAE, at most 18 distinct inputs carrying different affinity labels.

**[INFERENCE]** Two consequences, both consequential for this thesis:

1. The Davis protein set has an effective cardinality of 379, not 442, for representation-learning purposes.
2. Any target-wise split that separates members of a collapse group produces a nominally "unseen" test target whose exact sequence the model already trained on — see §7.2. This is a property of the **data**, and it will reappear in any reimplementation that reads these files, including the eventual hybrid method.

### 4.3 Order-stability caveat

`list_remove` (`datahelper.py:97-104`) rebuilds the entity list from `set(range(n)) - set(removelist)` — a **set difference**, whose iteration order is not contractually defined. `df_remove` uses `DataFrame.drop`, which *is* order-preserving. I verified the set path yields ascending order for these inputs (CPython hashes small ints to themselves). [DATA] [INFERENCE] Row/column alignment therefore holds **in this environment**; it rests on a CPython implementation detail rather than a guarantee, and is worth pinning if the pipeline is ever ported.

---

## 5. Fold Audit

### 5.1 The decisive finding: the fold files are never opened

`DataSet` has **no `read_sets` method**, and no path in the repository opens `folds/`. The one reference — `nfold_setting_sample:168-172` — is **commented out**:

```python
test_set = nfolds[5]
outer_train_sets = nfolds[0:5]
# test_set, outer_train_sets=dataset.read_sets(FLAGS)
# if FLAGS.problem_type==1:
#     test_set, outer_train_sets = dataset.read_sets(FLAGS)
```
[CODE] `run_experiments.py:166-172` [VERIFIED]

`nfolds` arrives from `experiment()`, which generates it at runtime (§6.1).

**[INFERENCE]** For Co-VAE the shipped fold files are **inert artifacts**. They are nonetheless audited below, because (a) they are the only reproducible split definition in the repository, (b) Stage 3 flagged them as incompatible and that claim needed measuring, and (c) they are a candidate for the controlled protocol in §11.

### 5.2 Fold file contents [DATA]

| Property | Davis | KIBA |
|---|---|---|
| `train_fold_setting1.txt` | 5 folds, sizes 5010/5009/5009/5009/5009 | 5 folds, sizes 19709 ×5 |
| `test_fold_setting1.txt` | 5,010 indices | 19,709 indices |
| Total indices | 30,056 | 118,254 |
| Unique | 30,056 — **0 duplicates** | 118,254 — **0 duplicates** |
| Index range | [0, 30055] | [0, 118253] |
| Zero-based | ✔ | ✔ |
| Exactly covers 0..N−1 | ✔ | ✔ |
| Train/test overlap | **0** | **0** |
| Inter-fold overlap | **0** — mutually exclusive | **0** — mutually exclusive |
| Indexing scheme | positions into the observed-pair array from `np.where(~isnan(Y))`, row-major | same |

Both files are internally clean: a complete, duplicate-free, mutually exclusive 6-way partition (5 train folds + 1 test fold).

### 5.3 Validity against the matrix the code actually builds

| | Davis | KIBA |
|---|---|---|
| Runtime observed pairs | **30,056** | **109,296** |
| Max fold index | 30,055 | **118,253** |
| `max index < n_observed`? | ✔ **valid** | ✘ **INVALID** |
| Fold cardinality == observed? | ✔ 30,056 = 30,056 | ✘ 118,254 ≠ 109,296 |

**Davis fold files are valid** for the runtime matrix — exactly consistent, because Davis is dense and unfiltered.

**KIBA fold files are structurally invalid** for the runtime matrix. They enumerate 118,254 pairs — the **pre-filter** count — while the code builds 109,296. Indices from 109,296 to 118,253 address positions that do not exist; 8,958 indices are out of range. Any attempt to use them would raise, or silently mis-address every pair after the first removed entity. [DATA]

This **confirms Stage 3's claim** that the bundled folds are incompatible with the runtime KIBA matrix, and quantifies the mismatch precisely: the fold files were generated against the **unfiltered 2,111 × 229** matrix, i.e. against the paper's stated dataset, before the undocumented length filter was introduced.

**[INFERENCE]** The KIBA fold files are strong evidence that the length filter was added *after* the split definitions were produced — the two artifacts come from different versions of the pipeline.

---

## 6. Fold Semantics

### 6.1 What the runtime actually constructs

`experiment()` (`run_experiments.py:508-517`) [CODE] [VERIFIED]:

```python
for i in range(1):                                    # ONE repetition
    random.seed(i + 1000)                             # seed 1000; numpy/torch unseeded
    if FLAGS.problem_type == 1: nfolds = get_random_folds(len(label_row_inds), 6)
    if FLAGS.problem_type == 2: nfolds = get_drugwise_folds(..., drugcount, 6)
    if FLAGS.problem_type == 3: nfolds = get_targetwise_folds(..., targetcount, 6)
```

| `problem_type` | Construction | Classification |
|:--:|---|---|
| 1 | random partition of the **observed-pair index array** into 6 | **PAIR-WISE (warm)** |
| 2 | random partition of the **drug index set** into 6, expanded to pairs | **COLD-DRUG** (entity-wise) |
| 3 | random partition of the **target index set** into 6, expanded to pairs | **COLD-TARGET** (entity-wise) |

Fold 5 is the test fold; folds 0–4 become the CV pool. `problem_type` has **no default** in `arguments.py` — it must be supplied. [CODE]

### 6.2 Measured structural properties (Davis)

Simulated with a Python-3.13-safe sampler. The code's own `random.sample(set, k)` raises on Python ≥ 3.11 (Stage 3 blocker B1), so exact membership is **not** reproducible here; the structural properties below are construction-invariant and hold for any seed. [DATA] [INFERENCE]

| `problem_type` | Test pairs | Train pairs | Drug overlap | Target overlap |
|:--:|---:|---:|---:|---:|
| 1 — pair-wise | 5,009 | 25,047 | 68 / 68 (all) | 442 / 442 (all) |
| 2 — drug-wise | 4,862 | 25,194 | **0** | 442 (all) |
| 3 — target-wise | 4,964 | 25,092 | 68 (all) | **0** |

Entity-wise splitting works as intended at the **identifier** level for all three settings.

### 6.3 Paper vs fold files vs code

| Dataset | Paper claims | Fold files imply | Code consumes | Match? |
|---|---|---|---|---|
| Davis | 6 entity-wise folds, 5 train + 1 test, **10 independent repetitions**, hyperparameters via 5-fold CV **on training data only** [PAPER] §3.3 | a **pair-wise** 6-way partition of all 30,056 observed pairs, valid for the runtime matrix | **generates its own folds**; files ignored. 6 folds ✔, entity-wise ✔ (types 2/3), **1 repetition ✘**, selection on **test** fold ✘ | **PARTIAL** |
| KIBA | same, on a 2,111 × 229 matrix | a pair-wise 6-way partition of **118,254** pairs — i.e. of the **paper's** matrix | generates its own folds over **109,296** pairs | **NO** — three-way disagreement |

Three distinct split definitions exist for KIBA: the paper's (entity-wise, 10×), the fold files' (pair-wise, over the unfiltered matrix), and the code's (entity-wise, 1×, over the filtered matrix). **No two agree.** [INFERENCE]

---

## 7. Leakage Analysis

Classified per the skill's scale. "Overlap" is reported factually; it is only called leakage where it defeats the protocol's own stated intent.

### 7.1 Pair / index level

| Level | Davis | KIBA | Class |
|---|---|---|---|
| Pair-level (same drug–target pair in train and test) | 0 | 0 | **NONE** |
| Index overlap in fold files | 0 | 0 | **NONE** |
| Inter-fold overlap | 0 | 0 | **NONE** |

Both the shipped folds and the runtime construction are clean partitions. [DATA] [VERIFIED]

### 7.2 Entity level

Under `problem_type 1` all 68 drugs and all 442 targets appear in both train and test. **This is not leakage** — pair-wise warm-start prediction is a legitimate protocol, and the paper's Table 2 reports new-drug/new-target settings separately. Reported factually.

Under `problem_type 2` and `3`, identifier-level disjointness is exact (§6.2). **But:**

> **CONFIRMED — sequence-level cold-target violation, Davis, `problem_type 3`.**
> In simulation, **13 of 73 test targets (17.8%)** had a sequence byte-identical to a target in the training set. [DATA]

The cause is §4.2: 18 collapse groups spanning 81 identifiers. Whenever a split separates group members — which random entity partitioning does with high probability — the "unseen" test target is one the encoder has already fitted, under a different label. The split logic is correct; **the data makes identifier-disjointness insufficient**.

**[INFERENCE]** Davis `problem_type 3` does not measure what [PAPER] §2.1 claims it measures ("targets not seen during training") for roughly a sixth of its test targets. The paper reports new-target results on Davis as a headline claim. Severity is bounded — 17.8% of test targets, not all — but it is systematic, it recurs under every seed, and it inflates the new-target metric in the optimistic direction.

KIBA is **unaffected**: all 229 sequences and all 2,111 SMILES are distinct. Davis drug-wise (`problem_type 2`) is likewise unaffected — 0 of 11 test drugs shared a SMILES with training. [DATA]

| Leakage channel | Davis | KIBA | Class |
|---|---|---|---|
| Cold-target defeated by duplicate sequences | 13/73 test targets | none possible | **CONFIRMED** (Davis only) |
| Cold-drug defeated by duplicate SMILES | 0/11 | none possible | **NONE** |

### 7.3 Model-selection contamination

`nfold_setting_sample:176-195` builds `val_sets` and then passes **`test_sets`** to both `general_nfold_cv` (grid search) and `general_nfold_cv_test` (final evaluation). `val_sets` is never consumed. [CODE] [VERIFIED]

**CONFIRMED.** This stage adds nothing new — it is Stage 3's C2 — but it is restated because it interacts with §7.2: hyperparameters, early-stopping epoch and the saved checkpoint are all selected on the same fold whose targets may already be sequence-visible.

### 7.4 Feature-construction and vocabulary leakage

| Channel | Finding | Class |
|---|---|---|
| Vocabulary | `CHARISOSMISET` (64) / `CHARPROTSET` (25) are **hard-coded constants**, not fitted to data | **NONE** |
| Normalisation | none applied to affinities at any point | **NONE** |
| Embeddings | learned inside the model, from training batches only | **NONE** |
| Length filter | **fitted on the full dataset before splitting** | see below |
| Similarity matrices | present on disk, **never read** | **NONE** |

The KIBA length filter runs in `parse_data`, i.e. **before any split exists** [CODE]. It is a deterministic function of individual string lengths and uses no affinity or split information, so it cannot transport label information across the boundary. **Class: NONE**, but it is a global preprocessing decision applied outside cross-validation and must be documented as such in any reproduction.

**The three similarity matrices are never opened** (§2.1), so the standard "similarity computed over train+test" concern **does not arise** in this repository. Had they been used, the Davis target-target matrix would warrant scrutiny; as shipped, it is inert. **Class: NONE.** [VERIFIED]

---

## 8. Paper vs Code vs Data

| # | Question | `[PAPER]` | `[CODE]` | `[DATA]` | Verdict |
|---|---|---|---|---|---|
| 1 | KIBA matrix size | 2,111 × 229, ≈117,954 pairs | filters by length at runtime | **1,954 × 217, 109,296 pairs** | ✘ mismatch, quantified |
| 2 | KIBA affinity filter | "removed pairs with affinity < 10" | no such filter | **434 values < 10 survive; min 0.0** | ✘ not implemented |
| 3 | Davis transform | pKd | `-log10(Kd/1e9)` | range 5.0–10.7959 | ✔ match |
| 4 | KIBA transform | KIBA score used directly | no transform | range 0.0–17.2002 | ✔ match |
| 5 | Davis density | not stated | — | **100% dense, 0 missing** | ✔ consistent |
| 6 | Davis censoring | "very uneven", ≈21,500 at pKd 5 (Fig. 5) | not handled | **20,931 (69.64%)** | ≈ refines Stage 2 |
| 7 | Split granularity | entity-wise, 6 folds | 6 folds, types 2/3 entity-wise | fold files are **pair-wise** | ✘ files ≠ paper ≠ code |
| 8 | Repetitions | **10** independent splits | `for i in range(1)` | — | ✘ mismatch |
| 9 | Hyperparameter selection | 5-fold CV **on training data only** | `test_sets` passed to both passes | — | ✘ contaminated |
| 10 | Validation partition | depicted in Fig. 6 caption | `val_sets` built, discarded | — | ✘ dead code |
| 11 | Fold files | not mentioned | **never opened** | valid for Davis, **invalid for KIBA** | ✘ inert |
| 12 | New-target setting | targets unseen in training | identifier-disjoint ✔ | **13/73 sequence-identical** | ✘ defeated by data |
| 13 | AUC threshold | not specified for KIBA | `affinities > 7` for both datasets | **KIBA: 109,286 pos vs 10 neg** | ✘ degenerate |

### 8.1 The AUC threshold is unusable on KIBA — measured

`run_experiments.py:163` and `:394` binarise at `affinities > 7` for **both** datasets. [CODE] Measured against the runtime matrices: [DATA]

| Dataset | Positives (> 7) | Negatives (≤ 7) | Positive rate |
|---|---:|---:|---:|
| Davis (pKd) | 2,457 | 27,599 | 8.17% |
| **KIBA (score)** | **109,286** | **10** | **99.9909%** |

Ten negatives in the entire KIBA matrix. A test fold of ~18,000 pairs contains **1.6 negatives in expectation**, and entity-wise folds will frequently contain **zero** — at which point `sklearn.roc_auc_score` raises `ValueError: Only one class present`.

**[INFERENCE]** This elevates Stage 3's B8 from "possible" to **near-certain**, and explains it: threshold 7 is meaningful for pKd but meaningless for a KIBA score whose 1st percentile is 10.1. This is a hard runtime failure, not merely a metric-quality concern.

---

## 9. Quantitative Statistics

Consolidated reference table. All values measured from local files at this stage. [DATA]

| Statistic | Davis | KIBA (shipped) | KIBA (runtime) |
|---|---:|---:|---:|
| Drugs | 68 | 2,111 | **1,954** |
| Unique SMILES | 68 | 2,111 | 1,954 |
| Targets | 442 | 229 | **217** |
| **Unique sequences** | **379** ⚠ | 229 | 217 |
| Matrix cells | 30,056 | 483,419 | 424,018 |
| Observed | **30,056** | 118,254 | **109,296** |
| Missing | **0 (0.00%)** | 365,165 (75.54%) | 314,722 (74.22%) |
| Density | **100%** | 24.46% | 25.78% |
| Min | 5.0000 | 0.0000 | 0.0000 |
| Max | 10.7959 | 17.2002 | 17.2002 |
| Mean | 5.4515 | 11.7199 | 11.7213 |
| Median | 5.0000 | 11.5000 | 11.5000 |
| Std | 0.8947 | 0.8369 | 0.8271 |
| q1 / q25 / q75 / q99 | 5.0 / 5.0 / 5.5229 / 8.8539 | 10.1 / 11.2 / 11.9239 / 14.6003 | 10.1 / 11.2 / 11.9 / 14.6003 |
| Distinct values | 456 | 2,884 | 2,688 |
| Modal value (count) | **5.0 (20,931 = 69.64%)** | 11.2 (15,046) | 11.2 (14,609) |
| Zero cells | 0 | 1 | 1 |
| Negative cells | 0 | 0 | 0 |
| Positives at `> 7` | 2,457 (8.17%) | — | **109,286 (99.99%)** |
| Fold-file indices | 30,056 | 118,254 | — |
| Fold files valid at runtime | **✔** | — | **✘** |
| Constant-predictor MSE (at mean) | **0.8005** | — | 0.6841 |
| Constant-predictor MSE (at median) | 1.0044 | — | 0.7331 |
| Constant-predictor MAE (at median) | 0.4515 | — | 0.5673 |

---

## 10. Reproduction Implications

1. **Davis is reproducible as data; KIBA is not.** Davis's runtime matrix is exactly the published one (68 × 442, 30,056, pKd 5.0–10.796). KIBA's is not, and no flag recovers the published shape — the filter is unconditional for the non-Davis branch. Reproducing the paper's KIBA column requires *editing* `datahelper.py`, which is a deviation, not a reproduction. [INFERENCE]

2. **The published fold definitions for KIBA cannot be used with the published code.** They index a matrix the code does not build (§5.3). Either the filter goes or the folds do.

3. **Ten repetitions are required and one is implemented.** Reported standard deviations in the paper are across 10 splits; the code's are across 5 CV folds that share a single test set. These are not the same quantity and should never be compared directly. [INFERENCE]

4. **Davis new-target results carry a known upward bias** of unmeasured magnitude, from the 17.8% sequence-visible test targets (§7.2). Quantifying it requires a run; bounding it does not — a sequence-aware grouped split is the fix, and it is a *deviation from the paper*, which used identifier-level splitting.

5. **AUC on KIBA will crash or be meaningless** at the hard-coded threshold of 7 (§8.1). Any reproduction must either change the threshold (deviation) or drop the metric (deviation).

6. **Stage 3's execution blockers are unaffected by this stage.** B1 (`random.sample(set)`), B2 (`--lamda`), B3 (CUDA) remain. Nothing found here makes the code more runnable.

---

## 11. Development-Dataset Implications

**Davis is the better Co-VAE development vehicle**, on evidence:

| Criterion | Davis | KIBA |
|---|---|---|
| Matches published dataset | ✔ exactly | ✘ (1,954 × 217 vs 2,111 × 229) |
| Fold files usable | ✔ valid | ✘ out-of-range indices |
| Size | 30,056 pairs, 68 × 442 — fast | 109,296 pairs, 1,954 × 217 — ~3.6× heavier |
| Affinity semantics | pKd, shared with PDBbind/BindingDB | KIBA score, shared with nothing |
| AUC metric | works (8.17% positive) | degenerate / raises |
| Preprocessing burden | low — one log transform, no filter | high — undocumented filter, mismatched folds |
| Known data defect | **379/442 unique sequences** (§4.2) | none |

Davis wins on five of seven criteria. Its one serious defect — target identity collapse — is *documented and detectable*, affects only the target-wise setting, and can be controlled for by a sequence-grouped split. KIBA's defects are structural and require code edits to work around.

**Caveat against over-reading this:** Davis's 100% density and 69.64% censoring make it an unusually easy and unusually degenerate regression target. It is the right dataset for *establishing that the pipeline runs and behaves sanely*. It is a poor dataset for claiming a method generalises. Do not let development convenience become the thesis's evidential base.

---

## 12. Open Questions

| # | Question | Why unresolved |
|---|---|---|
| **O1** | Does any flag combination recover the published 2,111 × 229 KIBA matrix? | Read of `parse_data` says no — the filter is unconditional on the non-Davis branch — but this is a static reading; execution would settle it |
| **O2** | What is the numerical magnitude of the test-fold-selection effect (§7.3)? | Requires two runs of one fold; execution is out of scope here |
| **O3** | How much does the 17.8% sequence-visible cold-target leak inflate Davis new-target CI? | Requires a run with and without sequence-grouped splitting |
| **O4** | What produced `kiba_binding_affinity_v2.txt` and the `_v2` suffix's meaning? | No preprocessing code ships; provenance unauditable |
| **O5** | Why do the KIBA fold files index the unfiltered matrix? | [INFERENCE] they predate the filter; unverifiable without upstream history |
| **O6** | Does `roc_auc_score` actually raise on a KIBA fold, or do all folds happen to catch a negative? | Depends on the realised split; 10 negatives across 6 folds makes raising likely but not certain |
| **O7** | Is the 442→379 collapse present in the original Davis release, or introduced by this repository's preprocessing? | Would need the upstream Davis source; out of scope |
| **O8** | Does the unnormalised target-target similarity matrix (range 80–16832) indicate an abandoned code path? | File is inert; no evidence either way |

---

## 13. Final Evidence-Based Findings

1. **`[DATA]` The bundled fold files are never read.** `read_sets` does not exist in Co-VAE; the only call site is commented out. Folds are generated at runtime from `random.seed(1000)`. The repository ships a reproducible split definition and then ignores it.
2. **`[DATA]` The KIBA fold files are invalid for the runtime matrix** — 118,254 indices against 109,296 observed pairs; 8,958 out of range. They were built against the paper's unfiltered matrix.
3. **`[DATA]` The KIBA runtime matrix is 1,954 × 217 / 109,296 pairs**, losing 8,958 pairs (7.58%) to an undocumented length filter whose thresholds (90, 1365) contradict the code's own comment (100, 1000).
4. **`[DATA]` The paper's KIBA affinity filter is absent**: 434 observed values below 10 survive, minimum 0.0.
5. **`[DATA]` Davis is fully dense and heavily censored**: 30,056 cells, zero missing, **69.64% at exactly pKd 5.0**. A constant mean-predictor scores MSE 0.8005.
6. **`[DATA]` Davis has 442 target identifiers but 379 unique sequences.** 18 collapse groups cover 81 identifiers (18.3%) — kinase mutants sharing a wild-type sequence.
7. **`[INFERENCE]` The Davis new-target protocol is defeated for ~17.8% of test targets** by finding 6. Identifier-disjoint splitting is insufficient on this data; the paper's flagship setting measures something weaker than it claims.
8. **`[DATA]` The KIBA AUC threshold is degenerate**: 109,286 positives against **10** negatives at `> 7`. Entity-wise folds will often contain no negative class at all.
9. **`[DATA]` Missing entries are excluded structurally**, never treated as negatives, and participate in neither the affinity loss nor the VAE terms.
10. **`[VERIFIED]` No similarity matrix is read**, so similarity-based leakage does not arise. Seven of twelve shipped files per dataset are inert.
11. **`[DATA]` Pair-level and index-level leakage is zero** in every fold file and every runtime construction inspected.
12. **`[INFERENCE]` Three mutually incompatible KIBA split definitions exist** — the paper's, the fold files', and the code's. No two agree.

---

*End of Stage 4 — Co-VAE. Read-only: no repository source file, dataset, or fold file was modified; no dependency installed; no model executed. Earlier audit reports are untouched.*

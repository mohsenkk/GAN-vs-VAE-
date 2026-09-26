# Stage 4 — Dataset & Fold Audit: DCGAN-DTA

**Paper-side specification:** `audit/02-paper-forensic/DCGAN-DTA.md` (Stage 2)
**Code-side tracing:** `audit/03-paper-code-traceability/DCGAN-DTA.md` (Stage 3)
**Data audited:** `DCGAN-DTA/DCGAN-DTA/data/bindingdb/`, `DCGAN-DTA/DCGAN-DTA/data/pdb/`
**Mode:** read-only. Every file opened `'r'`/`'rb'`. No repository file modified, no dependency installed, no model executed. All statistics computed by scripts held outside the repository.

**Evidence tags:** `[PAPER]` (publication) · `[CODE]` (implementation) · `[DATA]` (measured from local files) · `[INFERENCE]` (conclusion from comparison) · `[OPEN]` (undetermined).

---

## 1. Scope

Same eight questions as the companion Co-VAE report, applied to BindingDB and PDBbind. This stage does not repeat Stage 2 or Stage 3; it measures.

**Headline results of this stage for DCGAN-DTA:** the fold files are real, used, internally clean, and — unlike Co-VAE's — valid; `test_fold_setting2/3` are **confirmed drug cold-start** splits, closing Stage 3's U6; and the provenance of the GAN pretraining corpora (Stage 3's U2) is **substantially resolved** by a direct match against a third dataset.

---

## 2. Dataset Inventory

### 2.1 What the code actually opens

`datahelper.py:124-134` (`read_sets`) and `:136-…` (`parse_data`) are the data-reading paths. [CODE] Unlike Co-VAE, **every shipped data file is consumed** except `auc.jar` and `ligands_can.txt` (absent here).

| File | Consumed? | Role |
|---|---|---|
| `Y` | **YES** — `pickle.load(…, encoding='latin1')` | affinity matrix |
| `ligands.txt` | **YES** → `XD` | evaluation drugs |
| `proteins.txt` | **YES** → `XT` | evaluation targets |
| `ligands_train.txt` | **YES** → `XD_t` | **GAN pretraining corpus** |
| `proteins_train.txt` | **YES** → `XT_t` | **GAN pretraining corpus** |
| `protein_feature_vec.json` | **YES** — read in `parse_data` | pro2vec table |
| `protein_feature_vecblsm.json` | **YES** — read in `general_nfold_cv:598` (models B/C only) | BLOSUM table |
| `folds/train_fold_setting{N}.txt` | **YES** — `read_sets`, keyed on `--problem_type` | CV pool |
| `folds/test_fold_setting{N}.txt` | **YES** — same | test fold |
| `auc.jar` | **NO** — dead code (Stage 3 §E11) | — |

**[INFERENCE]** The fold files here are live infrastructure, not inert artifacts. This is the single largest structural difference from Co-VAE and it dominates every comparison in §11.

### 2.2 BindingDB inventory [DATA]

| Property | Value |
|---|---|
| Drugs (`ligands.txt`) | 9,864 entries / 9,864 unique keys / **9,864 unique SMILES** |
| Targets (`proteins.txt`) | 1,088 entries / **1,088 unique sequences** |
| Drug key format | PubChem CID with a float suffix (`'123446.0'`) |
| Target key format | UniProt accession (`'P79400'`) |
| SMILES length | min 8, max 914, mean 77.7, median 56 — **690 (7.0%) exceed the 200 cap** |
| Protein length | min 86, max 3969, mean 621.5, median 486 — **16 (1.5%) exceed the 2000 cap** |
| Matrix | 9,864 × 1,088 = 10,732,032 cells; **42,203 observed (0.39% dense)** |
| Entity collapse | **none** — all SMILES and all sequences distinct |
| Drugs with ≥1 observation | 9,862 / 9,864 — **2 drugs have no data at all** |
| Targets with ≥1 observation | 1,088 / 1,088 |

Counts match Supplementary Table 1 exactly (9,864 / 1,088 / 42,203). [PAPER] [DATA] ✔

### 2.3 PDBbind inventory [DATA]

| Property | Value |
|---|---|
| Drugs (`ligands.txt`) | 4,231 entries / 4,231 unique keys / **4,195 unique SMILES** ⚠ |
| Targets (`proteins.txt`) | 1,606 entries / **1,579 unique sequences** ⚠ |
| Drug key format | PDB ligand code + formula (`'BHM_C11H10BrF3N2O4'`) |
| Target key format | UniProt accession (`'K4KA16'`) |
| SMILES length | min 8, max 246, mean 58.7, median 53 — **10 (0.2%) exceed the 200 cap** |
| Protein length | min 14, max 7073, mean 505.4, median 366 — **32 (2.0%) exceed the 2000 cap** |
| Matrix | 4,231 × 1,606 = 6,794,986 cells; **5,014 observed (0.074% dense)** |
| Entity collapse | 35 SMILES groups / 71 ids (1.7%); 18 sequence groups / 45 ids (2.8%) |
| Drugs / targets with ≥1 obs | 4,231 / 4,231 and 1,606 / 1,606 — complete |

Counts match Supplementary Table 1 exactly (4,231 / 1,606 / 5,014). [PAPER] [DATA] ✔

**Both matrices are extremely sparse** — 0.39% and 0.074% respectively, versus Co-VAE's 100% (Davis) and 25.8% (KIBA). PDBbind averages **1.18 observations per drug** and 3.12 per target.

### 2.4 Truncation is silent and non-trivial on BindingDB

`label_smiles`/`label_sequence` apply `line[:MAX_LEN]` — **head truncation, no warning**. [CODE] At the README's caps: 690 BindingDB SMILES (7.0%) lose their tail, the longest being 914 characters truncated to 200 — **78% of the molecule discarded**. [DATA] [INFERENCE] The paper never mentions truncation (Stage 2, §B). For 7% of BindingDB drugs the model sees a fragment.

---

## 3. Affinity Representation

The two datasets differ fundamentally, and the difference is controlled by a flag with **no default**.

### 3.1 BindingDB — `--is_log 2` per README

| | RAW FILE FORMAT | RUNTIME REPRESENTATION |
|---|---|---|
| Quantity | **Kd in nanomolar** | pKd-like |
| Transform | none | `-log10((Y + 1) / 1e9)` at `datahelper.py` [CODE] — note the **+1**, not the standard `-log10(Y/1e9)` |
| Direction | lower = stronger | **higher = stronger** |
| Range | 0.0 – 10,000,000 | **2.0 – 9.0** |
| Mean / median / std | 46,434.89 / 10,000.0 / 427,061.56 | **5.7737 / 4.99996 / 1.2967** |
| Quantiles (0/1/25/50/75/99/100) | 0.0 / 0.3 / 300 / 10⁴ / 10⁴ / 951,960 / 10⁷ | 2.0 / 3.0214 / 5.0 / 5.0 / 6.5214 / 8.8861 / 9.0 |
| Distinct values | 2,742 | 2,742 |
| Missing | 10,689,829 (99.61%) | same |
| Zero cells | **2** | 0 |

**The `+1` is a real deviation.** [CODE] It is not `-log10(Kd/1e9)`. Its purpose is evidently to keep `Kd = 0` finite: the 2 zero-valued raw cells map to exactly 9.0, which is the observed maximum. [DATA] [INFERENCE] Two "infinitely strong binders" are thus silently converted into the dataset's ceiling value. For Kd ≫ 1 nM the `+1` is negligible; for the sub-nanomolar tail it is not.

**The pKd ≈ 5 spike is now measured.** Stage 2 flagged it as reproduction-critical item #7 without a number. **19,828 of 42,203 observed values (46.98%) sit at exactly Kd = 10,000 nM → pKd 5.0.** [DATA]

**[INFERENCE]** Nearly half of BindingDB is a single censored value, exactly as in Davis (69.64%). Kd = 10,000 nM is an assay ceiling, not a measurement. A constant mean-predictor achieves **MSE 1.6815**.

### 3.2 PDBbind — `--is_log 0` per README

| | RAW FILE FORMAT | RUNTIME REPRESENTATION |
|---|---|---|
| Quantity | **already log-transformed** (pKd/pKi/pIC50-style) | **identical — no transform** [CODE] |
| Direction | higher = stronger | higher = stronger |
| Range | **2.0 – 11.92** | 2.0 – 11.92 |
| Mean / median / std | **6.4002 / 6.4300 / 1.9471** | same |
| Quantiles (0/1/25/50/75/99/100) | 2.0 / 2.36 / 4.96 / 6.43 / 7.74 / 11.1274 / 11.92 | same |
| Distinct values | 807 | 807 |
| Missing | 6,789,972 (99.93%) | same |
| Modal value | 8.7 (45 occurrences — **0.90%**) | same |

**PDBbind is the only one of the four datasets that is not censor-dominated.** Its modal value accounts for 0.90% of observations, against 46.98% (BindingDB), 69.64% (Davis) and 13.4% (KIBA). Its distribution is broad and near-symmetric (mean 6.400 ≈ median 6.430), and its standard deviation (1.947) is the largest of the four. [DATA]

**`--is_log` has no default and is unguarded.** [CODE] Applying `is_log=1` to PDBbind — plausible for someone following the paper rather than the README — would compress its honest 2.0–11.92 range into a meaningless 7.92–8.70 (std 0.14). [DATA] I verified this counterfactually. **[INFERENCE]** A silent, catastrophic misconfiguration is one flag away, and nothing in the code would flag it.

### 3.3 Missing-value semantics

`run_experiments.py:955` computes `label_row_inds, label_col_inds = np.where(np.isnan(Y) == False)`; fold indices address that observed-pair array. [CODE] [VERIFIED] Missing cells enter neither the loss nor the GAN pretraining (which consumes the separate `*_train.txt` corpora, not the matrix). **Missing ≠ negative** holds — verified, not assumed.

Given 99.61% / 99.93% missingness, this is the dominant structural fact about both matrices.

---

## 4. Identifier Audit

| Check | BindingDB | PDBbind |
|---|---|---|
| Duplicate drug keys | none | none |
| Duplicate target keys | none | none |
| Counts match matrix dims | ✔ 9,864 × 1,088 | ✔ 4,231 × 1,606 |
| Row/col order = file insertion order | ✔ `OrderedDict` [CODE] | ✔ |
| **Duplicate drug SMILES** | none | **35 groups / 71 ids (1.7%)** |
| **Duplicate protein sequences** | none | **18 groups / 45 ids (2.8%)** |
| Entities with zero observations | **2 drugs** | none |

As in Co-VAE, identifiers are discarded — `parse_data` label-encodes **values** only. [CODE] Ordering is positional and stable.

**PDBbind's entity collapse is minor but not zero** (2.8% of targets, 1.7% of drugs). It matters only for cold-start settings, where it partially undermines disjointness — see §7.2. The magnitude is an order below Davis's 18.3%.

**The 2 BindingDB drugs with no observations** are inert: they occupy rows of an all-NaN matrix, are never indexed by any fold, but *are* counted in `drugcount` and so size the embedding table. Harmless, but it means "9,864 drugs" overstates the usable count by 2. [DATA]

### 4.1 Protein feature tables

| File | Keys | Vector dim | Absent symbols |
|---|---:|---:|---|
| `protein_feature_vec.json` | 23 | **24** | `O`, `U` |
| `protein_feature_vecblsm.json` | 22 | **20** | `B`, `O`, `U` |

Byte-identical between the two dataset directories. [DATA]

**This confirms Stage 3's C6 and sharpens it.** [PAPER] claims a 25-dimensional BLOSUM encoding; the shipped table is **20-dimensional over 22 symbols**, and `dataset.py:12` hardcodes `self.pro_emb = 20`. The paper's "25" matches neither. Separately, `protein_feature_vec.json` is **24**-dimensional over 23 symbols — a third figure, used by a different code path. Three dimensionalities are in play (25 claimed, 24 pro2vec, 20 BLOSUM). [DATA] [INFERENCE]

`B`/`O`/`U` absence is a live `KeyError` risk in `DataGenerator.get_pro_vec`, which has no guard (Stage 3 U10). Whether these symbols occur in the protein files is still **[OPEN]** — I did not scan for them, as it does not bear on the fold questions this stage targets.

---

## 5. Fold Audit

### 5.1 The fold files are read, and keyed on `--problem_type`

```python
test_fold  = json.load(open(fpath + "folds/test_fold_setting"  + str(setting_no) + ".txt"))
train_folds= json.load(open(fpath + "folds/train_fold_setting" + str(setting_no) + ".txt"))
```
[CODE] `datahelper.py:124-134`, where `setting_no = FLAGS.problem_type`. Consumed in `nfold_1_2_3_setting_sample:488`. [VERIFIED]

### 5.2 Measured contents [DATA]

| | BindingDB s1 | PDBbind s1 | PDBbind s2 | PDBbind s3 |
|---|---:|---:|---:|---:|
| Train folds | 5 | 5 | 5 | 5 |
| Fold sizes | 7034 ×5 | 836 ×5 | 709/709/709/709/708 | 689/689/689/689/690 |
| Test size | 7,033 | 834 | **1,470** | **1,568** |
| Total indices | 42,203 | 5,014 | 5,014 | 5,014 |
| Unique | 42,203 | 5,014 | 5,014 | 5,014 |
| **Duplicates** | **0** | **0** | **0** | **0** |
| Index range | [0, 42202] | [0, 5013] | [0, 5013] | [0, 5013] |
| Zero-based | ✔ | ✔ | ✔ | ✔ |
| **Covers 0..N−1 exactly once** | ✔ | ✔ | ✔ | ✔ |
| Train/test overlap | **0** | **0** | **0** | **0** |
| Inter-fold overlap | **0** | **0** | **0** | **0** |
| **Valid at runtime** | ✔ | ✔ | ✔ | ✔ |

`bindingdb/folds/` contains **only setting 1**. Settings 2 and 3 are **absent** — confirming Stage 3's C2 from the data side: no cold-start split exists for BindingDB. [DATA]

Indices are **zero-based positions into the observed-pair array** produced by `np.where(~isnan(Y))` in row-major order — not flattened matrix indices, not entity IDs. Verified: every index maps to a valid `(row, col)` and the union is exactly `range(n_observed)`. [DATA] [VERIFIED]

**All four fold files are internally clean and runtime-valid.** This is a clear positive, and the sharpest contrast with Co-VAE, whose KIBA folds are out of range and whose files are never read at all.

---

## 6. Fold Semantics

### 6.1 Reconstructed by entity overlap [DATA]

| Split | Train drugs | Test drugs | **Shared** | Train targets | Test targets | **Shared** | Pair overlap | Classification |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BindingDB s1 | 8,610 | 2,195 | **943** | 1,060 | 776 | **748** | 0 | **PAIR-WISE (warm)** |
| PDBbind s1 | 3,591 | 787 | **147** | 1,452 | 491 | **337** | 0 | **PAIR-WISE (warm)** |
| PDBbind s2 | 3,206 | 1,025 | **0** | 907 | 860 | 161 | 0 | **COLD-DRUG** |
| PDBbind s3 | 3,114 | 1,117 | **0** | 877 | 871 | 142 | 0 | **COLD-DRUG** |

**Stage 3's U6 is now closed.** `test_fold_setting2` and `test_fold_setting3` are **confirmed drug-disjoint**: zero drugs shared between the training union and the test fold, in both. Targets remain shared, as a drug cold-start requires. [DATA] [VERIFIED]

This matches the README's description of settings 2 and 3 as Open Babel logP and XLOGP3 cold-start splits. The two differ in composition (1,470 vs 1,568 test pairs; 1,025 vs 1,117 test drugs), consistent with two different logP tools partitioning the same molecules differently.

**[INFERENCE]** The fold *files* implement cold-start correctly. Stage 3's C10 — that `problem_type` does not implement cold-start — remains accurate but must be read precisely: `problem_type` performs **no cold-start logic of its own**; it merely selects which pre-built fold file to load. The cold-start lives entirely in the data. Stage 3's finding and this one are complementary, not contradictory: the mechanism is absent from the code and present in the files.

**The generating code is still absent** — no logP computation, no Open Babel or XLOGP3 call, no threshold. The split is reproducible only by *reusing these files*, never by regenerating them. [DATA] [OPEN]

### 6.2 Paper vs fold files vs code

| Dataset | Paper claims | Fold files imply | Code consumes | Match? |
|---|---|---|---|---|
| BindingDB | "five nearly equal-sized training and validation sets" for tuning; warm-start results in Figs. 2–3; **no test set described** [PAPER] §4 | 5 train folds + 1 held-out test fold, **pair-wise warm**, complete cover | `read_sets(problem_type=1)`; runs CV twice — once on `val_sets`, once on `test_sets` | **PARTIAL** — files supply a test fold the paper never describes |
| PDBbind s1 | same | same, pair-wise warm | same | **PARTIAL** |
| PDBbind s2/s3 | cold-start by logP; threshold, direction and split sizes **all absent** [PAPER] §4 | **confirmed COLD-DRUG**, 1,470 / 1,568 test pairs | `read_sets(problem_type=2|3)` | **✔ semantics match; ✘ construction unreproducible** |
| Target cold-start | never attempted [PAPER] | no such file | no code path | ✔ consistently absent |

### 6.3 Correction to Stage 3 on hyperparameter selection

Stage 3 (§A9) states that in the reported pass "early stopping and epoch selection run on the evaluation set". Re-reading `nfold_1_2_3_setting_sample:507-540` against the code, the picture is more specific: [CODE] [VERIFIED]

- The CV routine is called **twice** — first with `val_sets`, then with `test_sets`.
- `bestparamind` is taken from the **first (validation)** call; `all_predictions` from the **second (test)** call. `testperf = all_predictions[bestparamind]`.
- So **hyperparameter selection is on validation** — cleaner than Stage 3 implies.
- **But inside the second call**, `EarlyStopping(monitor='val_cindex_score', restore_best_weights=True)` and `rperf = max(history['val_cindex_score'])` both act on whatever set was passed — the **test fold**. [CODE] `:786-800`

**[INFERENCE]** Contamination is real but narrower than Stage 3 described: the **grid point** is chosen honestly on validation; the **stopping epoch and the reported value** are chosen as a maximum over epochs on the test fold. The reported CI is `max over epochs` of a test-fold metric — an optimistic estimator regardless of how the grid point was picked. Co-VAE's contamination (both search *and* final evaluation on the test fold) is strictly worse.

---

## 7. Leakage Analysis

### 7.1 Pair / index level

| Level | BindingDB s1 | PDB s1 | PDB s2 | PDB s3 | Class |
|---|---:|---:|---:|---:|---|
| Pair overlap train/test | 0 | 0 | 0 | 0 | **NONE** |
| Index overlap | 0 | 0 | 0 | 0 | **NONE** |
| Inter-fold overlap | 0 | 0 | 0 | 0 | **NONE** |
| Duplicate indices | 0 | 0 | 0 | 0 | **NONE** |

All four fold files are clean partitions. [DATA] [VERIFIED]

### 7.2 Entity level

Settings 1 (both datasets) share drugs and targets between train and test. **This is not leakage** — it is warm-start pair prediction, which is what the paper reports in Figs. 2–3. Reported factually.

Settings 2 and 3 are drug-disjoint by identifier. **However**, PDBbind has 35 duplicate-SMILES groups covering 71 identifiers (§4). **[INFERENCE]** Where a split separates members of such a group, a nominally cold test drug has a SMILES the model already trained on. I did not enumerate how many of the 71 straddle the s2/s3 boundary — a bounded, cheap follow-up. Upper bound is 71 of 1,025 (6.9%) and 71 of 1,117 (6.4%) test drugs; the realised figure will be lower. **Class: POSSIBLE**, magnitude bounded and small — materially milder than Co-VAE's confirmed Davis case (17.8%).

### 7.3 GAN-pretraining corpus overlap — CONFIRMED, and quantified

The GAN is pretrained on `ligands_train.txt` / `proteins_train.txt`, **not** on the fold-split training data. These corpora are not split-aware, so overlap with evaluation entities crosses the train/test boundary directly. [CODE] [DATA]

| | BindingDB | PDBbind |
|---|---:|---:|
| Eval drug SMILES also in `ligands_train.txt` | **126 / 9,864 (1.3%)** | 2 / 4,195 (0.05%) |
| Eval protein seqs also in `proteins_train.txt` | **461 / 1,088 (42.4%)** | 198 / 1,579 (12.5%) |

**Class: CONFIRMED.** For BindingDB, **42.4% of evaluation proteins appear verbatim in the GAN pretraining corpus** — including proteins that will land in the test fold. The discriminator whose final Conv1D layer is spliced into the DTA network has therefore seen a large share of the test proteins' exact sequences during unsupervised pretraining.

**[INFERENCE]** This is unsupervised representation leakage: no affinity labels cross the boundary, so it is weaker than label leakage, but it is not benign, and the paper's framing ("pretrained on external unlabeled corpora" [PAPER] §6.3) implies a separation that does not hold for 42.4% of BindingDB targets.

This **supersedes Stage 3's finding #8**, which identified the same channel only for variant B via `np.concatenate((XT, XT_t))` at `:603`. The overlap measured here exists in the **corpus files themselves** and therefore affects **all three variants**, not just B.

### 7.4 GAN-pretraining corpus provenance — Stage 3's U2, substantially resolved

Stage 3 left the origin of `ligands_train.txt` (50,068) and `proteins_train.txt` (50,202) open, noting no script links them to the claimed UniProt/ChEMBL sources. Direct measurement: [DATA] [VERIFIED]

| Finding | Evidence |
|---|---|
| `ligands_train.txt` **first 68 keys are exactly Co-VAE's Davis `ligands_iso.txt` keys, in order** | `['11314340','24889392','11409972','11338033','10184653', …]` — identical lists |
| All **68 Davis drug SMILES** are present in `ligands_train.txt` | 68/68 |
| `proteins_train.txt` **first 442 keys are exactly Co-VAE's Davis `proteins.txt` keys, in order** | `['AAK1','ABL1(E255K)','ABL1(F317I)','ABL1(F317I)p','ABL1(F317L)', …]` — identical lists |
| All **379 unique Davis protein sequences** are present | 379/379 |
| The files are **byte-identical between `pdb/` and `bindingdb/`** | verified key-by-key |

**[INFERENCE]** The GAN pretraining corpora are **not** clean UniProt/ChEMBL extracts. They are aggregates that **begin with the complete Davis dataset** — the very dataset Co-VAE uses — with roughly 50,000 further entries appended. The Davis-derived prefix is not incidental: it is the exact file, in the exact order, including Davis's gene-symbol-with-mutation key format (`ABL1(E255K)`), which is a Davis-specific convention with no place in a UniProt dump.

This is a **cross-paper finding of direct thesis relevance**: the two implementations this thesis compares share data at the corpus level, and the shared component is the development dataset the Co-VAE side would naturally use.

**Still [OPEN]:** the provenance of the remaining ~50,000 entries, and whether the paper's UniProt/ChEMBL claim covers them.

### 7.5 Corpus quality defects [DATA]

| Defect | Count |
|---|---:|
| **Zero-length SMILES** in `ligands_train.txt` | **650** of 50,068 (1.3%) |
| Protein sequences shorter than 10 residues in `proteins_train.txt` | **120** of 50,202 |
| Duplicate SMILES strings | 649 redundant entries |
| Duplicate protein sequences | **6,800** redundant entries (13.5%) |
| Longest protein sequence | 10,287 residues (truncated to 2000) |

**[INFERENCE]** 650 empty SMILES label-encode to all-zero vectors of length 200. The drug GAN — batch size 5, 5,000 iterations — trains on a corpus where 1.3% of real samples are indistinguishable from padding. The discriminator's notion of "real" is contaminated accordingly. No filtering code exists.

### 7.6 Feature-construction and normalisation leakage

| Channel | Finding | Class |
|---|---|---|
| Vocabulary | `CHARISOSMISET` (65) / `CHARPROTSET` (25) hard-coded | **NONE** |
| BLOSUM / pro2vec tables | static JSON, identical across datasets, not fitted | **NONE** |
| Affinity normalisation | none beyond the global `is_log` transform | **NONE** |
| Truncation / padding | deterministic per entity, split-independent | **NONE** |
| Similarity matrices | **none ship** with DCGAN-DTA | **N/A** |
| GAN pretraining corpus | see §7.3 | **CONFIRMED** |
| Epoch selection | `max` over epochs on the test fold (§6.3) | **CONFIRMED** |

No similarity matrices exist in this repository, so that leakage class does not arise.

---

## 8. Paper vs Code vs Data

| # | Question | `[PAPER]` | `[CODE]` | `[DATA]` | Verdict |
|---|---|---|---|---|---|
| 1 | Dataset counts | BindingDB 9864/1088/42203; PDB 4231/1606/5014 | — | **exact match** | ✔ |
| 2 | BindingDB transform | "logarithmic-transformed … pKd", formula never given | `-log10((Y+1)/1e9)` | range 2.0–9.0 | ≈ the `+1` is undocumented |
| 3 | PDBbind transform | same phrasing | **none** (`is_log=0`) | already 2.0–11.92 | ✘ paper implies a transform that does not run |
| 4 | BindingDB pKd≈5 mass | unquantified (Stage 2 #7) | not handled | **19,828 = 46.98%** | ✘ now measured |
| 5 | BLOSUM dimensionality | **25** | `pro_emb = 20` | **20-dim / 22 symbols** | ✘ confirms Stage 3 C6 |
| 6 | pro2vec dimensionality | not stated | used in `parse_data` | **24-dim / 23 symbols** | new — a third figure |
| 7 | Length caps | 200 / 2000 | `line[:MAX_LEN]` | 690 (7.0%) / 16 SMILES-proteins truncated | ✘ truncation undocumented |
| 8 | Warm-start test set | **never described** | `read_sets` loads one | test fold exists, clean | ✘ paper gap, data supplies it |
| 9 | Cold-start splits | logP, threshold absent | selects fold file only | **confirmed COLD-DRUG, s2 & s3** | ✔ semantics; ✘ unreproducible |
| 10 | Fold aggregation | undetermined (Stage 2 #5) | `max` over epochs, mean over 5 folds | — | ✘ optimistic estimator |
| 11 | Pretraining corpora | UniProt + ChEMBL | reads bundled files | **prefix is exactly Davis** | ✘ claim unsupported |
| 12 | Pretraining/eval separation | implied | corpora not split-aware | **42.4% of BindingDB proteins overlap** | ✘ confirmed leakage |
| 13 | AUPR threshold | positive class undefined | `affinity > 7`, both datasets | BindingDB 19.28% pos; PDB 38.55% pos | ✔ workable on both |

### 8.1 The AUPR threshold, unlike Co-VAE's, is viable

| Dataset | Positives (> 7) | Positive rate | Per test fold (s1) |
|---|---:|---:|---|
| BindingDB | 8,136 / 42,203 | **19.28%** | 1,328 / 7,033 (18.88%) |
| PDBbind s1 | 1,933 / 5,014 | **38.55%** | 312 / 834 (37.41%) |
| PDBbind s2 | — | — | 298 / 1,470 (20.27%) |
| PDBbind s3 | — | — | 347 / 1,568 (22.13%) |

Both classes are well represented in every fold. [DATA] **[INFERENCE]** The identical threshold of 7 that renders Co-VAE's KIBA AUC degenerate (109,286 positives vs **10** negatives) is perfectly serviceable here, because both DCGAN-DTA datasets live on a pKd-like scale where 7 is a meaningful potency cut. The threshold is not the problem; applying a pKd threshold to a KIBA score is.

---

## 9. Quantitative Statistics

| Statistic | BindingDB | PDBbind |
|---|---:|---:|
| Drugs | 9,864 | 4,231 |
| Unique SMILES | 9,864 | **4,195** |
| Targets | 1,088 | 1,606 |
| Unique sequences | 1,088 | **1,579** |
| Matrix cells | 10,732,032 | 6,794,986 |
| Observed | **42,203** | **5,014** |
| Missing | 10,689,829 (**99.61%**) | 6,789,972 (**99.93%**) |
| Density | 0.393% | 0.074% |
| Obs per drug (mean) | 4.28 | **1.18** |
| Obs per target (mean) | 38.79 | 3.12 |
| Runtime min / max | 2.0 / 9.0 | 2.0 / 11.92 |
| Mean / median / std | 5.7737 / 4.99996 / 1.2967 | 6.4002 / 6.4300 / **1.9471** |
| q1 / q25 / q75 / q99 | 3.0214 / 5.0 / 6.5214 / 8.8861 | 2.36 / 4.96 / 7.74 / 11.1274 |
| Distinct values | 2,742 | 807 |
| Modal value (count, share) | **5.0 (19,828 = 46.98%)** | 8.7 (45 = **0.90%**) |
| Positives at `> 7` | 8,136 (19.28%) | 1,933 (38.55%) |
| Constant-predictor MSE (mean) | 1.6815 | **3.7912** |
| Constant-predictor MSE (median) | 2.2802 | 3.7921 |
| Constant-predictor MAE (median) | 0.9219 | 1.5832 |
| Fold settings available | **1 only** | **1, 2, 3** |
| Fold files valid at runtime | ✔ | ✔ |
| SMILES over the 200 cap | 690 (7.0%) | 10 (0.2%) |
| Proteins over the 2000 cap | 16 (1.5%) | 32 (2.0%) |

---

## 10. Reproduction Implications

1. **The data layer is materially healthier than Co-VAE's.** All four fold files are valid, complete, duplicate-free, mutually exclusive and actually read. Entity counts match the paper exactly for both datasets. Nothing here requires editing code to obtain the published dataset — the opposite of Co-VAE's KIBA.

2. **Warm-start (setting 1) is reproducible as data on both datasets.** The published protocol gap — [PAPER] never describes a test set — is filled by the fold files, which supply a clean held-out fold. The reproduction will differ from the paper only in that the paper never said this is what it did.

3. **Cold-start is reproducible only by reusing the shipped files.** Settings 2 and 3 are confirmed drug cold-start, but the logP threshold, direction and tool are all absent. The splits can be *used*; they cannot be *regenerated* or extended to BindingDB, which has no settings 2/3 at all.

4. **The reported metric is an optimistic estimator by construction** (§6.3): `max` over epochs on the test fold. Reproducing the paper's numbers means reproducing that estimator; producing an honest number means deviating from it. These are two different experiments and must be reported as such.

5. **GAN pretraining leakage must be disclosed, not fixed silently.** 42.4% of BindingDB evaluation proteins are in the pretraining corpus. Removing them would change the method; leaving them in means the comparison inherits the leak. Either way it belongs in the thesis's protocol section.

6. **`--is_log` is a silent correctness hazard.** No default, and the wrong value destroys PDBbind's label distribution without error. Pin it explicitly in every run.

7. **Stage 3's C1 (the dimensional incompatibility of the transferred Conv1D layer) is untouched by this stage** and remains the binding constraint on whether any of this runs at all.

---

## 11. Development-Dataset Implications

### 11.1 Cross-dataset comparison

| Dataset | Drugs | Targets | Pairs | Affinity | Split type | Folds | Missingness | Main risk |
|---|---:|---:|---:|---|---|---:|---:|---|
| Davis (Co-VAE) | 68 | 442 | 30,056 | pKd 5.0–10.80 | runtime-generated, entity-wise | 6 (5+1) | **0.00%** | 69.64% censored at 5.0; 442→379 unique sequences |
| KIBA (Co-VAE) | 1,954 | 217 | 109,296 | KIBA 0–17.20 | runtime-generated, entity-wise | 6 (5+1) | 74.22% | matrix ≠ published; folds invalid; AUC degenerate |
| BindingDB (DCGAN) | 9,864 | 1,088 | 42,203 | pKd 2.0–9.0 | **file-based, pair-wise** | 5+1 | **99.61%** | 46.98% censored at 5.0; 42.4% pretrain overlap; warm-start only |
| PDBbind (DCGAN) | 4,231 | 1,606 | 5,014 | pKd 2.0–11.92 | **file-based, pair-wise + 2 cold-drug** | 5+1 ×3 | **99.93%** | 1.18 obs/drug; only 5,014 pairs |

### 11.2 Compatibility matrix

| Property | Davis | KIBA | BindingDB | PDBbind |
|---|---|---|---|---|
| Compatible with Co-VAE | **directly compatible** | requires adaptation (folds invalid, AUC degenerate) | requires adaptation (`Y` pickle + different loader) | requires adaptation |
| Compatible with DCGAN-DTA | requires adaptation (no `Y` pickle path, no `*_train` corpora, no fold settings 2/3) | requires adaptation | **directly compatible** | **directly compatible** |
| Same raw input available (SMILES + sequence) | yes | yes | yes | yes |
| Same affinity semantics | pKd | **KIBA score — incompatible** | pKd | pKd |
| Same split protocol possible | requires adaptation (Co-VAE generates its own) | fold mismatch | requires adaptation | requires adaptation |
| Computational cost | low | moderate | **computationally heavy** (42k pairs × 9,864 drugs) | **low** (5,014 pairs) |
| Preprocessing burden | **low** | high | moderate | **low** |
| Development suitability | good for Co-VAE only | poor | poor for iteration | **good for DCGAN-DTA only** |

### 11.3 Can one dataset serve both? — evidence-based answer

**No, not without adaptation that would itself need auditing.** The obstacles are concrete:

- **Loader incompatibility.** Co-VAE reads whitespace-delimited text (`pd.read_csv(sep='\s+')`); DCGAN-DTA reads a `latin1` pickle. Co-VAE reads `ligands_iso.txt`; DCGAN-DTA reads `ligands.txt`. Neither can open the other's directory as shipped. [CODE] [VERIFIED]
- **Fold incompatibility.** Co-VAE ignores fold files and generates entity-wise splits at runtime; DCGAN-DTA loads pair-wise files. Making them share a protocol means changing at least one — a deviation from both papers.
- **Missing infrastructure.** DCGAN-DTA additionally requires `Y` (pickle), `protein_feature_vec.json`, `protein_feature_vecblsm.json` and the two `*_train.txt` corpora. **None exists for Davis or KIBA**, and the pretraining corpora cannot be manufactured without knowing their provenance (§7.4).
- **Affinity semantics.** Three of the four datasets are on a pKd-like scale; **KIBA is not**, and no transform in either codebase converts it.

**[INFERENCE]** The honest conclusion is the one the task explicitly permits: **the first reproduction must use each paper's native datasets.** Davis (and, with documented deviations, KIBA) for Co-VAE; PDBbind and BindingDB for DCGAN-DTA. A common-dataset experiment is a **later, separate piece of work** requiring a shared loader and a shared split definition — which is properly part of the hybrid-method contribution, not of the reproduction.

Forcing a shared dataset now would mean rewriting one paper's data pipeline before either has been shown to run — inverting the dependency order and making any discrepancy uninterpretable.

### 11.4 Per-method development recommendation

Taking the eight criteria in the task's Step 8 in turn:

**For DCGAN-DTA: PDBbind.** It is the only one of the four with a broad, uncensored affinity distribution (modal value 0.90%, std 1.947); it has all three fold settings including two confirmed cold-drug splits; at 5,014 pairs it is by far the cheapest to iterate on (8.4× smaller than BindingDB); its fold files are valid; AUPR is well-conditioned (38.55% positive); and `--is_log 0` means no transform to get wrong. Its weakness — **1.18 observations per drug** — is severe for a warm-start setting and should be stated whenever setting-1 results are reported, but it does not impede *development*.

**For Co-VAE: Davis**, per the companion report §11 — with the 442→379 sequence collapse documented and controlled for in the target-wise setting.

**Shared caveat.** Davis and PDBbind are both good *development* vehicles and both poor *evidential* bases: Davis is 69.64% censored and dense; PDBbind is 99.93% empty with barely one observation per drug. They are for establishing that pipelines run and behave sanely. Headline claims need BindingDB and KIBA, which cost more and carry the defects catalogued above.

---

## 12. Open Questions

| # | Question | Why unresolved |
|---|---|---|
| **O1** | Provenance of the ~50,000 non-Davis entries in the pretraining corpora | §7.4 identifies the Davis prefix; the remainder is unattributed and no manifest ships |
| **O2** | How many of PDBbind's 71 duplicate-SMILES ids straddle the s2/s3 cold-drug boundary? | Bounded at ≤6.9%; cheap to compute, not required for this stage's conclusions |
| **O3** | Do `B`/`O`/`U` occur in `proteins.txt` / `proteins_train.txt`? | Would settle Stage 3's U10 `KeyError` risk; a static scan, deferred |
| **O4** | What logP threshold and direction generated settings 2 and 3? | Semantics confirmed (§6.1); the generating rule is absent from the repository |
| **O5** | Why does `bindingdb/` lack settings 2 and 3? | Consistent with the paper reporting cold-start on PDBbind only, but unstated |
| **O6** | Does the 2-cell `Kd = 0 → pKd 9.0` mapping affect any fold materially? | 2 of 42,203; almost certainly negligible, unverified |
| **O7** | What preprocessing produced `Y`, `ligands.txt`, `proteins.txt`? | Stage 3 D5 — no preprocessing code ships; harmonization claims unauditable |
| **O8** | Are the 650 empty SMILES a corrupted export or deliberate padding? | No evidence either way |

---

## 13. Final Evidence-Based Findings

1. **`[DATA]` All four fold files are valid, complete, duplicate-free and mutually exclusive**, covering `0..N−1` of the observed-pair array exactly once. Unlike Co-VAE's, they are actually read, and they work.
2. **`[DATA]` `test_fold_setting2` and `test_fold_setting3` are confirmed COLD-DRUG splits** — zero drug overlap with training. Stage 3's U6 is closed.
3. **`[DATA]` `bindingdb/folds/` contains only setting 1.** No cold-start split exists for BindingDB.
4. **`[DATA]` The GAN pretraining corpora begin with the complete Davis dataset** — first 68 ligand keys and first 442 protein keys are byte-identical to Co-VAE's Davis files, in order. Stage 3's U2 is substantially resolved: these are not clean UniProt/ChEMBL extracts.
5. **`[DATA]` 42.4% of BindingDB evaluation proteins appear verbatim in the GAN pretraining corpus** (461/1,088), and 12.5% for PDBbind. **CONFIRMED** unsupervised representation leakage affecting **all three variants**, not only variant B as Stage 3 reported.
6. **`[DATA]` BindingDB is 46.98% censored at pKd 5.0** (19,828/42,203) — Stage 2's reproduction-critical item #7, now measured.
7. **`[DATA]` PDBbind is the only uncensored dataset of the four**: modal value 0.90%, std 1.947, mean ≈ median. It is also the sparsest, at **1.18 observations per drug**.
8. **`[DATA]` Both matrices are extremely sparse** — 99.61% and 99.93% missing — and missing cells are excluded structurally, never treated as negatives.
9. **`[DATA]` Three BLOSUM/pro2vec dimensionalities are in play**: 25 claimed by the paper, 24 in `protein_feature_vec.json`, **20** in `protein_feature_vecblsm.json`. Confirms and extends Stage 3's C6.
10. **`[DATA]` The pretraining corpora contain 650 zero-length SMILES** and 6,800 duplicate protein sequences; the drug GAN trains on a corpus where 1.3% of "real" samples are empty.
11. **`[INFERENCE]` Stage 3's account of selection contamination is refined**: the grid point is chosen on validation, but the stopping epoch and the reported value are a `max` over epochs on the test fold. Narrower than Stage 3 stated, still an optimistic estimator.
12. **`[DATA]` `--is_log` is an unguarded correctness hazard**: no default, and `is_log=1` would compress PDBbind's 2.0–11.92 range to 7.92–8.70 silently.
13. **`[INFERENCE]` No single dataset can serve both methods without adaptation.** Loaders, fold mechanisms, required auxiliary files and — for KIBA — affinity semantics are all incompatible. The first reproduction must use native datasets per method.

---

*End of Stage 4 — DCGAN-DTA. Read-only: no repository source file, dataset, or fold file was modified; no dependency installed; no model executed. Earlier audit reports are untouched.*

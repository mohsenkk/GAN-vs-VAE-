# Stage 5 — Reproduction Feasibility Audit: DCGAN-DTA

**Inputs:** `audit/02-paper-forensic/DCGAN-DTA.md`, `audit/03-paper-code-traceability/DCGAN-DTA.md`, `audit/04-dataset-fold/DCGAN-DTA.md`, `THESIS_IMPLEMENTATION_PLAN.md`
**Implementation:** `DCGAN-DTA/DCGAN-DTA/` (upstream `github.com/mojtabaze7/DCGAN-DTA`, HEAD `453fe16`)
**Mode:** READ-ONLY. No source modified, no dataset modified, no fold file modified, no package installed, no environment created, nothing executed. Scratch analysis held outside the repository.

**Evidence tags:** `[PAPER]` · `[CODE]` · `[DATA]` · `[DOC]` (repository README or vendor release documentation) · `[INFERENCE]` · `[UNKNOWN]`

> **This stage overturns Stage 3's single most severe finding.** See §5.3 and the dated correction in §12.

---

## PART 1 — Environment Feasibility

### 1.1 Python

| Question | Finding | Tag |
|---|---|---|
| Version stated by repository | **None.** README lists "Python" with no version. | `[DOC]` |
| Version stated by paper | OS, CPU, GPU and "framework stack" named; no Python version. | `[PAPER]` |
| Syntax compatibility | `from __future__ import print_function` (`run_experiments.py:1`) is a Python-2 compatibility relic; harmless on 3.x. No 3.x-incompatible syntax found. | `[CODE]` |
| Python 3.11+ breakage | **None intrinsic to this code.** Unlike Co-VAE, no `random.sample(set)` pattern. The constraint is transitive, through TensorFlow. | `[CODE]` `[INFERENCE]` |
| Effective ceiling | Bounded by the TensorFlow window (§1.2), not by the source. | `[INFERENCE]` |

### 1.2 Framework — TensorFlow / Keras

This is the binding environment constraint. Four independent API usages bracket the viable window.

| # | API used | Location | Constraint implied | Tag |
|---|---|---|---|---|
| 1 | `Conv1DTranspose` | `:58, 62, 66` (drug gen), `:164, 168, 172` (target gen), `:268, 272, 276` (BLOSUM gen) | **TF ≥ 2.3** — the layer did not exist earlier | `[CODE]` `[DOC]` |
| 2 | `fit_generator`, `predict_generator` | `:660`, `:663` — **`general_nfold_cv` only, i.e. variants B and C** | **Keras 2** — both removed in Keras 3 | `[CODE]` `[DOC]` |
| 3 | `tf.compat.v1.keras.backend.{get_session, clear_session, set_session}` | `reset_keras():37-52`, called from **both** `general_nfold_cv:711` and `general_nfold_cv2:843` | **Keras 2** — the `tf.compat.v1.keras` namespace is absent under Keras 3 | `[CODE]` `[DOC]` |
| 4 | `tf2.ConfigProto`, `tf2.set_random_seed`, `tf2.matrix_band_part` via `import tensorflow.compat.v1 as tf2` | `:29, 31`, `cindex_score:875` | TF 2.x `compat.v1` — present throughout the 2.x line | `[CODE]` |

**[INFERENCE] Derived viable window: TensorFlow ≥ 2.3 and < 2.16, with Keras 2.** TF 2.16 is the release that made Keras 3 the default `keras` and removed the `tf.compat.v1.keras` shim. Constraint 3 binds **all three variants**, so no variant escapes the Keras-2 requirement.

**Transitively, Python ≤ 3.11**, since no TF release below 2.16 supports Python 3.12+. `[INFERENCE]` `[DOC]`

> **`[UNKNOWN]` — which specific version inside that window actually works is not established by the repository or by any documentation, and this audit does not assume one.** The window is a necessary condition derived from API usage, not a verified sufficient one. Marked **RUNTIME-ONLY UNKNOWN R1**.

**Keras/tf.keras mixing — present:**

```python
import keras                                    # :13  standalone
from keras import backend as K                  # :14
from keras.layers import Input, Reshape, ...    # :15
from keras.models import Sequential, Model      # :17
from tensorflow.keras.optimizers import Adam    # :18  ← tf.keras
from keras.callbacks import EarlyStopping       # :19
```

`Adam` is imported from `tensorflow.keras`; every layer and model class comes from standalone `keras`. `[CODE]`

**[INFERENCE]** Under TF 2.6–2.15 with Keras 2 this is benign — `tensorflow.keras` is a re-export of the same `keras` package, so the objects are identical. Under Keras 3 the two diverge into genuinely different classes and an optimizer from one cannot compile a model from the other. This mixing therefore *reinforces* the Keras-2 requirement rather than adding a new one.

**A further version-sensitive construct:** `build_dis` calls `model.add(Input(shape=(200,)))` (`:74`, `:178`) — passing a `KerasTensor` to `Sequential.add`. Keras 2's `Sequential.add` special-cases this and substitutes the originating `InputLayer`. `[CODE]` `[INFERENCE]` Keras 3 handles `Sequential` input specification differently. This matters because the **layer indexing in §5.3 depends on whether `InputLayer` occupies a slot in `model.layers`** — under Keras 2 it does not. Marked **RUNTIME-ONLY UNKNOWN R2**.

### 1.3 Other dependencies

Complete import inventory across all five modules. `[CODE]`

| Package | Imported in | Pinned? | API-compat sensitive? |
|---|---|---|---|
| `numpy` | all | **no** | moderate — NumPy 2.x removed aliases; none used here |
| `pandas` | `run_experiments` | **no** | low — imported but barely used |
| `tensorflow` / `tensorflow.compat.v1` | `run_experiments` | **no** | **critical** (§1.2) |
| `keras` | `run_experiments`, `dataset` | **no** | **critical** (§1.2) |
| `sklearn` | `run_experiments` (`average_precision_score`) | **no** | low |
| `matplotlib` | `run_experiments` | **`3.5.2`** — the only pinned version anywhere | moderate — `pil_kwargs` TIFF path |
| `seaborn` | `run_experiments` (`sns.set_theme`) | **no** | low; `set_theme` needs seaborn ≥ 0.11 |
| `Pillow` | **implicit** — `plt.savefig(..., pil_kwargs={"compression":"tiff_lzw"})` at `:833` | **no**, and **undeclared in the README** | moderate |
| `subprocess` + **Java** | `emetrics.get_aupr` | n/a | **not required** — see below |

**Java is NOT required.** `emetrics.get_aupr` shells out to `java -jar auc.jar`, but `run_experiments.py:20` imports **only** `get_rm2`; AUPR is computed with `sklearn.average_precision_score` at `:687`/`:819`. `auc.jar` and `emetrics.get_aupr` are dead code. `[CODE]` `[VERIFIED]` Confirms Stage 3 §E11. (Java is in fact absent from this machine — `java: command not found` — which would have been fatal had the path been live.) `[DATA]`

**Undeclared dependencies:** the README lists Python, TensorFlow, Keras, NumPy and matplotlib. It omits **seaborn**, **scikit-learn**, **pandas** and **Pillow**, all of which are imported at module scope and will therefore fail at import time if absent. `[CODE]` `[DOC]` `[INFERENCE]`

### 1.4 Filesystem and OS assumptions

| Assumption | Evidence | Consequence |
|---|---|---|
| `dataset_path` is concatenated with `+`, not `os.path.join` | `datahelper.py:140` etc. | trailing slash **mandatory**; README supplies it |
| `figdir = "figures/"` — POSIX-style relative literal | `:33` | created by `experiment()` via `os.makedirs`; works on Windows |
| `log_dir` default `/tmp` | `arguments.py:102` | **absolute POSIX path — invalid on Windows**; README overrides with `'logs/'` |
| `--dataset_path` default `/data/pdb/` | `arguments.py:72` | **leading slash = absolute path**, resolves to drive root; README uses relative `'data/pdb/'` |
| Working directory must be the repo root | all data paths relative | run from `DCGAN-DTA/DCGAN-DTA/` |
| GPU device 0 | `reset_keras():51` sets `visible_device_list = "0"` | **hard GPU requirement at that call site** — see §7-A |

**No OS-specific calls** beyond those path literals; no shell-outs on the live path.

---

## PART 2 — Entrypoint Feasibility

**Entrypoint:** `run_experiments.py:978` `if __name__ == "__main__"` → `argparser()` → `run_regression(FLAGS)` → `experiment(FLAGS, deepmethod)` → `nfold_1_2_3_setting_sample` → `general_nfold_cv2` (variant A) or `general_nfold_cv` (B/C). `[CODE]`

| Project | Entrypoint | Required args | Required files | Required environment | Status |
|---|---|---|---|---|---|
| DCGAN-DTA (variant A, PDBbind) | `run_experiments.py` | `--num_windows`, `--smi_window_lengths`, `--seq_window_lengths` (no defaults, `len()`-ed); `--max_seq_len 2000`, `--max_smi_len 200` (**defaults are 0**); `--dataset_path 'data/pdb/'` (**default is absolute `/data/pdb/`**); `--is_log 0`; `--problem_type 1`; `--log_dir 'logs/'` (**default `/tmp`**); `--model A` | `Y`, `ligands.txt`, `proteins.txt`, `ligands_train.txt`, `proteins_train.txt`, `protein_feature_vec.json`, `folds/{train,test}_fold_setting1.txt` — **all present** `[DATA]` | TF 2.3–2.15 + Keras 2, Python ≤ 3.11, seaborn, sklearn, pandas, Pillow. GPU needed only at `reset_keras` | **UNKNOWN UNTIL EXECUTION** |
| DCGAN-DTA (variant B/C, PDBbind) | same | same + `--model B\|C` | same + `protein_feature_vecblsm.json` | same, **plus** `fit_generator`/`predict_generator` ⇒ strictly Keras 2 | **UNKNOWN UNTIL EXECUTION** |

**Why not READY FOR SMOKE TEST:** no TensorFlow is installed on this machine (§1.5 below), no version inside the derived window has been verified to construct these graphs, and R2 (Keras-2 `Sequential.add(Input(...))` semantics) governs the layer indexing that §5.3 depends on. The status is *not* BLOCKED — no static defect prevents reaching model construction — but it cannot be called READY while a runtime-only unknown sits on the construction path.

### 1.5 Current machine state `[DATA]`

Measured, not assumed: Python **3.13.2**, numpy **2.2.3**, pandas **2.2.3**, matplotlib **3.10.3**. **`tensorflow`, `keras`, `sklearn`, `seaborn`, `tqdm` are NOT installed.** Java absent.

**[INFERENCE]** The ambient interpreter (3.13.2) is **outside** the derived window (≤ 3.11) and cannot host any TF release that satisfies §1.2. Stage 06 requires a separate interpreter. Creating one is explicitly out of scope for this stage.

---

## PART 3 — Argument Audit

All 16 arguments in `arguments.py`. `[CODE]`

| Argument | Default | Required in practice | Used? | README consistent? | Paper consistent? | Notes |
|---|---|---|---|---|---|---|
| `--seq_window_lengths` | **none → `None`** | **YES** | ✔ `general_nfold_cv:~580` | ✔ `4 8 16` | ✔ | `len(None)` ⇒ `TypeError` if omitted |
| `--smi_window_lengths` | **none → `None`** | **YES** | ✔ | ✔ `4 8 16` | ✔ | same |
| `--num_windows` | **none → `None`** | **YES** | ✔ | ✔ `128 32` | ⚠ paper states 128/256/384 filters; README grid is `128 32` | same |
| `--max_seq_len` | **`0`** ⚠ | **YES** | ✔ | ✔ `2000` | ✔ 2000 | **dangerous default**: 0 ⇒ zero-width tensors |
| `--max_smi_len` | **`0`** ⚠ | **YES** | ✔ | ✔ `200` | ✔ 200 | **dangerous default** |
| `--dataset_path` | **`/data/pdb/`** ⚠ | **YES** | ✔ | ✘ README uses `data/pdb/` | n/a | **dangerous default**: leading `/` = absolute |
| `--problem_type` | `1` | no | ✔ selects fold file | ✔ | ✔ | 1 = warm; 2/3 = cold-drug (PDBbind only) |
| `--is_log` | **`0`** | **YES for BindingDB** | ✔ | README: pdb `0`, bindingdb `2` | ⚠ paper implies a transform for both | **dangerous default for BindingDB** — see below |
| `--model` | `A` | no | ✔ `run_regression:971-977` | ✔ | ✔ Sup Table 3 | A/B/C |
| `--num_epoch` | `100` | no | ✔ | ✘ README says `300` | ✔ paper says 100 | README/paper conflict |
| `--batch_size` | `256` | no | ✔ | ✔ 256 | ✔ 256 | — |
| `--log_dir` | **`/tmp`** ⚠ | yes on Windows | ✔ | ✔ `'logs/'` | n/a | absolute POSIX default |
| `--learning_rate` | `0.001` | no | **✘ NEVER READ** | — | ⚠ paper states 0.001 | **ignored** — optimizer is the string `'adam'` (Keras default 0.001, coincidentally equal) |
| `--checkpoint_path` | `''` | no | **✘ NEVER READ** | — | — | ignored |
| `--num_hidden` | `0` | no | **✘ NEVER READ** | — | — | ignored |
| `--num_classes` | `0` | no | **✘ NEVER READ** | — | — | ignored |
| `--binary_th` | `0.0` | no | **✘ NEVER READ** | — | — | ignored; AUPR threshold is hard-coded `> 7` |

Confirms Stage 3 §E18 (five declared-but-unread flags) and adds the defaults analysis.

**Three dangerous defaults, in severity order:**

1. **`--dataset_path` defaults to `/data/pdb/`** — absolute. Running without it looks for `C:\data\pdb\` (or `/data/pdb/`), fails at the first `open()`. Loud, not silent.
2. **`--max_seq_len` / `--max_smi_len` default to `0`** — every sequence label-encodes to a length-0 array. Silent until a shape error much later. Stage 4 already flagged the *value* risk; the *default* is worse than the value.
3. **`--is_log` defaults to `0`** — correct for PDBbind, **wrong for BindingDB**, where raw Kd in nM (range 0–10⁷, mean 46,435) would be regressed directly against an MSE loss. Silent and catastrophic. Stage 4's D-5 is confirmed and its direction clarified: the hazard for **BindingDB** is the *default*; the hazard for **PDBbind** is accidentally passing `1`.

**`--problem_type` is not a cold-start switch.** It selects a fold filename. Stage 3's C10 stands; Stage 4 §6.1 established that the cold-start semantics live in the files. `[CODE]` `[DATA]`

---

## PART 4 — Data Pipeline Feasibility (PDBbind)

### 4.1 Traced path, raw files → first model-input tensor `[CODE]` `[DATA]`

```
argparser()
 └─ experiment(FLAGS, deepmethod)                              :928
     └─ DataSet(fpath, setting_no=problem_type, seqlen=2000, smilen=200)
     └─ dataset.parse_data(FLAGS)          with_label=2 (default)   datahelper.py:136
         ├─ open  data/pdb/protein_feature_vec.json     → pro2vec   23 keys × 24-dim
         ├─ json  data/pdb/ligands.txt                  → 4,231 SMILES
         ├─ json  data/pdb/proteins.txt                 → 1,606 sequences
         ├─ json  data/pdb/ligands_train.txt            → 50,068 SMILES   (GAN corpus)
         ├─ json  data/pdb/proteins_train.txt           → 50,202 sequences(GAN corpus)
         ├─ pickle data/pdb/Y  (latin1)                 → (4231,1606) float64, 5,014 observed
         └─ is_log==0 ⇒ NO transform                            ← values used verbatim, 2.0–11.92
         ├─ XD   = [label_smiles(s,200,CHARISOSMISET)   for s in ligands]        (4231,200)  float64
         ├─ XD_t = [label_smiles(s,200,CHARISOSMISET)   for s in ligands_train]  (50068,200) float64
         ├─ XT   = [label_sequence(p,2000,CHARPROTSET)  for p in proteins]       (1606,2000) float64
         └─ XT_t = [label_sequence(p,2000,CHARPROTSET)  for p in proteins_train] (50202,2000) float64
     └─ np.asarray on all four                                   :940-944
     └─ label_row_inds, label_col_inds = np.where(~isnan(Y))     :955   → 5,014 pairs
     └─ nfold_1_2_3_setting_sample(...)                          :486
         └─ dataset.read_sets(FLAGS)                             datahelper.py:124
             ├─ folds/test_fold_setting1.txt   → 834 indices
             └─ folds/train_fold_setting1.txt  → 5 folds × 836
         └─ builds train_sets / val_sets / test_sets (5 each)
         └─ general_nfold_cv2(... train_sets, val_sets)           :739   ← variant A
             ├─ trrows = label_row_inds[labeledinds]; trcols = label_col_inds[...]
             ├─ prepare_interaction_pairs(XD,XT,Y,trrows,trcols)  :908
             │    → train_drugs (N,200), train_prots (N,2000), train_Y (N,)
             └─ gridmodel = runmethod(FLAGS, XD_t, XT_t, p1,p2,p3)  ← build_GAN_A  :357
                  ⚠ TRAINS BOTH GANs BEFORE RETURNING (see §5.4)
             └─ gridmodel.fit([train_drugs, train_prots], train_Y, batch_size=256, ...)
                                                      ← FIRST MODEL-INPUT TENSOR
```

### 4.2 Required files — all present `[DATA]`

| File | Present | Size | Consumed by |
|---|:--:|---|---|
| `data/pdb/Y` | ✔ | 52 MB | `parse_data` |
| `data/pdb/ligands.txt` | ✔ | 336 KB | `XD` |
| `data/pdb/proteins.txt` | ✔ | 816 KB | `XT` |
| `data/pdb/ligands_train.txt` | ✔ | 3.6 MB | `XD_t` (GAN) |
| `data/pdb/proteins_train.txt` | ✔ | 25 MB | `XT_t` (GAN) |
| `data/pdb/protein_feature_vec.json` | ✔ | 4 KB | `pro2vec` — **read but unused by variant A** |
| `data/pdb/protein_feature_vecblsm.json` | ✔ | 4 KB | variants B/C only, read at `:598` |
| `folds/train_fold_setting1.txt` | ✔ | — | `read_sets` |
| `folds/test_fold_setting1.txt` | ✔ | — | `read_sets` |

**No missing file.** No preprocessing script is required to reach the first tensor — every input ships pre-built. `[DATA]`

### 4.3 Memory envelope `[DATA]` `[INFERENCE]`

`label_sequence`/`label_smiles` return `np.zeros(MAX_LEN)` — **float64**, not an integer dtype. `[CODE]` Materialised arrays:

| Array | Shape | dtype | Bytes |
|---|---|---|---:|
| `XT_t` | (50 202, 2 000) | float64 | **803 MB** |
| `XD_t` | (50 068, 200) | float64 | 80 MB |
| `Y` | (4 231, 1 606) | float64 | 54 MB |
| `XT` | (1 606, 2 000) | float64 | 26 MB |
| `XD` | (4 231, 200) | float64 | 7 MB |
| **Total resident** | | | **≈ 970 MB** |

Plus a transient Python list-of-arrays of comparable size during construction, before `np.asarray` copies it. **[INFERENCE]** Peak RSS during `parse_data` plausibly ~2 GB. Not a blocker on a normal machine, but it must be budgeted for, and it is incurred **before any model exists**.

### 4.4 Verdict

> **PDBbind can reach the first model-input tensor without modifying code** — provided the six non-default flags in §2 are supplied and a Keras-2 environment exists. Every required file is present, the fold files are valid (Stage 4 §5.2), no preprocessing step is missing, and no transform is applied. `[INFERENCE]`

The one caveat is §5.4: reaching the first *fit* tensor also triggers GAN pretraining.

---

## PART 5 — Model Construction Feasibility

### 5.1 Variant dispatch `[CODE]`

`run_regression:970` → `--model B` ⇒ `build_GAN_B`; `C` ⇒ `build_GAN_C`; anything else ⇒ `build_GAN_A` (default). Variant A routes to `general_nfold_cv2`; B and C to `general_nfold_cv`.

### 5.2 Discriminator layer stacks, enumerated

Under Keras 2, `Sequential.add(Input(...))` registers an `InputLayer`, which **does not appear in `model.layers`**. `[CODE]` `[INFERENCE]` (R2). Enumerating:

**`ganForDrug.build_dis` (`:72-94`)** — `Input(shape=(200,))` then:

| idx | layer | in-channels | out-channels |
|---:|---|---:|---:|
| 0 | `Reshape((200,1))` | — | 1 |
| 1 | `Conv1D(4, k=3, same)` | 1 | 4 |
| 2 | `Conv1D(8, k=3, same)` | 4 | 8 |
| 3 | `Conv1D(16, k=3, same)` | 8 | 16 |
| 4 | `Conv1D(32, k=3, same)` | 16 | 32 |
| **5 = [-3]** | **`Conv1D(64, k=3, same)`** | **32** | **64** |
| 6 = [-2] | `Flatten()` | — | — |
| 7 = [-1] | `Dense(1, tanh)` | — | — |

**`ganForTarget.build_dis` (`:176-198`)** — identical but `Input(shape=(2000,))` / `Reshape((2000,1))`. `layers[-3]` = **`Conv1D(64)`, built on 32 channels**.

**`ganForBlosumTarget.build_dis` (`:280-301`)** — **no `Input`, no `Reshape`**; channel progression 4 → 8 → 16 → **20** → **40**:

| idx | layer | in-channels | out-channels |
|---:|---|---:|---:|
| 3 | `Conv1D(20, k=3, same)` | 16 | 20 |
| **4 = [-3]** | **`Conv1D(40, k=3, same)`** | **20** | **40** |

All three `ganFor*` functions `return dis_v` — the generator is discarded. `[CODE]` `:151`, and confirms Stage 3 §E3.

### 5.3 The `gan_smiles.layers[-3]` question — RESOLVED, and Stage 3 is incorrect

| Variant | Transferred layer | **Built for** | **Receives** | Compatible? |
|---|---|---:|---|:--:|
| **A** — drug | `Conv1D(64)` from `ganForDrug` | **32 ch** | `Embedding(output_dim=32)` → **32 ch** (`:364-366`) | **✔ YES** |
| **A** — protein | `Conv1D(64)` from `ganForTarget` | **32 ch** | `Embedding(output_dim=32)` → **32 ch** (`:376-377`) | **✔ YES** |
| **B** — drug | `Conv1D(64)` from `ganForDrug` | **32 ch** | `Embedding(output_dim=32)` → **32 ch** (`:409-410`) | **✔ YES** |
| **B** — protein | `Conv1D(40)` from `ganForBlosumTarget` | **20 ch** | `Input(shape=(max_seq_len, 20))` → **20 ch** (`:406`, `:419`) | **✔ YES** |
| **C** — drug | `Conv1D(64)` from `ganForDrug` | **32 ch** | `Embedding(output_dim=32)` → **32 ch** (`:452-453`) | **✔ YES** |
| **C** — protein | *no transfer* — `XTinput` feeds `Conv1D` directly (`:461`) | — | — | n/a |

> ### ⚠ CORRECTION TO STAGE 3 — finding C1 is wrong
>
> Stage 3 §C1 states: *"`gan_smiles.layers[-3]` is the discriminator's `Conv1D(64, k=3)`, built on a **1-channel** input (`Reshape((200,1))`, `run_experiments.py:76,90`). It is then applied to a **32-channel** `Embedding(output_dim=32)` tensor"* — graded **CRITICAL**, and named in Stage 3 §F1 as the result that *"determines whether anything downstream is worth pursuing."*
>
> **The 1-channel reading is a misattribution.** `Reshape((200,1))` is `layers[0]`, and **four `Conv1D` layers (4 → 8 → 16 → 32) intervene** before `layers[-3]`. The kernel of `layers[-3]` is therefore shaped `(3, 32, 64)` — built for **32** input channels, which is exactly what `Embedding(output_dim=32)` supplies.
>
> The same correction applies to variant A's protein branch (`:376-377`, also 32 → 32), and Stage 3's own observation that variant B's protein transfer is "the only dimensionally consistent one (20 → 20)" is right about B but wrong to treat it as exceptional — **all six transfer sites are channel-consistent.**

**[INFERENCE] Static analysis does not prove incompatibility; it proves channel *compatibility* at every transfer site.** Per the operating rules I do not claim these variants construct successfully — that remains **RUNTIME-ONLY UNKNOWN R3** — but the specific defect Stage 3 identified as the project's principal blocker **does not exist**.

**Residual construction risks, all runtime-only:**

- **R2** — layer indexing depends on Keras 2 excluding `InputLayer` from `model.layers`. Under different semantics `layers[-3]` would select `Conv1D(32)` (drug) and the arithmetic changes. The table above is valid **for Keras 2**.
- **R4** — the transferred layer is *reused*, so it is a layer already built and already belonging to another model. Keras permits layer sharing, but the layer arrives with `trainable=False` inherited from `dis_v.trainable = False` (`:142, 246, 344`). Whether that propagates to the spliced instance is Stage 3's U5, unresolved and inspectable only at runtime via `model.trainable_weights`.
- **R5** — `Embedding(..., input_length=1)` on the drug branch (`:364`) while the input is `shape=(max_smi_len,)`. `input_length` is advisory in Keras 2 and generally ignored when it disagrees with the actual input; the mismatch is cosmetic but unverified.

### 5.4 Model construction is not separable from GAN training

`build_GAN_A` **begins** with:

```python
gan_smiles  = ganForDrug(XD_t)     # :358
gan_protein = ganForTarget(XT_t)   # :359
```

and `ganForDrug` ends with `dis_v = train(5000, 5, 500, XD); return dis_v` (`:150-151`). `ganForTarget` likewise runs `train(5000, 10, 500, XT)`. `[CODE]`

**[INFERENCE] There is no code path that constructs the DTA model without first running 5,000 adversarial iterations per GAN** — two GANs for variants A and B, one for variant C. The `build_dis`/`build_gen` closures are nested inside `ganFor*` and unreachable from outside without modifying source, which this stage forbids and Stage 06 should also avoid.

Consequences for Stage 06:

- The cheapest honest "can the model be constructed?" test **costs 10,000 GAN iterations** (variant A) or 5,000 (variant C).
- Variant C is therefore the **cheapest construction probe**, and it exercises the drug-branch transfer — the exact site of Stage 3's C1.
- `train()` calls `gen_v.predict(z)` every iteration (`:115`), which in Keras 2 carries per-call overhead; 5,000 iterations at batch 5 is many small graph executions. **Wall-clock is [UNKNOWN]** and is itself worth measuring.
- GAN pretraining re-runs **inside the innermost grid×fold loop** (Stage 3 §E5), so a full run multiplies this by (grid points × 5 folds × 2 passes).

---

## PART 6 — First-Smoke-Test Design

**Objective:** answer only — does it import, does the loader run, does one batch materialise, does the model instantiate, does one forward pass execute, does one loss compute. **Not** a paper result.

**Constraints honoured:** uses the repository's own data and its own documented invocation; introduces no synthetic dataset (the repository provides no such mechanism); modifies no production source.

### 6.1 Staged commands — to be run in Stage 06, NOT now

All from `DCGAN-DTA/DCGAN-DTA/`, in a Keras-2 environment (§1.2).

**S1 — import only.** Confirms the framework window and the four undeclared dependencies.
```
python -c "import tensorflow as tf, keras, sklearn, seaborn, pandas, matplotlib; print(tf.__version__, keras.__version__)"
```

**S2 — module import.** Reaches every module-scope side effect (`tf2.ConfigProto`, `sns.set_theme`) without running anything.
```
python -c "import run_experiments; print('module import OK')"
```

**S3 — loader + tensor shapes.** Exercises `parse_data` and `read_sets` end-to-end; captures the §4.3 memory envelope. Uses only public functions, no source change.
```
python -c "
from arguments import argparser; from datahelper import DataSet; import numpy as np, sys
sys.argv=['x','--dataset_path','data/pdb/','--max_seq_len','2000','--max_smi_len','200',
          '--is_log','0','--problem_type','1','--log_dir','logs/']
F=argparser(); F.charsmiset_size=64; F.charseqset_size=25
d=DataSet('data/pdb/',1,2000,200)
XD,XT,Y,XD_t,XT_t=d.parse_data(F)
for n,a in [('XD',XD),('XT',XT),('XD_t',XD_t),('XT_t',XT_t)]:
    a=np.asarray(a); print(n,a.shape,a.dtype,a.nbytes//2**20,'MB')
Y=np.asarray(Y); print('Y',Y.shape,'observed',int((~np.isnan(Y)).sum()))
te,tr=d.read_sets(F); print('test',len(te),'train folds',[len(f) for f in tr])
"
```

**S4 — model construction (variant C, cheapest).** Deliberately variant C: one GAN instead of two, and it still exercises the drug-branch `layers[-3]` transfer that §5.3 re-adjudicated. Expect 5,000 GAN iterations first; **time it**.
```
python -c "
from arguments import argparser; from run_experiments import build_GAN_C; import sys,time
sys.argv=['x','--max_seq_len','2000','--max_smi_len','200','--log_dir','logs/']
F=argparser(); F.charsmiset_size=64; F.charseqset_size=25
import numpy as np
from datahelper import DataSet
d=DataSet('data/pdb/',1,2000,200); XD,XT,Y,XD_t,XT_t=d.parse_data(F)
t=time.time(); m=build_GAN_C(F,np.asarray(XD_t),32,4,8); print('built in',round(time.time()-t,1),'s')
print([ (l.name, l.input_shape, l.output_shape) for l in m.layers ][:8])
print('trainable weights:',len(m.trainable_weights))
"
```

**S5 — one forward pass + one loss**, on a handful of real pairs from the real test fold.
```
python -c "
# ... construct m as in S4, then:
import numpy as np
from run_experiments import prepare_interaction_pairs
# take 8 pairs from fold file, predict, and evaluate the compiled MSE+cindex_score
"
```
(S5's exact body should be finalised in Stage 06 once S3/S4 confirm shapes; it must call only `m.predict` and `m.evaluate` on real indices.)

### 6.2 Evidence to capture

Framework versions; full `model.summary()`; the resolved `layers[-3]` name, `input_shape`, `output_shape` and kernel shape; `len(model.trainable_weights)` (settles U5/R4); peak RSS during `parse_data`; wall-clock of one `ganFor*` call; the exact traceback of any failure.

---

## PART 7 — Execution Blockers

### A. Environment blockers

| ID | Layer | Problem | Evidence | Stage 06 can test? | Source change? |
|---|---|---|---|:--:|:--:|
| **E1** | env | No TensorFlow/Keras installed; ambient Python 3.13.2 is outside the ≤3.11 window | `[DATA]` §1.5 | **yes — prerequisite** | no |
| **E2** | env | Exact working TF/Keras version unestablished; only `matplotlib 3.5.2` is pinned anywhere | `[DOC]` README; `[INFERENCE]` §1.2 | **yes** | no |
| **E3** | env | seaborn, sklearn, pandas, Pillow imported but undeclared in README | `[CODE]` §1.3 | yes | no |
| **E4** | env | `reset_keras()` sets `visible_device_list="0"` ⇒ requires GPU device 0 | `[CODE]` `:51` | **not in smoke scope** — called only after a full grid-point completes (`:711`, `:843`) | no |
| **E5** | env | `--log_dir` default `/tmp` and `--dataset_path` default `/data/pdb/` are absolute POSIX paths | `[CODE]` `arguments.py:72,102` | yes | no — pass flags |

### B. Data blockers

| ID | Layer | Problem | Evidence | Stage 06 can test? | Source change? |
|---|---|---|---|:--:|:--:|
| **D1** | data | BindingDB has no `folds/test_fold_setting{2,3}` ⇒ no cold-start possible there | `[DATA]` Stage 4 §5.2 | n/a — PDBbind unaffected | no |
| **D2** | data | `parse_data` materialises ≈970 MB (float64 encodings) before any model exists | `[DATA]` §4.3 | **yes — measure** | no |
| **D3** | data | Cold-start folds reusable but not regenerable (no logP rule ships) | `[DATA]` Stage 4 §6.1 | no | no |
| **D4** | data | `B`/`O`/`U` absent from both feature JSONs ⇒ latent `KeyError` in `DataGenerator.get_pro_vec` (variants B/C only) | `[DATA]` Stage 4 §4.1; Stage 3 U10 | **yes — static scan** | no |

*No missing file blocks PDBbind.* `[DATA]`

### C. Code blockers

| ID | Layer | Problem | Evidence | Stage 06 can test? | Source change? |
|---|---|---|---|:--:|:--:|
| **C1′** | code | ~~Transferred `layers[-3]` dimensionally incompatible~~ — **WITHDRAWN**, see §5.3 | `[CODE]` §5.2–5.3 | yes — confirm via `model.summary()` | **no** |
| **C2** | code | Three flags have no default and are `len()`-ed ⇒ `TypeError` if omitted | `[CODE]` §3 | yes | no — pass flags |
| **C3** | code | `--max_seq_len`/`--max_smi_len` default to `0` | `[CODE]` §3 | yes | no — pass flags |
| **C4** | code | Model construction inseparable from 5,000–10,000 GAN iterations | `[CODE]` §5.4 | **yes — measure cost** | no |
| **C5** | code | Variants B/C use `fit_generator`/`predict_generator` (Keras-3-removed) | `[CODE]` `:660,663` | yes | no |
| **C6** | code | Transferred layer's effective trainability unresolved | `[CODE]` §5.3 R4 | **yes — `trainable_weights`** | no |
| **C7** | code | TIFF save needs Pillow with LZW support | `[CODE]` `:833` | not in smoke scope | no |

### D. Scientific / reproduction blockers

*These do not prevent execution. They prevent faithful reproduction, and must not be conflated with A–C.*

| ID | Problem | Evidence |
|---|---|---|
| **S1** | Reported CI is a per-batch Keras average, not Eq. (1); `emetrics.get_cindex` is dead code | Stage 3 §A4 |
| **S2** | Stopping epoch and reported value are `max` over epochs on the **test** fold | Stage 4 §6.3 |
| **S3** | 42.4% of BindingDB eval proteins are inside the GAN pretraining corpus (12.5% PDBbind) — all three variants | Stage 4 §7.3 |
| **S4** | GAN pretraining corpora begin with the complete Davis dataset; UniProt/ChEMBL claim unsupported | Stage 4 §7.4 |
| **S5** | Three of five experiment families have no code path (cold-start generation, adversarial controls, all baselines, merge ablation) | Stage 3 §C2 |
| **S6** | The paper specifies **no training objective at all** | Stage 2 §C |
| **S7** | BLOSUM is 20-dim, not the paper's 25; FC widths follow Fig. 1 over the prose | Stage 3 §C6, §C8 |
| **S8** | 650 zero-length SMILES in the GAN corpus; 46.98% of BindingDB censored at pKd 5.0 | Stage 4 §7.5, §3.1 |

---

## PART 8 — What Can and Cannot Be Reproduced

### 8.1 Code execution feasibility

**NOT DETERMINABLE WITHOUT EXECUTION** — leaning favourable.

Every *static* obstacle is a supplied-flag or an environment-provisioning matter. All required files are present; the fold files are valid; no preprocessing is missing; and the one finding that Stage 3 graded CRITICAL is withdrawn in §5.3. What remains are R1–R5, all genuinely runtime-only.

### 8.2 Pipeline reproduction feasibility

**FEASIBLE WITH DOCUMENTED DEVIATIONS** — for the warm-start family on both datasets and the cold-drug family on PDBbind.

Deviations that must be declared: the paper describes no test set while the code uses one; reported CI is a batch-averaged approximation; epoch selection is on the test fold; FC widths follow the figure not the prose; BLOSUM is 20-dim; GAN pretraining overlaps evaluation entities.

**BLOCKED** for the other three experiment families (S5) — no code path exists.

### 8.3 Result reproduction feasibility

**BLOCKED** for the paper's figures as published.

Independent of execution: Figs. 5–8 and every baseline bar have no implementation (S5); the objective is unspecified (S6); the aggregation convention is undetermined (Stage 2 #5); and no seed/config/log/result artifact ships. Even a perfectly executing repository would reproduce **only Figs. 2–3's DCGAN-DTA bars**, and only up to the estimator in S2.

---

## PART 9 — Stage 06 Scope

| # | Item | Specification |
|---|---|---|
| 1 | **Environment candidate** | A **separate** Python **3.10** interpreter with **TensorFlow 2.15 / Keras 2.15** as the first candidate — the newest release inside the derived window (§1.2), maximising the chance of a working wheel. Fall back down the 2.x line on failure. Plus seaborn, scikit-learn, pandas, Pillow. **CPU-only is acceptable for S1–S5.** Do **not** modify the ambient 3.13 environment. |
| 2 | **First import test** | S1 → S2 (§6.1). Captures framework versions and any undeclared-dependency `ImportError`. |
| 3 | **First dataset-loader test** | S3. Expect `XD (4231,200)`, `XT (1606,2000)`, `XD_t (50068,200)`, `XT_t (50202,2000)`, `Y (4231,1606)` with **5,014** observed; folds `834` test and `5×836` train. |
| 4 | **First tensor-shape test** | Part of S3 — assert dtypes are `float64` and total ≈970 MB (§4.3). |
| 5 | **First model-construction test** | S4, **variant C first** (one GAN, still exercises the drug transfer), then variant A. Capture `model.summary()` and the `layers[-3]` resolved shapes. |
| 6 | **First forward-pass test** | S5 — `predict` on ~8 real pairs from `test_fold_setting1`, then `evaluate` for one MSE + `cindex_score`. |
| 7 | **Evidence to capture** | §6.2. |
| 8 | **What failure would mean** | An `ImportError` ⇒ the version window needs narrowing (E2). A construction failure at `layers[-3]` ⇒ R2's layer-indexing assumption is wrong under the tested Keras, **not** that Stage 3's C1 was right — the diagnosis would differ and must be recorded from the traceback. A loader failure ⇒ a Stage 4 conclusion needs revisiting. |
| 9 | **What failure would NOT mean** | It would **not** mean the paper is unreproducible (that is settled separately in §8.3 and is already BLOCKED for other reasons). It would **not** license editing source — a failure is a finding to record, and any patch is a Stage 07 decision. It would **not** invalidate Stage 4's data findings, which are independent of execution. |

**RUNTIME-ONLY UNKNOWNS carried into Stage 06:**

| ID | Unknown |
|---|---|
| **R1** | Which TF/Keras version inside 2.3–2.15 actually constructs and runs these graphs |
| **R2** | Whether `model.layers` excludes `InputLayer` under the tested version, on which §5.3's indexing depends |
| **R3** | Whether `build_GAN_A/B/C` construct successfully end-to-end |
| **R4** | Effective trainability of the spliced layer (`len(model.trainable_weights)`) — Stage 3 U5 |
| **R5** | Whether `Embedding(input_length=1)` against a `(max_smi_len,)` input is ignored or raises |
| **R6** | Wall-clock of one `ganFor*` call (5,000 iterations, batch 5/10) — governs whether any full run is affordable |
| **R7** | Peak RSS during `parse_data` |

**Out of scope for Stage 06:** full training, any paper figure, Kaggle or any remote compute, and **any source modification**.

---

## PART 10 — Development Dataset Decision

**DCGAN-DTA development dataset = PDBbind.** Reaffirmed on Stage 05 evidence, which adds three reasons beyond Stage 4's: `--is_log 0` means no transform to misconfigure; all required files are present and the folds are valid; and at 5,014 observed pairs the per-fold `fit` is the cheapest of the four datasets — material given that GAN pretraining re-runs inside the fold loop (§5.4).

> **PDBbind is a pipeline-validation dataset, not the scientific comparison dataset.** It exists to establish that the pipeline imports, loads, constructs, and runs one forward pass. Its 99.93% missingness and **1.18 observations per drug** (Stage 4 §9) make it a weak evidential base.
>
> **BindingDB remains a later-stage dataset**, subject to its known issues: 46.98% censored at pKd 5.0, 42.4% GAN-pretraining protein overlap, 7.0% of SMILES silently truncated, warm-start only (no settings 2/3), and an `--is_log` default that is wrong for it.

---

## PART 11 — Feasibility Matrix and Verdict

| Component | Status | Evidence |
|---|---|---|
| Environment | **RECOVERABLE** | window derived (§1.2), no version verified; nothing installed |
| Dependencies | **RECOVERABLE** | all on PyPI; four undeclared; only matplotlib pinned |
| Dataset | **READY** | all files present, counts match Sup Table 1 exactly |
| Preprocessing | **READY** to run / **BLOCKED** to audit | encoders ship; the script that produced `Y`/`ligands.txt` does not |
| Fold files | **READY** | all four valid, complete, mutually exclusive (Stage 4 §5.2) |
| Model | **READY** (static) | all six transfer sites channel-consistent (§5.3); R2–R5 runtime-only |
| Loss | **REQUIRES RECONSTRUCTION** | code uses MSE; the paper specifies no objective (S6) |
| Training | **READY** to run / deviant to report | S2's estimator |
| Evaluation | **RECOVERABLE** | CI is batch-averaged; exact `get_cindex` exists but is unused |
| Results | **BLOCKED** | no results, logs, seeds or configs ship; 3 of 5 families have no code |

**Effort to reach a green smoke test: LOW–MEDIUM.** Provisioning a Python 3.10 + TF 2.15 environment and supplying seven flags. The uncertainty is version-hunting (E2), not engineering.

**Effort to reach a faithful paper reproduction: VERY HIGH** — and partly impossible (S5, S6).

### Final verdict: 🟡 **YELLOW**

Upgraded from Stage 3's 🔴 RED, on one specific ground: the CRITICAL construction defect that anchored that verdict does not exist (§5.3). The released implementation now appears *executable* pending environment provisioning. It remains **YELLOW, not GREEN**, because faithful reproduction still requires controlled reconstruction — an unspecified objective, a batch-averaged CI, a test-fold-selected epoch, and three absent experiment families.

*The RED grade for **result** reproduction (§8.3) is unchanged.*

---

## PART 12 — Dated Corrections to Earlier Stages

Per the Stage 05 operating rules, prior findings are preserved and amended, not overwritten.

> **Correction 05-A — 2026-09-25 — supersedes Stage 3 §C1 (graded CRITICAL) and Stage 3 §F1.**
> **Original:** "The transferred discriminator layer is dimensionally incompatible with the drug branch in all three variants… built on a **1-channel** input (`Reshape((200,1))`)… applied to a **32-channel** `Embedding` tensor."
> **Correction:** `layers[-3]` is built on **32** channels, not 1. `Reshape((200,1))` is `layers[0]`; four `Conv1D` layers (4→8→16→32) intervene. All six transfer sites across variants A/B/C are channel-consistent (§5.3). The blocker is **withdrawn**. Stage 3's derived recommendation — that Stage 4 should first determine "whether `build_GAN_A/B/C` construct at all… this single result determines whether anything downstream is worth pursuing" — loses its premise, though construction remains worth testing as **R3**.
> **Status of the original finding:** superseded. Retained in `audit/03-*` unmodified.

> **Correction 05-B — 2026-09-25 — refines Stage 3 §A9 and Stage 4 §6.3.**
> Stage 4 already narrowed Stage 3's claim; Stage 05 confirms the mechanism from `arguments.py` and `:786-800`: the grid point is selected on `val_sets`, while `EarlyStopping` and `max(val_cindex_score)` operate on whichever set the second call passes — the test fold. Both statements stand as amended in Stage 4.

---

## Stage 05 Status

**COMPLETE — PARTIAL / RUNTIME UNKNOWN**

### What is known

All required PDBbind files are present and the fold files are valid. The full path from raw files to the first model-input tensor is traced and unobstructed (§4). The framework window is derived from four independent API usages: **TF ≥ 2.3, < 2.16, Keras 2, Python ≤ 3.11** (§1.2). The complete argument surface is mapped, including three dangerous defaults and five declared-but-unread flags (§3). Java is not required. **And the `layers[-3]` transfer is channel-consistent at all six sites — Stage 3's CRITICAL blocker is withdrawn** (§5.3).

### What is blocked

Nothing blocks *execution* statically. **Result** reproduction is blocked independently of execution: three of five experiment families have no code path, the training objective is unspecified in the paper, and no seed, config, log or result artifact ships (§8.3).

### What requires execution

R1–R7 (§9): the working framework version; whether `model.layers` excludes `InputLayer` under it; whether the variants construct; the spliced layer's trainability; the `input_length=1` behaviour; the cost of 5,000 GAN iterations; and peak loader RSS.

### What Stage 06 should test

S1→S5 in order (§6.1), variant C before variant A, on PDBbind, CPU-only, capturing §6.2's evidence. Stop at one forward pass and one loss.

### What must NOT be changed yet

Any file under `DCGAN-DTA/DCGAN-DTA/` — including the three no-default flags, the `/data/pdb/` and `/tmp` defaults, `reset_keras`'s GPU pin, and the `layers[-3]` splice. The ambient Python 3.13 environment. Every fold file and dataset. All Stage 02–04 reports.

### Next Single Step

**Provision an isolated Python 3.10 + TensorFlow 2.15 (Keras 2) environment and run smoke steps S1 and S2 only — import checks, nothing else.**

This is the prerequisite E1/E2 gate: until the framework window is narrowed from *derived* to *verified*, every downstream test (S3–S5) would attribute its failure to the wrong cause. It executes nothing from the repository beyond module import and modifies no source.

---

*End of Stage 5 — DCGAN-DTA. READ-ONLY: no source file, dataset, or fold file was modified; no package installed; no environment created; nothing executed.*

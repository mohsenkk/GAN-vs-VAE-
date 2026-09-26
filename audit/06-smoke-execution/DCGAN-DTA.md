# Stage 6 — Minimal Instrumented Smoke Execution: DCGAN-DTA

**Inputs:** Stage 02 forensic · Stage 03 traceability · Stage 04 dataset/fold · Stage 05 feasibility · `THESIS_IMPLEMENTATION_PLAN.md`
**Implementation:** `DCGAN-DTA/DCGAN-DTA/` (upstream `github.com/mojtabaze7/DCGAN-DTA`, HEAD `453fe16`)
**Development dataset:** PDBbind (pipeline-validation only — see §10)
**Date:** 2026-09-25

**Mode:** execution permitted for the first time. **No source file, dataset, or fold file was modified.** No hard-coded constant was edited. Both virtual environments and all downloaded wheels live outside the repository, in the session scratchpad.

**Evidence tags:** `[RUNTIME]` (observed this stage) · `[CODE]` · `[DATA]` · `[DOC]` · `[INFERENCE]`

---

## 1. Environment

### 1.1 Final environment

| Component | Version | Source of choice |
|---|---|---|
| OS | Windows 11 Enterprise 10.0.26200 | — |
| Python | **3.10.0** (`py -3.10`, pre-installed) | Stage 05 D5 — only version satisfying both projects |
| TensorFlow | **2.10.1** | see §1.2 |
| Keras | **2.10.0** (bundled) | follows TF |
| numpy | **1.26.4** (pinned `<2`) | see §1.3 — **required** |
| pandas | 2.3.3 | transitive |
| scikit-learn | 1.7.2 | undeclared repo dep (Stage 05 E3) |
| seaborn | 0.13.2 | undeclared repo dep |
| Pillow | 12.3.0 | undeclared repo dep |
| matplotlib | 3.10.9 | transitive |
| scipy | 1.15.3 | transitive |
| pip | 24.3.1 (upgraded from 21.2.3) | see §1.4 |
| GPU | NVIDIA GeForce **MX330**, cc **6.1**, 2048 MiB, driver 511.69 | hardware |
| TF GPU visibility | **none — CPU-only execution** | see §1.5 |

Environment root: `…/scratchpad/venv-dcgan` (outside the repository).

### 1.2 TensorFlow version decision — a documented deviation from plan D6

| | |
|---|---|
| **Package** | `tensorflow==2.10.1` |
| **Plan said** | D6 named TF **2.15** as first candidate |
| **Chosen instead** | **2.10.1** |
| **Reason** | On native Windows, **TF ≥ 2.11 is CPU-only**; 2.10 is the last release with native Windows GPU support. Both versions sit inside Stage 05's derived window (TF ≥ 2.3, < 2.16, Keras 2), so the compatibility analysis is unchanged, and 2.10.1 keeps a GPU path open for later stages at no cost. The smoke test itself needs only CPU. |
| **Evidence** | Stage 05 §1.2 derived window; TF release notes on Windows GPU support `[DOC]` |
| **Outcome** | `[RUNTIME]` Keras **2.10.0** confirmed — Keras 2, as the window requires. |

### 1.3 numpy — a required dependency-resolution choice

**This was a genuine runtime blocker, encountered and resolved.**

| | |
|---|---|
| **Package** | `numpy==1.26.4` |
| **Problem** | First smoke attempt aborted at `import tensorflow` with `AttributeError: _ARRAY_API not found` and *"A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6"* |
| **Root cause** | `[RUNTIME]` TF 2.10.1 declares its requirement as **`numpy (>=1.20)` — with no upper bound** (read from installed metadata). pip therefore legitimately resolved **numpy 2.2.6** as a transitive dependency of scikit-learn/pandas, which breaks TF's compiled C ABI. |
| **Evidence** | `importlib.metadata.requires('tensorflow')` → `numpy (>=1.20)`; traceback names the incompatibility explicitly |
| **Resolution** | Pin `numpy==1.26.4` (highest 1.x). This is **not** arbitrary version-shopping — it corrects a metadata defect in TensorFlow that the error message itself identifies. |
| **Verified** | `[RUNTIME]` TF imports cleanly at numpy 1.26.4 |

### 1.4 Network / pip findings (environmental, not repository-specific)

| Finding | Evidence |
|---|---|
| A local system proxy is enabled at `127.0.0.1:20808` (WinINET `ProxyEnable=1`), while `netsh winhttp` reports direct access | registry + `netsh` |
| pip **21.2.3** (bundled with Python 3.10.0) fails against it: `ValueError: check_hostname requires server_hostname` | traceback |
| Fixed by upgrading pip **in the venv** to 24.3.1 (a build tool, not a scientific dependency) | — |
| pip stalls indefinitely (0 KB/s, ~0 CPU) on **large** transfers through this proxy; small packages succeed | measured over 60 s windows |
| `curl` through the same proxy achieves **665 KB/s** on the same 435 MB wheel | measured |
| **Workaround used:** fetch the large wheel with `curl`, then `pip install <local wheel>` so pip still resolves and installs dependencies normally | — |

`[INFERENCE]` This is a property of this machine's network path, **not** of either repository. It will recur in Stage 07 and is worth recording for that reason alone.

### 1.5 GPU visibility

`[RUNTIME]` TF reports `built with CUDA: True` but `visible GPUs: none`, preceded by:

```
Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
```

**This is expected and acceptable.** TF 2.10 requires a CUDA 11.2 toolkit + cuDNN 8.1 installation, which Stage 06's rules direct me not to install unless genuinely required. Stage 05 §9 established that DCGAN-DTA needs the GPU **only** at `reset_keras()` (`:711`, `:843`), which runs after a complete grid point and is outside smoke scope. Every boundary reached below is CPU-legitimate.

---

## 2. Exact commands executed

All run with cwd = `DCGAN-DTA/DCGAN-DTA/`, interpreter = `…/venv-dcgan/Scripts/python.exe`.

**Non-default arguments supplied** (every one documented, none invented):

```
--dataset_path data/pdb/     # default is the ABSOLUTE "/data/pdb/" (Stage 05 §3, dangerous default)
--max_seq_len 2000           # default 0; value from README + paper
--max_smi_len 200            # default 0; value from README + paper
--is_log 0                   # README: PDBbind uses 0; Stage 04 confirmed values are already pKd
--problem_type 1             # warm-start; selects folds/test_fold_setting1.txt
--num_windows 32             # no default; len()-ed at :580. Smallest single grid point
--smi_window_lengths 4       # no default; smallest value in the README grid (4 8 16)
--seq_window_lengths 8       # no default; from the README grid
--batch_size 256             # repository default, matches paper
--num_epoch 100              # repository default, matches paper (not used — no training run)
--model C                    # Stage 05 D8: cheapest construction probe
--log_dir <scratchpad>/logs-dcgan/   # default "/tmp" is an invalid absolute POSIX path on Windows
```

`num_windows 32` and the two window lengths are reduced to **a single grid point each** purely to minimise work; all three are legitimate members of the README's documented grid, and no value outside the repository's own CLI was used.

The smoke script (in scratchpad, not the repository) calls only the repository's own public functions: `argparser`, `DataSet.parse_data`, `DataSet.read_sets`, `prepare_interaction_pairs`.

---

## 3. Boundary results

| Boundary | Result |
|---|---|
| Environment | **PASS** (after the numpy pin of §1.3) |
| Import | **PASS** |
| Argument parsing | **PASS** |
| Data loading | **PASS** |
| Feature construction | **PASS** |
| GAN construction | **NOT REACHED — deliberately, see §5** |
| Embedding transfer | **NOT REACHED in repository code**; layer semantics verified separately, §6 |
| Model forward | **NOT REACHED** |
| GPU memory | **not exercised** — CPU-only path; no GPU allocation occurred |

---

## 4. Runtime evidence — data pipeline

`[RUNTIME]` Every Stage 04 measurement was confirmed exactly. Nothing needed correcting.

### 4.1 Loader output

```
charsmiset_size=65  charseqset_size=25
Read data/pdb/ start
parse_data OK (3.8s)   np.asarray OK (0.4s)
  XD     (4231, 200)     float64      6.5 MiB
  XT     (1606, 2000)    float64     24.5 MiB
  XD_t   (50068, 200)    float64     76.4 MiB
  XT_t   (50202, 2000)   float64    766.0 MiB
  Y      (4231, 1606)    float64     51.8 MiB
XD max index 62  (Embedding input_dim = charsmiset_size+1 = 66)
XT max index 24  (Embedding input_dim = charseqset_size+1 = 26)
Y observed=5014  min=2.0000  max=11.9200  mean=6.4002
is_log=0 -> PDBbind values used VERBATIM (no transform)
label_row_inds/label_col_inds = 5014 pairs
```

| Quantity | Stage 04 predicted | Stage 06 observed | Match |
|---|---|---|:--:|
| XD | (4231, 200) | (4231, 200) float64 | ✔ |
| XT | (1606, 2000) | (1606, 2000) float64 | ✔ |
| XD_t | (50068, 200) | (50068, 200) float64 | ✔ |
| XT_t | (50202, 2000) | (50202, 2000) float64 | ✔ |
| Y | (4231, 1606) | (4231, 1606) float64 | ✔ |
| Observed pairs | 5,014 | **5,014** | ✔ |
| Affinity range | 2.0 – 11.92 | **2.0000 – 11.9200** | ✔ |
| Affinity mean | 6.4002 | **6.4002** | ✔ |
| Total array memory | ≈970 MB (Stage 05 §4.3) | **925.2 MiB** | ✔ close |

**`[RUNTIME]` The float64 encoding cost predicted in Stage 05 §4.3 is confirmed**: `XT_t` alone is **766 MiB**, because `label_sequence` returns `np.zeros(MAX_SEQ_LEN)` (float64) rather than an integer dtype. This is incurred before any model exists.

**`[RUNTIME]` Embedding index safety confirmed with margin**: XD max index 62 against `input_dim` 66; XT max index 24 against 26. DCGAN-DTA's `+1` guard (`charsmiset_size + 1`) gives it headroom that Co-VAE lacks.

### 4.2 Fold files

```
Reading data/pdb/ start
problem_type=1 -> folds/test_fold_setting1.txt
train folds=5 sizes=[836, 836, 836, 836, 836]  test=834
total=5014 unique=5014 dups=0
range=[0,5013]  covers 0..5013 exactly once: True
train/test index overlap = 0
DRUGS   train=3591 test=787 shared=147
TARGETS train=1452 test=491 shared=337
```

| Quantity | Stage 04 §5.2/§6.1 | Stage 06 observed | Match |
|---|---|---|:--:|
| Train fold sizes | 836 ×5 | 836 ×5 | ✔ |
| Test size | 834 | 834 | ✔ |
| Duplicates | 0 | 0 | ✔ |
| Covers 0..N−1 exactly once | ✔ | **True** | ✔ |
| Train/test index overlap | 0 | 0 | ✔ |
| Drugs train / test / shared | 3591 / 787 / 147 | **3591 / 787 / 147** | ✔ |
| Targets train / test / shared | 1452 / 491 / 337 | **1452 / 491 / 337** | ✔ |
| Classification | PAIR-WISE (warm) | consistent | ✔ |

**`[RUNTIME]` Stage 04's fold audit is confirmed exactly, entity counts included.**

### 4.3 One batch

```
prepare_interaction_pairs OK (0.00s)
  drugs  (256, 200)  float64
  prots  (256, 2000) float64
  labels (256,)      float64   min=2.440  max=11.680
```

A legitimate batch was produced from real fold indices via the repository's own function.

### 4.4 New runtime observation — a latent bug in `emetrics.py`

`[RUNTIME]` Python emitted, at import:

```
DCGAN-DTA\emetrics.py:38: SyntaxWarning: "is not" with a literal. Did you mean "!="?
  if pair is not 0:
```

`[CODE]` `emetrics.py:38` uses identity comparison against the literal `0`. In CPython this happens to work for small-int caching, but it is not language-guaranteed and is a genuine defect. **It sits in `emetrics.get_cindex`, which Stage 03 §A4 established is dead code** (`run_experiments.py:20` imports only `get_rm2`), so it has no effect on reported results. Recorded as a new, previously unreported finding.

---

## 5. Model construction — the boundary, and why it was not crossed

**`[CODE]` `[RUNTIME]` Model construction is inseparable from GAN pretraining in the supplied implementation.**

All three builders begin by calling a `ganFor*` function:

```
build_GAN_A :358-359   ganForDrug(XD_t) ; ganForTarget(XT_t)
build_GAN_B :404-405   ganForDrug(XD_t) ; ganForBlosumTarget(blosum_data)
build_GAN_C :447       ganForDrug(XD_t)
```

and each `ganFor*` ends with a hard-coded adversarial loop before returning its discriminator:

```
run_experiments.py:152   dis_v = train(5000,  5, 500, XD)   # drug GAN
run_experiments.py:256   dis_v = train(5000, 10, 500, XT)   # protein GAN
run_experiments.py:354   dis_v = train(5000, 10, 100, XT)   # BLOSUM GAN
```

**`[CODE]` Verified by signature inspection: `ganForDrug(XD)`, `ganForTarget(XT)` and `ganForBlosumTarget(XT)` take only the data array — none takes `FLAGS`.** No command-line argument reaches the iteration count, the batch size, or the reporting interval. A full grep of `FLAGS` usage confirms the first occurrence is at `:358`, inside `build_GAN_A`, after the GAN calls.

Therefore:

- **There is no legitimate CLI or configuration path that reduces this loop.** The only way to shorten it is to edit the literal `5000`, which Stage 06 explicitly forbids and which I did not do.
- `build_dis` / `build_gen` are **closures nested inside** `ganFor*`, so they cannot be reached from outside without modifying source.
- Per the Stage 06 operating rules, execution **stopped at the earliest safe boundary** — after feature assembly, before `build_GAN_C`.

`[INFERENCE]` The cheapest honest "can the model be constructed?" test in this repository costs **5,000 adversarial iterations** (variant C) or **10,000** (variants A and B), and Stage 03 §E5 established that this re-runs inside the innermost grid×fold loop. Runtime unknown **R6** (the wall-clock cost of one `ganFor*` call) therefore remains open and is the correct first target for a future stage that is permitted to run it.

---

## 6. `gan_smiles.layers[-3]` — Keras layer-semantics verification

### 6.1 Scope of this evidence — read before using it

This check **did not execute the repository**. It rebuilt, in a scratch script, the *same layer sequences* that `run_experiments.py`'s nested `build_dis` closures declare, to answer one question static reading could not: **under this installed Keras, does `model.layers` include the `InputLayer`, and therefore which layer does `layers[-3]` resolve to, with what kernel shape?** That is Stage 05's runtime unknown **R2**.

Layer shapes are fixed at build time, before any training, so untrained randomly-initialised layers are both sufficient and necessary for this question.

**It does NOT prove that the repository's `build_GAN_A/B/C` construct end-to-end.** That is **R3**, and it remains unverified because it is unreachable without the GAN loops of §5.

### 6.2 Result — R2 RESOLVED

`[RUNTIME]` Keras 2.10.0 / TF 2.10.1. **`InputLayer` is excluded from `model.layers`** in all three cases, exactly as Stage 05 §5.2 inferred.

**`ganForDrug.build_dis` — 8 layers:**

| idx | `[-n]` | layer | output |
|---:|---:|---|---|
| 0 | [-8] | `Reshape` | (None, 200, 1) |
| 1 | [-7] | `Conv1D` | (None, 200, 4) |
| 2 | [-6] | `Conv1D` | (None, 200, 8) |
| 3 | [-5] | `Conv1D` | (None, 200, 16) |
| 4 | [-4] | `Conv1D` | (None, 200, 32) |
| **5** | **[-3]** | **`Conv1D` `conv1d_4`** | **(None, 200, 64)** |
| 6 | [-2] | `Flatten` | (None, 12800) |
| 7 | [-1] | `Dense` | (None, 1) |

**Transfer-site table — all six sites:**

| Variant / branch | `layers[-3]` | **kernel shape** | Built for | Receives | Compatible |
|---|---|---|---:|---:|:--:|
| A/B/C drug | `Conv1D conv1d_4` | **(3, 32, 64)** | **32** | 32 — `Embedding(output_dim=32)` | **✔** |
| A protein | `Conv1D conv1d_9` | **(3, 32, 64)** | **32** | 32 — `Embedding(output_dim=32)` | **✔** |
| B protein | `Conv1D conv1d_14` | **(3, 20, 40)** | **20** | 20 — `Input(shape=(max_seq_len, 20))` | **✔** |
| C protein | *no transfer* | — | — | — | n/a |

**Functional re-application — the splice executes:**

```
OK   A/B/C drug   : in=(None, 200, 32)   ->  out=(None, 200, 64)
OK   A   protein  : in=(None, 2000, 32)  ->  out=(None, 2000, 64)
OK   B   protein  : in=(None, 2000, 20)  ->  out=(None, 2000, 40)
```

### 6.3 What this establishes

> **`[RUNTIME]` Stage 05's correction C-1 is confirmed by direct runtime evidence, and Stage 03's finding C1 is definitively refuted.**
>
> The kernel of `layers[-3]` is **(3, 32, 64)** — `(kernel_size, in_channels, out_channels)` — so it is built for **32** input channels, not the 1 that Stage 03 asserted. `Reshape((200,1))` is `layers[0]`; four `Conv1D` layers (4→8→16→32) intervene. Applying the layer to a 32-channel tensor succeeds and yields the expected shape.

`[RUNTIME]` Also confirmed incidentally: `Sequential.add(Input(shape=(200,)))` — the construct Stage 05 flagged as version-sensitive — is accepted without error by Keras 2.10.

---

## 7. GPU memory

**Not exercised.** The DCGAN-DTA smoke path is CPU-only (§1.5) and stops before any model is built, so no GPU allocation occurred and the 2 GiB limit was never approached. No OOM. GPU memory characterisation for this repository must wait for a stage permitted to run `build_GAN_*`.

---

## 8. Deviations from plan

| # | Deviation | Justification |
|---|---|---|
| D-a | TF **2.10.1** instead of D6's 2.15 candidate | §1.2 — both inside the derived window; 2.10 preserves a Windows GPU path |
| D-b | `numpy==1.26.4` pinned | §1.3 — required; TF 2.10.1's metadata omits the `<2` bound |
| D-c | pip upgraded 21.2.3 → 24.3.1 inside the venv | §1.4 — build tool; the old pip cannot reach the index through this proxy |
| D-d | Large wheel fetched with `curl`, installed from disk | §1.4 — pip stalls on large transfers through this proxy |
| D-e | Ran DCGAN before Co-VAE's environment finished provisioning | Co-VAE was blocked on a 2.27 GiB download; **executions remained strictly sequential** and DCGAN's path is CPU-only, so no GPU contention arose |
| D-f | `--num_windows 32`, `--smi_window_lengths 4`, `--seq_window_lengths 8` | Single grid point to minimise work; all are legitimate members of the README's documented grid |

**No deviation touched repository source, data, folds, or any hard-coded constant.**

---

## 9. Repository modification audit

`git status --short` after execution, and spot MD5s against the pre-execution baseline:

- `DCGAN-DTA/DCGAN-DTA/` — **unchanged** (no entry in `git status`)
- `data/pdb/folds/test_fold_setting1.txt` — MD5 `49024eef4d50f8346f524238ce24b778`, **identical to baseline**
- Both venvs, all wheels and all logs are under the session scratchpad — **nothing was written inside the repository**
- `--log_dir` was pointed at the scratchpad specifically so the repository's `logs/` stayed untouched

Full audit in the companion Co-VAE report §9 (single `git status` covers both).

---

## 10. Development dataset

**PDBbind remains the DCGAN-DTA development dataset** (plan D2), and Stage 06 strengthens the case: the loader runs in 4.2 s, every Stage 04 prediction held exactly, the fold files are valid at runtime, and `--is_log 0` means no transform to misconfigure.

> **PDBbind is a pipeline-validation dataset, not the scientific comparison dataset.** Its 99.93% missingness and 1.18 observations per drug make it a weak evidential base. **BindingDB remains a later-stage dataset**, with its 46.98% censoring, 42.4% GAN-pretraining protein overlap, warm-start-only folds, and an `--is_log` default that is wrong for it.

---

## Stage 06 Status

**COMPLETE — SMOKE PARTIALLY PASSED**

Every boundary that is reachable without triggering hard-coded GAN training **passed**. The remaining boundaries were not blocked by any defect — they were deliberately not crossed, because crossing them means running 5,000–10,000 adversarial iterations that no CLI argument can reduce.

### Runtime evidence

TF 2.10.1 / Keras 2.10.0 / Python 3.10.0 import the repository cleanly. `parse_data` completes in 3.8 s and reproduces **every** Stage 04 measurement exactly: 5,014 observed pairs, affinity 2.0000–11.9200, mean 6.4002, all five array shapes, and 925 MiB of float64 encodings. `read_sets` returns 5×836 train folds + 834 test, covering `0..5013` exactly once with zero overlap and the exact entity counts Stage 04 predicted. One real 256-pair batch was assembled. **`layers[-3]` has kernel shape (3, 32, 64)** and splices successfully onto a 32-channel tensor.

### Environment

Python 3.10.0 · TensorFlow 2.10.1 · Keras 2.10.0 · numpy 1.26.4 (pinned `<2`, required) · scikit-learn 1.7.2 · seaborn 0.13.2 · pandas 2.3.3 · Pillow 12.3.0 · CPU-only (no CUDA toolkit installed). Isolated venv outside the repository.

### Successful boundaries

Environment · Import · Argument parsing · Data loading · Feature construction · (separately) Keras layer-transfer semantics.

### First runtime blocker

**One genuine blocker was hit and resolved:** the NumPy 2.x / TF 2.10.1 C-ABI incompatibility (§1.3), caused by TF's own metadata omitting an upper bound. After pinning numpy `<2`, **no further runtime blocker was encountered on any boundary reached.**

The *stopping point* was not a blocker — it was the deliberate §5 boundary.

### GPU memory

**Not exercised.** CPU-only path, stopped before model construction. No allocation, no OOM.

### Deviations

Six, all listed in §8; none touched source, data, folds, or any hard-coded constant.

### What this proves

That the DCGAN-DTA repository imports and loads PDBbind correctly under a Keras-2 environment; that Stage 04's dataset and fold audit is accurate to the digit at runtime; that the fold files are valid and consumed as described; that one legitimate batch can be assembled; and that **the transferred `layers[-3]` is channel-compatible in fact, not merely in inference** — confirming Stage 05's C-1 and refuting Stage 03's C1.

### What this does NOT prove

It does **not** prove paper reproduction, numerical result reproduction, or the correctness of the published methodology. It does **not** prove that `build_GAN_A/B/C` construct end-to-end (**R3** — the §6 check verifies Keras semantics only, on a scratch replica). It does **not** prove that a forward pass or `fit()` works. It says nothing about GPU feasibility, about the cost of GAN pretraining (**R6**), about the absence of data leakage — Stage 04 in fact **confirmed** leakage at 42.4% of BindingDB proteins — or about either model's quality relative to the other.

### Next Single Step

**Measure the wall-clock cost of a single `ganForDrug(XD_t)` call (runtime unknown R6), by timing it to completion without modifying the hard-coded 5,000 iterations.**

This is the one measurement that determines whether *anything* downstream in this repository is affordable: GAN pretraining re-runs inside the innermost grid×fold loop, so its unit cost multiplies by (grid points × 5 folds × 2 passes). It is also the gateway to R3 — the call returns the discriminator whose `layers[-3]` would then be spliced for real. It needs explicit approval, because it is adversarial training, however small.

---

*End of Stage 6 — DCGAN-DTA. No repository source file, dataset, or fold file was modified; no hard-coded constant was edited; no training run to completion was performed.*

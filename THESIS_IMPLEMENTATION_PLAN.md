# Thesis Implementation Plan — DCGAN-DTA vs Co-VAE

**Created:** 2026-09-25 (at the close of Stage 04; no such file existed before)
**Thesis goal:** faithfully reproduce two DTA papers, compare them under a controlled experimental setup, then develop a hybrid representation as the proposed method.
**Workspace:** `D:\ut\thesis\refferences\final\GAN-vs-VAE-`

> This file records only what the completed audit stages have established. Future stages
> are listed as **NOT STARTED** and must not be marked otherwise until their artifacts exist.

---

## 1. Stage Status

| Stage | Skill | Output | Status |
|---|---|---|---|
| 01 | — | — | **does not exist** (numbering starts at 02) |
| 02 | `paper-forensic-audit` | `audit/02-paper-forensic/{Co-VAE,DCGAN-DTA}.md` | ✅ **COMPLETE** (both papers) |
| 03 | `paper-code-traceability` | `audit/03-paper-code-traceability/{Co-VAE,DCGAN-DTA}.md` | ✅ **COMPLETE** (both papers) |
| **04** | **`dataset-fold-audit`** | **`audit/04-dataset-fold/{Co-VAE,DCGAN-DTA}.md`** | ✅ **COMPLETE** — 2026-09-25 |
| **05** | **`reproduction-feasibility`** | **`audit/05-reproduction-feasibility/{Co-VAE,DCGAN-DTA}.md`** | ✅ **COMPLETE** — 2026-09-25. DCGAN-DTA: *PARTIAL / RUNTIME UNKNOWN*. Co-VAE: *STAGE 06 BLOCKED* (Tier 2) |
| **06** | — | **`audit/06-smoke-execution/{Co-VAE,DCGAN-DTA}.md`** | ✅ **COMPLETE** — 2026-09-25. Co-VAE: **SMOKE PASSED**. DCGAN-DTA: **SMOKE PARTIALLY PASSED** (stopped at the GAN-training boundary by design) |
| **07A** | — | **`audit/07-training-feasibility/Co-VAE.md`** | ✅ **COMPLETE** — 2026-09-25. Feasible boundary **`--batch_size 128`**; the published 256 **OOMs** |
| **07B** | — | **`audit/07-training-feasibility/DCGAN-DTA.md`** | ✅ **COMPLETE** — 2026-09-25. `ganForDrug` completed all 5000 iterations in **404.29 s on CPU**. **R6 resolved** |
| 08 | — | *(none yet)* | ⬜ **NOT STARTED** — first stage that could modify source, only on explicit approval |
| — | hybrid method | *(none yet)* | ⬜ **NOT STARTED** — explicitly out of scope until reproduction lands |

**Execution status: as of Stage 07B (2026-09-25), both implementations have been executed.** Stages 02–05 were read-only by design. Stage 06 reached the smoke boundaries; **07A** ran one Co-VAE training step (forward + backward + `optimizer.step()`); **07B** ran one full DCGAN-DTA `ganForDrug` call (5000 iterations). **No source file, dataset, fold file, architecture, or hard-coded constant has ever been modified.** No multi-step training loop, no epoch, and no reproduction run has been performed.

---

## 2. Stage 04 Summary

Read-only audit of all four local datasets. Statistics computed directly from the files with
numpy/pandas; no file in the repository was modified, no dependency installed, no model executed.

### 2.1 What the code actually consumes

| | Co-VAE | DCGAN-DTA |
|---|---|---|
| Fold files | **never opened** — the only call site is commented out; folds are generated at runtime from `random.seed(1000)` | **read** via `read_sets`, keyed on `--problem_type` |
| Inert shipped files | 7 of 12 per dataset (`Y`, all similarity matrices, `ligands_can.txt`, both fold files) | only `auc.jar` |
| Affinity source | `…Davis…v1.txt` (Davis) / `kiba_binding_affinity_v2.txt` (KIBA) | `Y` pickle (`latin1`) |

### 2.2 Measured dataset facts

| | Davis | KIBA (runtime) | BindingDB | PDBbind |
|---|---:|---:|---:|---:|
| Drugs × Targets | 68 × 442 | **1,954 × 217** | 9,864 × 1,088 | 4,231 × 1,606 |
| Observed pairs | 30,056 | **109,296** | 42,203 | 5,014 |
| Missing | **0.00%** | 74.22% | 99.61% | 99.93% |
| Affinity | pKd 5.0–10.80 | KIBA 0–17.20 | pKd 2.0–9.0 | pKd 2.0–11.92 |
| Censoring at modal value | **69.64%** @ 5.0 | 13.4% @ 11.2 | **46.98%** @ 5.0 | **0.90%** @ 8.7 |
| Unique sequences / targets | **379 / 442** ⚠ | 217 / 217 | 1,088 / 1,088 | 1,579 / 1,606 |
| Fold files valid at runtime | ✔ (unused) | ✘ **out of range** | ✔ | ✔ |
| Split type in fold files | pair-wise | pair-wise | pair-wise | pair-wise (s1), **cold-drug (s2, s3)** |

### 2.3 Findings that change the plan

1. **Co-VAE's KIBA fold files are invalid** — 118,254 indices against a 109,296-pair runtime matrix (8,958 out of range). They were built against the paper's unfiltered matrix, before the undocumented length filter existed.
2. **Davis has 442 target ids but 379 unique sequences** (18 collapse groups, 81 ids). Since the model consumes only sequences, the paper's flagship *new-target* protocol is defeated for **~17.8% of test targets**. This is a property of the data, and it will recur in the hybrid method.
3. **DCGAN-DTA's cold-start splits are real** — `test_fold_setting2/3` are confirmed drug-disjoint. Stage 03's U6 is closed. The generating logP rule is still absent, so they can be *reused* but not *regenerated*.
4. **The GAN pretraining corpora begin with the complete Davis dataset** — first 68 ligand keys and first 442 protein keys are byte-identical to Co-VAE's Davis files, in order. Stage 03's U2 is substantially resolved: they are not clean UniProt/ChEMBL extracts.
5. **42.4% of BindingDB evaluation proteins are in the GAN pretraining corpus** (461/1,088). Confirmed unsupervised representation leakage affecting **all three variants**, not only variant B as Stage 03 reported.
6. **Co-VAE's AUC on KIBA is degenerate** — threshold `> 7` yields 109,286 positives against **10** negatives. Entity-wise folds will often contain no negative class, raising `ValueError`.
7. **No single dataset can serve both methods** without adaptation: incompatible loaders, incompatible fold mechanisms, missing auxiliary files (`Y`, feature JSONs, pretraining corpora) for Davis/KIBA, and KIBA's non-pKd scale.

---

## 2A. Stage 05 Summary

Read-only feasibility audit. Nothing installed, no environment created, nothing executed.

### 2A.1 Environment findings

| | Co-VAE | DCGAN-DTA |
|---|---|---|
| Version documented anywhere | **none** — no README, no requirements file | README lists Python/TF/Keras/NumPy; **only `matplotlib 3.5.2` is pinned** |
| **Python ceiling** | **≤ 3.10** — `random.sample(set, k)` removed in 3.11 (`run_experiments.py:41`) | **≤ 3.11** — transitive, via the TF window |
| **Framework window** | **[UNKNOWN]** — no PyTorch version established | **TF ≥ 2.3, < 2.16, Keras 2** — derived from four API usages |
| Derivation | `Conv1DTranspose` ⇒ TF ≥ 2.3; `fit_generator`/`predict_generator` (variants B/C) and `tf.compat.v1.keras` in `reset_keras` (**all** variants) ⇒ Keras 2 ⇒ TF < 2.16 | |
| **GPU** | **unconditional, on the forward path** — `torch.cuda.FloatTensor` (`model.py:36`), `.cuda()` ×5 | needed **only** at `reset_keras()` (`:711`, `:843`), after a grid point completes ⇒ **CPU smoke test is viable** |
| Undeclared dependencies | all six (no README exists) | seaborn, scikit-learn, pandas, Pillow |
| Java | not required (`get_aupr` is sklearn) | not required (`auc.jar` path is dead code) |

**Ambient machine state (measured):** Python **3.13.2**, numpy 2.2.3, pandas 2.2.3, matplotlib 3.10.3. **`torch`, `tensorflow`, `keras`, `sklearn`, `seaborn`, `tqdm` are NOT installed. Java absent.** The ambient interpreter is outside both projects' windows; Stage 06 needs a separate interpreter. **Python 3.10 is the only version satisfying both projects simultaneously.**

### 2A.2 Findings that change the plan

1. **DCGAN-DTA's CRITICAL blocker does not exist.** Stage 3's C1 claimed `gan_smiles.layers[-3]` was built on a 1-channel input and applied to a 32-channel tensor. It is built on **32** channels — `Reshape((200,1))` is `layers[0]` and four `Conv1D` layers (4→8→16→32) intervene. **All six transfer sites across variants A/B/C are channel-consistent.** See correction C-1.
2. **DCGAN-DTA model construction cannot be separated from GAN training.** `build_GAN_A` calls `ganForDrug`/`ganForTarget`, which each run **5,000 adversarial iterations** before returning. Variant C (one GAN) is the cheapest construction probe and still exercises the drug-branch transfer.
3. **Co-VAE cannot reach a forward pass on CPU.** Unlike DCGAN-DTA, its CUDA dependency is on the hot path. Its smoke test splits into **Tier 1 (CPU: import → loader → one batch)** and **Tier 2 (GPU: construct → forward → loss)**.
4. **Co-VAE's `--lamda` does have a default** (`-5`); the defect is a scalar default under `nargs='+'`. See correction C-2.
5. **Co-VAE's `.exp_()` question resolves statically** — `Tensor.mul()` is out-of-place, so `logvar` is never mutated. See correction C-3.
6. **Co-VAE's length flags silently re-weight the loss**: `run_experiments.py:136` uses `max_smi_len / max_seq_len` as a coefficient on the target VAE term. Changing input length to fit a dataset changes the objective.
7. **Both models are dimensionally self-consistent on static trace.** No shape defect exists in either.
8. **Three dangerous defaults in DCGAN-DTA**: `--dataset_path` is the absolute `/data/pdb/`; `--max_seq_len`/`--max_smi_len` default to `0`; `--is_log` defaults to `0`, which is **wrong for BindingDB** (raw nM Kd into an MSE loss, silently).

### 2A.3 Reproduction feasibility — three separate conclusions per paper

| | DCGAN-DTA | Co-VAE |
|---|---|---|
| **Code execution** | NOT DETERMINABLE WITHOUT EXECUTION *(leaning favourable — every static obstacle is a flag or a provisioning matter)* | **BLOCKED on CPU**; not determinable given a GPU |
| **Pipeline reproduction** | FEASIBLE WITH DOCUMENTED DEVIATIONS (warm-start both datasets; cold-drug PDBbind) | FEASIBLE WITH DOCUMENTED DEVIATIONS (Davis); **BLOCKED** (KIBA as published) |
| **Result reproduction** | **BLOCKED** — 3 of 5 families have no code, objective unspecified, no artifacts ship | **BLOCKED** — 4 of 5 families have no code, MAE unimplemented, KIBA matrix differs |
| **Verdict** | 🟡 **YELLOW** *(upgraded from Stage 3's RED — see correction C-1)* | 🔴 **RED** *(unchanged)* |

---

## 2B. Stage 06 Summary — First Execution

Both repositories executed for the first time. **No source, dataset, fold file, or hard-coded constant was modified.**

### 2B.1 Environments (both isolated, outside the repository, ambient Python 3.13 untouched)

| | Co-VAE | DCGAN-DTA |
|---|---|---|
| Python | **3.10.0** | **3.10.0** |
| Framework | **torch 1.13.1+cu116** (CUDA runtime 11.6) | **TensorFlow 2.10.1 / Keras 2.10.0** |
| numpy | **1.26.4** (pinned `<2`) | **1.26.4** (pinned `<2` — **required**, see 2B.4) |
| Other | pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.9, tqdm 4.70.1 | scikit-learn 1.7.2, seaborn 0.13.2, pandas 2.3.3, Pillow 12.3.0 |
| Device | **GPU — NVIDIA GeForce MX330, cc 6.1, 2048 MiB, driver 511.69** | **CPU-only** (no CUDA toolkit installed; not needed for smoke) |
| pip | 24.3.1 (upgraded from the bundled 21.2.3) | 24.3.1 |

**No CUDA Toolkit was installed.** The `cu116` wheel bundles its own runtime and works with driver 511.69.

### 2B.2 Boundary results

| Boundary | Co-VAE | DCGAN-DTA |
|---|---|---|
| Environment | **PASS** | **PASS** (after the numpy pin) |
| Import | **PASS** | **PASS** |
| Argument parsing | **PASS** | **PASS** |
| Data loading | **PASS** | **PASS** |
| Batch / feature construction | **PASS** | **PASS** |
| Model construction | **PASS** | **NOT REACHED — by design** (see 2B.5) |
| Forward pass | **PASS** | NOT REACHED |
| Loss | **PASS** | NOT REACHED |
| GPU memory | **measured, no OOM** | not exercised (CPU path) |

### 2B.3 Stage 04's data audit confirmed exactly at runtime

Every predicted figure held, for both projects, to the digit:

- **Davis** — XD (68, 85), XT (442, 1200), Y (68, 442), **30,056 observed (0% missing)**, pKd **5.0000–10.7959**, mean **5.4515**.
- **PDBbind** — XD (4231, 200), XT (1606, 2000), XD_t (50068, 200), XT_t (50202, 2000), Y (4231, 1606), **5,014 observed**, **2.0000–11.9200**, mean **6.4002**; folds 5×836 + 834 covering `0..5013` exactly once, drugs 3591/787/**147 shared**, targets 1452/491/**337 shared**.
- **Float64 encoding cost confirmed**: `XT_t` alone is **766 MiB**; PDBbind arrays total **925 MiB** before any model exists (Stage 05 estimated ≈970 MB).
- **Davis zero-margin embedding index confirmed**: `XT max index 24` against `Embedding limit 24` — the last valid slot, exactly as Stage 04 §4.3 warned.

### 2B.4 Findings that change the plan

1. **Co-VAE's three CRITICAL blockers were all neutralised without touching source.** **B1** cleared by Python 3.10 (`random.sample` on a set runs); **B2** cleared by passing `--lamda -5` explicitly; **B3** satisfied by real NVIDIA hardware. Stage 05's Co-VAE verdict of "STAGE 06 BLOCKED (Tier 2)" was contingent on GPU availability — hardware was present, and **every boundary passed**.
2. **`layers[-3]` kernel shape measured: (3, 32, 64)** — built for **32** input channels. Runtime confirmation of correction C-1 and definitive refutation of Stage 03's C1. All three splices execute. *(Keras-semantics check on a scratch replica — does NOT prove `build_GAN_*` construct end-to-end, which is still R3.)*
3. **DCGAN model construction is confirmed inseparable from GAN pretraining.** `ganForDrug/Target/BlosumTarget` take only the data array — no `FLAGS` — so no CLI argument reaches the hard-coded `train(5000, …)` calls at `:152`, `:256`, `:354`. Execution stopped at the earliest safe boundary, as the rules require.
4. **NEW — Co-VAE's co-regularization is numerically negligible at the paper's own λ.** At λ = −5, the two VAE terms contribute **0.0216%** of the objective; the affinity MSE term is **99.978%** (stable across batch 256 and 64). This gives an empirical magnitude to Stage 03's C6. *Measured at initialisation — not a claim about trained behaviour.*
5. **NEW — the 2 GB GPU is near-saturated by a single Co-VAE forward pass.** Batch 256 forward-only (`no_grad`) peaks at **805.77 MiB allocated / 980 MiB reserved**, leaving **86 MiB** free. Training at batch 256 is unlikely to fit; batch 64 peaks at 248.76 MiB.
6. **NEW — latent bug in DCGAN `emetrics.py:38`**: `if pair is not 0:` — identity comparison against a literal (`SyntaxWarning` at import). It sits in `get_cindex`, which Stage 03 §A4 established is dead code, so it does not affect reported results.
7. **Environment/network findings** (recur in Stage 07): a local proxy at `127.0.0.1:20808`; bundled pip 21.2.3 fails with `check_hostname requires server_hostname`; pip **stalls on large transfers** through this proxy while `curl` succeeds; a resumed download can reach the **exact correct size yet be corrupt** — verify integrity before installing.

### 2B.5 Where DCGAN-DTA stopped, and why

Not a failure. `build_GAN_C` was not invoked because invoking it runs **5,000 adversarial iterations** (10,000 for A and B) that no command-line argument can reduce, and editing the literal is forbidden. Everything up to and including feature assembly passed.

---

## 2C. Stage 07A Summary — Co-VAE One-Step Training Feasibility

Measures whether the repository's **native** training step fits the 2 GB MX330. Environment unchanged from Stage 06; only the documented `--batch_size` flag varied. **No source, data, fold, architecture, or hard-coded constant was changed; no AMP, checkpointing, accumulation, or allocator tuning was used.**

One step = `zero_grad → forward → 3-term loss → backward → optimizer.step()`, replicating `train():115-141` for a single batch, then stop.

| `--batch_size` | Result | Peak allocated | Peak reserved | Step time |
|---:|---|---:|---:|---:|
| 64 | ✅ completed | 529.66 MiB | 724.00 MiB | 1.525 s |
| **128** | ✅ **completed** | **1009.29 MiB** | **1072.00 MiB** | **1.700 s** |
| 256 | ❌ **CUDA OOM** | 838.72 MiB at failure | 944.00 MiB | — |

Usable GPU budget: **1661.87 MiB** of 2047.88 (≈386 MiB held by display + CUDA context).

### 2C.1 Findings

1. **The published batch size (256) is not trainable on this GPU.** `arguments.py:49` defaults to 256 and the paper states 256. Training here requires **≤ 128** — a CLI change, not a code change, but a **declared deviation from the published configuration**.
2. **OOM at 256 occurred inside the forward pass, before backward.** Stage 06 showed the *same* forward under `no_grad` peaked at 805.77 MiB and fit; with gradients retained it does not. Retained activations roughly double the forward cost (batch 64: 248.76 → 482.08 MiB).
3. **Peak occurs during backward, not forward**, in both successful runs.
4. **Adam is lazily allocated** — `optimizer init` adds 0 MiB; the ≈99.8 MiB of state appears on the first `step()`. Combined with parameters + gradients, ≈200 MiB persists for all of training, independent of batch size.
5. **Scaling is near-affine**: ≈50 MiB fixed + ≈7.5 MiB per sample. Extrapolating to 256 predicts ≈1969 MiB — beyond budget, consistent with the observed failure.
6. **Headroom at 128 is ≈590 MiB (~35%)** — adequate for one step, **not** validated for a sustained run (excludes eval batches, checkpoint serialisation at `:289`/`:376`, and long-run fragmentation).

The exact maximum lies in **(128, 256)**; it was not bisected, as the procedure specified stopping at the first OOM.

### 2C.2 Scope limit

This is **training-feasibility evidence, not reproduction evidence.** It does not show convergence, any epoch completing, or any published number being recoverable — Stage 05 rated Co-VAE result reproduction **BLOCKED** independently of hardware. The scientific effect of batch 128 vs 256 is unmeasured.

---

## 2D. Stage 07B Summary — DCGAN-DTA `ganForDrug` Timing (R6)

One bounded measurement of the **real, unmodified** repository call, on PDBbind, in a fresh subprocess with a 600 s external safety limit. **No source, data, fold, architecture, hard-coded constant, or CLI argument changed; `ganForDrug` was not patched or wrapped.**

**Result: STOP A — normal completion. R6 RESOLVED.**

| Measurement | Value |
|---|---|
| Path | `ganForDrug(XD_t)` → `train(5000, 5, 500, XD)` (`:152`, verified against source) |
| Device | **CPU only** — `visible_GPUs=NONE`, `nvidia-smi` 0 MiB / 0% at every sample |
| **Call elapsed** | **404.29 s (6 min 44 s)** — full 5000 iterations |
| Iterations observed | all **10** checkpoints (500 … 5000); repository prints only every 500 |
| Mean per iteration | 80.9 ms |
| Peak GPU VRAM | **0 MiB** — GPU never used; 1800 MiB budget never approached |
| Process RSS | **765 → 967 MiB** (directly observed) |
| Termination | normal exit, return code 0; wall-clock limit not reached |

### 2D.1 Findings

1. **A single GAN pretraining call is affordable** — ~6.7 min on CPU, no OOM, no exception.
2. **Per-iteration cost degrades within the call**: 64.5 → 104.7 ms/iteration, a **1.55× slowdown**, with RSS rising ~202 MiB under a fixed model set. *Inferred* to be session/graph accumulation across repeated `predict()`/`train_on_batch()`; mechanism not instrumented, **not claimed as a defect**.
3. **The GAN does not appear to learn in this run.** Discriminator 100.00% accuracy and ~0 loss from the first checkpoint; **generator loss constant at exactly 15.424948 across all ten checkpoints**. `[CODE]` the discriminator ends in `Dense(1, activation='tanh')` (`:94`) yet compiles with `binary_crossentropy` (`:144`). *Inferred* discriminator dominance with vanishing generator gradient. **NOT established** as a bug or as affecting published results — ten sampled checkpoints, one run, no generated outputs inspected.
4. **Nothing is persisted** — no checkpoint, weights, figure or log; confirms Stage 03 §E17 at runtime.
5. **The GPU is unused by this path as configured** (TF 2.10 cannot see it without the CUDA 11.2 toolkit). Not moved to GPU, per instruction.

### 2D.2 Reproduction implication — clearly labelled ESTIMATE

`build_GAN_*` runs inside the innermost grid×fold loop, and the CV routine is called twice. README PDBbind grid = 2 × 3 × 3 = 18 points × 5 folds × 2 passes = **180 invocations**.

> **ESTIMATE, not an observed result:** for **variant C** (one GAN per build), 180 × 404.29 s ≈ **20.2 hours of drug-GAN pretraining alone**, excluding DTA training and evaluation. It assumes constant per-call cost — which finding 2 gives direct reason to doubt. **No estimate is given for variants A/B**, which pretrain two GANs each; the protein-GAN cost is unmeasured and `ganForTarget` was **not run**.

**Bottom line:** one call is feasible; the full grid is **not established** as feasible on this hardware.

---

## 3. Decisions Log

| # | Date | Decision | Basis | Status |
|---|---|---|---|---|
| **D1** | 2026-09-25 | **The first reproduction will use each paper's native datasets, not a forced common dataset.** Co-VAE on Davis; DCGAN-DTA on PDBbind. A common-dataset experiment is deferred to the hybrid-method work. | Stage 04 §11.3 (DCGAN report): loaders, fold mechanisms, required auxiliary files and KIBA's affinity scale are all incompatible. Forcing a shared dataset would mean rewriting a data pipeline before either has been shown to run, making any discrepancy uninterpretable. | **ADOPTED** |
| **D2** | 2026-09-25 | **Development datasets: Davis (Co-VAE) and PDBbind (DCGAN-DTA).** | Davis: matches the published matrix exactly, valid folds, low preprocessing burden, workable AUC, smallest. PDBbind: only uncensored distribution of the four, all three fold settings, 8.4× cheaper than BindingDB, `is_log=0` so no transform to get wrong. | **ADOPTED** |
| **D3** | 2026-09-25 | **Both development datasets are for pipeline validation, not for headline evidence.** Davis is 69.64% censored and fully dense; PDBbind is 99.93% empty at 1.18 observations per drug. Headline claims require KIBA and BindingDB. | Stage 04 §11.4 (both reports). | **ADOPTED** |
| **D4** | 2026-09-25 | **Reproduce-the-paper and produce-an-honest-number are two separate experiments** and will be reported separately. The papers' reported metric is a `max`-over-epochs statistic on the test fold. | Stage 04 §6.3 (DCGAN), §7.3 (Co-VAE). | **ADOPTED** |
| **D5** | 2026-09-25 | **Stage 06 targets Python 3.10 for both projects** — the only version clearing Co-VAE's `random.sample(set)` ceiling (≤ 3.10) while remaining inside DCGAN-DTA's TF window (≤ 3.11). Two separate isolated environments, since the frameworks differ. The ambient 3.13 interpreter is not modified. | Stage 05 §1.1–1.2 both reports. | **ADOPTED** |
| **D6** | 2026-09-25 | **DCGAN-DTA's first framework candidate is TensorFlow 2.15 / Keras 2.15** — newest inside the derived TF 2.3–2.15 window, maximising wheel availability; fall back down the 2.x line on failure. **This is a candidate, not an established working version** (R1). | Stage 05 §1.2, §9 (DCGAN report). | **ADOPTED** |
| **D7** | 2026-09-25 | **Co-VAE's smoke test is split into Tier 1 (CPU) and Tier 2 (GPU).** Tier 2 must not be attempted without confirmed NVIDIA hardware. Reaching a forward pass without a GPU would require editing six `.cuda()` sites — a **Stage 07 deviation decision**, not a setup step. | Stage 05 §6, §8.1 (Co-VAE report). | **ADOPTED** |
| **D8** | 2026-09-25 | **Stage 06 tests DCGAN-DTA variant C before variant A.** Variant C builds one GAN instead of two (halving the 5,000-iteration cost) while still exercising the drug-branch `layers[-3]` transfer — the exact site of the withdrawn C1. | Stage 05 §5.4, §6.1 (DCGAN report). | **ADOPTED** |
| **D9** | 2026-09-25 | **No source modification in Stage 06.** Blockers are findings to record, not defects to repair. Every candidate patch (Co-VAE B1/B2/B3, any DCGAN flag default) is deferred to Stage 07 and requires explicit approval, with each deviation logged. | Standing constraint, reaffirmed against Stage 05's blocker tables. | **ADOPTED** |

---

## 4. Blockers

### 4.1 Hard execution blockers — carried from Stage 03, unchanged by Stage 04

| ID | Method | Blocker | Severity |
|---|---|---|---|
| B1 | Co-VAE | `random.sample(set, k)` raises on Python ≥ 3.11 | **CRITICAL** — stands; clears on Python ≤ 3.10 |
| B2 | Co-VAE | ~~`--lamda` has no default~~ → `len()` on an int | **CRITICAL** — stands, **cause re-diagnosed: see correction C-2** |
| B3 | Co-VAE | unconditional CUDA, no CPU fallback | **CRITICAL** — stands; Stage 05 confirms it is on the **forward path** (6 sites) |
| B4 | Co-VAE | `--num_windows`, `--smi_window_lengths`, `--seq_window_lengths` have no defaults and **no README** documents them | HIGH — stands; clears by passing the flags |
| B11 | Co-VAE | ~~in-place `.exp_()` may alter the KL input~~ | **WITHDRAWN — see correction C-3** |
| ~~C1~~ | DCGAN-DTA | ~~transferred `Conv1D(64)` built on 1 channel, applied to a 32-channel embedding, in all three variants~~ | **WITHDRAWN — see correction C-1** |
| E2 | DCGAN-DTA | TF/Keras window derived but **no version verified**: TF ≥ 2.3 (`Conv1DTranspose`), < 2.16 (`fit_generator`, `tf.compat.v1.keras`) | HIGH |

### 4.1a Dated corrections to earlier stages

> Prior findings are preserved above and amended here, never silently replaced. Full reasoning in the Stage 05 reports §12.

| ID | Date | Supersedes | Correction |
|---|---|---|---|
| **C-1** | 2026-09-25 | Stage 3 §C1 (CRITICAL) and §F1; restated in Stage 04 and §4.1 above | `gan_smiles.layers[-3]` is built on **32** input channels, not 1. `Reshape((200,1))` is `layers[0]`; four `Conv1D` layers (4→8→16→32) intervene. All six transfer sites in variants A/B/C are channel-consistent. **Blocker withdrawn**; DCGAN-DTA's verdict moves 🔴 RED → 🟡 YELLOW. Construction remains untested (**R3**). |
| **C-2** | 2026-09-25 | Stage 3 §13 blocker B2; restated in Stage 04 §2 and §4.1 above | `--lamda` **does** have a default (`default=-5`, `nargs='+'`, `arguments.py:83-88`). The defect is a **scalar default under `nargs='+'`**, so `len()` fails only on the default path. Blocker and severity stand; cause re-diagnosed. Clears by passing `--lamda -5`. |
| **C-3** | 2026-09-25 | Stage 3 §13 blocker B11 (HIGH, unresolved) and Stage 3 §14 Phase-4 target #3 | `Tensor.mul(0.5)` is **out-of-place**; `.exp_()` mutates the temporary, never `logvar`. `loss_f` receives the true `logvar`. **Withdrawn — resolved statically, no runtime check needed.** Stage 3's C5 (ε mis-scaling) is unaffected and remains open. |
| **C-4** | 2026-09-25 | **Confirms C-1 with runtime evidence** (Stage 3 §C1 remains refuted) | `[RUNTIME]` Under Keras 2.10.0, `InputLayer` is excluded from `model.layers`; `layers[-3]` is `Conv1D` with **kernel shape (3, 32, 64)** — built for **32** input channels, not 1 — and applying it to a 32-channel tensor yields (None, 200, 64). Verified on a scratch replica of the same layer sequence, **not** by running `build_GAN_*` (R3 still open). |
| **C-5** | 2026-09-25 | Refines Stage 05 Co-VAE verdict "COMPLETE — STAGE 06 BLOCKED (Tier 2)" | That verdict was explicitly conditional on GPU availability. **NVIDIA hardware was present, and Co-VAE's Tier 2 passed in full** — construction, forward pass and loss all succeeded on a 2 GB MX330 with no OOM and no source change. The Stage 05 blocker assessment was correct as written; the condition simply resolved favourably. Co-VAE's **result-reproduction** verdict (BLOCKED) is unchanged and unaffected. |
| **C-6** | 2026-09-25 | Resolves Stage 05 runtime unknowns | **R2** resolved (C-4 above). **CV-R1** resolved — GPU available. **CV-R4** resolved — `weights_init` covers `Linear`/`BatchNorm1d`/`LSTM`/`Conv1d` but **not** `Embedding` or `ConvTranspose1d`, which keep PyTorch defaults. **CV-R5** resolved — real fold sizes `[5010, 5010, 5009×4]` → test 5,009 / train 25,047, matching Stage 04's simulation. **CV-R6** resolved — all three loss terms compute; `10**lamda` scales the **VAE** terms, which are 0.0216% of the objective. **R7** resolved — PDBbind arrays total 925 MiB. Still open: **R1** (no other framework version tested), **R3**, **R5**, **R6**, **CV-R3**. |

### 4.2 Data-layer blockers — established by Stage 04

| ID | Method | Blocker | Severity |
|---|---|---|---|
| **D-1** | Co-VAE | KIBA fold files unusable (8,958 out-of-range indices). Reproducing the paper's KIBA column requires editing `datahelper.py` — a deviation, not a reproduction. | **HIGH** |
| **D-2** | Co-VAE | KIBA AUC raises or is meaningless at threshold 7 (10 negatives in 109,296 pairs). | **HIGH** |
| **D-3** | Co-VAE | Davis new-target results carry an upward bias from 17.8% sequence-visible test targets. Magnitude unmeasured. | MEDIUM |
| **D-4** | DCGAN-DTA | Cold-start splits cannot be regenerated or extended to BindingDB (no logP rule, no settings 2/3 for BindingDB). | MEDIUM |
| **D-5** | DCGAN-DTA | `--is_log` has no default; the wrong value silently destroys PDBbind's label distribution. | MEDIUM |
| **D-6** | both | No `requirements.txt`, `environment.yml`, Dockerfile or lockfile ships with either repository. No working environment has been established for either. | **HIGH** |

### 4.3 Stage 05 blockers — environment and code

| ID | Project | Problem | Stage 06 can test? | Needs source change? |
|---|---|---|:--:|:--:|
| **E1** | both | Ambient Python 3.13.2 is outside both windows; neither framework installed | **yes — prerequisite** | no |
| **E3** | DCGAN | seaborn, sklearn, pandas, Pillow imported but undeclared in README | yes | no |
| **E4** | DCGAN | `reset_keras()` pins `visible_device_list="0"` ⇒ GPU required **at that call site only** (`:711`, `:843`) | out of smoke scope | no |
| **E5** | both | Absolute POSIX defaults: `--log_dir /tmp` (both), `--dataset_path /data/pdb/` (DCGAN) | yes | no — pass flags |
| **C4** | DCGAN | Model construction inseparable from 5,000–10,000 GAN iterations | **yes — measure** | no |
| **C6** | DCGAN | Spliced layer's effective trainability unresolved (Stage 3 U5) | **yes — `trainable_weights`** | no |
| **CV-3** | Co-VAE | Length flags silently re-weight the loss (`:136` uses `max_smi_len/max_seq_len` as a coefficient) | yes — record values | no |
| **CV-4** | Co-VAE | Davis protein embedding index maxes at **24 of 24** — zero margin before `IndexError` | yes — assert | no |
| **CV-5** | Co-VAE | `--dataset_path` defaults to **KIBA**, whose folds are invalid | yes — pass Davis | no |

### 4.4 Runtime-only unknowns carried into Stage 06

| ID | Project | Unknown |
|---|---|---|
| **R1** | both | Which framework version actually works (TF 2.3–2.15 window / PyTorch unestablished) |
| **R2** | DCGAN | Whether `model.layers` excludes `InputLayer` under the tested Keras — the §5.3 layer indexing depends on it |
| **R3** | DCGAN | Whether `build_GAN_A/B/C` construct end-to-end |
| **R4** | DCGAN | Effective trainability of the spliced layer |
| **R5** | DCGAN | Whether `Embedding(input_length=1)` against a `(max_smi_len,)` input is ignored or raises |
| **R6** | DCGAN | Wall-clock of one `ganFor*` call — governs whether any full run is affordable |
| **R7** | DCGAN | Peak RSS during `parse_data` (≈970 MB resident, ~2 GB peak estimated) |
| **CV-R1** | Co-VAE | Whether NVIDIA hardware is available at all — **determines whether Tier 2 exists** |
| **CV-R3** | Co-VAE | Whether `.squeeze()` breaks on a trailing batch of size 1 |
| **CV-R5** | Co-VAE | Realised fold sizes under the code's own RNG (Stage 04 simulated these with a substitute sampler) |
| **CV-R6** | Co-VAE | Which loss term `10**lamda` actually scales, and whether all three carry gradient |

---

## 5. Dataset Findings — Reference

Full detail in `audit/04-dataset-fold/`. Key numbers for citation:

- **Davis** — 68 × 442, 30,056 pairs, **0% missing**, pKd 5.0–10.7959 (mean 5.4515, median 5.0000, std 0.8947). **20,931 (69.64%) at exactly pKd 5.0.** Constant-mean-predictor MSE **0.8005**. 379 unique sequences among 442 ids. AUC positives at `>7`: 8.17%.
- **KIBA (runtime)** — 1,954 × 217, 109,296 pairs, 74.22% missing, KIBA score 0.0–17.2002 (mean 11.7213, median 11.5000, std 0.8271). Published matrix is 2,111 × 229; the gap is an undocumented length filter (caps 90 / 1365) costing 8,958 pairs. 434 values below 10 survive the filter the paper says removes them.
- **BindingDB** — 9,864 × 1,088, 42,203 pairs, **99.61% missing**, runtime pKd 2.0–9.0 via `-log10((Y+1)/1e9)` (mean 5.7737, median 4.99996, std 1.2967). **19,828 (46.98%) at exactly pKd 5.0.** 690 SMILES (7.0%) exceed the 200 cap and are silently head-truncated. Fold setting 1 only.
- **PDBbind** — 4,231 × 1,606, 5,014 pairs, **99.93% missing**, pKd 2.0–11.92 used verbatim (mean 6.4002, median 6.4300, std 1.9471). Modal value 0.90% — the only uncensored dataset. **1.18 observations per drug.** Fold settings 1 (warm), 2 and 3 (confirmed cold-drug).

---

## 6. Roadmap

| Step | Description | Gate |
|---:|---|---|
| ~~Prev~~ | ~~Stage 05 — `reproduction-feasibility`~~ | ✅ done 2026-09-25 |
| ~~Prev~~ | ~~Stage 06 — minimal instrumented smoke execution~~ | ✅ done 2026-09-25 (§2B) |
| ~~Prev~~ | ~~Stage 07A — Co-VAE one-step training feasibility~~ | ✅ done 2026-09-25 (§2C) |
| ~~Prev~~ | ~~Stage 07B — DCGAN-DTA `ganForDrug` timing (R6)~~ | ✅ done 2026-09-25 (§2D) |
| **Next** | **Stage 08 — not yet defined.** Both §7A runtime candidates are now answered | **requires explicit approval; NOT started** |
| Then | Deviation decisions: whether to patch anything at all. **Stage 06 showed Co-VAE needs no patch** — B1/B2/B3 all cleared by environment alone | **requires explicit approval — first stage that modifies source** |
| Then | Native-dataset reproduction: Co-VAE/Davis, DCGAN-DTA/PDBbind, per D1–D2 | gated on Stages 06–07 |
| Then | Controlled comparison design — shared loader, shared split definition, shared metric implementation | gated on both reproductions landing |
| Later | Hybrid representation method | **out of scope until the above completes** |

---

## 7A. Stage 07 Candidates — both now answered

Stage 06 answered every question it was scoped to answer. Two measurements now gate everything downstream, one per project. **Both require explicit approval; neither requires a source change.**

| | Co-VAE — ✅ **ANSWERED, §2C** | DCGAN-DTA — ✅ **ANSWERED, §2D** |
|---|---|---|
| **Question** | ~~Does one forward + backward step fit in 2 GB?~~ **Yes, at ≤ 128; 256 OOMs.** | ~~What does one `ganForDrug(XD_t)` call cost?~~ **404.29 s on CPU, full 5000 iterations.** |
| **Why it gates everything** | Forward-only at batch 256 already leaves 86 MiB free (§2B.4 #5). If no batch size trains, native-dataset reproduction needs different hardware. | GAN pretraining re-runs inside the innermost grid×fold loop, so its unit cost multiplies by (grid points × 5 folds × 2 passes). It is also the gateway to **R3**. |
| **Method** | Start at `--batch_size 64`, one step, stop at first OOM. Documented flag only. | Time one call to completion. **Do not edit the hard-coded 5000.** |
| **Bounded?** | One step, not one epoch | One GAN pretrain, not a grid |
| **Risk** | OOM (recorded, not worked around) | Cost may prove prohibitive — itself a finding |

Neither is a reproduction run, a benchmark, or a paper result.

---

## 7. Stage 06 Entry Criteria and Scope *(satisfied — retained for the record)*

**Stage 06 is the first stage permitted to execute code. It is not permitted to modify source (D9).**

### 7.1 Entry criteria — all must hold before Stage 06 begins

1. **Explicit user approval** to execute.
2. **Co-VAE only:** GPU availability determined (CV-R1). With NVIDIA hardware ⇒ Tier 1 + Tier 2. Without ⇒ **Tier 1 only**, and reaching a forward pass becomes a Stage 07 decision.
3. Two isolated **Python 3.10** environments provisioned (D5) — neither touching the ambient 3.13 interpreter.
4. DCGAN-DTA: TensorFlow 2.15 / Keras 2.15 as first candidate (D6), plus seaborn, scikit-learn, pandas, Pillow.
5. Co-VAE: a CUDA-enabled PyTorch build, plus numpy, pandas, scikit-learn, matplotlib, tqdm.

### 7.2 Scope — ordered, stop at first forward pass

| | DCGAN-DTA (PDBbind, variant C then A) | Co-VAE (Davis) |
|---|---|---|
| S1 | import framework; record versions | import torch; **record `torch.cuda.is_available()` first** |
| S2 | `import run_experiments` | `import run_experiments` |
| S3 | `parse_data` + `read_sets` — expect `XD (4231,200)`, `XT (1606,2000)`, `XD_t (50068,200)`, `XT_t (50202,2000)`, **5,014** observed, folds `834` / `5×836` | `parse_data` on Davis at `--max_smi_len 85 --max_seq_len 1200` — expect `(68,85)`, `(442,1200)`, **30,056** observed, pKd **5.0–10.7959** |
| S4 | `build_GAN_C` — capture `model.summary()`, resolved `layers[-3]` shapes, `len(trainable_weights)`, **wall-clock** | fold generation — the **B1 probe** on Python 3.10 |
| S5 | one `predict` + one `evaluate` on ~8 real pairs from `test_fold_setting1` | one `DataLoader` batch — `(256,85)`, `(256,1200)`, `(256,)` float32 |
| S6–S8 | — | **Tier 2, GPU only:** construct `net(...).cuda()`, one forward pass, one loss with the three terms printed separately |

### 7.3 Explicitly out of scope for Stage 06

Full training · any paper figure or number · Kaggle or any remote compute · **any source modification**, including the three one-line changes that would clear Co-VAE's B1–B3.

### 7.4 Interpreting failure

A failure is **evidence to record, not a defect to repair**. In particular: a DCGAN construction failure at `layers[-3]` would indicate that **R2**'s layer-indexing assumption is wrong under the tested Keras — **not** that the withdrawn C1 was correct; the diagnosis must come from the traceback. A Co-VAE CUDA failure means only that no GPU is present. Neither would invalidate Stage 04's data findings, which are execution-independent, nor change §2A.3's result-reproduction verdicts, which are already BLOCKED for independent reasons.

### Open questions carried forward

Co-VAE O1–O8 and DCGAN-DTA O1–O8 are listed in §12 of their respective Stage 04 reports. The two most likely to change the plan:

- Does any flag combination recover the published 2,111 × 229 KIBA matrix? (Co-VAE O1 — static reading says no.)
- What is the provenance of the ~50,000 non-Davis entries in the GAN pretraining corpora? (DCGAN-DTA O1.)

---

## 9. Standing Constraints

- `Co-VAE/CoVAE/` and `DCGAN-DTA/DCGAN-DTA/` are **vendored upstream clones with their own git history**. `Co-VAE/CoVAE` is untracked by the parent repo for this reason. Do not refactor; any edit is a logged deviation, not a fix.
- `audit/02-*`, `audit/03-*` and `audit/04-*` are **final**. Later stages cite, refine and correct them **by dated correction** (§4.1a); they never overwrite them.
- Fold files are **audit evidence**. Do not regenerate or normalise them.
- Later stages inherit the read-only expectation unless explicitly lifted. Stage 06 lifts *execution* only; Stage 07 is the earliest that may lift *modification*, and only on explicit approval.
- The ambient Python 3.13 environment is not to be modified. Stage 06 environments are isolated and additional.

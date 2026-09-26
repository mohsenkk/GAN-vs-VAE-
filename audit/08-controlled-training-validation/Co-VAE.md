# Stage 08 — Controlled Training Validation: Co-VAE

**Date:** 2026-09-25/26
**Purpose:** establish whether the native Co-VAE training path can execute repeated, numerically stable optimization under controlled conditions. **Not** a reproduction of published results.

---

## 1. Exact commit used

| | |
|---|---|
| Implementation | `Co-VAE/CoVAE/` |
| Upstream | `github.com/LiminLi-xjtu/CoVAE`, branch `master` |
| **HEAD** | **`2c17268`** ("Add files via upload") |
| Source modified | **No** — verified by `git status` and MD5 (§12) |

---

## 2. Environment

Identical to the Stage 06/07A validated environment. Nothing installed, upgraded, or downgraded.

| Component | Version |
|---|---|
| OS | Windows 11 Enterprise 10.0.26200 |
| Python | 3.10.0 (isolated venv, outside the repository) |
| PyTorch | **1.13.1+cu116** |
| CUDA runtime (torch) | 11.6 |
| NumPy | 1.26.4 |
| GPU | **NVIDIA GeForce MX330**, cc 6.1, 2047.88 MiB total, **1661.87 MiB free** |
| Driver | 511.69 |

---

## 3. Exact command / configuration

```
--dataset_path ./data/davis/
--max_smi_len 85
--max_seq_len 1200
--num_windows 32
--smi_window_lengths 4
--seq_window_lengths 8
--lamda -5
--problem_type 1
--batch_size 64          <-- DEVIATION from the requested 128; see §6
```

### 3.1 The external runner

The repository exposes no short-training CLI option, so the native functions were invoked through an external runner (`scratchpad/stage08_covae.py`, **outside** the repository). It calls, unchanged:

`RE.get_random_folds` · `RE.prepare_interaction_pairs` · `model.net` · `RE.weights_init` · **`RE.train`** · **`RE.test`** · `RE.loss_f` · `RE.get_cindex`

Two experiments were run:

| | What it is |
|---|---|
| **A1 — NATIVE** | 5 × [`RE.train()` → `RE.test()`], i.e. 5 native epochs over a documented subset. `RE.train()` is used **verbatim**, including its own re-creation of `Adam` on every call (`run_experiments.py:118`) — the repository's native per-epoch behaviour. Ordering matches `general_nfold_cv:280-281` (train, then test). |
| **A2 — INSTRUMENTED** | A step loop replicating `train():126-138` so that per-step loss components, gradient norms and NaN/Inf can be observed. **External instrumentation, not the native function.** It keeps **one** optimizer across all 30 steps to expose a continuous trajectory. |

---

## 4. Dataset

Davis, via the native `DataSet.parse_data`. Runtime values match Stage 04/06 exactly: `XD (68, 85)`, `XT (442, 1200)`, `Y (68, 442)`, **30,056 observed pairs**, pKd 5.0000–10.7959.

---

## 5. Fold / split

Native fold generation, `problem_type 1` (pair-wise/warm), `random.seed(1000)`, `RE.get_random_folds(30056, 6)`:

```
native folds: train=25047  test=5009  overlap=0
```

Identical to Stage 06. **A controlled subset** was then taken for the short run:

| | pairs | batches | drugs | targets |
|---|---:|---:|---:|---:|
| train subset | **1,280** | 20 | 68 | 423 |
| test subset | **640** | 10 | 68 | 339 |

Both subsets are prefixes of the native fold-1 splits; train/test disjointness is inherited from the native folds.

---

## 6. Deviations

| # | Deviation | Why | Reported as |
|---|---|---|---|
| **D-1** | **`--batch_size 64` instead of the requested 128** | Batch 128 **OOMs inside native `RE.train()` on its first batch** — see §6.1. Reduced via the documented CLI flag only; no source change. | **Material deviation** |
| D-2 | Training restricted to a 1,280-pair subset, 5 epochs | Stage 08 is explicitly a controlled validation, not a reproduction | Intended by design |
| D-3 | External runner used to invoke native functions | Repository exposes no short-training option; §3.1 | Documented |
| D-4 | A2 keeps one optimizer across steps | To expose a continuous trajectory; native re-creates Adam per epoch. A1 preserves the native behaviour. | Clearly separated from A1 |

### 6.1 Batch 128 fails in the native training loop — a refinement of Stage 07A

`[OBSERVED]` Two independent attempts at `--batch_size 128` both terminated with `torch.cuda.OutOfMemoryError`, raised at **`run_experiments.py:128`** inside native `RE.train()`, on the **first batch**, in `model.py:78` (`decoder2` → `ConvTranspose1d`):

```
CUDA out of memory. Tried to allocate 76.00 MiB (GPU 0; 2.00 GiB total capacity;
932.41 MiB already allocated; 0 bytes free; 1.04 GiB reserved in total by PyTorch)
```

Stage 07A measured a **single** step at batch 128 succeeding with peak allocated **1009.29 MiB**. The failure here occurs at 932.41 + 76 ≈ **1008 MiB** — essentially the same peak. 

`[INFERRED]` Batch 128 sits exactly on the device limit. An isolated step fits; the same step inside `RE.train()` does not, because the surrounding process holds slightly more (DataLoader machinery, tqdm's reference to the batch, allocator fragmentation). **This confirms, and makes concrete, Stage 07A's own caveat** that its 128 figure was "adequate for one step, not validated for a sustained run."

**No memory workaround was applied.** `max_split_size_mb` was not set, and no AMP, checkpointing, accumulation or offloading was used. The batch size was reduced through the repository's own CLI.

---

## 7. Runtime

| Experiment | Work | Wall-clock | Rate |
|---|---|---:|---:|
| A1 native | 5 epochs × 20 steps = 100 optimizer steps | **38.1 s** total training | **7.6 s / epoch** (≈380 ms/step) |
| A1 evaluation | 6 × `RE.test()` over 640 pairs | ~2 s each | — |
| A2 instrumented | 30 steps | **11.1 s** | **370 ms / step** |

---

## 8. Memory

| Measurement | A1 (native, 5 epochs) | A2 (30 steps) |
|---|---:|---:|
| After construction — allocated | 49.89 MiB | 49.89 MiB |
| **Peak allocated** | **679.44 MiB** | **678.29 MiB** |
| **Peak reserved** | **834.00 MiB** | **834.00 MiB** |
| Device free at end | 202.43 MiB | 202.43 MiB |
| Untrained-baseline run (separate process) | peak alloc 258.4 / reserved 350.0 MiB | — |

`[OBSERVED]` **No memory leak.** Across all five native epochs, steady-state allocated stayed at **99.9 MiB** and peak allocated moved only from 678.29 → 679.44 MiB (+1.15 MiB total). Reserved was flat at 834.00 MiB from epoch 1 onward.

`[OBSERVED]` Checkpointing works: `torch.save(model, …)` → **49.4 MiB**, and `torch.load` reloaded a `net` instance successfully. This is the native mechanism used at `general_nfold_cv:289`.

---

## 9. Numerical stability

`[OBSERVED]` Across all 30 instrumented steps:

| Check | Result |
|---|---|
| NaN in any loss component | **none** |
| Inf in any loss component | **none** |
| All values finite (total, affinity, drug, target, gradient norm) | **True, every step** |
| Parameters with `grad is None` | **0 of 52** |
| Zero gradient norm at any step | **none** |
| Exploding loss | **none** — losses decreased |
| Gradient norm range | first 21.95, max 67.42 (step 5), min 1.78, last 3.10 |

Every parameter tensor received a gradient, and the gradient norm settled to ~2.5–3.1 after the initial transient rather than vanishing or exploding.

---

## 10. Training curves

### 10.1 A1 — native path (`RE.train` → `RE.test` per epoch)

The untrained baseline was measured in a **separate process** so the native train→test ordering was not perturbed.

| Epoch | CI | MSE | r²m | AUC | train time |
|---|---:|---:|---:|---:|---:|
| **0 (untrained)** | 0.483355 | **29.672663** | −0.009807 | 0.426820 | — |
| 1 | **0.549246** | **0.747112** | 0.006781 | 0.617455 | 9.9 s |
| 2 | 0.526349 | 1.041189 | 0.000701 | 0.548614 | 6.9 s |
| 3 | 0.497765 | 0.820926 | 0.001034 | 0.460583 | 7.2 s |
| 4 | 0.527894 | **0.743612** | 0.007044 | 0.595477 | 7.0 s |
| 5 | 0.484336 | 0.770579 | 0.001183 | 0.471452 | 7.2 s |

MSE trajectory: `29.67 → 0.747 → 1.041 → 0.821 → 0.744 → 0.771`
CI trajectory: `0.483 → 0.549 → 0.526 → 0.498 → 0.528 → 0.484`

### 10.2 A2 — instrumented per-step trajectory

| Step | total | affinity | drug (recon+KL) | target (recon+KL) | grad norm | finite |
|---:|---:|---:|---:|---:|---:|:--:|
| 1 | 27.831682 | 27.825401 | 355.85 | 3842.28 | 21.95 | ✔ |
| 2 | 22.606665 | 22.600412 | 354.29 | 3824.47 | 23.58 | ✔ |
| 3 | 14.967410 | 14.961190 | 352.67 | 3802.02 | 48.13 | ✔ |
| 4 | 4.492173 | 4.485989 | 350.84 | 3778.33 | 39.43 | ✔ |
| 5 | 5.070323 | 5.064183 | 349.31 | 3736.97 | 67.42 | ✔ |
| 10 | 0.660400 | 0.654884 | 328.23 | 3153.37 | 7.09 | ✔ |
| 15 | 1.186110 | 1.181314 | 258.90 | 3115.34 | 2.99 | ✔ |
| 20 | 0.905582 | 0.901153 | 232.01 | 2978.25 | 2.56 | ✔ |
| 25 | 0.599135 | 0.595074 | 213.08 | 2725.69 | 2.57 | ✔ |
| 30 | 0.886171 | 0.882100 | 199.28 | 2934.73 | 3.10 | ✔ |

- total loss: first 27.831682 → last 0.886171 (min 0.526184), **Δ = −26.95**
- **Both VAE terms also decrease monotonically-ish**: drug 355.85 → 199.28, target 3842.28 → 2934.73 — they do receive and act on gradient.
- **VAE-term share of the total loss: 0.0226% → 0.4595%.** It rises only because the affinity term fell ~30×; in absolute terms the co-regularization remains well under 1% of the objective at λ = −5.

---

## 11. Comparison with Stage 07

| Experiment | Stage 07A | Stage 08 | Difference |
|---|---|---|---|
| **Batch 128, one step** | ✅ completed, peak alloc **1009.29 MiB** | ❌ **OOM inside native `RE.train()`**, first batch, at ≈1008 MiB | Single-step headroom **does not transfer** to the native loop — Stage 07A's caveat confirmed |
| **Batch 64, one step** | ✅ peak alloc **529.66 MiB** / reserved 724.00 | ✅ 100 native steps, peak alloc **679.44 MiB** / reserved 834.00 | +149.78 MiB alloc, +110 MiB reserved — the native loop plus evaluation costs more than one isolated step |
| Step time, batch 64 | 1.525 s (incl. cold-start autotune) | **370–380 ms** steady-state | Stage 07A's figure was dominated by one-off CUDA autotuning |
| Successful optimizer steps | 1 | **130** (100 native + 30 instrumented) | repeated optimization validated |
| Loss behaviour | single value, 28.30 | **27.83 → 0.886**; MSE 29.67 → 0.75 | optimization trajectory now observed |
| Memory stability | not assessed | **peak +1.15 MiB over 5 epochs** | no leak |
| VAE-term share | 0.0216% (initialisation) | 0.0226% → 0.4595% | consistent; rises as affinity falls |

---

## 12. Repository integrity

```
 M AGENTS.md          <- pre-existing (GitNexus stat-line)
 M CLAUDE.md          <- pre-existing
 ? Co-VAE/CoVAE       <- pre-existing (nested git repo)
?? THESIS_IMPLEMENTATION_PLAN.md
?? audit/04-…/ 05-…/ 06-…/ 07-…/ 08-…/
```

No source, dataset, or fold modification. Davis data MD5s unchanged from the Stage 06 baseline. `logs/`, `figures/`, `result/` still contain only their original 2-byte placeholders. The 49.4 MiB checkpoint and the run logs were written to the **scratchpad**, outside the repository.

---

## 13. Interpretation

### Observed

1. The native path — `RE.train()` and `RE.test()` — executes repeatedly and returns without error for 5 consecutive epochs (100 optimizer steps).
2. MSE falls from **29.67 (untrained) to 0.747 after one epoch**, then oscillates in **0.744–1.041** for epochs 2–5.
3. CI moves within **0.484–0.549**; AUC within **0.427–0.617**; r²m within **−0.010 to 0.007**.
4. All losses and gradients finite at every step; every parameter receives gradient; no zero or exploding gradient norms.
5. Memory is bounded and stable: peak allocated grows **1.15 MiB across five epochs**.
6. Native checkpointing round-trips successfully.
7. Batch 128 — the Stage 07A boundary — **OOMs inside the native training loop**.

### Inferred

8. **Optimization is genuinely occurring**, not merely executing: a 30× drop in the affinity term, non-zero settled gradient norms, and a decrease in both VAE terms are together inconsistent with a frozen model.
9. **After the first epoch, the model sits close to a trivial-predictor regime.** Stage 04 measured the constant-mean-predictor MSE on full Davis as **0.8005**; epochs 1–5 here range **0.744–1.041**, and CI hovers near 0.5 (random ranking). The rapid collapse from 29.67 to ~0.75 is consistent with the model quickly learning the dataset mean — unsurprising given Stage 04's finding that **69.64% of Davis sits at exactly pKd 5.0**.
10. The co-regularization remains numerically minor at λ = −5 (<0.5% of the objective), consistent with Stage 07A.

### Not established

- **Nothing about model quality.** 100 steps on a 1,280-pair subset cannot show what this model achieves when trained properly (the paper specifies 100 epochs over 25,047 pairs, ×6 folds, ×10 repetitions).
- **Not** that the model cannot exceed the trivial baseline — the run is far too short, and oscillation across five epochs is expected this early.
- **Not** that the implementation is incorrect, that the paper is wrong, or anything comparative about Co-VAE versus DCGAN-DTA.
- **Not** the largest trainable batch size in the native loop: 128 fails and 64 works; the boundary between them was not bisected.
- **Not** whether full-length training is numerically stable — only 130 steps were observed.

---

## 14. Unresolved questions

| # | Question |
|---|---|
| Q1 | What is the largest batch size that sustains the **native** train→test loop? (128 fails, 64 works; not bisected) |
| Q2 | Does the model escape the trivial-predictor regime with longer training, or does MSE ≈ 0.8 persist? Requires a full-length run. |
| Q3 | Does numerical stability hold over 100 epochs rather than 5? |
| Q4 | Does the per-epoch re-creation of `Adam` (`:118`) materially affect convergence versus a persistent optimizer? A1 and A2 differ in exactly this respect but are not otherwise comparable. |
| Q5 | Would a GPU with more VRAM permit the paper's `--batch_size 256`, and would that change the trajectory? |

---

## 15. Recommendation for Stage 08.5

**Co-VAE contributes a moderate, not decisive, argument for Kaggle GPU validation.**

The native training path is validated locally at batch 64: it optimizes, it is numerically stable, it is memory-bounded, and it checkpoints. What **cannot** be done locally is running the **published configuration** — `--batch_size 256` OOMs (Stage 07A), and even 128 fails inside the native loop (§6.1). Any local reproduction would therefore carry a permanent, declared deviation in batch size, which interacts with the optimizer and with every reported metric.

A GPU with ≥ 8 GB would remove that deviation and allow Q2/Q3/Q5 to be answered at the published settings.

---

## Stage 08 Decision — Co-VAE

# CONDITIONAL PASS

The native training path executes repeatedly and demonstrates meaningful optimization behaviour — a 30× reduction in the affinity loss, finite non-zero gradients throughout, bounded memory with no leak, working checkpoints, and a reachable native evaluation path.

The condition is a **documented hardware limitation**: the 2 GB MX330 cannot run the published `--batch_size 256`, and cannot sustain `--batch_size 128` inside the native training loop. Validation was therefore performed at **batch 64**, a documented CLI value, with no source modification.

---

*No repository source file, dataset, fold file, architecture, loss function, or training logic was modified. No memory workaround was applied. This is training-validation evidence only.*

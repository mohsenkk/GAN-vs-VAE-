# Stage 07B — DCGAN-DTA `ganForDrug` Timing

**Date:** 2026-09-25 · **Scope:** Stage 07B only. No Co-VAE work, no `ganForTarget`, no Stage 08.

---

## Objective

Measure the real computational cost of **one** `ganForDrug()` call on the DCGAN-DTA development configuration, under a hard safety budget, to resolve runtime unknown **R6** — and thereby determine whether the repository's hard-coded GAN training schedule is practically executable on the current machine.

This is **not** a reproduction of the DCGAN-DTA paper, not a benchmark, and not a training run of the DTA model.

---

## Environment

Unchanged from Stage 06. Nothing was installed, upgraded, or reconfigured.

| Component | Version |
|---|---|
| Python | 3.10.0 (isolated venv, outside the repository) |
| TensorFlow | 2.10.1 |
| Keras | 2.10.0 |
| NumPy | 1.26.4 |
| OS | Windows 11 Enterprise 10.0.26200 |

---

## Hardware

| | |
|---|---|
| GPU | NVIDIA GeForce MX330, 2048 MiB, compute capability 6.1, driver 511.69 |
| **Device actually used** | **CPU only** |

`[OBSERVED]` The child reported, before any model was built:

```
built_with_cuda=True
visible_GPUs=NONE
logical_devices=['/device:CPU:0']
```

`[OBSERVED]` `nvidia-smi` sampled every ~5 s for the whole run: **`vram_used=0 MiB, gpu_util=0%` at every sample**, including the peak.

**The repository selects CPU naturally.** TF 2.10 cannot see the GPU here because the CUDA 11.2 toolkit / `cudart64_110.dll` is not installed (Stage 06 §1.5). Per the brief, the GAN was **not** moved to the GPU. The GPU memory budget of 1800 MiB was therefore never approached — it was never touched.

---

## Repository Path Tested

The **real, unmodified** repository call:

```
run_experiments.py:54   ganForDrug(XD)
                        └── :152   dis_v = train(5000, 5, 500, XD)
```

invoked as `RE.ganForDrug(XD_t)`.

`[OBSERVED]` **`XD_t`, not `XD`, is the correct argument** — `build_GAN_A/B/C` all pass the GAN pretraining corpus (`:358`, `:404`, `:447`). Measured input: `XD_t shape=(50068, 200) dtype=float64, 76.4 MiB` — the 50,068-entry drug corpus, on the PDBbind development dataset (`--dataset_path data/pdb/`, `--is_log 0`), identical to Stage 06.

`ganForDrug` was **not** monkey-patched, wrapped, shortened, or reimplemented. The iteration count was not touched.

---

## Static Training-Loop Verification

Verified against current source before execution, not assumed from the earlier audit.

| Question | Finding |
|---|---|
| Hard-coded schedule | `[CODE]` `:152` — **`train(5000, 5, 500, XD)`** → `iterations=5000`, `batch_size=5`, `interval=500`. Confirms the Stage 05/06 reading. |
| Any repository-supported way to limit iterations? | **No.** `train` is a closure defined at `:103` inside `ganForDrug`; all three arguments are literals at the call site; `ganForDrug(XD)` takes **only the data array — no `FLAGS`**. No CLI argument reaches the loop. |
| One GAN or several? | **One.** Three models are built **once** each, before the loop: `dis_v` (`:143`), `gen_v` (`:147`), `gan_v` (`:149`). No model is rebuilt inside the loop. |
| Work per iteration | `[CODE]` 4 graph executions at batch 5: `gen_v.predict(z)` (`:115`), `dis_v.train_on_batch` ×2 (`:118-119`), `gan_v.train_on_batch` (`:122`). |
| Progress logging | Only when `(iteration+1) % 500 == 0` (`:125-129`) → **exactly 10 observable checkpoints** for a 5000-iteration run. Keras itself also prints a one-line progress bar per `predict()` call. |
| Memory released between sub-runs? | **Not applicable** — there are no sub-runs. `reset_keras()` exists but is called at `:711`/`:843`, after a whole grid point, never inside `ganForDrug`. |
| Existing callback or timeout? | **None.** No `EarlyStopping`, no callback list, no time limit anywhere in the GAN path. |

---

## Safety Budget

Defined before execution, enforced entirely from the parent process:

| Limit | Value | Outcome |
|---|---|---|
| Wall-clock | **600 s (10 min)** | **not reached** |
| GPU VRAM | 1800 MiB allocated | **not reached** (GPU unused, 0 MiB throughout) |
| Iterations | stop early if wall-clock would be exceeded | **not needed** |

No memory workaround was used: no memory-growth change, no mixed precision, no CPU offload, no checkpointing, no batch or model modification.

---

## Execution Method

A **fresh subprocess**, as required — TF state cannot leak into this session, and a runaway call stays killable.

- **Parent** (`gan_timing_parent.py`, scratchpad): launches the child with `CREATE_NEW_PROCESS_GROUP`, reads stdout line-by-line and **timestamps each line on arrival** (external instrumentation — the repository's own prints carry no timestamps), samples RSS and `nvidia-smi` every ~5 s, and enforces the 600 s limit with `terminate()`.
- **Child** (`gan_timing_child.py`, scratchpad): sets the Stage 06 PDBbind configuration, runs `DataSet.parse_data`, then calls `RE.ganForDrug(XD_t)` directly.

Neither script is inside the repository, and neither alters it.

---

## Observed Iterations

`[OBSERVED]` **All 10 progress checkpoints appeared; the loop ran to its full 5000 iterations.**

| Iteration | Parent timestamp | Interval | ms / iteration |
|---:|---:|---:|---:|
| 500 | 46.29 s | 33.78 s | 67.6 |
| 1000 | 78.53 s | 32.24 s | 64.5 |
| 1500 | 111.29 s | 32.76 s | 65.5 |
| 2000 | 145.66 s | 34.37 s | 68.7 |
| 2500 | 181.22 s | 35.56 s | 71.1 |
| 3000 | 223.45 s | 42.23 s | 84.5 |
| 3500 | 267.56 s | 44.11 s | 88.2 |
| 4000 | 314.44 s | 46.88 s | 93.8 |
| 4500 | 364.46 s | 50.02 s | 100.0 |
| **5000** | **416.79 s** | 52.33 s | **104.7** |

- First observable iteration: **500** (at 46.29 s)
- Last observable iteration: **5000** (at 416.79 s)
- Iterations 1–499 are **not individually observable** — the repository prints only every 500.

`[OBSERVED]` Loss values at every checkpoint from 500 onward:

```
500  [D loss: 0.000012 , acc: 100.00] [G loss: 15.424948]
1000 [D loss: 0.000001 , acc: 100.00] [G loss: 15.424948]
...
5000 [D loss: 0.000000 , acc: 100.00] [G loss: 15.424948]
```

**The generator loss is identical to six decimal places — 15.424948 — at all ten checkpoints**, while discriminator accuracy is 100.00% and its loss is ~0 throughout.

---

## Timing

`[OBSERVED]`

| Measurement | Value |
|---|---|
| `ganForDrug()` call elapsed | **404.29 s (6 min 44 s)** |
| Total experiment elapsed (incl. TF import + `parse_data`) | 417.81 s |
| TF import → device enumeration | 7.8 s |
| `parse_data` (PDBbind) | 2.82 s |
| Mean per iteration | **80.9 ms** |
| Fastest 500-block | 64.5 ms/iter (iterations 500–1000) |
| Slowest 500-block | **104.7 ms/iter** (iterations 4500–5000) |
| **Slowdown, first block → last** | **1.55×** |
| Child return code | **0** |

---

## Memory

| Measurement | Value |
|---|---|
| Peak GPU VRAM used | **0 MiB** (GPU never used) |
| GPU utilisation | 0% at every sample |
| CUDA OOM | **none** |
| Child process RSS — *direct observations* | **765 MiB** at t ≈ 100 s; **967 MiB** at t ≈ 250 s |

**A measurement caveat, stated plainly.** The parent's automated RSS sampler reported a constant `4.2 MiB`, which is **invalid**. The venv's `python.exe` is a small launcher shim (PID 10960) that spawns the real interpreter (PID 15068); `Popen.pid` returned the shim, so the sampler tracked the wrong process. The two RSS figures above come from direct `Get-Process` queries against the real worker during the run and are reliable; the automated series is not, and is discarded rather than reported.

`[OBSERVED]` RSS grew from 765 MiB to 967 MiB between t ≈ 100 s and t ≈ 250 s — **+202 MiB while the model set was fixed and no new data was loaded.**

---

## Termination Reason

**STOP A — normal completion.**

```
completed            = yes
iterations           = 5000 (full hard-coded schedule)
elapsed              = 404.29 s (call) / 417.81 s (experiment)
peak GPU memory      = 0 MiB (GPU unused)
child_returncode     = 0
termination_reason   = child_stdout_closed (i.e. normal exit)
```

The 600 s wall-clock limit was **not** reached; the VRAM threshold was **not** reached; no CUDA OOM; no unhandled exception. `ganForDrug` returned a `Sequential` model with **8 layers** — matching the discriminator stack enumerated in Stage 06 §6.2.

---

## Runtime Findings

### F1 — The call completes, on CPU, in under 7 minutes

`[OBSERVED]` 404.29 s for the full hard-coded 5000-iteration schedule. R6 is **resolved**: a single `ganForDrug` call is affordable in isolation.

### F2 — Per-iteration cost degrades monotonically within the call

`[OBSERVED]` 64.5 → 104.7 ms/iteration, a **1.55× slowdown**, rising in every block after iteration 1500. `[OBSERVED]` RSS rose ~202 MiB over the same period.

`[INFERRED]` The pattern — steadily increasing step time plus steadily growing memory, with a fixed model set and no new data — is consistent with graph/tensor accumulation in the TF/Keras session across repeated `predict()` and `train_on_batch()` calls. `[NOT ESTABLISHED]` The precise mechanism was not instrumented, and this is **not** claimed to be a defect: it is an observed runtime characteristic of this loop under TF 2.10.1.

**Consequence for extrapolation:** because the cost is not constant within a call, per-call cost cannot be assumed constant across repeated calls in one process either.

### F3 — The GAN does not appear to learn during this run

`[OBSERVED]` Discriminator accuracy **100.00%** and loss ≈ 0 from the first checkpoint; generator loss **exactly 15.424948 at all ten checkpoints**, unchanged to six decimal places across 4,500 iterations.

`[CODE]` Two relevant construction facts: the discriminator's output layer is `Dense(1, activation='tanh')` (`:94`) while the model is compiled with `binary_crossentropy` (`:144`) — `tanh` ranges over (−1, 1) whereas BCE expects (0, 1); and real inputs are rescaled `Xtrain / 32.5 - 1.0` (`:105`) to match the generator's `tanh` output range.

`[INFERRED]` A frozen generator loss alongside a saturated, 100%-accurate discriminator is the classic signature of discriminator dominance with vanishing generator gradient. The `tanh` + BCE pairing is a plausible contributing mechanism.

`[NOT ESTABLISHED]` This run does **not** establish that the GAN is broken, that this is a bug, or that the published results are affected. Only ten sampled checkpoints were observed, no generated outputs were inspected, no weight statistics were collected, and a single run on one dataset cannot characterise training dynamics. **Stated as an observation requiring dedicated investigation, not as a defect finding.**

### F4 — Nothing is persisted

`[OBSERVED]` The call wrote no checkpoint, no weights, no figure, and no log file. The only filesystem effect inside the repository was Python bytecode (`__pycache__`). This confirms Stage 03 §E17 at runtime.

---

## Interpretation

### Observed

- One `ganForDrug(XD_t)` call on PDBbind completes **all 5000 hard-coded iterations in 404.29 s on CPU**, returning an 8-layer `Sequential` discriminator, exit code 0.
- The GPU is **not used at all** by this path as the repository is configured here.
- Per-iteration time rises from 64.5 ms to 104.7 ms within the call; RSS rises ~202 MiB.
- Discriminator saturates at 100% accuracy; generator loss is constant at 15.424948 across all checkpoints.
- Peak VRAM 0 MiB; no OOM; no exception; wall-clock budget not reached.

### Inferred

- A single GAN pretraining call is **individually affordable** on this machine.
- The within-call degradation suggests the cost of *repeated* calls in one process may exceed a simple multiple of 404 s.
- The loss pattern suggests discriminator dominance and a non-learning generator in this configuration.

### Not established

- **Not** that 5000 iterations "take X hours" in any other context — this figure is for **one** call, on **CPU**, on **PDBbind**, for the **drug** GAN only.
- **Not** the cost of `ganForTarget` or `ganForBlosumTarget`. Those were **not run** (§8 forbids it). The protein GANs operate on 2000-length sequences at batch 10 and would plausibly cost more, but **no measurement exists** and none is offered.
- **Not** the cost of a full `build_GAN_A/B/C` call (A and B each build **two** GANs).
- **Not** GPU feasibility of this path — unmeasured, because the repository does not use the GPU here.
- **Not** whether the GAN is defective, or whether its behaviour affects published results.
- **Not** anything about DTA-network training, which follows GAN pretraining and was not exercised.

---

## Reproduction Implication

`[CODE]` Stage 03 §E5 established that `build_GAN_*` is invoked as `runmethod` **inside the innermost grid×fold loop** (`:781` / `:654`), and `nfold_1_2_3_setting_sample` calls the CV routine **twice** (validation pass, then test pass).

The README's PDBbind grid is `--num_windows 128 32` (2) × `--smi_window_lengths 4 8 16` (3) × `--seq_window_lengths 4 8 16` (3) = **18 grid points**, × **5 folds**, × **2 passes** = **180 invocations**.

> ### ESTIMATE — not an observed result
> For **variant C** (one GAN per build), 180 × 404.29 s ≈ **72,772 s ≈ 20.2 hours of drug-GAN pretraining alone**, excluding the DTA network's own 100-epoch training and all evaluation.
>
> This is an **estimate**, obtained by multiplying **one measured call** by a call count read from the code and README. It is offered because the per-call figure is a completed, fully-timed observation — but it assumes each call costs the same, which **F2 gives direct reason to doubt**: cost rose 1.55× *within* a single call.
>
> For **variants A and B**, which each pretrain **two** GANs, no estimate is given: the protein-GAN cost is unmeasured.

`[NOT ESTABLISHED]` Whether `reset_keras()` (called at `:711`/`:843` after each grid point) would reset the degradation of F2 — and indeed whether it runs at all on this machine, since it sets `config.gpu_options.visible_device_list = "0"` and no GPU is visible. That is a separate, untested question.

**Bottom line:** one GAN pretraining call is affordable; the repository's *full* grid, which re-runs pretraining 180 times, is not obviously affordable on this hardware and has not been measured end-to-end.

---

## Repository Integrity

`git status --short` after the experiment:

```
 M AGENTS.md          <- pre-existing (GitNexus stat-line, predates Stage 06)
 M CLAUDE.md          <- pre-existing (same)
 ? Co-VAE/CoVAE       <- pre-existing (nested git repo, untracked by parent)
?? THESIS_IMPLEMENTATION_PLAN.md
?? audit/04-dataset-fold/
?? audit/05-reproduction-feasibility/
?? audit/06-smoke-execution/
?? audit/07-training-feasibility/
```

**No source, dataset, or fold modification.**

Artifact scan of `DCGAN-DTA/DCGAN-DTA/`:

| Artifact | Found | Action |
|---|---|---|
| TensorFlow logs | none | — |
| Checkpoints / saved models | none | — |
| Generated model files | none | — |
| `figures/`, `logs/` directories | **not created** | — |
| Temporary files | none | — |
| `__pycache__/` (5 × `cpython-310.pyc`) | **yes — created by this stage** (21:25) | **removed**; the directory contained no pre-existing file |

`[PRESERVED]` The full 15,158-line run log is kept **outside** the repository at `…/scratchpad/gan_timing.log` (692 KB). It contains the timestamped iteration checkpoints and is the primary evidence for this report — it was deliberately **not** deleted.

The external `logs-dcgan/` directory is empty: `ganForDrug` never calls `logging()`.

---

## Stage 07B Verdict

**COMPLETE — STOP A (normal completion). R6 RESOLVED.**

| Item | Result |
|---|---|
| Path tested | `ganForDrug(XD_t)` → `train(5000, 5, 500, XD)` — unmodified |
| Schedule confirmed | **5000 iterations, batch 5, log interval 500** |
| Device | **CPU only** (GPU visible to `nvidia-smi` but not to TF; 0 MiB used throughout) |
| Elapsed | **404.29 s** for the call; 417.81 s for the experiment |
| Last observable iteration | **5000** — the full schedule; all 10 checkpoints seen |
| Peak memory | GPU **0 MiB**; process RSS **765 → 967 MiB** (directly observed) |
| Termination | **Normal completion**, exit code 0 |
| Completed fully? | **Yes** |
| Practical feasibility | **One call: feasible** (~6.7 min, CPU, no OOM). **Full grid: not established** — ~180 invocations are implied, estimated ≈20.2 h for variant C's drug GAN alone, and F2 suggests that estimate is optimistic. |
| Repository state | **Clean** |

**Not started, per instruction:** `ganForTarget`, `ganForBlosumTarget`, `build_GAN_*`, full DCGAN-DTA training, Stage 08, and any retry or parameter variation. One controlled measurement was taken and the stage stopped.

---

*End of Stage 7B — DCGAN-DTA. No repository source file, dataset, fold file, architecture, hard-coded constant, or CLI argument was modified; `ganForDrug` was not patched or wrapped; no retry was attempted.*

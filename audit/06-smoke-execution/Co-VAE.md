# Stage 6 — Minimal Instrumented Smoke Execution: Co-VAE

**Inputs:** Stage 02 forensic · Stage 03 traceability · Stage 04 dataset/fold · Stage 05 feasibility · `THESIS_IMPLEMENTATION_PLAN.md`
**Implementation:** `Co-VAE/CoVAE/` (upstream `github.com/LiminLi-xjtu/CoVAE`, branch `master`, HEAD `2c17268`)
**Development dataset:** Davis (pipeline-validation only — see §10)
**Date:** 2026-09-25

**Mode:** execution permitted for the first time. **No source file, dataset, or fold file was modified.** No `.cuda()` call was patched, no architecture changed. The virtual environment and all wheels live outside the repository, in the session scratchpad.

**Evidence tags:** `[RUNTIME]` (observed this stage) · `[CODE]` · `[DATA]` · `[DOC]` · `[INFERENCE]`

> **Headline: Stage 05 predicted Co-VAE's Tier 2 would be blocked without NVIDIA hardware. Hardware was available, and every boundary passed — construction, forward pass, and loss — with no OOM.**

---

## 1. Environment

### 1.1 Final environment (A1)

| Component | Version | Source of choice |
|---|---|---|
| OS | Windows 11 Enterprise 10.0.26200 | — |
| Python | **3.10.0** (`py -3.10`, pre-installed) | Stage 05 §1.1 — **hard ceiling ≤ 3.10** (`random.sample` on a set) |
| **PyTorch** | **1.13.1+cu116** | see §1.2 |
| torchvision | **not installed** | not imported by the repository `[CODE]` |
| CUDA runtime (reported by torch) | **11.6** | matches driver's native CUDA |
| numpy | **1.26.4** (pinned `<2`) | torch 1.13.1 has a NumPy 1.x ABI |
| pandas | 2.3.3 | repository dep |
| scikit-learn | 1.7.2 | repository dep |
| matplotlib | 3.10.9 | repository dep |
| tqdm | 4.70.1 | repository dep |
| pip | 24.3.1 (upgraded from 21.2.3) | build tool; see DCGAN report §1.4 |
| **GPU** | **NVIDIA GeForce MX330** | hardware |
| Compute capability | **6.1** (Pascal) | `[RUNTIME]` |
| GPU memory | **2048 MiB total, 1662 MiB free at start** | `[RUNTIME]` |
| Driver | **511.69** | `nvidia-smi` |

Environment root: `…/scratchpad/venv-covae` (outside the repository).

### 1.2 PyTorch version decision

| | |
|---|---|
| **Package** | `torch==1.13.1+cu116` |
| **Reason** | (a) `cu116` matches driver 511.69's **native CUDA 11.6**, avoiding reliance on minor-version compatibility; (b) 1.13.1 is the last release with `cu116` wheels; (c) it retains every legacy API the repository uses — `torch.cuda.FloatTensor` (`model.py:36`) and `torch.autograd.Variable`; (d) it is **below PyTorch 2.6**, avoiding the `torch.load(weights_only=True)` default that Stage 05 §1.2 flagged as blocker C7 against the repository's whole-module `torch.save`; (e) sm_61 is supported. |
| **Evidence** | Stage 05 §1.2 API inventory; driver's reported CUDA version `[DOC]` |
| **Verified** | `[RUNTIME]` `torch.version.cuda = 11.6`, `cuda.is_available() = True`, device `NVIDIA GeForce MX330`, capability `(6, 1)`. A GPU sanity matmul (1000×1000) executed successfully. |

**`[RUNTIME]` The driver/runtime pairing works. No CUDA Toolkit installation was required** — the wheel bundles its own runtime, exactly as the task anticipated.

### 1.3 Diagnostic output (A1, verbatim)

```
torch             : 1.13.1+cu116
torch.version.cuda: 11.6
cuda.is_available : True
device name       : NVIDIA GeForce MX330
capability        : (6, 1)
GPU memory        : free=1662 MiB / total=2048 MiB
sanity matmul     : OK (1000, 1000) allocated=7.6 MiB
```

Note the real budget is **1662 MiB**, not 2048 — roughly 386 MiB is already held by the display/context.

### 1.4 Environment provisioning note

The wheel is 2,433,843,745 bytes. A first download — assembled across several interrupted/resumed `curl` passes during session restarts — produced a file of **exactly the right size but corrupt** (`zlib.error: Error -3 while decompressing data: invalid code lengths set`). It was discarded and re-fetched in a single pass, then **integrity-verified before installation**:

```
members: 9491 ; first corrupt member: None ; ZIP INTEGRITY: OK
```

`[INFERENCE]` Size alone is not an integrity check for resumed downloads. Worth remembering for Stage 07.

Network findings (proxy, pip stalls) are environmental and identical to those recorded in the companion DCGAN-DTA report §1.4; they are not repeated here.

---

## 2. Exact commands executed (A3)

Run with cwd = `Co-VAE/CoVAE/`, interpreter = `…/venv-covae/Scripts/python.exe`.

**Non-default arguments supplied** — every one documented, none invented:

```
--dataset_path ./data/davis/   # default is ./data/kiba/ — the WRONG dataset, and the one
                               #   whose fold files are invalid (Stage 04 §5.3)
--max_smi_len 85               # default 100; 85 is the paper's Davis value (Stage 02)
--max_seq_len 1200             # default 1000; 1200 is the paper's Davis value
--num_windows 32               # NO DEFAULT; len()-ed at :248. 32 gives latent 3*32=96,
                               #   the value Stage 03 identified as the code's intended one
--smi_window_lengths 4         # NO DEFAULT; smallest value in the paper's filter-length grid
--seq_window_lengths 8         # NO DEFAULT; from the paper's grid
--lamda -5                     # HAS a default (-5) but it is a SCALAR under nargs='+',
                               #   so the default path raises. Supplied explicitly (Stage 05 C-2)
--problem_type 1               # pair-wise/warm — the simplest split
--batch_size 256               # repository default, matches paper
--log_dir <scratchpad>/logs-covae/   # default "/tmp" is an invalid absolute POSIX path on Windows,
                               #   and logging() does not makedirs. Pointed outside the repo
                               #   so the repository's logs/ placeholder stayed untouched
```

A second run used `--batch_size 64` to characterise memory scaling (§7). `--batch_size` is a documented repository flag; no other configuration differed.

The smoke script (in scratchpad, not the repository) calls only the repository's own functions: `argparser`, `DataSet.parse_data`, `get_random_folds`, `prepare_interaction_pairs`, `net`, `weights_init`, `loss_f`, `get_cindex`. **No `loss.backward()`, no `optimizer.step()`** — forward-only validation, as Part A8 prefers.

---

## 3. Boundary results (A9)

| Boundary | Result |
|---|---|
| Environment | **PASS** |
| Import | **PASS** |
| Argument parsing | **PASS** |
| Data loading | **PASS** |
| Batch creation | **PASS** |
| Model construction | **PASS** |
| Forward pass | **PASS** |
| Loss | **PASS** |
| GPU memory | **measured** — peak 805.77 MiB allocated / 980 MiB reserved at batch 256; **no OOM** |

---

## 4. Runtime evidence — data and folds (A4)

### 4.1 Davis loader — every Stage 04 measurement confirmed

```
parse_data OK (0.1s)
XD (68, 85)    float64  max index 49  (Embedding limit 63)
XT (442, 1200) float64  max index 24  (Embedding limit 24)
Y  (68, 442)   float64  observed=30056  min=5.0000  max=10.7959  mean=5.4515
observed pairs (label_row_inds) = 30056
```

| Quantity | Stage 04 predicted | Stage 06 observed | Match |
|---|---|---|:--:|
| XD | (68, 85) | (68, 85) float64 | ✔ |
| XT | (442, 1200) | (442, 1200) float64 | ✔ |
| Y | (68, 442) | (68, 442) float64 | ✔ |
| Observed pairs | 30,056 (**0% missing**) | **30,056** | ✔ |
| Affinity min / max | 5.0 / 10.7959 | **5.0000 / 10.7959** | ✔ |
| Affinity mean | 5.4515 | **5.4515** | ✔ |
| pKd transform applied | `-log10(Kd/1e9)` | confirmed by the range | ✔ |

### 4.2 The zero-margin embedding index — confirmed at runtime

`[RUNTIME]` **`XT max index 24` against `Embedding limit 24`.** Stage 04 §4.3 predicted exactly this: Davis's protein encoding sits on the **last valid index**, with zero headroom, because `net` uses `nn.Embedding(charseqset_size, 128)` **without** the `+1` guard that DCGAN-DTA applies.

It does not fail on this data. `[INFERENCE]` It would raise `IndexError` on any protein file containing the symbol `Z` (index 25). Safe here, fragile by construction — and a real hazard for the eventual hybrid method if it reuses this vocabulary.

### 4.3 Fold generation — the B1 probe (Stage 03's first CRITICAL blocker)

```
get_random_folds OK (0.02s)  sizes=[5010, 5010, 5009, 5009, 5009, 5009]
test=5009  train=25047  overlap=0
drugs train/test=68/68  targets train/test=442/442
```

> **`[RUNTIME]` Blocker B1 is CLEARED on Python 3.10.** `random.sample(indices, k)` with `indices` a `set` (`run_experiments.py:41`) executes without error. Stage 03 graded this **CRITICAL** and Stage 05 confirmed the ≤ 3.10 ceiling; the runtime now confirms that the ceiling is **sufficient** — no source change was needed, only the correct interpreter.

**`[RUNTIME]` Runtime unknown CV-R5 is resolved.** Stage 04 §6.2 had to simulate these folds with a substitute sampler because the repository's own sampler could not run. The real sizes are `[5010, 5010, 5009, 5009, 5009, 5009]`, and the derived split is **test = 5,009 / train = 25,047 — exactly the figures Stage 04's simulation predicted for `problem_type 1`**, with zero overlap. Entity overlap is 68/68 drugs and 442/442 targets, correct for a warm pair-wise split.

### 4.4 One batch (A5)

```
drug   (256, 85)   torch.float32  device=cpu
target (256, 1200) torch.float32  device=cpu
label  (256,)      torch.float32  device=cpu
```

`[RUNTIME]` Confirms Stage 05 §4.3: `prepare_interaction_pairs` casts to **float32** per pair, and the batch is assembled on **CPU** — `.cuda()` is invoked only inside `net.forward`.

---

## 5. Model construction (A6)

```
NUM_FILTERS=32  FILTER_LENGTH1=4  FILTER_LENGTH2=8
[before construction] allocated=    0.00 MiB  reserved=   0.00 MiB  free=1661.87 MiB  total=2047.88 MiB
construction OK (1.17s)
[after construction]  allocated=   49.89 MiB  reserved=  66.00 MiB  free=1276.43 MiB  total=2047.88 MiB
parameters: 12,931,834  (49.33 MiB as fp32)
latent dim (3*NUM_FILTERS) = 96
```

**`[RUNTIME]` `net(FLAGS, 32, 4, 8).cuda()` constructs successfully.** Stage 05 §5.1 traced the architecture as dimensionally self-consistent; runtime confirms it. **12,931,834 parameters (49.33 MiB fp32)**, and **latent dimension 96** — matching the value Stage 03 recovered from code.

`model.apply(weights_init)` ran without error. `[CODE]` `weights_init` (`:99-111`) covers `Linear`, `BatchNorm1d`, `LSTM` and `Conv1d` but **not** `Embedding` or `ConvTranspose1d`, which therefore keep PyTorch's default initialisation — resolving runtime unknown **CV-R4** by inspection, corroborated by the clean run.

Note the 386 MiB gap between `free` before (1661.87) and after (1276.43) construction while only 66 MiB was reserved: that is the **CUDA context**, not the model.

---

## 6. Forward pass and loss (A7, A8)

### 6.1 Forward — all nine outputs, correct shapes

```
forward OK (1.695s)
  pre_affinity   (256,)          float32  cuda:0
  new_drug       (256, 85, 64)   float32  cuda:0
  new_target     (256, 1200, 25) float32  cuda:0
  drug(int)      (256, 85)       int64    cuda:0
  target(int)    (256, 1200)     int64    cuda:0
  mu_drug        (256, 96)       float32  cuda:0
  logvar_drug    (256, 96)       float32  cuda:0
  mu_target      (256, 96)       float32  cuda:0
  logvar_target  (256, 96)       float32  cuda:0
```

`[RUNTIME]` The decoder output shapes confirm Stage 05 §5.1's trace precisely: `new_drug` reconstructs to `(batch, max_smi_len, charsmiset_size)` = (256, 85, **64**) and `new_target` to (256, 1200, **25**), i.e. the decoders restore the input length exactly and project onto the vocabulary. The latent tensors are (256, **96**).

**`[RUNTIME]` No OOM at batch 256, and `.squeeze()` (runtime unknown CV-R3) did not misfire** — batch 256 is not size-1, so the hazard Stage 05 flagged did not arise. It remains untested for a trailing batch of size 1.

### 6.2 Loss — forward-only, replicating `train():130-136`

```
loss_affinity (MSE)    = 29.122047
loss_drug   (recon+KL) = 354.366150
loss_target (recon+KL) = 3887.318848
lamda=-5   10**lamda=1e-05   max_smi_len/max_seq_len=0.0708333
TOTAL                  = 29.128345
-> affinity term share = 99.9784%
get_cindex (untrained) = 0.524125
```

All three terms compute. The total reproduces `run_experiments.py:136` exactly:
`loss_affinity + 10**lamda * (loss_drug + (max_smi_len/max_seq_len) * loss_target)`.

> ### `[RUNTIME]` A new, quantified finding — the co-regularization is numerically negligible at the paper's own λ
>
> At **λ = −5, which is one of the two values Table 1 reports**, the two VAE terms together contribute
> `1e-05 × (354.366 + 0.0708 × 3887.319) = 0.00630` out of a total of **29.128** — i.e. **0.0216%**.
> The affinity MSE term is **99.978%** of the objective.
>
> Reproduced at batch 64: affinity share **99.9789%** — stable, not a batch artefact.
>
> `[INFERENCE]` This gives an empirical magnitude to Stage 03's **C6** (λ weights the VAE terms rather than the affinity term, with an extra undocumented `max_smi_len/max_seq_len` factor on the target branch). At the paper's own λ, the "co-regularized" objective is, numerically, almost pure affinity regression. This is a measurement of the objective's composition at initialisation, **not** a claim about trained behaviour or about what the paper intended — the terms do carry gradient, and their relative scale could change during training.

`get_cindex = 0.524` on an untrained model is consistent with near-random ranking, as expected.

---

## 7. GPU memory (A6, A7, and the 2 GB constraint)

| Configuration | Peak **allocated** | Peak **reserved** | Device free after forward | OOM |
|---|---:|---:|---:|:--:|
| batch **256** (repository default) | **805.77 MiB** | **980.00 MiB** | **86.43 MiB** | **no** |
| batch **64** | **248.76 MiB** | **302.00 MiB** | 764.43 MiB | no |

Model parameters are 49.33 MiB in both cases; the rest is activations.

**`[RUNTIME]` Batch 256 forward-only leaves just 86 MiB free on the device.** With ~386 MiB of CUDA context plus 980 MiB reserved, the 2048 MiB card is close to saturated by a *single forward pass under `torch.no_grad()`*.

`[INFERENCE]` **Training at batch 256 on this GPU is very unlikely to fit.** A backward pass must retain activations that `no_grad` discards, and the peak here is already ~805 MiB allocated. Scaling is roughly linear in batch size (4× batch → ~3.2× peak). This is a hardware-capacity observation for planning, **not** an attempt to work around anything: no architecture was changed, and `--batch_size` is a documented repository flag. Establishing the actual training-memory ceiling is a task for a later stage that is permitted to run a backward pass.

---

## 8. Deviations from plan

| # | Deviation | Justification |
|---|---|---|
| D-g | `torch==1.13.1+cu116` selected (Stage 05 left the version `[UNKNOWN]`) | §1.2 — matches driver's native CUDA 11.6; retains legacy APIs; below the 2.6 `torch.load` change |
| D-h | `numpy==1.26.4` pinned | torch 1.13.1 has a NumPy 1.x ABI |
| D-i | pip upgraded 21.2.3 → 24.3.1 in the venv; large wheel fetched with `curl` | build-tool/network workaround, see DCGAN report §1.4 |
| D-j | Second run at `--batch_size 64` | memory characterisation using a documented repository flag; scientific configuration otherwise identical |
| D-k | `--log_dir` pointed outside the repository | the default `/tmp` is invalid on Windows and `logging()` does not `makedirs`; keeps the repository's `logs/` placeholder untouched |

**No deviation touched repository source, data, folds, `.cuda()` calls, or the architecture.**

---

## 9. Repository modification audit (Part D) — covers both projects

`git status --short` at the project root, after all execution:

```
 M AGENTS.md          <- pre-existing (GitNexus stat-line rewrite, predates Stage 06)
 M CLAUDE.md          <- pre-existing (same)
 ? Co-VAE/CoVAE       <- pre-existing (nested git repo, untracked by parent)
?? THESIS_IMPLEMENTATION_PLAN.md
?? audit/04-dataset-fold/
?? audit/05-reproduction-feasibility/
?? audit/06-smoke-execution/
```

**No source file, dataset, or fold file appears as modified.**

**Checksums re-verified against the pre-execution baseline:**

| File | Baseline MD5 | After execution | |
|---|---|---|:--:|
| `Co-VAE/CoVAE/data/davis/ligands_iso.txt` | `eb7c083b…f50d0` | `eb7c083b…f50d0` | ✔ |
| `Co-VAE/CoVAE/data/davis/proteins.txt` | `104e19e7…70a` | `104e19e7…70a` | ✔ |
| `DCGAN-DTA/DCGAN-DTA/data/pdb/folds/test_fold_setting1.txt` | `49024eef…4b778` | `49024eef…4b778` | ✔ |

**Output directories untouched:** `Co-VAE/CoVAE/{logs,figures,result}/` each still contain only their original 2-byte `readme` placeholder, dated Sep 11. Nothing was written into either repository.

**Environment artifacts.** Importing the modules created `__pycache__` bytecode. These were removed **selectively**:

- `DCGAN-DTA/DCGAN-DTA/__pycache__/` — the **directory itself** was created by this stage (Sep 25 18:37); removed entirely.
- `Co-VAE/CoVAE/__pycache__/` — **pre-existed**, and contains `datahelper.cpython-313.pyc` dated **Sep 11**, visible in the Stage 04 file inventory. Only the five `*.cpython-310.pyc` files created by this stage were deleted; **the pre-existing 313 artifact was preserved.**

Verified after cleanup: `Co-VAE/CoVAE/__pycache__/` contains exactly `datahelper.cpython-313.pyc`, and `DCGAN-DTA/DCGAN-DTA/__pycache__` no longer exists.

Both virtual environments, all wheels (2.9 GB), and all logs are under the session scratchpad — **entirely outside the repository**.

---

## 10. Development dataset

**Davis remains the Co-VAE development dataset** (plan D2), and Stage 06 strengthens it: `parse_data` completes in 0.1 s, every Stage 04 figure held exactly, fold generation works on Python 3.10, and the full construct→forward→loss path fits in 2 GB at the repository's default batch size.

> **Davis is a pipeline-validation dataset, not the scientific comparison dataset.** Its **100% density**, **69.64% censoring at pKd 5.0**, and **442 → 379 target-sequence collapse** (Stage 04) make it a weak evidential base, and the collapse actively compromises the paper's flagship new-target setting.
>
> **KIBA remains a later-stage dataset**, with its non-published runtime matrix (1,954 × 217), 8,958 out-of-range fold indices, degenerate AUC threshold (109,286 positives vs 10 negatives), and its status as `--dataset_path`'s default — a trap to guard against in every invocation.

---

## Stage 06 Status

**COMPLETE — SMOKE PASSED**

Every defined boundary was crossed successfully, including the full Tier 2 that Stage 05 expected might be unreachable.

### Runtime evidence

torch 1.13.1+cu116 on Python 3.10.0 drives an MX330 (cc 6.1, 2048 MiB). `parse_data` reproduces **every** Stage 04 Davis measurement exactly — (68, 85), (442, 1200), (68, 442), 30,056 observed, pKd 5.0000–10.7959, mean 5.4515. **`get_random_folds` runs, clearing blocker B1**, yielding sizes `[5010, 5010, 5009×4]` → test 5,009 / train 25,047 with zero overlap, matching Stage 04's simulation. `net(...).cuda()` builds **12,931,834 parameters**, latent **96**. One forward pass returns all nine tensors with correct shapes in 1.70 s. All three loss terms compute; the affinity term is **99.978%** of the objective at λ = −5.

### Environment

Python 3.10.0 · **torch 1.13.1+cu116** · CUDA runtime 11.6 · numpy 1.26.4 · pandas 2.3.3 · scikit-learn 1.7.2 · matplotlib 3.10.9 · tqdm 4.70.1 · torchvision not installed (not needed). GPU: NVIDIA GeForce MX330, cc 6.1, driver 511.69, 2048 MiB. Isolated venv outside the repository.

### Successful boundaries

Environment · Import · Argument parsing · Data loading · Batch creation · **Model construction** · **Forward pass** · **Loss**.

### First runtime blocker

**None was encountered.** No boundary failed. The three blockers Stage 03 graded CRITICAL were all neutralised without touching source: **B1** cleared by running Python 3.10; **B2** cleared by passing `--lamda -5` explicitly (confirming Stage 05's C-2 re-diagnosis — the flag has a default, it is merely a scalar under `nargs='+'`); **B3** satisfied by real NVIDIA hardware.

### GPU memory

**Measured, no OOM.** Batch 256: peak **805.77 MiB allocated / 980.00 MiB reserved**, leaving **86.43 MiB** free on device. Batch 64: peak **248.76 MiB / 302.00 MiB**. Parameters 49.33 MiB. The card is near-saturated by one forward pass at batch 256, so training at that batch is unlikely to fit — a capacity observation, not a change made.

### Deviations

Five, all listed in §8; none touched source, data, folds, `.cuda()` calls, or the architecture.

### What this proves

That the Co-VAE repository, **entirely unmodified**, imports, loads Davis, generates folds, assembles a batch, constructs its model on a 2 GB GPU, executes a forward pass, and computes its three-term loss — under Python 3.10 with a CUDA-enabled PyTorch. It proves Stage 04's Davis audit is accurate at runtime, that Stage 05's architectural trace was correct, and that the objective at λ = −5 is 99.978% affinity MSE at initialisation.

### What this does NOT prove

It does **not** prove paper reproduction, numerical result reproduction, or the correctness of the published methodology. It does **not** prove that training converges, that a backward pass fits in 2 GB, or that any published number is recoverable — Stage 05 §8.3 already rated **result** reproduction BLOCKED for reasons execution cannot change (four of five experiment families have no code path, MAE is unimplemented, KIBA's runtime matrix is not the published one). It does **not** demonstrate absence of data leakage — Stage 04 **confirmed** leakage, with 13/73 test targets sequence-identical to training under the target-wise setting. It says nothing about either model's quality relative to the other, and the single-batch loss values are initialisation artefacts, not results.

### Next Single Step

**Measure whether one training step — a single forward *plus backward* on one batch — fits in the 2 GB GPU, starting at `--batch_size 64` and stopping at the first OOM.**

This is the one unknown that governs every later stage for Co-VAE: §7 shows a forward-only pass at batch 256 already leaves 86 MiB free, so the repository's default batch is likely untrainable on this hardware, and the largest workable batch determines whether native-dataset reproduction is feasible here at all or needs different hardware. It uses only the documented `--batch_size` flag, needs no source change, and is bounded — one step, not one epoch. It requires explicit approval, since it is the first backward pass.

---

*End of Stage 6 — Co-VAE. No repository source file, dataset, or fold file was modified; no `.cuda()` call was patched; no architecture was changed; no training step was executed.*

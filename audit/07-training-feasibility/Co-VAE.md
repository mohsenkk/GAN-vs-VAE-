# Stage 7A — One-Step Training Feasibility: Co-VAE

**Question:** what is the largest `--batch_size` at which Co-VAE can complete **exactly one training step** on the available NVIDIA GeForce MX330 (2 GB)?
**Dataset:** Davis (development dataset, consistent with Stage 06)
**Date:** 2026-09-25
**Scope:** Stage 07A only. No DCGAN-DTA work was performed.

**Mode:** execution. **No source file, dataset, or fold file was modified.** No architecture change, no hard-coded constant edited. Only the repository's own documented `--batch_size` flag was varied.

> ### This is training-feasibility evidence, not reproduction evidence.
> It measures whether the hardware can execute one optimizer step of the repository's native training loop. It says nothing about convergence, about any published number, or about the correctness of the methodology. Stage 05 §8.3 rated Co-VAE's **result reproduction BLOCKED** for reasons no amount of execution can change, and Stage 07A does not alter that.

**Evidence tags:** `[RUNTIME]` · `[CODE]` · `[INFERENCE]`

---

## 1. Environment

Unchanged from Stage 06 — the same isolated venv was reused, nothing installed or upgraded.

| Component | Version |
|---|---|
| OS | Windows 11 Enterprise 10.0.26200 |
| Python | 3.10.0 (isolated venv, outside the repository) |
| PyTorch | **1.13.1+cu116** |
| CUDA runtime (torch) | **11.6** |
| numpy / pandas / scikit-learn | 1.26.4 / 2.3.3 / 1.7.2 |
| GPU | **NVIDIA GeForce MX330**, compute capability **(6, 1)** |
| GPU memory | **2047.88 MiB total; 1661.87 MiB free before allocation** |
| Driver | 511.69 |

`[RUNTIME]` The usable budget is **1661.87 MiB**, not 2048 — roughly 386 MiB is held by the display and CUDA context before PyTorch allocates anything.

---

## 2. Exact configuration

Identical across all three runs except `--batch_size`:

```
--dataset_path ./data/davis/
--max_smi_len 85
--max_seq_len 1200
--num_windows 32
--smi_window_lengths 4
--seq_window_lengths 8
--lamda -5
--problem_type 1
--batch_size {64 | 128 | 256}      <-- the ONLY variable
--log_dir <scratchpad>/logs-covae/
```

Each batch size ran in a **fresh process**, so no measurement inherited a warm allocator.

### 2.1 What "one training step" means here

The driver script replicates `run_experiments.py` `train():115-141` for a single batch and then stops:

```
model = net(FLAGS, 32, 4, 8).cuda()      # general_nfold_cv:276
model.apply(weights_init)                #                 :277
model.train()                            # train():116
loss_func  = nn.MSELoss()                #        :117
optimizer  = optim.Adam(model.parameters())  #    :118
optimizer.zero_grad()                    #        :126
affinity   = Variable(affinity).cuda()   #        :127
... = model(drug, target, FLAGS, ...)    #        :128
loss_affinity = loss_func(pre_affinity, affinity)         # :131
loss_drug     = loss_f(new_drug,   drug,   mu_d, logvar_d) # :132
loss_target   = loss_f(new_target, target, mu_t, logvar_t) # :133
loss = loss_affinity + 10**lamda * (loss_drug
                    + max_smi_len/max_seq_len * loss_target)  # :136
loss.backward()                          #        :137
optimizer.step()                         #        :138
<< STOP >>
```

**One batch. One step. No loop, no epoch, no second batch.** `get_cindex` (`:134`) was omitted as it is a metric, not part of the memory path.

**No workarounds were used, as required:** no AMP or `autocast`, no gradient checkpointing, no gradient accumulation, no CPU offload, no `max_split_size_mb` tuning, no architecture change. This measures the repository's **native** training footprint.

---

## 3. Results

### 3.1 Summary

| `--batch_size` | Result | Peak **allocated** | Peak **reserved** | Device free after | Step wall-clock |
|---:|---|---:|---:|---:|---:|
| **64** | ✅ **STEP COMPLETED** | **529.66 MiB** | **724.00 MiB** | 312.43 MiB | **1.525 s** |
| **128** | ✅ **STEP COMPLETED** | **1009.29 MiB** | **1072.00 MiB** | 222.43 MiB | **1.700 s** |
| **256** | ❌ **CUDA OUT OF MEMORY** | 838.72 MiB *(at failure)* | 944.00 MiB | 122.43 MiB | — |

Per the specified procedure, testing stopped at the first OOM.

### 3.2 Timing breakdown (successful runs)

| batch | forward | backward | `optimizer.step()` | whole step | process wall-clock incl. setup |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.207 s | 0.270 s | 0.038 s | **1.525 s** | 2.64 s |
| 128 | 1.128 s | 0.523 s | 0.037 s | **1.700 s** | 2.85 s |

`[INFERENCE]` The forward time is dominated by one-off CUDA kernel autotuning, which is why it barely changes between 64 and 128. Backward scales as expected (0.270 → 0.523 s, ≈1.9× for 2× batch). `optimizer.step()` is essentially constant at ~0.037 s, being a function of parameter count, not batch size.

### 3.3 Phase-by-phase memory — batch 64

```
[start               ] alloc=   0.00  reserved=   0.00  peakAlloc=   0.00  devFree=1661.87
[after construction  ] alloc=  49.89  reserved=  66.00  peakAlloc=  49.89  devFree=1276.43
[after optimizer init] alloc=  49.89  reserved=  66.00  peakAlloc=  49.89  devFree=1276.43
[after forward       ] alloc= 451.40  reserved= 554.00  peakAlloc= 482.08  devFree= 512.43
[after loss          ] alloc= 460.38  reserved= 554.00  peakAlloc= 482.08  devFree= 512.43
[after backward      ] alloc= 109.26  reserved= 636.00  peakAlloc= 529.66  devFree= 400.43
[after optimizer.step] alloc= 208.08  reserved= 724.00  peakAlloc= 529.66  devFree= 312.43
```

### 3.4 Phase-by-phase memory — batch 128

```
[after construction  ] alloc=  49.89  reserved=  66.00  peakAlloc=  49.89  devFree=1276.43
[after forward       ] alloc= 854.43  reserved= 994.00  peakAlloc= 915.78  devFree=  72.43
[after loss          ] alloc= 872.74  reserved= 994.00  peakAlloc= 915.78  devFree=  72.43
[after backward      ] alloc= 118.58  reserved= 808.00  peakAlloc=1009.29  devFree= 228.43
[after optimizer.step] alloc= 217.25  reserved= 814.00  peakAlloc=1009.29  devFree= 222.43
```

`[RUNTIME]` Three structural observations, identical in both runs:

1. **`Adam` is lazy.** `optimizer init` adds **0 MiB** — the exponential-moving-average buffers are allocated on the first `step()`, which is why `alloc` rises from ~109–119 MiB to ~208–217 MiB afterwards. That residual is parameters (49.89) + gradients (49.89) + two Adam state tensors (≈99.8) ≈ 199.6 MiB, matching the observed ~208/217 MiB. It is **independent of batch size** and persists for the whole of training.
2. **Peak occurs during backward, not forward.** Activations retained from forward are still live while gradient buffers are being filled.
3. **Retained activations roughly double the forward cost.** Stage 06's `no_grad` forward at batch 64 peaked at 248.76 MiB; with gradients enabled the same forward peaks at **482.08 MiB**.

### 3.5 The OOM at batch 256 — verbatim

```
RESULT: CUDA OUT OF MEMORY  batch=256
CUDA out of memory. Tried to allocate 226.00 MiB (GPU 0; 2.00 GiB total capacity;
838.72 MiB already allocated; 122.43 MiB free; 944.00 MiB reserved in total by PyTorch)
If reserved memory is >> allocated memory try setting max_split_size_mb to avoid
fragmentation. See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

`[RUNTIME]` **The failure occurred inside the forward pass — the run never reached `loss.backward()`.** The phase log ends at `after optimizer init`; no `after forward` line was emitted.

`[INFERENCE]` This is a sharper result than "training does not fit". At batch 256, **grad-enabled forward alone exceeds the device**, even though Stage 06 showed that the *same* forward under `torch.no_grad()` completed with a peak of 805.77 MiB. The difference is precisely the retained activation graph.

The suggestion in the error text about `max_split_size_mb` was **deliberately not acted on**: it is a memory-management workaround, and the brief is to measure native feasibility. Reserved (944 MiB) is not >> allocated (838.72 MiB) here in any case, so fragmentation is not the limiting factor — the requirement genuinely exceeds the budget.

---

## 4. Interpretation

### 4.1 The feasible boundary

> **Largest batch size verified to complete one training step: `--batch_size 128`.**
> **First failure: `--batch_size 256` — CUDA OOM during forward.**

The repository's **default and the paper's stated batch size is 256** (`arguments.py:49`; Stage 02). `[INFERENCE]` **Co-VAE cannot be trained at its published batch size on this GPU.** Training on this hardware requires `--batch_size 128` or lower — a documented CLI change, not a code change, but a **deviation from the published configuration** that must be declared in any results derived from it.

The exact maximum lies somewhere in the interval **(128, 256)**. It was not bisected, because the specified procedure was to stop at the first OOM. 128 is the largest value on the prescribed 64/128/256 ladder that works.

### 4.2 Measured scaling

| batch | peak allocated | Δ vs previous |
|---:|---:|---:|
| 64 | 529.66 MiB | — |
| 128 | 1009.29 MiB | +479.63 MiB |
| 256 | *(OOM)* | — |

`[INFERENCE]` Peak allocation is close to affine in batch size: roughly **≈50 MiB fixed (parameters) + ≈7.5 MiB per sample**. Extrapolating to 256 predicts ≈**1969 MiB**, well beyond the 1661.87 MiB available — consistent with the observed failure, and consistent with the failure arriving early, inside forward.

### 4.3 Headroom at the feasible boundary

At batch 128 the peak reserved is 1072.00 MiB against 1661.87 MiB available — roughly **590 MiB of headroom**, about 35%. `[INFERENCE]` That margin is adequate for a single step but should not be assumed adequate for a full training run: this measurement excludes evaluation batches, checkpoint serialisation (`torch.save(model, …)` at `:289`/`:376`), and any allocator fragmentation that accumulates over thousands of steps. A sustained run may behave differently, and that remains unmeasured.

### 4.4 What this does not establish

- **Not** that training converges, or that any epoch completes.
- **Not** any statement about paper reproduction — Stage 05 rated Co-VAE's result reproduction **BLOCKED** independently of hardware (four of five experiment families have no code path, MAE is unimplemented, KIBA's runtime matrix is not the published one).
- **Not** a throughput or benchmark figure. The timings are single-step, cold-cache, and include one-off autotuning.
- **Not** a claim about DCGAN-DTA, which was untouched in this stage.
- **Not** a statement about the *scientific* consequences of batch 128 vs 256. Batch size interacts with the optimiser and the reported metrics; changing it is a protocol deviation whose effect on results is unmeasured.

---

## 5. Repository modification audit

`git status --short` after all three runs:

```
 M AGENTS.md          <- pre-existing (GitNexus stat-line rewrite, predates Stage 06)
 M CLAUDE.md          <- pre-existing (same)
 ? Co-VAE/CoVAE       <- pre-existing (nested git repo, untracked by parent)
?? THESIS_IMPLEMENTATION_PLAN.md
?? audit/04-dataset-fold/
?? audit/05-reproduction-feasibility/
?? audit/06-smoke-execution/
```

**No source file, dataset, or fold file is modified.**

Checksums re-verified against the Stage 06 baseline:

| File | MD5 | |
|---|---|:--:|
| `data/davis/ligands_iso.txt` | `eb7c083b7ea6943bb63dc4f06bbf50d0` | ✔ unchanged |
| `data/davis/proteins.txt` | `104e19e7ffa8240c606e22ac5d66870a` | ✔ unchanged |
| `data/davis/drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt` | `c04de163c5faaa98b9d4cadcbe475bc9` | ✔ unchanged |

`Co-VAE/CoVAE/{logs,figures,result}/` still contain only their original 2-byte `readme` placeholders dated Sep 11 — **nothing was written into the repository**, and `--log_dir` was pointed at the scratchpad for exactly this reason.

**Artifacts removed:** importing the modules regenerated five `*.cpython-310.pyc` files in `Co-VAE/CoVAE/__pycache__/`. These were created by this stage and were deleted. The **pre-existing** `datahelper.cpython-313.pyc` (dated Sep 11, present since before Stage 04) was **preserved**. Verified after cleanup: the directory contains only that one file.

All logs and the driver script remain in the session scratchpad, outside the repository.

---

## Stage 07A Status

**COMPLETE**

| Item | Result |
|---|---|
| Batch sizes tested | **64, 128, 256** |
| Succeeded | **64, 128** |
| First OOM | **256** (during forward, before backward) |
| Peak allocated | 64 → **529.66 MiB** · 128 → **1009.29 MiB** · 256 → 838.72 MiB at failure |
| Peak reserved | 64 → **724.00 MiB** · 128 → **1072.00 MiB** · 256 → 944.00 MiB at failure |
| Step wall-clock | 64 → **1.525 s** · 128 → **1.700 s** |
| **Feasible boundary** | **`--batch_size 128`** (exact maximum lies in (128, 256), un-bisected by design) |
| Published batch size | **256 — not trainable on this GPU** |
| Repository state | **clean** — no source, data, or fold modification |

### Next Single Step

**None initiated.** Stage 07B (timing one `ganForDrug(XD_t)` call for DCGAN-DTA, runtime unknown R6) remains the other open candidate and was **not started**, per instruction.

---

*End of Stage 7A — Co-VAE. No repository source file, dataset, or fold file was modified; no architecture or hard-coded constant was changed; no memory workaround was applied; no training loop or epoch was run.*

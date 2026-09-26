# Stage 5 — Reproduction Feasibility Audit: Co-VAE

**Inputs:** `audit/02-paper-forensic/Co-VAE.md`, `audit/03-paper-code-traceability/Co-VAE.md`, `audit/04-dataset-fold/Co-VAE.md`, `THESIS_IMPLEMENTATION_PLAN.md`
**Implementation:** `Co-VAE/CoVAE/` (upstream `github.com/LiminLi-xjtu/CoVAE`, branch `master`, HEAD `2c17268`)
**Mode:** READ-ONLY. No source modified, no dataset modified, no fold file modified, no package installed, no environment created, nothing executed. Scratch analysis held outside the repository.

**Evidence tags:** `[PAPER]` · `[CODE]` · `[DATA]` · `[DOC]` · `[INFERENCE]` · `[UNKNOWN]`

> **Two Stage 3 blockers are re-diagnosed here.** B2 is mischaracterised (§3.2) and B11 is resolvable statically (§5.3). Both are recorded as dated corrections in §12.

---

## PART 1 — Environment Feasibility

### 1.1 Python

| Question | Finding | Tag |
|---|---|---|
| Version stated by repository | **None. There is no README, no requirements file, no setup file, no documentation of any kind.** | `[DOC]` |
| Version stated by paper | Not mentioned. No hardware, no software versions. | `[PAPER]` Stage 2 §17 |
| **Python 3.11+ breakage** | **YES — confirmed, and intrinsic to this source.** `get_random_folds:41` calls `random.sample(indices, k)` where `indices` is a **`set`** (`:33`, `:42`). `random.sample` began deprecating set input in 3.9 and **removed it in 3.11**, raising `TypeError: Population must be a sequence`. | `[CODE]` `[DOC]` |
| Effective ceiling | **Python ≤ 3.10** | `[INFERENCE]` |
| Effective floor | f-strings and `.format` only; no walrus, no match. No evidence of a floor above 3.6. | `[CODE]` |
| Version-sensitive behaviour beyond the above | `list_remove:99-103` rebuilds entity order from a **set difference** (`set(range(n)) - set(removelist)`). Set iteration order is not contractually defined; it holds ascending for small ints in CPython. Stage 4 §4.3 verified ascending order for these inputs. | `[CODE]` `[DATA]` `[INFERENCE]` |

**[INFERENCE]** This is a **hard** ceiling, not a soft one, and it is the first thing the program hits after data loading (§2.2). The ambient interpreter on this machine is **3.13.2** — incompatible. `[DATA]`

### 1.2 Framework — PyTorch and CUDA

| Question | Finding | Tag |
|---|---|---|
| Version stated anywhere | **None.** | `[DOC]` |
| Deprecated APIs in use | `from torch.autograd import Variable` (`model.py:3`, `run_experiments.py:6`) — deprecated since PyTorch 0.4 (2018); still functional in 2.x as a passthrough. | `[CODE]` |
| | `torch.cuda.FloatTensor(size)` (`model.py:36`) — the legacy typed-tensor constructor, long discouraged in favour of `torch.empty(..., device=...)`. Still present in 2.x. | `[CODE]` |
| | `torch.save(model, path)` / `torch.load(path)` — **whole-object pickling**, not `state_dict` (`:289, 376, 382`). In **PyTorch 2.6+, `torch.load` defaults to `weights_only=True`**, which refuses to unpickle a whole `nn.Module`. | `[CODE]` `[DOC]` |
| **CUDA requirement** | **Unconditional, six sites, no fallback:** `model.py:36` `torch.cuda.FloatTensor`; `model.py:128,131` `.cuda()` on inputs; `run_experiments.py:127` `.cuda()` on affinity; `:276, :361` `net(...).cuda()`. No `device=` parameter, no `torch.cuda.is_available()` guard anywhere. | `[CODE]` `[VERIFIED]` |

**[INFERENCE] Derived constraint: Python ≤ 3.10, a CUDA-enabled PyTorch build, and physical NVIDIA hardware.** An upper PyTorch bound is implied by the `torch.load` default change — **PyTorch < 2.6** avoids it, though the checkpoint path is only reached after training a grid point, so it does not block a smoke test.

> **`[UNKNOWN]` — no PyTorch version is established by the repository or any documentation, and this audit assumes none.** Marked **RUNTIME-ONLY UNKNOWN R1**.

**The CUDA requirement is qualitatively worse than DCGAN-DTA's.** There, GPU is needed only at `reset_keras()`, after a full grid point completes — a CPU smoke test is viable. Here, `torch.cuda.FloatTensor` sits inside `CNN.reparametrize`, on **every forward pass**. `[CODE]` **[INFERENCE] No forward pass is possible on a CPU-only machine without modifying source** — which is forbidden in this stage and out of scope for Stage 06. This is the single most consequential feasibility difference between the two projects.

### 1.3 Other dependencies

Complete import inventory. `[CODE]`

| Package | Imported in | Pinned? | Notes |
|---|---|---|---|
| `torch` (+ `nn`, `autograd`, `utils.data`, `optim`, `nn.init`) | `model`, `run_experiments` | **no** | **critical** (§1.2) |
| `numpy` | all | **no** | low risk |
| `pandas` | `datahelper` | **no** | `read_csv(sep='\s+')` — `sep='\s+'` is the modern spelling; the deprecated `delim_whitespace` is **not** used, so pandas 2.x is safe |
| `sklearn` | `emetrics` (`auc`, `precision_recall_curve`), `run_experiments` (`roc_auc_score`, `accuracy_score`) | **no** | low |
| `matplotlib` (+ `pyplot`, `cm`) | `datahelper`, `run_experiments` | **no** | imported at module scope; `from matplotlib.pyplot import cm` is unused |
| `tqdm` | `run_experiments:12` | **no** | **not installed here** `[DATA]` |

**No Java, no external binary, no subprocess.** `emetrics.get_aupr` is pure scikit-learn (`emetrics.py:61-64`) — unlike DCGAN-DTA's Java-shelling variant. `[CODE]` `[VERIFIED]`

**Undeclared everything.** With no README and no requirements file, all six third-party packages are undeclared. `[DOC]`

### 1.4 Filesystem and OS assumptions

| Assumption | Evidence | Consequence |
|---|---|---|
| `dataset_path` concatenated with `+` | `datahelper.py:140-150` | trailing slash **mandatory** |
| `--dataset_path` default `./data/kiba/` | `arguments.py:55` | **relative** — safer than DCGAN's absolute default, but points at **KIBA**, whose folds are invalid (Stage 4 §5.3) |
| `--log_dir` default `/tmp` | `arguments.py:80` | **absolute POSIX — invalid on Windows**; and `logging()` opens `log_dir/log.txt` with no `makedirs`, so a non-existent dir raises |
| `figdir = "figures/"` | `run_experiments.py:28` | created by `experiment()` |
| `torch.save(model, 'checkpoint.pth')` | `:289, 376` | **fixed filename, repo root**, overwritten across grid points, folds and passes (Stage 3 B5) |
| `np.savetxt("./result/iter{i}…")` | `:418-419` | `result/` ships with a placeholder; not auto-created |
| POSIX separators in path literals | throughout | Stage 3 B7 — benign on Windows, which accepts `/` |
| Working directory must be `Co-VAE/CoVAE/` | all paths relative | — |

---

## PART 2 — Entrypoint Feasibility

**Entrypoint:** `run_experiments.py:540` `if __name__ == "__main__"` → `argparser()` → `experiment(FLAGS)` → `nfold_setting_sample` → `general_nfold_cv` → `general_nfold_cv_test`. `[CODE]`

| Project | Entrypoint | Required args | Required files | Required environment | Status |
|---|---|---|---|---|---|
| Co-VAE (Davis) | `run_experiments.py` | `--num_windows`, `--smi_window_lengths`, `--seq_window_lengths` (**no defaults**, `len()`-ed); `--lamda` (**scalar default — see §3.2**); `--dataset_path './data/davis/'`; `--max_smi_len 85`+, `--max_seq_len 1200`+ (defaults 100/1000); `--problem_type 1\|2\|3` (default 2); `--log_dir` (default `/tmp`) | `ligands_iso.txt`, `proteins.txt`, `drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt` — **all present** `[DATA]` | **Python ≤ 3.10**, CUDA-enabled PyTorch, **NVIDIA GPU**, numpy, pandas, sklearn, matplotlib, tqdm | **BLOCKED BEFORE MODEL CONSTRUCTION** |

**Status justification.** The pipeline gets *further* than one might expect — imports succeed, `parse_data` completes, `np.where` completes — and then fails deterministically inside fold generation, **before** `net(...)` is ever instantiated. On Python ≤ 3.10 that particular failure clears, and the next is the `len()` on a no-default flag (§3.1), still before model construction. Only after both are supplied does execution reach `net(FLAGS, ...).cuda()`.

### 2.1 Current machine state `[DATA]`

Measured: Python **3.13.2**; numpy **2.2.3**, pandas **2.2.3**, matplotlib **3.10.3** present. **`torch`, `sklearn`, `tqdm` are NOT installed.** No CUDA toolchain probed (no GPU check performed — out of scope).

### 2.2 Deterministic failure order `[CODE]` `[INFERENCE]`

On the ambient 3.13 interpreter with `torch` installed and the README-less defaults:

| # | Site | Failure |
|---:|---|---|
| 1 | `run_experiments.py:12` | `ImportError: tqdm` (not installed here) |
| 2 | `experiment:489` `parse_data` | succeeds — but on **KIBA** by default, not Davis |
| 3 | `experiment:509` `get_drugwise_folds` (default `problem_type 2`) → `get_random_folds:41` | **`TypeError` — `random.sample` on a `set`** ← **B1, first hard stop** |
| 4 | `general_nfold_cv:248` `len(paramset1)` | `TypeError: object of type 'NoneType' has no len()` ← **B4** |
| 5 | `general_nfold_cv:248` `len(lamda_set)` | `TypeError: object of type 'int' has no len()` ← **B2, and see §3.2** |
| 6 | `general_nfold_cv:276` `net(...).cuda()` | `RuntimeError` / `AssertionError` if no CUDA ← **B3** |

Steps 4 and 5 are the *same expression*; `paramset1` is evaluated first, so B4 masks B2 unless both are supplied.

---

## PART 3 — Argument Audit

All 12 arguments in `arguments.py`. `[CODE]`

| Argument | Default | Required in practice | Used? | README consistent? | Paper consistent? | Notes |
|---|---|---|---|---|---|---|
| `--seq_window_lengths` | **none → `None`** | **YES** | ✔ `:241` | **no README exists** | ✔ grid searched | `len(None)` ⇒ `TypeError` |
| `--smi_window_lengths` | **none → `None`** | **YES** | ✔ `:240` | — | ✔ | same |
| `--num_windows` | **none → `None`** | **YES** | ✔ `:239` | — | ⚠ paper's filter counts are fixed, not a grid | same |
| **`--lamda`** | **`-5`** with `nargs='+'` ⚠ | **YES** | ✔ `:242, 248, 275` | — | ✔ λ ∈ {−3,−5} as exponents | **has a default, but it is a scalar — see §3.2** |
| `--max_seq_len` | `1000` | **YES for Davis** | ✔ `:483` | — | ⚠ paper: 1200 Davis / 1000 KIBA | **dangerous**: also scales the loss (§3.3) |
| `--max_smi_len` | `100` | **YES for Davis** | ✔ `:484` | — | ⚠ paper: 85 Davis / 100 KIBA | same |
| `--dataset_path` | `./data/kiba/` ⚠ | **YES for Davis** | ✔ | — | n/a | default selects the dataset whose folds are invalid |
| `--problem_type` | `2` | no | ✔ `:510-513` | — | ✔ new-drug / new-target | 1 = pair, 2 = drug-wise, 3 = target-wise |
| `--num_epoch` | `100` | no | ✔ `:279` | — | ✔ 100 | — |
| `--batch_size` | `256` | no | ✔ `:243` | — | ✔ 256 | — |
| `--log_dir` | `/tmp` ⚠ | yes on Windows | ✔ | — | n/a | `logging()` does not `makedirs` |
| `--checkpoint_path` | `''` | no | **✘ NEVER READ** | — | — | ignored; checkpoint filename is hard-coded |

### 3.2 `--lamda` — Stage 3's B2 is mischaracterised

Stage 3 B2 states: *"`--lamda` omitted → `len()` on an int"*, described as **"`--lamda` has no default"** (Stage 3 §13, Execution Blockers) and repeated in Stage 4 §2 and the master plan.

**The flag does have a default.** `[CODE]` `arguments.py:83-88`:

```python
parser.add_argument('--lamda', type=int, default=-5, nargs='+',)
```

The defect is subtler and worth stating precisely: `nargs='+'` makes any **supplied** value a *list* (`--lamda -5` → `[-5]`), but the **default is the bare int `-5`**, not `[-5]`. Consumption at `:242` `lamda_set = FLAGS.lamda`, then `:248` `len(lamda_set)` and `:275` `for lamda in lamda_set`. `len(-5)` ⇒ `TypeError`.

**[INFERENCE]** The failure and its consequence are exactly as Stage 3 said; the *cause* is not "missing default" but "scalar default under `nargs='+'`". This matters practically: a reader told "no default" might add one; the actual fix is to pass the flag, and the default value `-5` is already the paper's λ. The blocker stands, re-diagnosed.

### 3.3 The length flags silently change the loss — a dangerous default

`run_experiments.py:136`:

```python
loss = loss_affinity + 10**lamda * (loss_drug + FLAGS.max_smi_len / FLAGS.max_seq_len * loss_target)
```

**[CODE]** `max_smi_len / max_seq_len` is a **loss weight**, not only a tensor-shape parameter. `[INFERENCE]` At defaults (100/1000) the target branch is weighted 0.1; at Davis's paper values (85/1200) it is 0.0708. Changing input length to accommodate a dataset silently re-weights the objective. Confirms and sharpens Stage 3 B6 and C6.

Stage 4 established (§2.2) that Davis proteins reach **2549** residues and SMILES reach **92**. At the defaults (1000/100) **115 of 442 proteins** are head-truncated and no SMILES is; at the paper's 1200/85, 61 proteins **and 3 SMILES** are truncated. `[DATA]` Neither configuration is lossless, and the two differ in both truncation and loss weighting.

---

## PART 4 — Data Pipeline Feasibility (Davis)

### 4.1 Traced path, raw files → first model-input tensor `[CODE]` `[DATA]`

```
argparser()
 └─ experiment(FLAGS)                                            :470
     └─ DataSet(fpath, setting_no=problem_type, seqlen=max_seq_len, smilen=max_smi_len)
     └─ dataset.parse_data(FLAGS)                          datahelper.py:136
         ├─ json  data/davis/ligands_iso.txt   → OrderedDict → orderdict_list → 68 SMILES (VALUES only)
         ├─ json  data/davis/proteins.txt      → 442 sequences (VALUES only; 379 unique)
         ├─ read_csv drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt
         │         sep='\s+', header=None, latin1        → (68,442) Kd in nM, 0 missing
         ├─ affinities = -(log10(affinities / 1e9))      → pKd 5.0 – 10.7959      ← RUNTIME TRANSFORM
         │  (Davis branch only; NO length filter — that is KIBA-only)
         ├─ XD = [label_smiles(s, max_smi_len, CHARISOSMISET) for s in ligands]   (68, L_d)  float64
         └─ XT = [label_sequence(p, max_seq_len, CHARPROTSET) for p in proteins]  (442, L_t) float64
     └─ np.asarray XD, XT, Y                                     :492-494
     └─ label_row_inds, label_col_inds = np.where(~isnan(Y))     :504  → 30,056 pairs (ALL cells)
     └─ random.seed(1000);  problem_type ⇒ get_{random,drugwise,targetwise}_folds(..., 6)
                                                                 :509-513   ⚠ B1 FAILS HERE on Py≥3.11
     └─ nfold_setting_sample(...)                                :165
         ├─ test_set = nfolds[5]; outer_train_sets = nfolds[0:5]
         ├─ builds val_sets (NEVER CONSUMED) and test_sets       :176-186
         └─ general_nfold_cv(..., train_sets, test_sets)         :235
             ├─ len(paramset1)*len(paramset2)*len(paramset3)*len(lamda_set)   ⚠ B4/B2 FAIL HERE
             ├─ prepare_interaction_pairs(XD,XT,Y,trrows,trcols) :457
             │    → list of [drug (L_d,) float32, target (L_t,) float32, affinity () float32]
             ├─ DataLoader(dataset, batch_size=256, shuffle=True)
             └─ net(FLAGS, p1, p2, p3).cuda()                    :276  ⚠ B3 FAILS HERE without CUDA
                 └─ forward: x.long().cuda() → Embedding → …     ← FIRST MODEL-INPUT TENSOR
```

### 4.2 Required files — all present `[DATA]`

| File | Present | Consumed |
|---|:--:|---|
| `data/davis/ligands_iso.txt` | ✔ | **yes** — values only |
| `data/davis/proteins.txt` | ✔ | **yes** — values only |
| `data/davis/drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt` | ✔ | **yes** |
| `data/davis/Y` | ✔ | **no** — never opened |
| `data/davis/folds/*` | ✔ | **no** — never opened (Stage 4 §5.1) |
| `data/davis/*similarities*` | ✔ | **no** |

**No missing file.** Only three of the nine Davis files are consumed. `[DATA]`

### 4.3 Tensor shapes, dtypes and index safety

| Object | Shape (defaults 100/1000) | Shape (paper 85/1200) | dtype |
|---|---|---|---|
| `XD` | (68, 100) | (68, 85) | float64 (`np.zeros`) |
| `XT` | (442, 1000) | (442, 1200) | float64 |
| `Y` | (68, 442) | same | float64 |
| per-pair drug / target / affinity | (L_d,) / (L_t,) / () | same | **float32** (`prepare_interaction_pairs:461-465`) |
| batch | (256, L_d) / (256, L_t) / (256,) | same | float32 → `.long()` in `forward` |

**Embedding index safety — verified from data.** `net` uses `nn.Embedding(FLAGS.charsmiset_size, 128)` and `nn.Embedding(FLAGS.charseqset_size, 128)` — **without the `+1` guard** that DCGAN-DTA applies (`input_dim=charsmiset_size + 1`). Valid indices are therefore 0…63 and 0…24. `[CODE]`

Measured over the actual files: `[DATA]`

| Dataset | Max SMILES index (limit 63) | Max protein index (limit 24) | Unknown chars |
|---|---:|---:|---|
| Davis | **49** | **24** | none |
| KIBA | **54** | **23** | none |

No overflow, confirming Stage 3's B13 clearance. **But Davis's protein encoding sits exactly at the boundary** — index 24 (`X`) is the last valid slot, zero margin. `[INFERENCE]` Any protein file containing `Z` (index 25) would raise `IndexError`. Safe on the shipped data, fragile by construction.

### 4.4 Verdict

> **Davis can reach `prepare_interaction_pairs` and the `DataLoader` without modifying code, on Python ≤ 3.10 with the six required flags supplied.** It **cannot** reach the first *model-input* tensor without CUDA, because `net(...).cuda()` at `:276` precedes any forward pass and `torch.cuda.FloatTensor` sits inside every forward. `[CODE]` `[INFERENCE]`

This is a sharper statement than Stage 3's: the *data* path is clean and complete; the barrier is the device, not the data.

---

## PART 5 — Model Construction Feasibility

### 5.1 Static dimensional trace — `net` (`model.py:116`)

`net` composes: `embedding1`, `embedding2`, `cnn1` (`CNN`), `cnn2` (`CNN`), `reg` (`net_reg`), `decoder1`, `decoder2`. With `NUM_FILTERS = F`, `FILTER_LENGTH = k`:

**Encoder `CNN` (`:6-55`) — gated-CNN, traced channel by channel:**

| Step | Operation | Channels in → out | After gating |
|---|---|---|---|
| `conv1` | `Conv1d(128, 2F, k, pad=k//2)` | 128 → **2F** | split in half ⇒ **F** |
| `conv2` | `Conv1d(F, 4F, k, pad=k//2)` | **F → 4F** ✔ matches | split ⇒ **2F** |
| `conv3` | `Conv1d(2F, 6F, k, pad=k//2)` | **2F → 6F** ✔ matches | split ⇒ **3F** |
| `out` | `AdaptiveAvgPool1d(1)` | (B, 3F, 1) | `.squeeze()` ⇒ (B, 3F) |
| `layer1`/`layer2` | `Linear(3F → 3F)` ×2 | ✔ | μ and logvar, **latent dim = 3F** |

**Every channel count matches.** The gating (`x.split(x.size(1)/2, 1)` then `out * sigmoid(gate)`) halves channels at each stage, and `conv2`/`conv3` are declared with exactly the post-gate counts. `[CODE]` `[INFERENCE]` With `F = 32`, latent = 96 — consistent with Stage 3's reported latent dimension.

**`net_reg` (`:84-113`):** `reg1`, `reg2` each `Linear(3F → 3F)`; `torch.cat` on dim 1 ⇒ **6F**; `reg` is `Linear(6F → 1024) → ReLU → Dropout(0.1) → Linear(1024 → 512) → ReLU → Dropout(0.1) → Linear(512 → 1)`. ✔ consistent.

**`decoder` (`:58-81`):** `Linear(3F → 3F·(D − 3(k−1)))` → `view(-1, 3F, D − 3(k−1))` → three `ConvTranspose1d(k, stride 1, pad 0)`, each adding `(k−1)` ⇒ length returns to **D** → `permute` → `Linear(128 → vocab)`. ✔ consistent, and the output length reconstructs the input length exactly.

> **The Co-VAE model is dimensionally self-consistent on static trace.** No shape defect analogous to the one Stage 3 alleged for DCGAN-DTA exists here — and, per the companion report §5.3, none exists there either.

### 5.2 Static construction risks

| ID | Risk | Evidence | Class |
|---|---|---|---|
| **R2** | `decoder` requires `D − 3(k−1) > 0`. At Davis paper values `max_smi_len=85`, `k=16` ⇒ 85 − 45 = 40 ✔. At `max_smi_len=100`, k=16 ⇒ 55 ✔. A small `max_smi_len` with a large filter yields a **negative dimension** and a `view` error. | `[CODE]` `:62,77` | latent, config-dependent |
| **R3** | `output.squeeze()` (`:51`) removes **all** size-1 dims. For a trailing batch of size 1, (1, 3F, 1) ⇒ (3F,), and `torch.cat(..., 1)` in `net_reg` then fails on a 1-D tensor. Davis train folds are 25,047 pairs; 25047 mod 256 = **199**, so this does not arise for Davis at batch 256 — but it is size-dependent. | `[CODE]` `[DATA]` `[INFERENCE]` | **RUNTIME-ONLY R3** |
| **R4** | `weights_init` is applied via `model.apply` (`:277`); its coverage of `Embedding`/`ConvTranspose1d` is unverified. | `[CODE]` `:99` | minor |

### 5.3 The `.exp_()` question — Stage 3's B11 resolves statically

Stage 3 B11 — graded **HIGH (unresolved)** — asks whether `logvar.mul(0.5).exp_()` mutates the tensor that later reaches `loss_f`'s KL term, and recommends it as Phase-4 target #3.

```python
def reparametrize(self, mean, logvar):
    std = logvar.mul(0.5).exp_()          # model.py:35
```

**`Tensor.mul(other)` is out-of-place** — it returns a **new** tensor. `.exp_()` then mutates *that new temporary*, never `logvar` itself. `[CODE]` `[DOC]` `[INFERENCE]`

`forward` returns `output2` (the `layer2` output, i.e. logvar) **unmodified** (`:53-55`), and `loss_f` receives it as `logvar`. **The KL term is computed on `logvar`, not on `exp(logvar/2)`.**

Nor is there an autograd hazard: the backward of `mul(0.5)` needs only the scalar, so in-place `exp_()` on its output does not invalidate a saved tensor.

**[INFERENCE] B11 is resolved in the code's favour and needs no runtime check.** Stage 3's Phase-4 target #3 can be struck. *(This says nothing about C5 — `eps` sampled from `N(0, 0.1²)` against a KL derived for `N(0, I)` — which remains a genuine and unresolved mis-scaling, and is a scientific defect, not an execution one.)*

---

## PART 6 — First-Smoke-Test Design

**Blunt constraint:** steps 4–6 of the standard smoke sequence (**instantiate the model, one forward pass, one loss**) are **not reachable on a CPU-only machine**, because `net(...).cuda()` and `torch.cuda.FloatTensor` are unconditional (§1.2). Making them reachable requires editing source, which is forbidden now and out of Stage 06's scope.

**Therefore the Co-VAE smoke test splits into two tiers.**

### 6.1 Tier 1 — CPU-only, available immediately after provisioning

Answers questions 1–3 (import, loader, one batch). From `Co-VAE/CoVAE/`, on **Python 3.10**.

**S1 — imports.**
```
python -c "import torch, numpy, pandas, sklearn, matplotlib, tqdm; print(torch.__version__, torch.cuda.is_available())"
```

**S2 — module import.**
```
python -c "import run_experiments; print('module import OK')"
```

**S3 — loader, Davis, paper-faithful lengths.**
```
python -c "
import sys; sys.argv=['x','--dataset_path','./data/davis/','--max_smi_len','85','--max_seq_len','1200',
 '--problem_type','1','--num_windows','32','--smi_window_lengths','4','--seq_window_lengths','8',
 '--lamda','-5','--log_dir','logs/']
from arguments import argparser; from datahelper import DataSet
import numpy as np
F=argparser(); d=DataSet('./data/davis/',F.problem_type,F.max_seq_len,F.max_smi_len)
F.charsmiset_size=d.charsmiset_size; F.charseqset_size=d.charseqset_size
XD,XT,Y=d.parse_data(F)
XD,XT,Y=np.asarray(XD),np.asarray(XT),np.asarray(Y)
print('XD',XD.shape,XD.dtype,'max idx',XD.max(),'limit',d.charsmiset_size-1)
print('XT',XT.shape,XT.dtype,'max idx',XT.max(),'limit',d.charseqset_size-1)
print('Y',Y.shape,'observed',int((~np.isnan(Y)).sum()),'min',np.nanmin(Y),'max',np.nanmax(Y))
"
```
**Expected** (from Stage 4): `XD (68,85)`, `XT (442,1200)`, `Y (68,442)`, **30,056** observed, min **5.0**, max **10.7959**.

**S4 — fold generation, the B1 probe.** The single most informative CPU test: it confirms whether Python 3.10 clears B1, and lets fold structure be compared against Stage 4 §6.2.
```
python -c "
import sys; sys.argv=['x','--dataset_path','./data/davis/','--problem_type','1']
import random, numpy as np
from run_experiments import get_random_folds, get_drugwise_folds, get_targetwise_folds
random.seed(1000)
f=get_random_folds(30056,6); print('random folds', [len(x) for x in f])
"
```
**A `TypeError` here on Python ≤ 3.10 would overturn the B1 diagnosis** and must be recorded.

**S5 — one batch** via `prepare_interaction_pairs` + `DataLoader`, asserting `(256, 85)`, `(256, 1200)`, `(256,)` float32. Reachable on CPU — the tensors are plain, and `.cuda()` is only invoked inside `net.forward`.

### 6.2 Tier 2 — requires an NVIDIA GPU; do not attempt without one

**S6 — model construction:** `net(FLAGS, 32, 4, 8).cuda()`, then print `sum(p.numel() for p in model.parameters())`.
**S7 — one forward pass** on the S5 batch: assert nine return values and that `out` is `(256,)`, `x` is `(256, 85, 64)`, `y` is `(256, 1200, 25)`.
**S8 — one loss:** `loss_f` on each branch plus the `10**lamda` combination at `:136`, printing the three terms separately — which also settles Stage 3's C6 (which term λ weights) empirically.

### 6.3 Evidence to capture

`torch.__version__` and `torch.cuda.is_available()`; all shapes and dtypes; max embedding indices vs limits; fold sizes and drug/target overlap per `problem_type`; on Tier 2, parameter count and the three loss terms. Full traceback for any failure.

---

## PART 7 — Execution Blockers

### A. Environment blockers

| ID | Layer | Problem | Evidence | Stage 06 can test? | Source change? |
|---|---|---|---|:--:|:--:|
| **E1** | env | Ambient Python **3.13.2**; code requires **≤ 3.10** | `[DATA]` §2.1; `[CODE]` §1.1 | **yes — prerequisite** | no |
| **E2** | env | No PyTorch version established anywhere; `torch` not installed | `[DOC]` `[DATA]` | yes | no |
| **E3** | env | **CUDA + NVIDIA GPU unconditionally required**, six sites, no fallback | `[CODE]` §1.2 | **only with hardware** | **yes, to avoid** |
| **E4** | env | `tqdm`, `sklearn` not installed; all deps undeclared (no README) | `[DATA]` `[DOC]` | yes | no |
| **E5** | env | `--log_dir` default `/tmp`; `logging()` does not `makedirs` | `[CODE]` `arguments.py:80, 99` | yes | no — pass flag |

### B. Data blockers

| ID | Layer | Problem | Evidence | Stage 06 can test? | Source change? |
|---|---|---|---|:--:|:--:|
| **D1** | data | `--dataset_path` defaults to **KIBA**, whose fold files are invalid and whose AUC is degenerate | `[CODE]` `[DATA]` Stage 4 §5.3, §8.1 | yes — pass Davis | no |
| **D2** | data | KIBA runtime matrix ≠ published (1,954×217 vs 2,111×229) | `[DATA]` Stage 4 §2.4 | n/a for Davis | **yes**, to recover the paper's matrix |
| **D3** | data | Davis protein embedding index maxes at **24 of 24** — zero margin | `[DATA]` §4.3 | yes — assert in S3 | no |
| **D4** | data | Length flags silently re-weight the loss (§3.3) | `[CODE]` `:136` | yes — record chosen values | no |

*No file is missing for Davis.* `[DATA]`

### C. Code blockers

| ID | Layer | Problem | Evidence | Stage 06 can test? | Source change? |
|---|---|---|---|:--:|:--:|
| **C1** | code | `random.sample(set, k)` ⇒ `TypeError` on Python ≥ 3.11 (**B1**) | `[CODE]` `:41` | **yes — S4** | no, if ≤ 3.10 |
| **C2** | code | Three flags have no default and are `len()`-ed (**B4**) | `[CODE]` §3 | yes | no — pass flags |
| **C3** | code | `--lamda` default is a **scalar** under `nargs='+'` (**B2, re-diagnosed §3.2**) | `[CODE]` `arguments.py:83-88` | yes | no — pass flag |
| **C4** | code | `net(...).cuda()` before any forward (**B3**) | `[CODE]` `:276, 361` | Tier 2 only | **yes, to avoid** |
| **C5** | code | `.squeeze()` breaks on a trailing batch of size 1 (**R3**) | `[CODE]` `:51` | Tier 2 | no |
| **C6** | code | Fixed `checkpoint.pth`, overwritten across grid points/folds/passes (**B5**) | `[CODE]` `:289, 376` | not in smoke scope | no |
| **C7** | code | `torch.load` default `weights_only=True` in PyTorch ≥ 2.6 vs whole-module `torch.save` | `[CODE]` `[DOC]` §1.2 | not in smoke scope | no — pin torch |
| **C8′** | code | ~~In-place `.exp_()` may corrupt the KL input~~ — **WITHDRAWN**, see §5.3 | `[CODE]` `[DOC]` | n/a | no |

### D. Scientific / reproduction blockers

*These do not prevent execution. They prevent faithful reproduction.*

| ID | Problem | Evidence |
|---|---|---|
| **S1** | Hyperparameter search, early stopping, checkpointing **and** final reporting all use the test fold; `val_sets` discarded | Stage 3 C2; Stage 4 §7.3 |
| **S2** | One repetition (`range(1)`), not the paper's ten | Stage 3 C3 |
| **S3** | MAE reported in Table 2, never implemented | Stage 3 C4 |
| **S4** | `eps ~ N(0, 0.1²)` against a KL derived for `N(0, I)` | Stage 3 C5 |
| **S5** | λ weights the VAE terms, not the affinity term, with an undocumented extra factor | Stage 3 C6; §3.3 |
| **S6** | Average pooling, not the paper's max pooling; no ReLU in the encoder | Stage 3 C7, C8 |
| **S7** | **Davis target-wise leakage** — 13/73 test targets sequence-identical to training (442 ids → 379 sequences) | Stage 4 §7.2 |
| **S8** | KIBA AUC degenerate: 109,286 positives vs **10** negatives at `> 7` | Stage 4 §8.1 |
| **S9** | Only Co-VAE is implemented — no baselines, no generation, no RDKit, no Tables 3–5, no COVID study | Stage 3 C11 |
| **S10** | Three mutually incompatible KIBA split definitions (paper / fold files / code) | Stage 4 §6.3 |

---

## PART 8 — What Can and Cannot Be Reproduced

### 8.1 Code execution feasibility

**BLOCKED** on CPU-only hardware; **NOT DETERMINABLE WITHOUT EXECUTION** given an NVIDIA GPU.

This is a materially worse position than DCGAN-DTA's. There, a CPU smoke test reaches a forward pass. Here, **no forward pass exists without CUDA**, and the barrier is unconditional and on the hot path. Everything up to and including the `DataLoader` batch is reachable on CPU (Tier 1); nothing beyond it is.

### 8.2 Pipeline reproduction feasibility

**FEASIBLE WITH DOCUMENTED DEVIATIONS** for Davis; **BLOCKED** for KIBA as published.

Davis deviations to declare: one repetition instead of ten; selection and reporting on the test fold; average rather than max pooling; ε mis-scaling; λ weighting the opposite term; MAE absent; and the target-wise sequence leakage (S7).

KIBA is blocked because the runtime matrix is not the published one and cannot be made so without editing `datahelper.py` (D2) — and its fold files are invalid regardless.

### 8.3 Result reproduction feasibility

**BLOCKED**, independent of execution.

Four of five experiment families have no code path (S9). Table 2's MAE column is not implemented (S3). The KIBA rows would be computed on a different matrix (D2). Even a perfect GPU run reproduces **only the Co-VAE row of Table 2 for Davis**, and only up to the estimator in S1 and the leakage in S7.

---

## PART 9 — Stage 06 Scope

| # | Item | Specification |
|---|---|---|
| 1 | **Environment candidate** | An isolated **Python 3.10** interpreter — the highest version clearing B1. PyTorch: **no version is established**; start from a current CUDA build and, if the checkpoint path is later exercised, pin **< 2.6** (C7). Plus numpy, pandas, scikit-learn, matplotlib, tqdm. **Do not modify the ambient 3.13 environment.** |
| 2 | **First import test** | S1 → S2 (§6.1). `torch.cuda.is_available()` **determines whether Tier 2 is attemptable at all** and must be recorded first. |
| 3 | **First dataset-loader test** | S3 on Davis with `--max_smi_len 85 --max_seq_len 1200`. Assert against Stage 4: (68,85), (442,1200), 30,056 observed, pKd 5.0–10.7959. |
| 4 | **First tensor-shape test** | S5 — one `DataLoader` batch: (256,85), (256,1200), (256,) float32. |
| 5 | **First model-construction test** | S6 — **Tier 2, GPU only.** |
| 6 | **First forward-pass test** | S7, then S8 for one loss. **Tier 2, GPU only.** |
| 7 | **Evidence to capture** | §6.3. |
| 8 | **What failure would mean** | A `TypeError` at S4 on Python 3.10 ⇒ the B1 diagnosis is wrong and the root cause must be re-derived from the traceback. A shape mismatch at S3 ⇒ a Stage 4 measurement needs revisiting. A CUDA failure at S6 ⇒ only that no GPU is present — **not** that the model is defective. |
| 9 | **What failure would NOT mean** | It would **not** mean the paper is unreproducible — §8.3 is already BLOCKED for independent reasons. It would **not** license patching B1/B2/B3: any patch is a Stage 07 decision with each deviation logged. It would **not** invalidate Stage 4's findings, which are execution-independent. |

**RUNTIME-ONLY UNKNOWNS carried into Stage 06:**

| ID | Unknown |
|---|---|
| **R1** | Which PyTorch version works; whether CUDA hardware is available at all |
| **R3** | Whether `.squeeze()` breaks on a trailing batch of size 1 for the realised fold sizes |
| **R4** | Whether `weights_init` covers `Embedding` and `ConvTranspose1d` |
| **R5** | The realised fold sizes and entity overlap under the code's own RNG on Python 3.10 (Stage 4 §6.2 simulated these with a substitute sampler) |
| **R6** | Whether the three loss terms carry non-zero gradient, and which term `10**lamda` actually scales (Stage 3 C6) |

**Out of scope for Stage 06:** full training, any paper number, Kaggle or remote compute, and **any source modification** — including the three one-line changes that would clear B1–B3.

---

## PART 10 — Development Dataset Decision

**Co-VAE development dataset = Davis.** Reaffirmed on Stage 05 evidence: all three consumed files are present; the runtime matrix matches the published one exactly; embedding indices are in range (§4.3); and the data path to the `DataLoader` is clean and CPU-reachable.

> **Davis is a pipeline-validation dataset, not the scientific comparison dataset.** Its role is to establish that the pipeline imports, loads, folds and batches — and, given a GPU, constructs and runs one forward pass. Its **100% density**, **69.64% censoring at pKd 5.0**, and **442 → 379 target-sequence collapse** (Stage 4) make it a weak evidential base, and the collapse actively compromises the paper's flagship new-target setting.
>
> **KIBA remains a later-stage dataset**, subject to: a runtime matrix that is not the published one, fold files with 8,958 out-of-range indices, a degenerate AUC threshold, and its status as `--dataset_path`'s default — a trap to guard against in every invocation.

---

## PART 11 — Feasibility Matrix and Verdict

| Component | Status | Evidence |
|---|---|---|
| Environment | **RECOVERABLE**, conditional on GPU | Python ≤ 3.10 derivable; CUDA is hardware, not configuration |
| Dependencies | **RECOVERABLE** | six packages, all on PyPI, **none declared** — no README exists |
| Dataset (Davis) | **READY** | three consumed files present; matrix matches the paper exactly |
| Dataset (KIBA) | **REQUIRES RECONSTRUCTION** | runtime ≠ published; needs a source edit to recover |
| Preprocessing | **READY** to run / **BLOCKED** to audit | transform is in-code; KIBA's offline filtering is unauditable |
| Fold files | **BLOCKED / irrelevant** | never read; KIBA's are invalid (Stage 4 §5) |
| Model | **READY** (static) | dimensionally self-consistent (§5.1); B11 withdrawn (§5.3) |
| Loss | **READY** to run / deviant | three terms live, but λ weights the wrong term and ε is mis-scaled |
| Training | **BLOCKED without GPU** | `.cuda()` on the hot path |
| Evaluation | **RECOVERABLE** | CI/MSE/r²m implemented; **MAE absent**; AUC degenerate on KIBA |
| Results | **BLOCKED** | no results ship; 4 of 5 experiment families have no code |

**Effort to reach a green Tier-1 smoke test: LOW.** A Python 3.10 environment plus six flags.
**Effort to reach Tier 2: LOW given an NVIDIA GPU; otherwise BLOCKED** — the alternative is patching six call sites, which is a deviation, not a setup step.
**Effort to reach a faithful paper reproduction: VERY HIGH**, and partly impossible (S9).

### Final verdict: 🔴 **RED** — unchanged from Stage 3

Unlike DCGAN-DTA, nothing found in Stage 05 lifts this grade. Two blockers were re-diagnosed (B2) or withdrawn (B11), but both were secondary. The three that matter stand: **Python ≤ 3.10**, **three no-default `len()`-ed flags**, and an **unconditional CUDA dependency on the forward path**. The dataset-level defects (S7, S8, S10, D2) and the four missing experiment families are unchanged.

---

## PART 12 — Dated Corrections to Earlier Stages

Prior findings are preserved and amended, not overwritten.

> **Correction 05-C — 2026-09-25 — refines Stage 3 §13 blocker B2, and its restatements in Stage 4 §2 and `THESIS_IMPLEMENTATION_PLAN.md` §4.1.**
> **Original:** "`--lamda` omitted → `len()` on an int" / "**`--lamda` has no default**".
> **Correction:** `--lamda` **does** have a default — `default=-5` with `nargs='+'` (`arguments.py:83-88`). The defect is that the default is a **scalar** while `nargs='+'` yields a list for supplied values, so `len(FLAGS.lamda)` raises only on the default path. The blocker and its severity stand; the cause is re-diagnosed. Practically: supply `--lamda -5` and it clears — no code change needed.
> **Status:** amended, not withdrawn.

> **Correction 05-D — 2026-09-25 — withdraws Stage 3 §13 blocker B11 ("HIGH (unresolved)") and Stage 3 §14 Phase-4 target #3.**
> **Original:** "in-place `.exp_()` on `logvar` may alter the KL input… This determines whether the KL is computed on `logvar` or on `exp(logvar/2)`," recommended for runtime instrumentation.
> **Correction:** `Tensor.mul(0.5)` is **out-of-place**, returning a new tensor; `.exp_()` mutates that temporary, never `logvar`. `forward` returns `output2` unmodified, so `loss_f` receives the true `logvar`. No mutation, and no autograd hazard. Resolved statically (§5.3); **no runtime check required.**
> **Status:** withdrawn. *(Stage 3's C5 — ε mis-scaling — is unaffected and remains open.)*

---

## Stage 05 Status

**COMPLETE — STAGE 06 BLOCKED** *(Tier 1 executable; Tier 2 blocked without NVIDIA hardware)*

### What is known

All three consumed Davis files are present, and the data path from raw files to a `DataLoader` batch is clean, complete and CPU-reachable (§4). The environment constraint is derived and hard: **Python ≤ 3.10** (`random.sample` on a set) plus a **CUDA-enabled PyTorch and physical NVIDIA hardware** (§1). The full argument surface is mapped, including the loss-weighting side effect of the length flags (§3.3). The model is dimensionally self-consistent (§5.1), embedding indices are in range with **zero margin on Davis proteins** (§4.3), and **B11 is withdrawn** (§5.3).

### What is blocked

**Model construction and every forward pass, on CPU-only hardware** — `net(...).cuda()` at `:276` and `torch.cuda.FloatTensor` at `model.py:36` are unconditional. **Result** reproduction is blocked independently: four of five experiment families have no code path, MAE is unimplemented, and KIBA's runtime matrix is not the published one (§8.3).

### What requires execution

R1 and R3–R6 (§9): the working PyTorch version and whether CUDA hardware exists; the `.squeeze()` trailing-batch behaviour; `weights_init` coverage; the realised fold sizes under the code's own RNG; and which loss term `10**lamda` actually scales.

### What Stage 06 should test

**Tier 1 only, unless an NVIDIA GPU is confirmed:** S1–S5 (§6.1) — imports, module import, the Davis loader against Stage 4's expected shapes, the B1 fold-generation probe, and one `DataLoader` batch. **Tier 2 (S6–S8) requires a GPU and must not be attempted without one.**

### What must NOT be changed yet

Any file under `Co-VAE/CoVAE/` — in particular the `random.sample(set)` call, the three no-default flags, the `--lamda` scalar default, and all six `.cuda()` sites. The ambient Python 3.13 environment. Every dataset and fold file. All Stage 02–04 reports.

### Next Single Step

**Confirm whether an NVIDIA GPU with a working CUDA driver is available on the target machine — before provisioning anything.**

This single fact determines Stage 06's entire shape for Co-VAE: with it, Tier 1 and Tier 2 run and the project reaches a forward pass; without it, Stage 06 is capped at Tier 1 and reaching a forward pass becomes a *source-modification decision* — a Stage 07 matter needing your approval, not a setup step. Provisioning a Python 3.10 environment before knowing this risks building the wrong one. It requires no installation and no repository execution.

---

*End of Stage 5 — Co-VAE. READ-ONLY: no source file, dataset, or fold file was modified; no package installed; no environment created; nothing executed.*

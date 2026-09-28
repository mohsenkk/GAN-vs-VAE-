# Stage 08.5B — Co-VAE Batch-256 Kaggle GPU Validation

## Decision and scope

**PASS for the bounded execution/resource question.** On one Kaggle Tesla T4, Co-VAE constructed its native model, consumed an observed first training batch of 256, completed all 98 native optimizer updates in one epoch, and reached its native `RE.test()` path. Recorded training and evaluation outputs were finite, the worker returned 0, and no sampled resource safety threshold was crossed. This supports the interpretation that the earlier local batch-256 OOM was a resource limitation under the MX330 environment. It does **not** isolate GPU memory as the only causal difference, because the operating system, Python, PyTorch, CUDA runtime, and run scope also differed.

This is **not** a paper reproduction, convergence study, model-quality result, cross-baseline comparison, or full grid/all-fold experiment. Stage 09 was not started.

## Evidence and identity

The evidence is the user-returned complete `summary.json`, last 60 lines of `training.log`, last 10 lines of `memory.jsonl`, last 10 lines of `events.jsonl`, and notebook stdout. They were pasted into `C:/Users/Snapp/.codex/attachments/5fdd809d-f5e6-4eb1-8dac-1491e67ee916/Pasted text.txt` (SHA-256 `3e21c75c9823d499aa0c858f5c3bcaa1493ba690a5b4d06388ebed9e65220557`). The full Kaggle log and event files were not available locally for independent review; the summary and supplied excerpts are the measurement sources for this report.

| Item | Recorded identity |
|---|---|
| Runner input | `/kaggle/input/models/mohsenkashefikia/covae/pytorch/default/1/stage08_5b_covae_kaggle.py` |
| Co-VAE input | `/kaggle/input/models/mohsenkashefikia/covae-model/pytorch/default/1/CoVAE` |
| Output directory | `/kaggle/working/stage08_5b/` |
| Runner SHA-256 | `0991cff4ea481316380f9c53fac474a2eadc235eee80da2e832875e596f850ee` |
| Original `run_experiments.py` SHA-256 | `39546fa554e628bf4f30dc492d152ed1235fbeedfc83c713218dd0d40d108043` |
| Root thesis HEAD at local review | `ac5809b8fa4204f5c44c4faf3f5f5bbeb9539c28` |
| Nested Co-VAE HEAD at runner preparation | `2c172682e502e3d39f6cc6a6d5bca428326e7d0b` |

The uploaded runner and `run_experiments.py` hashes match their locally inspected files. The summary records matching SHA-256 values for the other four source files and three Davis input files, and `source.unchanged_after_run: true` for those protected files. The **Kaggle input's Git commit identity itself was not verified** (`input_git_identity_verified: false`); the nested HEAD above is the local preparation identity. The user provided a runner path and notebook stdout, but not the exact notebook cell text beyond the prior handoff instructions.

## Environment, dataset, and selected path

| Item | Recorded value |
|---|---|
| Python / OS | 3.12.13 / Linux 6.12.90+ |
| PyTorch / bundled CUDA runtime | 2.10.0+cu128 / 12.8 |
| NumPy / pandas / scikit-learn | 2.0.2 / 2.3.3 / 1.6.1 |
| Other packages | matplotlib 3.10.0; tqdm 4.67.3 |
| Device | Exactly one visible Tesla T4 (`CUDA_VISIBLE_DEVICES=0`), 15,360 MiB; four CPUs reported |
| Davis arrays | `XD (68, 85)`, `XT (442, 1200)`, `Y (68, 442)`; 30,056 observed pairs |
| Configuration | `problem_type=1`, `is_log=0`, 32 filters, SMILES kernel 4, sequence kernel 8, λ exponent −5, lengths 85/1200, batch 256, one epoch |

The source's runtime-generated pair-wise split was made with Python `random.seed(1000)`. Folds 0–4 supplied **25,047 training pairs** and fold 5 supplied **5,009 test pairs**. The split-index SHA-256 values matched the local Python 3.10 reference for the first native split: train `74cd831aa2f08dc322e4d2183eb9dc691adf45306ad903a29d9f4f6931e40e5f`, test `57e590130330d13049c835073aa566496138b51105a5a9ebe1b932b6ee2579a8`. The Python 3.12 runner used an external set-to-tuple shim for `random.sample`; the matching hashes establish the selected index membership for this run. PyTorch's random seed and exact stochastic training order were **not** controlled by the runner.

The harness called the original `DataSet.parse_data`, `get_random_folds`, `prepare_interaction_pairs`, `net`, `weights_init`, **`RE.train()`**, and **`RE.test()`**. It used the source's native model, Adam step, regression/VAE loss, DataLoader shuffle, and evaluation functions. Its wrappers observed return values and optimizer completion without replacing the loss arithmetic. The original all-fold/grid driver was not invoked.

## Completion and numerical evidence

| Boundary | Returned evidence |
|---|---|
| Model construction | Completed on `cuda:0`; 12,931,834 parameters |
| Training | 98/98 batches and 98 optimizer updates completed; first observed batch size 256; final partial batch 215 |
| First training batch | Combined objective 31.131548; regression MSE 31.125298; drug reconstruction+KL 352.730865; target reconstruction+KL 3844.067627; c-index 0.469403 |
| Last training batch | Combined objective 1.105266; regression MSE 1.102009; drug reconstruction+KL 146.703476; target reconstruction+KL 2526.640381; c-index 0.592540 |
| Native evaluation | `RE.test()` reached after the epoch, over all 5,009 selected test pairs |
| Returned test metrics | CI 0.57814795; MSE 0.80474102; RM2 0.01909421; AUC 0.61456223 |
| Numerical/exit state | All 98 recorded training batch metrics and returned test metrics finite; exception null; worker return code 0; runner exit code 0 |

The supplied training-log tail confirms progress through batch 98/98 and prints completion plus the native evaluation result. The returned summary records all-batch finiteness; the complete per-batch records reside in Kaggle `events.jsonl`, of which only the tail was pasted. The native `RE.test()` path does **not** return AUPR at this one-epoch boundary. `loss_f` returns reconstruction-plus-KL per branch; this run did not separate the reconstruction and KL terms further. The first-to-last loss change is execution evidence, not a convergence or quality claim.

## Timing, memory, and safety boundary

| Measurement | Recorded value |
|---|---:|
| Native training epoch | 28.534 s |
| Native evaluation | 1.795 s |
| Worker total / runner parent total | 35.104 s / 37.653 s |
| Peak sampled worker RSS | 1,920,434,176 bytes = 1.789 GiB, below the 12 GiB limit |
| PyTorch peak allocated / reserved | 1,993.93 MiB / 2,322 MiB |
| Peak sampled physical GPU use | 2,465 MiB, below the 14,848 MiB safety threshold |

The physical T4 capacity was 15,360 MiB. The PyTorch allocator peaks are worker-specific; `nvidia-smi` readings are physical-device samples and can include other processes. Periodic host/device sampling cannot rule out higher unsampled instantaneous values, but no OOM, safety stop, or numerical failure was recorded. The original local MX330 had about 1,662 MiB free before allocation; this Kaggle run's observed GPU footprint exceeds that local free-memory figure. Different PyTorch/CUDA stacks also prevent a strict hardware-only causal comparison.

## Controlled deviations and revised conclusion

Only the first six-way split, one configuration, and one epoch were used. The full experiment's hyperparameter grid, all-fold/repetition schedule, later early-stop checks, and five-epoch checkpoint/reload boundary were not reached. Unlike Stage 08's local 1,280/640-pair controlled subsets at batch 64, this Kaggle run used the full selected 25,047/5,009-pair split at batch 256. External read-only monitoring and the Python 3.11+ set-sampling compatibility shim were declared. No model architecture, optimizer mathematics, loss, source, data, or shipped folds were changed, and no package installation or retry was reported.

**Previous local findings:** Stage 07A observed batch-256 OOM during an isolated forward pass on the 2 GB MX330; Stage 08 observed batch-128 OOM on the first native training batch but completed repeated native train/test work at batch 64. **New evidence:** the selected native path completed at batch 256 on a T4 with finite outputs and memory below its Kaggle limits. **Revised scoped conclusion:** the batch-256 native path is executable under this higher-memory Kaggle environment, supporting a local resource-limit explanation. The earlier local failures remain valid for their hardware and run boundaries. This single cross-environment run does not establish that GPU memory alone explains every difference, nor any published-result reproduction or advantage over DCGAN-DTA.

Only this new Stage 08.5B report was added to the thesis repository during this review. Pre-existing working-tree changes, the original Co-VAE implementation, datasets, folds, and earlier audit reports were preserved. No local or Kaggle experiment was started by this review.

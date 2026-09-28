# Stage 09B preparation — Co-VAE Davis new-drug full-training reference

**Status (2026-09-28): PREPARED; static protocol audit PASS. Stage 09B execution NOT RUN.**
This note authorizes no execution by itself and records no full-training result. Stage 09A remains closed; its documentation and returned artifacts are unchanged.

## Files and identity

- Runner: [stage09b_covae_full_kaggle.py](../../scratchpad/stage09b_covae_full_kaggle.py)
- Four-cell Kaggle notebook: [stage09b_covae_full_notebook.ipynb](../../scratchpad/stage09b_covae_full_notebook.ipynb)
- Runner SHA-256: `c5be744f18ed0cf6298ecbc32c50380362122adf8c34988745b7ad023b31c7b2`
- Local Co-VAE nested HEAD: `2c172682e502e3d39f6cc6a6d5bca428326e7d0b`; attached Kaggle Git checkout identity is not assumed.
- Output: `/kaggle/working/stage09b_full/`; supervisor console: `/kaggle/working/stage09b_full_console.log`.

The notebook pins the runner hash. The runner verifies it and the eight immutable input hashes below before training, and rechecks source/data after execution. A mismatch fails loudly without repair.

| Immutable input, relative to CoVAE | Expected SHA-256 |
|---|---|
| `run_experiments.py` | `39546fa554e628bf4f30dc492d152ed1235fbeedfc83c713218dd0d40d108043` |
| `model.py` | `a6dc416b3c4257d67e226fb1888d6236c6203f483d943415c5c1a8879e32e8fc` |
| `arguments.py` | `1b9416dfb212a9c92bda76ec5eea18561530055b9b442d6c0c07e4a483fc388a` |
| `datahelper.py` | `eb7f1245531be9d5cf9e8099e5063125eac0c5dc3b6dbfe5fda9947f797b917c` |
| `emetrics.py` | `031696e05fb3c40ea3607ec76c76dcc5079d888d8a7ac2c60658e0cd041fc7f9` |
| `data/davis/ligands_iso.txt` | `9789900f1e723226187215235b379998e5735a3ffec0e9487bad27c1c6a64161` |
| `data/davis/proteins.txt` | `187cedeee10cd3b58af915a2861a5ed0c4ff42a2d728f4dea5e0b47e3d75b291` |
| `data/davis/drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt` | `31b3f3efea9afe53239fa9fe735aedd6195586b99261319917b55b1d1c9e8392` |

## Exact controlled protocol

Davis, **new-drug / problem_type=2**, native six-way drug split at Python split seed **1000**. Folds 0–4 together supply training; fold 5 supplies the held-out test. No shipped fold file, grid, ten repetitions, inner validation/model selection, or repeated-test native driver is used.

Fixed configuration: **32 filters, drug kernel 5, target kernel 7, lambda exponent -5, batch 256, native Adam default learning rate 0.001, exactly 100 full epochs**, SMILES/protein caps **85/1200**. Native argument names are `--num_windows 32 --smi_window_lengths 5 --seq_window_lengths 7 --lamda -5 --batch_size 256 --num_epoch 100 --max_smi_len 85 --max_seq_len 1200 --problem_type 2 --is_log 0`, with explicit dataset/log paths. There is **no native learning-rate CLI argument**; every actual Adam construction is checked for lr 0.001.

**Paper rationale:** Table 1 / Section 3.3 disclose drug kernels {5,7}, target kernels {7,11}, lambda settings {-3,-5}, and base filters 32. The winning Davis configuration is not disclosed sufficiently to reconstruct it. The exponent interpretation `10**lamda` is source behavior, not an explicit paper convention. This is a **representative paper-grid full-training reference**, not the authors' best configuration or exact paper reproduction. Evidence: [Stage 09 plan, Minimum 09B](PLAN.md), [paper audit, Sections 9/13](../02-paper-forensic/Co-VAE.md), and [traceability audit](../03-paper-code-traceability/Co-VAE.md).

**Native behavior preserved:** source `run_experiments.py:85–140` defines branch loss as mean(sequence-summed cross entropy + KL), with KL `-0.5*sum(1+logvar-mu**2-exp(logvar))`. Training optimizes `MSE + 10**(-5)*(drug_loss + 85/1200*target_loss)`. `RE.train` recreates Adam on each call; the runner calls it 100 times, preserving this reset. Construction uses `RE.net(flags,32,5,7).cuda()` and the same `model.apply(RE.weights_init)`. No scheduler, clipping, new initialization, extra regularization, changed loss, early stop, or selected checkpoint is introduced. The full released grid/test-informed selection driver is deliberately bypassed, as predeclared in Stage 09.

Python, NumPy, PyTorch CPU and CUDA seeds are explicitly **1000**; split construction resets Python to 1000 immediately before its native call. `PYTHONHASHSEED=0` is set at child-process launch. These added controls do not claim the paper's undisclosed seed. DataLoader shuffle remains native, with the seeded global Torch RNG and default zero workers. No determinism/autotuning/backend flags are changed. Native CUDA reparameterization noise has std 0.1 and remains stochastic even in `eval/no_grad`; final prediction capture uses that single native pass without rerunning the model.

The only syntax compatibility shim temporarily converts sets to tuples for `random.sample` during native split generation and restores the original sampler afterward. This reuses [Stage 08.5B](../08-5-kaggle-gpu-validation/Co-VAE.md)'s mechanism. Baseline source is never rewritten.

## Split evidence and held-out isolation

A bounded local diagnostic extracted only native split/parser functions with AST; it did not import the baseline or Torch, construct a model, or train. Python 3.13.2 produced fold pair counts **[5304,5304,4862,4862,4862,4862]**, **25,194 train / 4,862 test pairs**, **57 train / 11 test drugs**, zero drug overlap, and repeatable membership under the shim:

- Local train ordered-index SHA-256: `07765b179f3c72d0382185f3ac56adb3cc0d2a306590bf8ef00edfefb47f05e7`
- Local test ordered-index SHA-256: `50692467100063bc6b005795b6827611bafce4e5f03ec77624ff3704e39ffa02`

These are local diagnostic references, not hard-coded Kaggle counts or a claimed Python-3.12 membership verification. Runtime records actual entity/pair counts, each fold's indices/hash, ordered train/test hashes, bounds/coverage/duplicates, drug-ID separation, target overlap, unique/duplicate protein sequences, and shared duplicate-sequence groups. Shared targets/sequences are expected for new-drug and do not establish cold-target separation.

Native parsing loads the full affinity matrix; isolation means held-out membership metadata may be inspected, but held-out pair tensors and the DataLoader are not constructed until the epoch-100 gate passes. The gate requires 100 completed epochs, all runtime-derived expected optimizer updates, 100 native Adam constructions, and saved final-state weights. It rejects premature access, a second loader, and a second traversal. `RE.test` is also guarded against early/repeated calls.

The epoch-100 state is saved once as `final_model.state_dict.pt`. One native `RE.test` call runs afterward; a read-only forward hook and loader observer capture its predictions/targets in native order. No second forward pass, reseeding, checkpoint selection, or test access during training occurs.

## Metrics, monitoring, and retained artifacts

Native test returns **CI, MSE, RM2, ROC AUC at pKd > 7** (`run_experiments.py:142–164`). This path does not return AUPR; the paper's KIBA AUC sweep is not claimed for Davis. External **MSE, MAE**, and a chunked full-vector check of the already audited native CI estimator are computed from the saved prediction CSV. The CI check preserves the source's lower-triangle/order convention; it is not a new standardized Stage 10 metric. Native MSE/CI and saved-vector checks must agree within declared numerical tolerances.

History records batch-size-weighted epoch means of native total loss, regression MSE, drug/target reconstruction-plus-KL, and native batch CI, plus actual optimizer construction/update counts, runtime, CUDA current/cumulative-peak allocated/reserved memory, and RSS. Separate reconstruction/KL terms are not exposed by native training; they are not fabricated or obtained by rewriting the loop.

Required artifacts: `environment.json`, `protocol.json`, `dataset_summary.json`, `split_summary.json`, `split_membership.json`, `history.csv`, `history.jsonl`, `live_status.json`, `resource_history.csv`, `milestones.jsonl`, `final_summary.json`, `test_predictions.csv`, `artifact_hashes.json`, final state_dict, architecture/parameter summary, worker summary, event log, and native training log. Final evidence includes runtime test/prediction/finite counts and min/max/mean/population-std statistics. The hash manifest includes the finalized summary/history/predictions/state and immutable inputs; it excludes itself to avoid circular hashing.

The supervisor monitors host RSS (12 GiB cap) and physical GPU 0 (capacity minus 512 MiB; 14,848 MiB on the prior T4). Samples may include other processes and miss instantaneous peaks; Torch allocator peaks are also retained. A worker-exit RSS race is handled, and three consecutive live-monitor failures stop safely. Numerical/count/overlap/hash/missing-artifact failures invalidate or fail the run; none are auto-repaired. **There is no wall-time kill, metric-based stop, retry, or overwrite of an existing output directory.**

## Kaggle execution cells

Attach this runner and the unchanged `CoVAE/` input with `data/davis/`. Import the prepared notebook and choose a T4 session, even if two GPUs are offered.

1. **Cell 1 — preflight:** verify the pinned runner and eight input hashes; print runner/source paths, Python/PyTorch/CUDA/package versions and GPU; check exactly one logical T4. Runs `--preflight` in a fresh process; no baseline import/model/training.
2. **Cell 2 — launch:** start one background supervisor so Cell 3 can monitor concurrently. The supervisor owns one worker and writes persistent logs. Refuse nonempty output and existing console files; no automatic retry.
3. **Cell 3 — monitor:** read live status and resource/history observations every five seconds; print epoch/100, elapsed time, loss components, CUDA current/peak memory, physical GPU, RSS, and warnings. It cannot modify training state or inspect test predictions during training. Interrupting this cell leaves the job running.
4. **Cell 4 — finalize:** after exit, print final summary/native metrics/external checks, list artifacts and hashes, verify saved file hashes, and show console/native log tails. A missing summary, failed status, nonzero exit, or hash mismatch is reported without rerunning.

The runtime launch command is `[sys.executable, '-B', '-u', str(RUNNER)]`, with `CUDA_VISIBLE_DEVICES=0`, `PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`, and the pinned `STAGE09B_EXPECTED_RUNNER_SHA256`. The notebook contains the exact executable cells; no local execution or package installation is required. Download the output directory and console after completion for forensic review before closing Stage 09B.

## Expected cost and unresolved risks

**Planning estimate only:** Stage 08.5B measured a 28.534 s epoch / 98 updates and 1.795 s test evaluation for warm 32/4/8. The new-drug diagnostic implies 99 training batches; simple scaling gives about **48 minutes** for 100 epochs before setup/monitoring. Plan around **50 minutes**, with a practical **45–90 minute allowance**, and replace this estimate with actual epoch timings. Changing kernels, setting, stack, warm-up and instrumentation prevents exact extrapolation.

Prior physical GPU peak was **2,465 MiB**, Torch peak allocated **1,993.93 MiB** and reserved **2,322 MiB**. Budget roughly **2.5–3 GiB** as a planning reference, not a 5/7 memory guarantee. Decoder dimensions 73/1182 remain positive and odd-kernel encoder padding preserves length, but the 5/7 forward path and 100-epoch stability have not been executed. Kaggle versions/session availability may change; Cell 1 records the actual stack without installing anything. Seeded execution is not bitwise determinism. Paper-selected hyperparameters, full selection and ten-split aggregation remain unrecoverable; no numerical reproduction claim is prewritten.

## Static forensic audit

All PASS entries below apply to preparation/static controls, not successful future execution. Locations are in the runner unless stated otherwise.

| # | Requirement | Static result and evidence |
|---|---|---|
| 1 | Davis | PASS — `native_flags:182`, Davis input hashes, native loader path |
| 2 | New-drug / problem_type 2 | PASS — parser assertion and `get_drugwise_folds` at `build_training_inputs:268` |
| 3 | Split seed 1000 | PASS — `SEED`, `random.seed:296`, native six-way call |
| 4 | Fixed 32/5/7/-5 | PASS — constants, singleton parser assertions, native constructor/train/test arguments |
| 5 | Batch 256 | PASS — parser assertion and both native DataLoaders |
| 6 | Learning rate .001 | PASS — `ObservedAdam:428` preserves native defaults and asserts every actual group lr |
| 7 | Exactly 100 full epochs | PASS — `worker:485` bounded loop; batch/sample/update/construction invariants |
| 8 | No test-driven early stop | PASS — no evaluation or break in training loop; no paper/metric stopping condition |
| 9 | No training-time test evaluation | PASS — one test call outside loop; guarded `RE.test` at `worker:474` |
| 10 | Held-out isolation | PASS — `HeldoutGate:228`; deferred factory at 347; final-state save before loader construction |
| 11 | One final held-out evaluation | PASS — sole call at 540; native-call and single-traversal counters; capture hook returns no replacement |
| 12 | Native model/loss/train preserved | PASS — original native calls; observations delegate original optimizer/loss/progress methods; no source edit |
| 13 | External compatibility only | PASS — temporary set-to-tuple shim at 294–302, restored in finally |
| 14 | One GPU | PASS — CUDA environment before Torch; child launch forces GPU 0; logical-device/T4 assertions |
| 15 | Sufficient evidence | PASS — required artifacts and finalization checks; source/data and finalized-output SHA-256 manifests |
| 16 | Existing changes preserved | PASS — only the three new preparation files created; protected documentation, baseline inputs, legacy runner and all Stage 09A results rehashed unchanged |

Additional checks passed: Python syntax; four unexecuted notebook code cells and matching runner pin; source-native argument resolution; repeatable native split/partition/disjoint drugs; gate failure injection (epoch 99, missing optimizer update, access outside native call, repeated traversal/open); and saved-vector checks (prediction ties, no comparable CI pairs, multiple chunks, MSE/MAE, NaN/Inf rejection). These diagnostics used standard-library/NumPy helpers only, without Torch import, model construction, or optimizer execution.

GitNexus supplied native train/test context. It has no indexed symbol for the new runner, so a graph impact result is unavailable (UNKNOWN); local call/control-flow review was used, with no reindex and no modification of existing indexed baseline symbols. No baseline source, dataset, fold, existing audit report, Stage 09A evidence, or implementation plan was changed. No full training, Stage 10, representation extraction, or hybrid work was started.

# Stage 08 — Controlled Training Validation: DCGAN-DTA Variant C

**Date:** 2026-09-27. **Experiment status: STOPPED BY SAFETY LIMIT.**
**Interpretation: BLOCKED under the specified local resource boundary.**

Variant C constructed successfully and entered native DTA training. One optimizer update was confirmed with finite training loss and c-index. The worker exceeded the 2.5 GiB RSS limit before that update completed; the monitor's termination command was denied by Windows. Elevated termination was subsequently successful. No validation evaluation was reached, and repeated DTA updates were not established. This is not a successful validation within the requested resource boundary.

## 1. Objective and prior evidence

Validate the smallest bounded native Variant C path: drug GAN, Variant C construction, one DTA training epoch including native validation. No published-result reproduction, grid search, or full cross-validation was intended.

Reviewed `AGENTS.md`, `audit/07-training-feasibility/DCGAN-DTA.md`, and `audit/08-controlled-training-validation/DCGAN-DTA.md` (Stage 08B), plus the relevant source and existing Stage 04 fold findings. Stage 08B established generator parameter/output change despite constant logged generator loss; it did not exercise DTA construction or downstream training. Those findings and reports remain unchanged. Co-VAE was not rerun.

## 2. Source identity and selected native path

| Item | Identity |
|---|---|
| Repository root | `D:/ut/thesis/refferences/final/GAN-vs-VAE-` |
| Root HEAD | `ac5809b8fa4204f5c44c4faf3f5f5bbeb9539c28` |
| Baseline directory | `DCGAN-DTA/DCGAN-DTA/` |
| Baseline HEAD | `453fe16a3c6279dff3ddd4bd28ee63f50465b799` |
| Native constructor | `run_experiments.py:445`, `build_GAN_C` |
| Native training/evaluation driver | `general_nfold_cv`; `fit_generator` at line 660 |

Static inspection established that `nfold_1_2_3_setting_sample` ordinarily calls the CV driver twice, first for validation folds and then test sets. The external runner instead called the unchanged `general_nfold_cv` directly with singleton configuration lists and one train/validation pair of index lists. This avoided both the multi-fold/grid schedule and the second test pass.

The native driver itself created interaction pairs and the original `DataGenerator`, called the original `build_GAN_C`, and entered its original `fit_generator` call with native EarlyStopping. Its validation data argument was retained. The planned external stop was immediately after that fit returned, before `predict_generator`, extra metrics, plotting, and cleanup. That planned boundary was not reached.

## 3. Exact configuration and fold

The first README grid combination was selected; no search or alternative configuration was tried.

```text
--dataset_path data/pdb/
--problem_type 1 --is_log 0 --model C
--num_windows 128 --smi_window_lengths 4 --seq_window_lengths 4
--max_smi_len 200 --max_seq_len 2000
--batch_size 256 --num_epoch 1
--log_dir C:/Users/Snapp/AppData/Local/Temp/codex-variant-c-20260927-084354
```

This is the README's first filter/kernel combination, with Variant C explicitly selected and the epoch count bounded to one rather than the README's 300. No assertion is made that this single combination reproduces a canonical published result.

Native `DataSet.parse_data` loaded PDBbind with default `with_label=2`. `XD_t` was `(50068, 200)` and used unchanged for the drug GAN. The unused target-GAN corpus `XT_t` was released externally; Variant C does not consume it. Observed affinity entries were identified by the native `np.where(np.isnan(Y) == False)` rule.

| Split property | Measured selection |
|---|---|
| Fold | First shipped validation fold, native index 0 (human fold 1) |
| Source | `data/pdb/folds/train_fold_setting1.txt` and `test_fold_setting1.txt` |
| Observed affinity pairs | 5,014 |
| Training pairs | 3,344; concatenation of the other four shipped training folds, preserving order |
| Validation pairs | 836; first shipped training fold |
| Held-out test pairs | 834; recorded, not executed |
| Planned training batches | 14 at batch size 256 |
| Planned validation batches | 4 |
| Split origin | Shipped folds, not newly generated indices |

The runner retained native pair ordering, generator behavior, default Keras fit behavior, and the native validation set. It did not truncate the fold or substitute a smaller batch. Existing Stage 04 evidence was reused; no full leakage/fold audit was repeated.

## 4. Environment and command

| Component | Actual value |
|---|---|
| Python | 3.10.0, 64-bit |
| TensorFlow / Keras | 2.10.1 / 2.10.0 |
| NumPy | 1.26.4 |
| OS platform string | `Windows-10-10.0.26200-SP0` |
| CPU | 11th Gen Intel Core i7-1165G7 @ 2.80 GHz |
| TensorFlow physical/logical devices | CPU only; no visible GPU |

The existing Stage 07B/08B environment was reused, with no installation, update, device override, or GPU repair:

```text
C:/Users/Snapp/AppData/Local/Temp/claude/D--ut-thesis-refferences-final-GAN-vs-VAE-/fcfe04c0-12b6-44a3-9bd5-a8840772c0a8/scratchpad/venv-dcgan/Scripts/python.exe
```

Launch command:

```powershell
python -B C:/Users/Snapp/AppData/Local/Temp/codex-variant-c-20260927-084354/monitor.py
```

The parent launched the virtual-environment interpreter with `-B -u` and the external `worker.py`, with the baseline directory as working directory and `PYTHONDONTWRITEBYTECODE=1`. Source-native NumPy/Python/TensorFlow seed calls were retained; the observer did not reseed training. This does not establish cross-run bitwise reproducibility.

## 5. GAN reuse decision and result

**One new native GAN call was required and executed.** `build_GAN_C` unconditionally invokes `ganForDrug(XD_t)`. Despite the variable name `gan_smiles`, this function returns the trained **discriminator**, and Variant C applies `gan_smiles.layers[-3]` to the drug embedding. It does not consume the generator object.

Stage 08B saved weight arrays, not a live object plus a native constructor restore interface. Reusing those arrays through this constructor would require replacing its GAN call or reconstructing part of its behavior. No such patch or reconstruction was made. The new call retained 5,000 iterations, batch size 5, interval 500, native sampling, input scaling, architectures, Adam optimizers, labels, and losses.

The GAN reached all ten checkpoints. Its training closure began at worker elapsed 18.516 s; iteration 5,000 was observed at 620.141 s (approximately 601.625 s between those events). Generator loss at every checkpoint was `15.424947738647461`, printed as `15.424948`; discriminator accuracy was 100%. Aggregate discriminator loss was zero at the final checkpoint. Observed aggregate GAN losses/accuracy were finite. No new generator-weight claim is made from this run.

## 6. Construction and DTA update evidence

The native Variant C constructor returned successfully at worker elapsed **620.563 s**:

| Model property | Observed value |
|---|---:|
| Total parameters | 3,070,593 |
| Trainable parameters | 3,064,385 |
| Non-trainable parameters | 6,208 |
| Optimizer iterations at training entry | 0 |

Native compile configuration was Adam, mean-squared-error loss, and `cindex_score`. `fit_generator` was entered at **620.579 s**. The native model summary and external model JSON were retained. Keras warned that `fit_generator` is deprecated; no construction/compile exception occurred.

At **648.782 s**, the original Keras `CallbackList.on_train_batch_end` returned for batch index 0. External observation recorded:

| Evidence | Value |
|---|---:|
| Optimizer iteration counter | **1** |
| Training loss | **43.4715576171875** |
| Training `cindex_score` | **0.5101507902145386** |
| Finiteness of recorded values | Both finite |
| Training entry to first completed-batch observation | **28.203 s** |

The counter is direct evidence of **one confirmed completed optimizer step**, not an inference from calling fit. No second batch-end event was recorded. Because termination occurred during execution and no final optimizer counter could be read, the exact terminal counter is unavailable; repeated completed updates are not established. These values are training metrics, not validation/test results or evidence of model quality.

Crucially, the first completed update occurred **after the memory threshold had already been exceeded**. It must not be represented as an update validated within the requested RSS limit.

## 7. Native evaluation

Native validation was configured inside the existing `fit_generator` call, but **not reached**. There was no `on_test_begin`, validation batch, epoch-end history, or returned validation metric. No validation/test metric values are available.

The later native `predict_generator`, RM2, AUPR, plotting, test-set pass, and cross-fold aggregation were not executed. MSE/c-index names were configured in the model, but configured names do not establish returned evaluation results.

## 8. Resource trajectory and safety-stop failure

The monitor sampled Windows working-set size for actual worker PID **22536**, obtained from the worker itself, rather than the launcher. It collected **1,249 samples**, approximately every 0.5 s. The first sample was 34.332 MiB during startup. Below are nearest samples to worker events; offsets are within approximately 0.25 s, except the separately identified threshold event.

| Event | Worker elapsed (s) | RSS (MiB) |
|---|---:|---:|
| GAN training start | 18.516 | 697.56 |
| GAN 500 | 69.125 | 963.84 |
| GAN 1000 | 126.969 | 1107.08 |
| GAN 1500 | 184.610 | 1234.70 |
| GAN 2000 | 233.782 | 1360.70 |
| GAN 2500 | 296.438 | 1499.90 |
| GAN 3000 | 351.282 | 1623.86 |
| GAN 3500 | 410.094 | 1750.54 |
| GAN 4000 | 468.985 | 1874.27 |
| GAN 4500 | 543.329 | 2012.97 |
| GAN 5000 | 620.141 | 2065.86 |
| DTA construction/training entry | 620.563 / 620.579 | 2088.31 |

RSS increased gradually through GAN training, then increased sharply during DTA entry. At **parent elapsed 626.060 s**, the monitor observed **2,813,566,976 bytes = 2,683.227 MiB = 2.620338 GiB**, exceeding the 2.5 GiB threshold. This is the **maximum sampled RSS**, not the true peak or final RSS: sampling stopped at that point.

The monitor attempted `taskkill /PID <launcher> /T /F`. Windows returned **`ERROR: Access denied`**. The parent then failed with `subprocess.TimeoutExpired` while waiting 15 seconds for the worker. Thus the intended immediate safety termination **did not succeed**. The worker continued and recorded its first completed DTA batch at 648.782 s; memory after threshold detection was not sampled.

The exact worker was subsequently terminated using elevated `Stop-Process -Id 22536 -Force`. Absence was verified at **2026-09-27 08:56:17.9726559 +03:30**; a follow-up process query found no process for this experiment script. From the worker's recorded start timestamp, confirmed absence was approximately **664.314 s** after startup. Exact termination time was not captured: runtime is bounded by the last live callback at **648.782 s** and confirmed absence at **664.314 s**. It was below 900 s, but the memory limit was exceeded and enforcement was delayed.

TensorFlow logged large CPU allocation warnings for 522,715,136-byte and 782,893,056-byte allocations. These are allocation warnings, not a demonstrated TensorFlow OOM exception. No native model exception or non-finite training value was observed before forced termination. The monitor exception is distinct from the model's behavior. No complete epoch runtime, final model state, final RSS, or normal worker exit result exists.

This trajectory does not establish a memory leak. The retention of DTA data, GAN/runtime state, framework allocations, and observer overhead were not isolated experimentally. No automatic cleanup, smaller-batch fallback, configuration switch, or retry was attempted.

## 9. External controls and deviations

- One native configuration and one native validation fold were supplied externally; no source loop was rewritten.
- The supported `num_epoch` argument was set to one. Native EarlyStopping and validation arguments remained intact.
- The unused target-GAN corpus was released; the driver received `None` for that unused Variant C argument.
- Python tracing observed native GAN checkpoints, constructor return, optimizer counters, original callback logs, and the intended post-fit stop line. It did not add callbacks, patch training methods, replace optimizers, change model architecture, or modify sampling/order/losses.
- Logs and observer files were directed outside the repository. The driver would have been interrupted before plotting or further experiment work.
- Tracing and monitoring add overhead, so this is not an uninstrumented performance benchmark.
- The RSS safeguard detected its threshold but failed to enforce an immediate stop. Elevated manual termination was necessary; resource compliance cannot be claimed for the update observed afterward.

## 10. Interpretation and remaining uncertainty

**BLOCKED under the current configuration and resource boundary.** Construction succeeded, the original training path was entered, and one completed finite update was directly observed. The memory threshold was exceeded before that update completed, repeated updates were not demonstrated, and native validation was not reached. Therefore neither PASS nor CONDITIONAL PASS is supported for the requested downstream validation.

This is not a claim that Variant C cannot train on other hardware or under a separately authorized configuration. Open questions include repeated-update stability, one-epoch completion, native validation behavior, actual DTA peak memory, and the contributions of retained GAN/framework state to memory use. Any future bounded execution also needs a termination mechanism verified to work under Windows permissions. No additional experiment is authorized by this report.

The run establishes no paper reproduction, model quality, convergence, generalization, advantage over Co-VAE, complete CV/grid feasibility, GPU feasibility, or target-GAN behavior. No Variant A/B, target GAN, additional fold/configuration, Kaggle job, Stage 08.5, or Stage 09 work was performed. No Stage 08 summary was created.

## 11. Artifacts and repository integrity

External evidence is retained at:

```text
C:/Users/Snapp/AppData/Local/Temp/codex-variant-c-20260927-084354/
```

Files include `worker.py`, `monitor.py`, `worker.json`, `environment.json`, `configuration.json`, `selected-indices.json`, `events.jsonl`, `rss.jsonl`, `native.log`, native `log.txt`, `model.json`, `event-memory.json`, `termination-verification.json`, `protected-before.json`, and `integrity.json`.

`event-memory.json` records nearest-sample offsets, including a stale nearest sample for the late DTA batch; that row is **not** a contemporaneous DTA-batch RSS measurement. The monitor failed before producing its normal `monitor.json`, and the worker was killed before producing `results.json`; their absence must not be interpreted as successful completion. The monitor traceback was captured in the task's command output. These external temporary-directory artifacts were retained, but are not protected against later OS cleanup.

All **70 protected manifest entries** matched their pre-execution SHA-256 values, including the existing Stage 08B report, source, datasets/folds, instructions, and Codex skills. The manifest is scoped to those entries, not a complete repository-wide hash claim. Existing working-tree changes and the baseline's pre-existing untracked `__pycache__/` were preserved. Only this new report was added inside the repository by this task. No source patch, prior-report edit, installation, GitNexus reindex, or commit was made.

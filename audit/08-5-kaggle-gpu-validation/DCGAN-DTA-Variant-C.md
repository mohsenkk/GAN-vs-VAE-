# Stage 08.5A — DCGAN-DTA Variant C Kaggle GPU Validation

## Decision and scope

**PASS for the bounded execution question.** On one Kaggle Tesla T4, the selected Variant C path completed one native drug-GAN call, constructed the DTA model, performed 14 optimizer updates across one DTA epoch, and reached the repository's native validation path. Recorded training and validation values were finite. This resolves the Stage 08C uncertainty about whether this selected path can progress beyond the local resource boundary in a higher-resource environment. It does **not** establish paper reproduction, model quality, full-grid or all-fold feasibility, or that the GPU alone caused the difference from local execution.

The returned `summary.json` says `status: stopped_by_safety_limit` and the notebook runner exited 1. Those are **parent-monitor classifications**, not evidence that training failed: the worker returned 0, its exception field is null, and completed GAN, epoch, and validation evidence is present. The monitor recorded one missing RSS value during worker shutdown and overwrote the worker's completed status with `worker RSS monitor unavailable`. The raw reported status is preserved here; it is not silently relabeled.

## Evidence and identity

The evidence is the user-returned Kaggle `summary.json`, the last 60 lines of `training.log`, the last 10 lines of `memory.jsonl`, and notebook stdout, pasted into `C:/Users/Snapp/.codex/attachments/8d52f9df-e54c-407b-ac97-c7a0cce78667/Pasted text.txt` (SHA-256 `68ecf565bab87345b76389d47add7b76269d7b98a200cc6d24bdc0e546746e67`). The full Kaggle artifact files were not available locally for independent review. The runner used `/kaggle/input/models/mohsenkashefikia/stage08-5a-kaggle/tensorflow2/default/1/stage08_5a_kaggle.py`; its working outputs were under `/kaggle/working/stage08_5a/`. The command was a single Python runner invocation from the Kaggle notebook; its exact cell text was not included with the returned evidence.

| Item | Recorded value |
|---|---|
| Root thesis HEAD at local review | `ac5809b8fa4204f5c44c4faf3f5f5bbeb9539c28` |
| Baseline HEAD at runner preparation | `453fe16a3c6279dff3ddd4bd28ee63f50465b799` |
| Kaggle baseline input | `/kaggle/input/models/mohsenkashefikia/dcgan/tensorflow2/default/1/DCGAN-DTA` |
| Baseline `run_experiments.py` SHA-256 | `c05b14bd62fd0fb90419868e64e50bfbd9628401013e817001058eccf28a7058` |
| Python / platform | 3.12.13 / Linux 6.12.90+ |
| TensorFlow / `tf_keras` / NumPy | 2.20.0 / 2.20.0 / 2.0.2 |
| Other recorded libraries | pandas 2.3.3, scikit-learn 1.6.1, matplotlib 3.10.0 |
| Device | One visible Tesla T4, compute capability 7.5, 15,360 MiB; four CPUs reported |

The Kaggle input's `run_experiments.py` hash matches the locally inspected baseline file. The runner also recorded hashes for four other source files and eight PDBbind data/fold files, with `unchanged_after_run: true` for the source copy check. The Kaggle input's **Git commit identity was not verified** (`input_git_identity_verified: false`); the baseline HEAD above is the local preparation identity, not a verified Kaggle checkout claim. The uploaded runner itself was not fingerprinted in the returned summary.

## Selected method and observations

The controlled path used PDBbind (`is_log=0`), Variant C, 128 filters, drug and protein kernel lengths 4, batch size 256, the first shipped validation fold (zero-based index 0), and exactly one DTA epoch. Native source seed calls use NumPy/Python seed 1 and TensorFlow seed 0; this run does not establish bitwise repeatability across environments.

The loaded arrays were `XD (4231, 200)`, `XT (1606, 2000)`, `Y (4231, 1606)`, and drug-GAN corpus `XD_t (50068, 200)`. There were 5,014 observed affinity pairs; 3,344 training pairs, 836 validation pairs, and 834 held-out test pairs. The test set was not used. The shipped fold files were selected rather than regenerated; their full Stage 04 validity was not re-audited here.

| Boundary | Returned evidence |
|---|---|
| Native drug GAN | Exactly one call completed; 630.993 s recorded GAN time |
| Variant C | Constructed with 3,070,593 parameters |
| DTA epoch | 14/14 training batches and 14 optimizer updates completed |
| Training metrics | Loss 23.3320; c-index 0.54151; all 14 recorded batch loss/c-index pairs finite |
| Native validation | Fit validation and post-fit native metric path reached; 836 predictions; selected epoch index 0 |
| Validation metrics | CI 0.57333684; MSE/validation loss 15.95072460; AUPR 0.48635567; RM2 0.04664521; all finite |
| Exception / worker exit | No recorded exception; worker return code 0 |

The log excerpt independently shows the 14/14 fit line with finite training and validation loss/c-index, best-epoch weight restoration, and entry to `predict_generator`. The other validation metrics and GAN completion are evidenced by the returned summary, not by the excerpt alone. Deprecation warnings for `fit_generator` and `predict_generator` did not stop execution. These metrics establish finite execution, **not** predictive quality or comparability to published results.

## Runtime and resource boundary

| Measurement | Recorded value |
|---|---:|
| Worker total | 679.189 s |
| Parent total | 683.291 s |
| Environment initialization | 13.976 s |
| Dataset loading | 4.131 s |
| Variant C construction, excluding GAN | 0.261 s |
| DTA fit including in-fit validation | 25.521 s |
| Post-fit prediction and metrics | 2.815 s |
| Peak sampled worker RSS | 2,766,475,264 bytes = 2.576 GiB, below the 12 GiB limit |
| Peak sampled physical GPU use | 13,883 MiB, 965 MiB below the 14,848 MiB safety threshold |

The physical GPU total was 15,360 MiB. GPU use is a device-level observation that can include other processes, not a precise attribution to the worker. Periodic samples do not prove that unsampled peaks stayed below these values. The run completed without a recorded OOM or numerical failure. Its sampled host RSS exceeded the **separate** 2.5 GiB cap used in Stage 08C locally; Kaggle's limit for this run was 12 GiB.

Notebook stdout recorded `dta_epoch_completed` at parent elapsed 679.1 s and `native_validation_completed` at 681.1 s. RSS was still 2,487,062,528 bytes at 682.062 s; the next and last sample, at 683.097 s, had `worker_rss_bytes: null`. The parent loop in the executed runner treated a null `/proc/<pid>/status` RSS while `proc.poll()` had not yet reported exit as a safety failure. It then preserved that reason after reading the worker result and forced `stopped_by_safety_limit`, despite worker return code 0. This is consistent with a worker-exit monitoring race; the exact process-state transition was not captured. There is no sampled RSS or GPU threshold crossing that explains a true resource-limit stop.

## Deviations, revision, and limits

This was one configuration, one fold, one DTA epoch, and no final test pass, cross-validation aggregate, grid search, plotting, or session reset. The read-only Kaggle input was accessed through a writable source copy. Standalone `keras` imports were mapped to `tf_keras` 2.20 without editing original source. An unused native `np.mat(Y copy)` allocation was omitted because NumPy 2 removed `np.mat`. These are declared environment/control deviations; the Kaggle path is not an exact historical environment or full paper protocol.

**Previous Stage 08C conclusion:** on the local Windows CPU setup, the 2.5 GiB host-RSS limit was crossed before a second confirmed update, and native validation was not reached; that local path was resource-blocked under its specified boundary. **New evidence:** this Kaggle Linux/T4/12 GiB-bound path completed repeated updates and native validation. **Revised scoped conclusion:** the selected Variant C validation path is executable with these higher-resource and compatibility conditions; the Stage 08C local boundary finding remains valid. Hardware, operating system, framework versions, and compatibility adaptations changed together, so this single run cannot isolate which difference was decisive.

After reviewing the returned evidence, the local scratchpad runner's monitor was amended to wait briefly for normal worker exit when RSS disappears and to apply resource thresholds only while the worker is alive. This is a **post-run instrumentation correction**; it did not change the Kaggle-executed file or its raw `summary.json`, and no rerun was performed. No baseline source, data, folds, prior audit report, or original paper was changed. No Stage 09 or thesis-novelty work was started.

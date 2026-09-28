# Stage 09B — Co-VAE Davis new-drug controlled full training

**Closure (2026-09-28): Stage 09B: PASS for controlled full training and held-out evaluation.** The completed run independently passed forensic review. Its claim is a **representative paper-grid full-training reference**, not exact numerical paper reproduction, the paper's ten-split aggregate, or its undisclosed selected configuration.

## Completed-run evidence and provenance

Returned artifacts are preserved under [CoVAE-resluts/stage09b_full/](CoVAE-resluts/stage09b_full/). The [final summary](CoVAE-resluts/stage09b_full/final_summary.json), [protocol](CoVAE-resluts/stage09b_full/protocol.json), [environment](CoVAE-resluts/stage09b_full/environment.json), [CSV history](CoVAE-resluts/stage09b_full/history.csv), JSONL history, [events](CoVAE-resluts/stage09b_full/events.jsonl), [native training log](CoVAE-resluts/stage09b_full/training.log), [supervisor console](CoVAE-resluts/stage09b_full_console.log), resource samples, saved final state, and prediction vectors were independently inspected. The accepted review reused the returned execution evidence; it did not rerun training or make another model prediction.

| Identity | Recorded evidence |
|---|---|
| Local runner | [stage09b_covae_full_kaggle.py](../../scratchpad/stage09b_covae_full_kaggle.py); SHA-256 `c5be744f18ed0cf6298ecbc32c50380362122adf8c34988745b7ad023b31c7b2` |
| Kaggle notebook | [stage09b_covae_full_notebook.ipynb](../../scratchpad/stage09b_covae_full_notebook.ipynb); pins that runner hash |
| Kaggle baseline input | `/kaggle/input/models/mohsenkashefikia/covae-model/pytorch/default/1/CoVAE` |
| Kaggle output | `/kaggle/working/stage09b_full/`; supervisor console `/kaggle/working/stage09b_full_console.log` |
| Local nested Co-VAE HEAD at preparation | `2c172682e502e3d39f6cc6a6d5bca428326e7d0b` |
| Thesis root HEAD at documentation closure | `d79b452e38b164cf8bb2a254861efe5ce934bc93`; pre-existing working-tree changes are separate |
| Kaggle input Git identity | **Unverified** (`input_git_identity_verified: false`); file fingerprints establish the inspected inputs, not an on-Kaggle commit |

## Exact controlled protocol

| Item | Verified protocol |
|---|---|
| Dataset / setting | **Davis, new-drug / `problem_type=2`, `is_log=0`**; 68 drugs, 442 target IDs, 30,056 observed pairs |
| Labels | Native Davis transform `-log10(Kd/1e9)` from nanomolar Kd to pKd |
| Fixed configuration | **32 base filters, drug kernel 5, target kernel 7, lambda exponent -5**, batch **256** |
| Sequence lengths | Drug **85**, target **1200**; native encoding/truncation preserved |
| Model / initialization | Source-native `RE.net(flags,32,5,7).cuda()` and `model.apply(RE.weights_init)`; 12,931,834 parameters |
| Objective | Native `MSE + 10**(-5) * (drug_loss + 85/1200 * target_loss)`; each branch loss is mean(sequence-summed reconstruction cross entropy + KL) |
| Optimizer | Native Adam default learning rate **0.001**, checked on each construction; **recreated per epoch**, preserving the released `RE.train` behavior |
| Training / selection | **100 complete epochs** on folds 0–4 combined; fixed epoch 100 reference; no early stopping, internal validation selection, grid, or test-driven checkpoint selection |
| RNG controls | Python, NumPy, PyTorch CPU/CUDA and split seed **1000**; `PYTHONHASHSEED=0` at process launch |
| Compatibility deviation | Temporary external `random.sample` set-to-tuple shim during native split construction, then restored; no baseline source patch |

The notebook launched the pinned external runner as `[sys.executable, '-B', '-u', str(RUNNER)]`, with `CUDA_VISIBLE_DEVICES=0`, `PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`, and `STAGE09B_EXPECTED_RUNNER_SHA256` set to the fingerprint above. The original parser resolved these arguments, recorded in `protocol.json` (this is not an instruction to launch the released grid driver):

```text
--dataset_path /kaggle/input/models/mohsenkashefikia/covae-model/pytorch/default/1/CoVAE/data/davis/
--problem_type 2 --is_log 0 --num_windows 32
--smi_window_lengths 5 --seq_window_lengths 7 --lamda -5
--batch_size 256 --num_epoch 100 --max_smi_len 85 --max_seq_len 1200
--log_dir /kaggle/working/stage09b_full
```

The released grid and repeated-test selection driver was deliberately bypassed under the predeclared [Stage 09 plan](PLAN.md). Native model construction, loss, optimizer updates, and `RE.train`/`RE.test` remained operative. The exponent interpretation of lambda is repository behavior; the paper does not explicitly specify that convention. No backend determinism flags were changed, and seeded execution does not establish bitwise GPU repeatability.

## Split evidence and held-out isolation

The [split summary](CoVAE-resluts/stage09b_full/split_summary.json) and [membership record](CoVAE-resluts/stage09b_full/split_membership.json) identify `RE.get_drugwise_folds -> RE.get_random_folds`, six drug-wise folds, and split seed 1000. Ordered fold pair counts are **[5304, 5304, 4862, 4862, 4862, 4862]**. Independent inspection confirmed valid bounds, unique complete coverage of all 30,056 observed pairs, correct entity ordering, and prediction identities matching fold 5 in order.

| Boundary | Verified evidence |
|---|---|
| Training | Folds **0–4** combined: **25,194 pairs**, **57 drugs**, **99 batches per epoch** |
| Held-out test | Fold **5**: **4,862 pairs**, **11 drugs**, **19 batches** |
| Drug / pair overlap | **Zero train/test drug overlap** and zero pair overlap; no overlap in raw or length-85 truncated SMILES |
| Target / sequence overlap | All **442 target IDs / 379 unique sequences** shared, including 18 duplicate-sequence groups; expected for new-drug, not a cold-target claim |
| Ordered training indices SHA-256 | `07765b179f3c72d0382185f3ac56adb3cc0d2a306590bf8ef00edfefb47f05e7` |
| Ordered held-out indices SHA-256 | `50692467100063bc6b005795b6827611bafce4e5f03ec77624ff3704e39ffa02` |

The returned Python 3.12 split hashes match the Python 3.13 [preparation reference](Co-VAE-PREPARATION.md). This cross-runtime comparison was established during forensic review; the runner's `cross_Python_reference_hash_verified: false` field records that it did not itself perform that comparison. The artifact is preserved unchanged, and no general cross-version guarantee is inferred.

**Held-out isolation applied to training, model selection, tensors and prediction access.** The native parser loaded the full raw affinity matrix, and membership/identity metadata was available before training. Held-out pair tensors and the loader were deferred until the gate verified 100 epochs, all required updates, 100 optimizer constructions, and the saved final state. No held-out evaluation occurred during training. This does not claim that the raw test labels were never loaded by the native parser.

## Training completion and optimizer behavior

CSV and JSONL histories agree on **100 contiguous epochs**, each covering all **25,194 training pairs** and **99 optimizer updates**. The final counts are **9,900 updates** and **100 optimizer constructions**. Native Adam's per-epoch reset was preserved rather than replaced with a persistent optimizer. All recorded loss components and numerical history fields are finite; the recorded total agrees with the native loss arithmetic within floating-point tolerance.

| Training observation | Epoch 1 | Epoch 100 |
|---|---:|---:|
| Sample-weighted total loss | 2.497931 | 0.250505 |
| Sample-weighted regression MSE | 2.493926 | 0.249276 |
| Native training batch CI, sample-weighted | 0.514456 | 0.861111 |

These are training observations, not held-out metrics or proof of convergence. Branch diagnostics retain reconstruction-plus-KL together, as exposed by the native loss; separate reconstruction and KL measurements were not fabricated. The stopping condition was completion of the predefined epoch budget, with no metric-based early stop.

## Final evaluation and independent metric verification

[Final epoch-100 weights](CoVAE-resluts/stage09b_full/final_model.state_dict.pt) were saved before held-out loader construction. The same completed model underwent **exactly one native `RE.test` call and one held-out prediction pass**. Events place evaluation after epoch 100; the summary records `test_calls=1` and `test_passes=1`. There was no selection of an earlier checkpoint. Native evaluation uses `eval`/`no_grad`, while the VAE's CUDA reparameterization noise remains stochastic under those modes.

The [prediction CSV](CoVAE-resluts/stage09b_full/test_predictions.csv) contains **4,862 finite predictions and targets** in the exact held-out membership order. Drug/target indices and IDs were independently checked, and saved labels match the original Davis pKd values at the loader's float32 precision. Prediction range is **4.685180–8.986460**; target range is **5.000000–9.795880**.

| Held-out metric | Verified result | Davis new-drug paper Table 2 reference (context only) |
|---|---:|---:|
| Native CI | **0.664941** | 0.712 ± 0.063 |
| MSE | **0.863547** | 0.724 ± 0.096 |
| External MAE | **0.664325** | 0.550 ± 0.048 |
| Native RM2 | **0.143994** | 0.107 |
| ROC AUC at pKd > 7 | **0.732374** | No Davis value used for this comparison |

Independent calculations from the saved vectors agree with native CI/MSE/RM2/ROC AUC and external MAE within numerical tolerance, without another prediction pass. Float64 MSE is **0.8635470959052146** versus native **0.8635472059249878**; the independent native-convention CI is **0.6649408166358756** versus native **0.6649408340454102**.

**CI preserves the native order-sensitive lower-triangle convention:** only pairs with `j <= i` and `y_i > y_j` enter the denominator, with half credit for prediction ties. It is a full-vector check of that estimator, not a newly standardized order-independent benchmark. **ROC AUC is not AUPR**; native `RE.test` returns ROC AUC at pKd > 7, with 553 positives and 4,309 negatives here. MAE was calculated externally from the same saved predictions.

## Runtime, environment, memory and integrity

Execution used **Python 3.12.13, PyTorch 2.10.0+cu128, PyTorch CUDA build 12.8**, NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, matplotlib 3.10.0, and tqdm 4.67.3 on Linux 6.12.90+ x86-64, four logical CPUs and **one logical Tesla T4** (`CUDA_VISIBLE_DEVICES=0`). These versions and RNG controls are provenance, not a reconstruction of the authors' undisclosed historical environment.

| Resource / exit evidence | Verified result |
|---|---|
| Total parent runtime | **41m44s** (2,504.426 s, rounded); worker 2,501.623 s |
| Final evaluation and checks | 2.072 s, as recorded by the runner |
| Peak sampled host RSS | **1.69 GiB** (1,819,246,592 bytes), below the **12 GiB** host limit |
| Sampled GPU peak | **2,463 MiB**, below the **14,848 MiB** safety limit |
| Torch allocator cumulative peaks | Allocated **1,989.99 MiB**; reserved **2,320 MiB** |
| Resource sample count | **2,416**; independently checked counts and maxima agree with the summary |
| Exit / stopping | **Worker exit code 0**; status `completed`; no recorded exception; 100 epochs and one native held-out evaluation completed |

Physical GPU monitoring covers GPU 0 and may include other processes; sampled peaks do not prove the instantaneous maximum. The run neither reached a sampled safety threshold nor invoked a wall-time stop.

All **18 entries** in [artifact_hashes.json](CoVAE-resluts/stage09b_full/artifact_hashes.json) were independently verified, including the finalized summary, histories, events, resource samples, prediction CSV and final state_dict. The manifest excludes itself to avoid circular hashing; the separate supervisor console is not a manifest entry. The recorded runner fingerprint matches the prepared local runner. Five source and three Davis input fingerprints match the preparation reference and current local files; the runner also reports `immutable_inputs_unchanged: true` after its post-run check. Full fingerprints remain in the manifest, protocol and historical preparation note.

## Historical evidence, limitations and final decision

**Previous finding:** [Co-VAE-PREPARATION.md](Co-VAE-PREPARATION.md) recorded only a prepared runner/static PASS, a local split diagnostic, and estimated cost; full 5/7 training stability and held-out metrics were unverified. **New evidence:** the returned run completed 100 epochs, passed the integrity and isolation checks, and produced one independently cross-checked held-out vector. **Revised conclusion:** controlled full training and held-out evaluation are established for this specific configuration, split and stack. The preparation file remains unchanged as historical evidence; its “NOT RUN” and cost-estimate statements describe preparation time, not the completed run.

**Numerical paper reproduction remains unverified.** Only one predeclared configuration and one deterministic split were used. The paper does not disclose its selected hyperparameter combination, reports means/stds across ten random splits, and its full fivefold model-selection protocol was not reproduced. The paper's Table 2 numbers above are contextual references, not PASS thresholds or evidence of close agreement. Existing architecture/objective deviations remain documented in [Stage 03](../03-paper-code-traceability/Co-VAE.md); the present run does not resolve them by successful execution. Native stochastic evaluation and metric conventions further limit inference. This result does not reproduce the ten-split aggregate, establish all-split stability, or rank Co-VAE against Stage 09A's PDBbind warm result.

**Final decision: Stage 09B: PASS for controlled full training and held-out evaluation. CLOSED.** Together with the accepted Stage 09A closure, the planned minimum Stage 09 controlled baseline-credibility work is **complete**, while numerical paper reproduction remains unverified for both baselines. **Stage 10 has NOT STARTED.** This documentation closure does not design the common Davis benchmark, extract representations, or begin hybrid/fusion or convex-optimization work. No baseline source or result artifact was modified, and no training was run during closure.

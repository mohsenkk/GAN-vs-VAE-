# Stage 09A — DCGAN-DTA Variant C 128/4/8 controlled full training

**Closure (2026-09-28): PASS for controlled full training and held-out evaluation.** The accepted forensic review establishes completion of the planned single-fold Stage 09A run. Numerical paper reproduction remains unverified.

## Completed-run evidence

Returned artifacts are preserved under [results/stage09a_full/](results/stage09a_full/). The [summary](results/stage09a_full/summary.json), [CSV history](results/stage09a_full/history.csv), JSONL history, [training log](results/stage09a_full/training.log), [events](results/stage09a_full/events.jsonl), memory samples, checkpoints, and saved prediction/target vectors were independently inspected.

| Item | Verified full-run evidence |
|---|---|
| Dataset / split | PDBbind, warm setting / `problem_type=1`, `is_log=0`; shipped validation fold 0: 836 pairs; training folds 1–4: 3,344 pairs |
| Configuration | DCGAN-DTA Variant C; 128 filters / drug kernel 4 / protein kernel 8; batch size 256; SMILES/protein caps 200/2000 |
| Native drug GAN | One fresh call; 5,000 iterations, batch 5; no protein GAN or reused preflight GAN state |
| Training | **150 completed epochs**, **2,100 optimizer updates** (14 batches per epoch); maximum allowed 300 epochs |
| Checkpoint selection | Strict maximum native validation `val_cindex_score`; **best epoch 75**, **best native validation CI 0.763699** |
| Stopping | Native validation-CI early stopping after **75 consecutive non-improving epochs**, at epoch 150 |
| Held-out evaluation | **Best checkpoint restored before held-out evaluation**; **exactly one held-out prediction pass**, **834 pairs** |
| Exit / numerical state | Worker return code 0; no recorded exception or numerical failure; predictions finite |

The history's maximum native validation CI occurs at epoch 75. All 50 recorded best-checkpoint updates strictly increased that metric. The training log confirms restoration of epoch 75; the event sequence places best-checkpoint validation before opening the held-out fold and before the single test pass. Native parsing loads the full affinity matrix, but held-out fold membership is not opened until after selection. Final-epoch weights remain separately saved; their lower validation MSE does not replace the predefined CI-selected checkpoint.

| Held-out metric | Verified result | Paper Fig. 3 Variant C reference (context only) |
|---|---:|---:|
| Global CI | **0.767271** | 0.768 |
| Reconstructed native batch CI | **0.766640** | Estimator alignment unverified |
| MSE | **2.113802** | 1.907 |
| AUPR (`y > 7`) | **0.781662** | 0.787 |
| RM2 | **0.405841** | 0.451 |

Global CI uses the repository's full-vector `emetrics.get_cindex`. Native batch CI was reconstructed with the source `cindex_score` on each native batch, sample-count weighted, from the same single held-out prediction vector; it does not imply a second model prediction or evaluation pass.

| Resource measurement | Verified result |
|---|---|
| Total parent runtime | **47m15s** (2,834.550 s, rounded) |
| Peak host RSS | **3.77 GiB** (4,051,161,088 bytes), below the 12 GiB limit |
| Sampled GPU peak | **13,883 MiB**, below the **14,848 MiB** safety threshold by 965 MiB |

All 2,809 parent memory samples stayed within the thresholds. GPU monitoring covers physical GPU 0, including other processes; sampled peaks do not establish an instantaneous maximum.

## Integrity, provenance, and claim boundary

CSV and JSONL histories agree on epochs and native validation values. Saved validation and held-out vectors contain 836 and 834 finite predictions respectively; independently recomputed MSE and MAE agree with the summary. Returned source-file hashes and the runner hash match their recorded identities. The runner SHA-256 is `0d79880083c723bef7c0f3072ea855a761a14537c0e4eaf0f40fdba3558ce966`; local nested HEAD at preparation is `453fe16a3c6279dff3ddd4bd28ee63f50465b799`. Kaggle checkout Git identity remains unverified; the summary retains source/data/fold hashes and reports the source copy unchanged.

Execution used Python 3.12.13, TensorFlow/`tf_keras` 2.20.0, NumPy 2.0.2, and one visible Tesla T4. Native NumPy/Python seeds were 1 and TensorFlow-v1 seed 0; deterministic GPU execution is not claimed. Declared controls include the fixed single-fold configuration, external legacy-Keras mapping, omission of the unused NumPy-2-incompatible `np.mat` allocation, and additional validation diagnostics. Baseline source, datasets, and folds were not patched.

**Numerical paper reproduction remains unverified because the paper's complete selection, aggregation, and metric protocol is not fully recoverable.** Published values above are contextual references, never PASS thresholds. This run establishes controlled full-training and held-out-evaluation credibility for this named configuration and split; it does not reproduce the historical search or published aggregate, establish all-fold convergence, or support a direct comparison with Co-VAE on another dataset.

**Revision of the previous scoped conclusion:** the retained preflight below established only one-epoch feasibility and left full training and held-out evaluation unverified. The completed-run artifacts now establish those two boundaries, so **Stage 09A is closed as PASS**. Stage 09 as a whole is not closed. This documentation update does not start Stage 09B, Stage 10, representation extraction, or hybrid implementation.

---

## Historical bounded preflight record

The following preflight findings are retained unchanged. Statements about an unperformed full run or unopened test fold describe that earlier preflight, not the completed run documented above.

**Decision: PASS for the bounded preflight.** On the selected Tesla T4, the fixed paper-reported 128/4/8 Variant C path completed one fresh native drug-GAN call, constructed the DTA model, completed one batch-256 epoch with 14 optimizer updates, and evaluated the first shipped validation fold. All returned training and validation values were finite. The worker and runner exited 0; the runner reported `preflight_pass: true` with every recorded PASS criterion true. This establishes one-epoch execution and sampled resource feasibility for this configuration. It is **not** full Stage 09A training, test evaluation, convergence evidence, model-quality evidence, or paper-number reproduction.

## Evidence and identity

The evidence is the user-returned full `summary.json`, last 60 lines of `training.log`, last 10 lines each of `memory.jsonl` and `events.jsonl`, and notebook stdout in `C:/Users/Snapp/.codex/attachments/4c3f0def-f986-4874-b9e2-29e9d8ff65b4/Pasted text.txt` (SHA-256 `a5177bf37ddbf630af4f8bf89f062acd2d224b477235fa623074e1d1c10ee226`). The complete Kaggle log and event files were not available locally for independent review; summary fields and pasted tails are the measurement sources. The log tail confirms all 14 fit batches and native validation, but does not show the earlier GAN's ten printed checkpoints independently.

| Item | Returned identity |
|---|---|
| Kaggle runner input | `/kaggle/input/models/mohsenkashefikia/stage09a-dcgan-preflight-kaggle/tensorflow2/default/1/stage09a_dcgan_preflight_kaggle.py` |
| Runner SHA-256 | `6f1a2fff496f6579e32cf4565f3535cb9838dee590439de8506405245553f466`; matches the locally prepared runner |
| DCGAN-DTA Kaggle input | `/kaggle/input/models/mohsenkashefikia/dcgan/tensorflow2/default/1/DCGAN-DTA` |
| Original `run_experiments.py` SHA-256 | `c05b14bd62fd0fb90419868e64e50bfbd9628401013e817001058eccf28a7058`; matches the inspected local file and Stage 08.5A input |
| Source/data integrity | Five source and eight PDBbind input/training-fold hashes in the returned summary match the predeclared runner hashes; `source.unchanged_after_run: true` |
| Thesis root HEAD at local review | `ac5809b8fa4204f5c44c4faf3f5f5bbeb9539c28`; working-tree changes remain separate from this commit |
| Local nested DCGAN-DTA HEAD at preparation | `453fe16a3c6279dff3ddd4bd28ee63f50465b799` |
| Kaggle input Git identity | **Not verified** as a checkout (`input_git_identity_verified: false`); file hashes establish the inspected files, not an on-Kaggle commit |
| Output directory | `/kaggle/working/stage09a_preflight/` |

The Kaggle notebook listed **two** Tesla T4 devices. The worker set `CUDA_VISIBLE_DEVICES=0`, TensorFlow reported exactly one visible GPU, and the parent monitored physical GPU 0. No conclusion is drawn about GPU 1.

## Environment, data, and exact path

| Item | Returned value |
|---|---|
| Python / platform | 3.12.13 / Linux 6.12.90+ x86-64 |
| TensorFlow / `tf_keras` / NumPy | 2.20.0 / 2.20.0 / 2.0.2; legacy Keras mapping and memory growth active |
| Other libraries | pandas 2.3.3; scikit-learn 1.6.1; matplotlib 3.10.0 |
| Selected GPU | Tesla T4, 15,360 MiB physical memory, compute capability 7.5 |
| PDBbind arrays | `XD (4231, 200)`, `XT (1606, 2000)`, `Y (4231, 1606)`, drug-GAN corpus `XD_t (50068, 200)`; 5,014 observed pairs |
| Fixed configuration | Warm `problem_type=1`, Variant C, `is_log=0`, filters **128**, drug kernel **4**, protein kernel **8**, batch **256**, lengths 200/2000, **one** DTA epoch |
| Optimizer | Native model compiles with the string `adam` and MSE loss; `--learning_rate 0.001` was passed and logged, but the released constructor does not consume that flag. The native Adam default is the operative setting. |
| Selected fold | First shipped training fold, zero-based fold 0: **836 validation** pairs; other four shipped training folds: **3,344 training** pairs |
| Held-out set | **834 pairs inferred** as the remainder of observed pairs. The held-out test-fold file was not opened or hashed, and no held-out pairs were constructed or evaluated. Native `parse_data` still loaded the full affinity matrix before selection by training-fold indices. |

The native `DataGenerator.__len__` uses ceiling division: 3,344 training pairs at batch 256 yield **14** training batches, and 836 validation pairs yield **4** validation batches. The runner observed a 256-pair first batch and recorded the actual optimizer counter rather than treating the derived count as proof of execution. Source-native Python/NumPy/TensorFlow seed calls remained in the imported baseline; their presence does not establish bitwise repeatability on this Kaggle stack.

## Completed boundaries and numerical evidence

| Boundary | Returned evidence |
|---|---|
| Drug-GAN pretraining | Exactly **one** call to `ganForDrug(XD_t)` through `build_GAN_C`; returned after **618.010 s**. Inspected source fixes 5,000 iterations, batch 5, logging every 500. No protein GAN, other variant, reused GAN state, or grid search. |
| Variant C construction | Completed with **3,605,121 parameters**; the earlier 128/4/4 model had 3,070,593, a difference of **534,528** between the returned model counts. |
| DTA training | One native `fit_generator` epoch; **14/14** batches and **14** optimizer updates recorded. All 14 returned batch loss/CI pairs finite. Final train loss **22.46318245**, native train CI **0.53452641**. |
| Native in-fit validation | Reached all **4** validation batches. Validation loss/MSE **6.26194715**, native Keras CI **0.56408370**; the log tail and event tail agree with the summary. |
| Native post-fit validation | **836** validation predictions. Native-path AUPR **0.47050731** (`y > 7`) and RM2 **0.04002332**; all returned validation metrics finite. Selected epoch index 0, as expected with one epoch. |
| Held-out test | `test_set_used: false` in the summary and validation result; `heldout_test_fold_file_read: false`; no test metric was produced. |
| Exit state | Worker return code **0**, notebook runner exit code **0**, no recorded exception or failure classification; all 11 returned PASS criteria true. |

These are **one-epoch validation values**, not estimates of converged performance or comparable results against the paper's Fig. 3. The native CI is Keras's batch-averaged estimator. AUPR and RM2 here are for validation, never the held-out test set.

## Time, memory, and safety boundary

| Measurement | Returned value |
|---|---:|
| Native drug GAN | 618.010 s |
| DTA fit including in-fit validation | 29.088 s (25.595 s before validation; 3.413 s validation) |
| Post-fit validation prediction/metrics | 2.718 s |
| Worker / parent elapsed | 672.408 s / 678.094 s; both below the **1,800 s** limit |
| Peak sampled worker RSS | 2,858,930,176 bytes = **2.663 GiB**, below the **12 GiB** cap |
| Peak sampled physical GPU 0 use | **13,883 MiB**, **965 MiB below** the 14,848 MiB safety threshold (15,360 MiB device minus 512 MiB headroom) |

Worker boundary samples recorded GPU use of **105 MiB before GAN**, **257 MiB after GAN and after model construction**, **8,323 MiB during DTA training**, and **13,883 MiB after the epoch and after validation**. The monitor's physical GPU figure can include allocations by other processes; samples do not prove that no higher instantaneous peak occurred. No sampled threshold crossing or OOM was reported. The corrected shutdown monitor finished normally here, unlike the Stage 08.5A parent status that was affected by a missing-RSS exit race.

## Comparison with Stage 08.5A and decision

Stage 08.5A measured the same Variant C path for one epoch at **128/4/4**, not the paper's protein kernel 8. Its physical GPU peak was also **13,883 MiB**; its fit including validation took **25.521 s**, compared with **29.088 s** here. The present **128/4/8** run resolves the specific uncertainty of whether kernel 8 can construct, update repeatedly, and reach validation within this sampled single-T4 boundary. Identical sampled physical GPU peaks do **not** establish identical true memory requirements; the model has more parameters, and a single epoch cannot establish long-run memory or timing stability. The measured timing difference should not be attributed to the kernel alone from these two runs.

The preflight used a read-only Kaggle input through a writable source copy, mapped original standalone `keras` imports to the verified `tf_keras` runtime, and omitted an unused `np.mat(Y copy)` allocation removed in NumPy 2. These were external compatibility/control deviations; the baseline implementation, input data, and shipped folds were not patched. No automatic retry, reduced batch, changed kernel, held-out test evaluation, or full Stage 09A run occurred.

**Revised scoped conclusion:** the Stage 09A **preflight gate passes** for one native drug GAN plus one 128/4/8 DTA epoch and validation on this Kaggle T4 setup. The full one-fold run is now technically reasonable to **prepare for separate authorization** under the [Stage 09 plan](PLAN.md); this evidence does not itself authorize or establish that full run, its early-stopping behavior, or its final test result. No result is compared to Co-VAE or carried into Stage 10.

Only this Stage 09A evidence report was created in the thesis repository during the present review. Pre-existing working-tree changes and all prior audit findings were preserved; no local training or new Kaggle job was started.

# Stage 09A — DCGAN-DTA Variant C 128/4/8 Kaggle preflight

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

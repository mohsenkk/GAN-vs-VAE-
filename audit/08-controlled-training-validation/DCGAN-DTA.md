# Stage 08 — Controlled Training Validation: DCGAN-DTA

**Date:** 2026-09-27. **Status: COMPLETED — one native drug-GAN call, 5,000 iterations.**

**Finding:** constant checkpoint generator loss did not mean a frozen generator in this run. All 12 generator trainable parameter tensors changed; their combined difference had L2 norm **9.714797289**. The same fixed latent input produced a mean absolute output difference of **0.905347842**. This establishes parameter and output change, not healthy training, convergence, or paper reproduction.

## 1. Objective and relationship to Stage 07B

Test only `ganForDrug(XD_t)` using the original implementation and the Stage 07B environment/configuration. Determine whether generator parameters and fixed-input outputs change despite a generator loss printed as `15.424948` at every native reporting checkpoint.

The previous finding in `audit/07-training-feasibility/DCGAN-DTA.md` was an observed constant checkpoint generator loss and 100% discriminator accuracy, with an **inference** of discriminator dominance and a non-learning generator. That stage did not measure generator weight or output differences.

**New evidence:** generator trainable parameters, BatchNormalization state, and fixed-input outputs all changed in this run. **Revised conclusion:** the checkpoint loss pattern alone cannot establish a frozen/non-learning generator. It remains insufficient evidence of useful learning. This is a new run; it does not retrospectively measure Stage 07B's weights. The earlier report is preserved unchanged.

The existing Stage 08 Co-VAE conditional-pass report was read for context. Co-VAE was not rerun.

## 2. Repository identity and preservation

| Item | Identity |
|---|---|
| Thesis repository | `D:/ut/thesis/refferences/final/GAN-vs-VAE-` |
| Root HEAD | `ac5809b8fa4204f5c44c4faf3f5f5bbeb9539c28` |
| Baseline repository | `DCGAN-DTA/DCGAN-DTA/` |
| Baseline HEAD | `453fe16a3c6279dff3ddd4bd28ee63f50465b799` |
| Native entry point | `run_experiments.py:54`, `ganForDrug`, including its nested `train` closure |
| Source SHA-256 | `C05B14BD62FD0FB90419868E64E50BFBD9628401013E817001058ECCF28A7058` |

The initial root working tree already contained modified `AGENTS.md` and `CLAUDE.md`, nested-repository untracked content, `.codex/`, and `project-structure.txt`. The baseline nested repository reported only pre-existing `__pycache__/` as untracked. These were preserved. Python ran with `-B` and `PYTHONDONTWRITEBYTECODE=1`.

Before/after SHA-256 checks matched for all **47 protected-file manifest entries** (instructions, Codex skills, audit reports, and non-ignored baseline files) and **22 dataset/fold manifest entries**. These manifests can overlap; 69 entries is not a claim of 69 distinct files or a hash audit of the entire repository. No original source, data, fold, or previous report was modified. This report is the only new repository file from this task; observation artifacts are outside the repository.

## 3. Actual environment

| Component | Observed value |
|---|---|
| Python | 3.10.0, 64-bit, MSC v.1929 |
| TensorFlow | 2.10.1 |
| Keras | 2.10.0 |
| NumPy | 1.26.4 |
| OS platform string | `Windows-10-10.0.26200-SP0` (Windows API/build identification) |
| CPU | 11th Gen Intel Core i7-1165G7 @ 2.80 GHz |
| TensorFlow physical devices | `/physical_device:CPU:0` only |
| TensorFlow logical devices | `/device:CPU:0` only |
| GPU visible to TensorFlow | No |

The existing Stage 07B virtual environment was reused, without installations or updates:

```text
C:/Users/Snapp/AppData/Local/Temp/claude/D--ut-thesis-refferences-final-GAN-vs-VAE-/fcfe04c0-12b6-44a3-9bd5-a8840772c0a8/scratchpad/venv-dcgan/Scripts/python.exe
```

TensorFlow logged a missing `cudart64_110.dll`. No GPU configuration changes or GPU repair were attempted. CPU identity was read from the Windows registry after a CIM query was denied; the registry returned the processor name alongside an unrelated property-conversion warning.

## 4. Exact execution and configuration

The external parent was launched with:

```powershell
python -B C:/Users/Snapp/AppData/Local/Temp/codex-dcgan-stage08-20260927-002551/monitor.py
```

It launched the virtual-environment Python listed above with `-B -u` and the external `worker.py`, with working directory `D:/ut/thesis/refferences/final/GAN-vs-VAE-/DCGAN-DTA/DCGAN-DTA`. The worker passed these arguments to the native parser:

```text
--dataset_path data/pdb/
--max_seq_len 2000 --max_smi_len 200 --is_log 0 --problem_type 1
--num_windows 32 --smi_window_lengths 4 --seq_window_lengths 8
--batch_size 256 --num_epoch 100 --model C
--log_dir C:/Users/Snapp/AppData/Local/Temp/codex-dcgan-stage08-20260927-002551
```

The worker used the native `DataSet` and `parse_data(FLAGS)` with default `with_label=2`, then `np.asarray(XD_t)`. Measured drug corpus: **(50068, 200), float64**; parse time **3.109 s**. Configuration is PDBbind (`data/pdb/`), using its shipped GAN drug corpus, not an alternative corpus. No affinity split evaluation was performed. Unused `XT`, `Y`, and `XT_t` were released, as in the Stage 07B runner; `XD_t` was unchanged.

Only one call to `RE.ganForDrug(XD_t)` was made. The CLI DTA batch/epoch arguments above do not override the native GAN closure's hard-coded schedule:

| Effective native setting | Value |
|---|---|
| Iterations / batch / interval | 5000 / 5 / 500 |
| Latent dimension and sampling | 100; native normal sampling and native row sampling |
| Input scaling | `Xtrain / 32.5 - 1.0` |
| Real/fake labels | Ones / zeros, shape `(5, 1)` |
| Discriminator | Native Conv1D stack; final Dense tanh; binary cross-entropy; accuracy |
| Generator | Native Dense, Conv1DTranspose, BatchNormalization, tanh architecture |
| Generator training loss | Combined generator/discriminator binary cross-entropy against real labels |
| Optimizers | Separate native Adam instances for discriminator and combined model |
| Observed Adam configuration | learning rate 0.001, beta1 0.9, beta2 0.999, epsilon 1e-7, decay 0, amsgrad false |

The native code compiles the discriminator, then sets `dis_v.trainable=False` before compiling the combined model. This ordering was retained. The module's native seed calls (NumPy 1, Python random 1, TensorFlow compat.v1 0) were not replaced. These seeds alone do not establish bitwise reproducibility across framework/device environments.

## 5. External instrumentation and deviations

No native function was replaced, patched, shortened, or edited. An external Python trace callback recognized the native nested `train` code object. At its entry it captured generator/discriminator arrays and metadata; after each native generator training call it checked the existing real/fake discriminator losses and accuracies, aggregate discriminator loss/accuracy, and generator loss for finiteness. It recorded the original ten checkpoints without adding a loss definition. After the native call returned, it captured final arrays.

The fixed input was sampled once with **an independent `np.random.RandomState(808)`**, shape `(5, 100)`, and saved. Before/after inference used `gen_v(z_fixed, training=False)` on that same unchanged input. The initial probe was checked to leave every generator weight unchanged and leave NumPy's global and Python's RNG states unchanged. The generator's inspected inference path has no stochastic sampling layer. The probe did not reseed native training. No exhaustive TensorFlow internal-state equivalence test was performed.

Differences were calculated after casting saved arrays to float64. A tensor was counted as changed when at least one stored element differed exactly; there was **no semantic small-change threshold**. Trainable parameter variables were separated from BatchNormalization moving statistics using variable metadata, rather than treating all Keras weights as learned parameters. For the discriminator this does not imply its top-level `trainable` flag was true during the combined-model phase.

An external parent sampled Windows `GetProcessMemoryInfo` working-set size (RSS) approximately every 0.5 s. The worker wrote its own PID **21672**, distinct from launcher PID **15364**, and that actual worker PID was monitored. The monitor collected **1,133 samples**. It would terminate the launched process tree if RSS exceeded 2 GiB or elapsed time exceeded 600 s; this timer included startup/loading, making it stricter than a call-only budget. Observed non-finite loss/metric values would raise and stop the call. Exceptions/OOM would end the run; no retries were configured.

**Declared observation overhead:** tracing, numerical checks, weight snapshots, two inference probes, and external memory sampling add runtime/memory cost. Architecture, optimization, ordering, sampling, losses, labels, scaling, and iteration counts were unchanged. This is a native-path controlled validation, not an uninstrumented timing benchmark or paper reproduction. No causal attribution of differences from Stage 07B runtime/RSS is established.

## 6. Native checkpoint evidence

The table preserves native printed loss/accuracy precision; elapsed time is from call entry. Raw native output was cross-checked against observer events for all ten checkpoints. At every checkpoint the unrounded observed G loss was exactly **15.424947738647461**.

| Iteration | D loss (printed) | D accuracy (%) | G loss (printed) | Call elapsed (s) |
|---:|---:|---:|---:|---:|
| 500 | 0.000012 | 100.00 | 15.424948 | 44.203 |
| 1000 | 0.000001 | 100.00 | 15.424948 | 85.813 |
| 1500 | 0.000000 | 100.00 | 15.424948 | 128.860 |
| 2000 | 0.000000 | 100.00 | 15.424948 | 172.391 |
| 2500 | 0.000001 | 100.00 | 15.424948 | 218.360 |
| 3000 | 0.000000 | 100.00 | 15.424948 | 268.766 |
| 3500 | 0.000000 | 100.00 | 15.424948 | 319.203 |
| 4000 | 0.000000 | 100.00 | 15.424948 | 394.860 |
| 4500 | 0.000000 | 100.00 | 15.424948 | 481.469 |
| 5000 | 0.000000 | 100.00 | 15.424948 | 537.922 |

Printed zeros are not always numerical zeros: at iteration 1500 the observed D loss was `2.3245817715178418e-7`. Unrounded checkpoint values are in `events.jsonl`. All checked native training loss/metric values across **5,000 iterations** were finite; no observed NaN/Inf, exception, or OOM stopped the run. This does not assert that every internal activation/gradient was inspected.

## 7. Generator and discriminator differences

| Variable group | Total L2 difference | Maximum absolute difference | Changed tensors | Fraction |
|---|---:|---:|---:|---:|
| Generator trainable parameters | 9.714797289 | 0.122624308 | 12 / 12 | 100% |
| Generator BN moving state | 12.341732667 | 0.974319147 | 4 / 4 | 100% |
| Generator all stored weights/state | 15.706548048 | 0.974319147 | 16 / 16 | 100% |
| Discriminator parameters (all stored weights) | 4.381905928 | 0.212716162 | 12 / 12 | 100% |

All differences were finite. Independent offline recalculation from the saved before/after arrays reproduced the aggregate trainable-parameter differences. A changed-tensor fraction is not a changed-scalar fraction.

| Model / layer / group | L2 difference | Max absolute difference | Changed tensors |
|---|---:|---:|---:|
| G dense_1 parameters | 6.764633077 | 0.082120754 | 2 / 2 |
| G conv1d_transpose parameters | 5.834961935 | 0.085246913 | 2 / 2 |
| G batch_normalization parameters | 0.421476743 | 0.081225224 | 2 / 2 |
| G batch_normalization moving state | 10.981206243 | 0.974319147 | 2 / 2 |
| G conv1d_transpose_1 parameters | 3.692614556 | 0.092829056 | 2 / 2 |
| G batch_normalization_1 parameters | 0.620888137 | 0.122624308 | 2 / 2 |
| G batch_normalization_1 moving state | 5.633069739 | 0.842356011 | 2 / 2 |
| G conv1d_transpose_2 parameters | 0.609670346 | 0.116048902 | 2 / 2 |
| D conv1d parameters | 0.154371608 | 0.082631797 | 2 / 2 |
| D conv1d_1 parameters | 0.410142311 | 0.105592117 | 2 / 2 |
| D conv1d_2 parameters | 0.885433250 | 0.167300962 | 2 / 2 |
| D conv1d_3 parameters | 1.512668784 | 0.212716162 | 2 / 2 |
| D conv1d_4 parameters | 2.553359264 | 0.191835895 | 2 / 2 |
| D dense parameters | 3.068753792 | 0.205331709 | 2 / 2 |

## 8. Fixed generator output

For the identical saved latent batch, both outputs had shape `(5, 200, 1)`:

- Mean absolute output difference: **0.9053478417888382**.
- Maximum absolute output difference: **1.3043068051338196**.
- Before output range: **[-0.32616102695, 0.16589926183]**.
- After output range: **[-1.0, 1.0]**.
- Output differences were finite; offline recalculation reproduced the mean difference.

The output comparison demonstrates change on this fixed batch. Both trainable parameters and BatchNormalization state changed, so this probe does not isolate their separate contributions. Endpoint extrema do not measure how frequently outputs saturate or establish distributional quality/diversity.

## 9. Worker memory, runtime, and stopping condition

RSS below is the nearest periodic sample to each event, not an assertion of perfectly synchronized memory measurement. Maximum absolute sample/event offset in this table was **0.198 s**. MiB uses 2^20 bytes.

| Event | Worker elapsed (s) | Worker RSS (MiB) |
|---|---:|---:|
| Native call entry | 28.828 | 434.832 |
| Training-closure start / initial probe completed | 29.469 | 444.621 |
| Iteration 500 | 73.031 | 712.227 |
| Iteration 1000 | 114.641 | 855.254 |
| Iteration 1500 | 157.703 | 982.625 |
| Iteration 2000 | 201.219 | 1108.664 |
| Iteration 2500 | 247.188 | 1248.918 |
| Iteration 3000 | 297.594 | 1375.090 |
| Iteration 3500 | 348.031 | 1500.563 |
| Iteration 4000 | 423.688 | 1624.609 |
| Iteration 4500 | 510.297 | 1764.832 |
| Iteration 5000 | 566.750 | 1891.234 |
| Worker evidence capture finished | 566.781 | 1891.234 |

The first startup sample was **32.074 MiB**, before framework/data loading. Maximum observed RSS was **1891.234 MiB (1.847 GiB)**. The last sample during process teardown was **1739.023 MiB**; the training-end sample above is the relevant ending RSS for the native run. The training trajectory was **gradually increasing**, with a larger initial increase. This measurement does not establish a memory leak or its cause.

Native call wall time: **537.922 s** (including in-call observer overhead). Parent launch-to-exit time: **568.546 s**. Stop condition: **normal native completion**, returning an 8-layer `Sequential` discriminator; worker exit code **0**. Neither the 600 s budget nor 2 GiB RSS limit was reached. Exactly one native call was run, without a second seed or retry.

Stage 07B's reported 404.29 s call and limited RSS observations are historical context. Instrumentation and uncontrolled host/runtime variability mean this run does not isolate a performance regression or establish uninstrumented memory cost.

## 10. Interpretation, unresolved questions, and limits

**Case A is supported:** constant G loss at the ten checkpoints coexisted with changed generator trainable parameters and changed fixed-input outputs. The generator was not unchanged between the measured endpoints. Discriminator parameters also changed, providing the requested reference measurement.

These endpoint measurements do not locate when parameter updates occurred. They cannot establish continued useful updates throughout all 5,000 iterations, exclude later stagnation, explain the constant loss mechanism, or establish healthy adversarial balance. No per-step gradients, intermediate weight snapshots, output-distribution analysis, or downstream affinity performance were measured. Those remain unresolved; no additional experiment is authorized by this report.

This task establishes neither paper-faithful reproduction nor convergence, useful molecular representations, generalization, baseline superiority, GPU feasibility, target-GAN feasibility, or full-grid readiness. No claim that the paper is wrong or the implementation is broken follows from these results. Native binary-cross-entropy/tanh behavior was preserved rather than repaired or substituted.

Not run: Co-VAE, target GANs, full DTA construction/training, cross-validation, grids, Kaggle/cloud, or thesis novelty. No Stage 08 summary was created, and no later thesis stage was started.

## 11. Evidence locations

All external observation files are retained under:

```text
C:/Users/Snapp/AppData/Local/Temp/codex-dcgan-stage08-20260927-002551/
```

- `worker.py`, `monitor.py`: exact external runner and safety monitor.
- `native.log`: unmodified native stdout/stderr, including ten printed checkpoints.
- `events.jsonl`, `rss.jsonl`, `worker.json`, `monitor.json`: timestamps, actual PID, memory trajectory, termination metadata.
- `environment.json`, `configuration.json`, `optimizers.json`: measured environment and effective configuration.
- `gen_v_before.npz`, `gen_v_after.npz`, `dis_v_before.npz`, `dis_v_after.npz`, `weights_metadata.json`: before/after variable evidence.
- `z_fixed.npy`, `output_before.npy`, `output_after.npy`: fixed-input/output evidence.
- `results.json`, `verify.py`, `verification.json`: computed differences and independent offline checks.
- `protected-before.json`, `data-before.json`, `integrity.json`, `host.json`: preservation and host metadata.

Runner SHA-256: `73DADECE3E82CF2E5D0F2FB2370113156C906FD8F37416C4D3FB8C3916182D48`.
Monitor SHA-256: `6DC073ECD2E216C04977A3058A4A3008151667BB4C79CEA940E59D67DE304D27`.

These artifacts are outside version control in a temporary-directory location; their retention is not guaranteed against later operating-system cleanup. They were not deleted or relocated in this task. No commit was made.

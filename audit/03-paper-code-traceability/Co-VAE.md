# Phase 3 — Co-VAE: Paper ↔ Code Traceability Audit

**Paper-side specification:** `audit/02-paper-forensic/Co-VAE.md` (Phase 2)
**Implementation:** `Co-VAE/CoVAE/` (upstream `github.com/LiminLi-xjtu/CoVAE`, branch `master`, HEAD `2c17268`)
**Mode:** read-only static tracing plus lightweight shape/data inspection. No training run, no full reproduction, no file modified.

Tags: `[PAPER]` `[CODE]` `[VERIFIED]` `[INFERRED]` `[OPEN]`.
`[VERIFIED]` marks facts confirmed by executing the repository's own data-loading code or by direct inspection of the bundled data files.

---

## 1. Executive Summary

**Overall traceability verdict: 🔴 RED**

The repository implements a genuine co-regularized two-VAE architecture whose skeleton follows the paper — two encoders, two decoders, a joint regression head, and a live three-term objective. But the runtime dataset does not match the published dataset, the evaluation protocol differs from the published protocol in ways that affect every reported number, and several architectural details contradict both the prose and Fig. 4.

**Most important implementation discrepancies**

1. **The KIBA matrix the code builds is 1954 × 217 with 109,296 observed pairs — not the paper's 2111 × 229.** [VERIFIED] The bundled `ligands_iso.txt`/`proteins.txt` already contain exactly 2111 and 229 entries, so the paper's published filtering happened offline; the code then applies a *second*, undocumented **length** filter (`get_removelist(ligands, 90)`, `get_removelist(proteins, 1365)`) that removes a further 157 drugs and 12 targets.
2. **The paper's stated KIBA affinity filter is not applied anywhere.** The runtime KIBA matrix still spans `[0.0000, 17.2002]` [VERIFIED], contradicting §3.1's "We removed the drug-target pairs which have known affinities less than 10" and Fig. 5's histogram, which starts at ≈9.5.
3. **Hyperparameter selection and final evaluation both run on the test fold.** `nfold_setting_sample` builds `val_sets` (`run_experiments.py:176-181`) and then passes **`test_sets`** to both `general_nfold_cv` and `general_nfold_cv_test` (`:188-195`). `val_sets` is never consumed. Early stopping, checkpoint saving and epoch selection all key off that same test fold.
4. **One repetition, not ten.** `for i in range(1)` with `random.seed(i+1000)` (`:508-509`) — the paper reports means ± std over ten random splits; the code performs a single split, and the reported std is over the 5 CV folds, which all share one test set.
5. **MAE is never computed.** The paper's Table 2 reports MAE for every method and setting; no MAE implementation exists in the repository. [VERIFIED]
6. **Pooling is average, not max.** `nn.AdaptiveAvgPool1d(1)` (`model.py:24`) against the paper's "max-pooling layer" in §2.5 and Fig. 4.
7. **ε is sampled from N(0, 0.1²), not N(0, I).** `torch.cuda.FloatTensor(std.size()).normal_(0, 0.1)` (`model.py:36`) — PyTorch's second argument is the standard deviation.
8. **No baselines and no drug-generation experiment exist.** No KronRLS, DeepDTA, DeepAffinity, GraphDTA, GAN, SeqGAN, AAE or VAE code; no RDKit, no SMILES decoding, no validity/uniqueness. [VERIFIED]

**Major execution blockers.** The code cannot run on Python ≥ 3.11 (`random.sample` on a `set`), cannot run without `--lamda` supplied explicitly, and cannot run without CUDA. All three are confirmed below.

**One Phase-2 ambiguity is resolved in the paper's favour:** λ *is* used as an exponent — `10**lamda` (`run_experiments.py:136`) — so Table 1's `[-3,-5]` are exponents and the apparent sign paradox in Phase 2's A1 does not materialize at runtime.

---

## 2. Repository and Entry Points

| Role | File / symbol | Evidence |
|---|---|---|
| Main & training entry point | `run_experiments.py:540` `if __name__ == "__main__"` → `experiment(FLAGS)` | [CODE] |
| Argument parsing | `arguments.py:5` `argparser()`; `arguments.py:98` `logging()` | [CODE] |
| Model definitions | `model.py` — `CNN` (`:6`, encoder), `decoder` (`:58`), `net_reg` (`:84`), `net` (`:116`) | [CODE] |
| Dataset loading / preprocessing | `datahelper.py:122` `DataSet`, `:137` `parse_data`; helpers `get_removelist` (`:88`), `list_remove` (`:97`), `df_remove` (`:106`), `label_smiles` (`:71`), `label_sequence` (`:79`) | [CODE] |
| Objective | `run_experiments.py:91` `loss_f` (recon + KL); `:136` the combined loss line | [CODE] |
| Metrics | `emetrics.py` — `get_cindex` (`:4`), `get_rm2` (`:54`), `get_aupr` (`:61`); MSE via `nn.MSELoss`; AUC via `sklearn.roc_auc_score` | [CODE] |
| Fold generation | `run_experiments.py:30` `get_random_folds`, `:55` `get_drugwise_folds`, `:73` `get_targetwise_folds` | [CODE] |
| Split assembly | `run_experiments.py:165` `nfold_setting_sample` | [CODE] |
| Hyperparameter search | `run_experiments.py:235` `general_nfold_cv` | [CODE] |
| Final evaluation | `run_experiments.py:326` `general_nfold_cv_test` | [CODE] |
| Checkpointing | `torch.save(model, 'checkpoint.pth')` at `:289` and `:376`; `torch.load('checkpoint.pth')` at `:382` | [CODE] |
| Result output | `np.savetxt("./result/iter{i}affinities.txt")` and `…preaffinities.txt` (`:418-419`); text log via `logging()` | [CODE] |
| Bundled outputs | `result/`, `figures/`, `logs/` each contain only a 2-byte `readme` placeholder — **no results ship with the repo** | [VERIFIED] |

**Execution flow**

```
argparser()  →  experiment()
   → DataSet.parse_data()            load ligands_iso.txt / proteins.txt / affinity txt
                                     (KIBA only) length-filter drugs & targets
                                     (Davis only) pKd = −log10(Kd/1e9)
                                     label-encode + truncate/pad to max_smi_len / max_seq_len
   → np.where(~isnan(Y))             observed (row, col) pairs
   → get_drugwise_folds / get_targetwise_folds / get_random_folds   (6 folds, seed 1000)
   → nfold_setting_sample()          nfolds[5] = test ; nfolds[0:5] = train
                                     builds val_sets  (never used)
                                     builds test_sets (test fold × 5)
        → general_nfold_cv(train_sets, test_sets)        grid search over p1×p2×p3×λ
        → general_nfold_cv_test(train_sets, test_sets)   retrain at best params
   → metrics logged to logs/<timestamp>/log.txt ; affinities to result/
```
[CODE] `run_experiments.py:470-538`

---

## 3. Paper ↔ Code Traceability Matrix

| # | Area | Paper claim | Code implementation | Status | Evidence |
|---|---|---|---|---|---|
| 1 | **Dataset — Davis** | 68 drugs × 442 targets, Kd | Runtime `XD (68,85)`, `XT (442,1200)`, `Y (68,442)`, 30,056 observed | **MATCH** | [PAPER] §3.1; [VERIFIED] `parse_data` executed |
| 2 | **Dataset — KIBA** | 2111 drugs × 229 targets, density 24.4% (≈117,954 pairs) | Runtime `XD (1954,100)`, `XT (217,1000)`, `Y (1954,217)`, **109,296** observed | **MISMATCH** | [PAPER] §3.1; [VERIFIED] |
| 3 | **Preprocessing — Davis pKd** | `pKd = −log₁₀(Kd/10⁹)` | `affinities = -(np.log10(affinities / 10**9))`, `datahelper.py:148`. Runtime `Y` range **5.0000 – 10.7959** | **MATCH** | [PAPER] §3.1; [CODE]; [VERIFIED] |
| 4 | **Preprocessing — KIBA filter** | "removed the drug-target pairs which have known affinities less than 10" | No affinity filter exists. Only `get_removelist(ligands, 90)` / `(proteins, 1365)` — a **string-length** filter. Runtime `Y` range **0.0000 – 17.2002** | **MISMATCH** | [PAPER] §3.1; [CODE] `datahelper.py:154-159`; [VERIFIED] |
| 5 | **Preprocessing — Davis filter** | SMILES < 85, sequences < 1200 | **No filter on the Davis branch at all**; only truncation via `line[:MAX_LEN]` | **PARTIAL** | [PAPER] §3.1; [CODE] `datahelper.py:146-149`, `:71-85` |
| 6 | **Input encoding — vocabularies** | 64 SMILES chars, 25 protein chars | `CHARISOSMILEN = 64`, `CHARPROTLEN = 25` | **MATCH** | [PAPER] §3.1; [VERIFIED] |
| 7 | **Input encoding — embedding** | dim 128 | `nn.Embedding(charsmiset_size, 128)`, `nn.Embedding(charseqset_size, 128)`; `Conv1d(in_channels=128, …)` | **MATCH** | [PAPER] §3.3; [CODE] `model.py:119-120`, `:10` |
| 8 | **Input encoding — lengths** | Davis (85,128)/(1200,128); KIBA (100,128)/(1000,128) | CLI `--max_smi_len` default **100**, `--max_seq_len` default **1000** (the KIBA values). No per-dataset defaults; Davis requires explicit flags | **PARTIAL** | [PAPER] §3.3; [CODE] `arguments.py:27-38` |
| 9 | **Encoder — GatedCNN layers** | 3 gated 1D-conv layers, filters 32×1/32×2/32×3 | 3 `Conv1d` with `out_channels` = `2·nf`, `4·nf`, `6·nf`, halved by gating → effective **nf, 2nf, 3nf** | **MATCH** | [PAPER] §2.5, Table 1; [CODE] `model.py:9-22, 41-49` |
| 10 | **GatedCNN gating** | `A ⊙ sigmoid(B)` on a channel split | `out, gate = x.split(int(x.size(1)/2), 1)` then `x = out * torch.sigmoid(gate)` | **MATCH** | [PAPER] §2.4, Fig. 3; [CODE] `model.py:42-43` |
| 11 | **Encoder — ReLU count** | "two ReLU"/"a ReLU after each conv layer" (Phase 2 **A4**) | **Zero** ReLUs after the conv layers; ReLU appears only inside the μ/log-var heads (`layer1`, `layer2`) and in `net_reg` | **MISMATCH** | [PAPER] §2.5; [CODE] `model.py:9-22, 25-32` |
| 12 | **Encoder — pooling** | max-pooling | `nn.AdaptiveAvgPool1d(1)` — **average** pooling | **MISMATCH** | [PAPER] §2.5, Fig. 4; [CODE] `model.py:24` |
| 13 | **Encoder — FC after pooling** (Phase 2 **A5**) | Fig. 4 shows an FC; §2.5 omits it | Two parallel `Linear(nf·3, nf·3)+ReLU` heads produce μ and log-var directly from the pooled vector | **PARTIAL** | [FIGURE] Fig. 4; [CODE] `model.py:25-32, 50-53` |
| 14 | **Encoder — dropout** | Dropout described only in the Reg block, rate 0.2 | `nn.Dropout(0.2)` inside `conv2` and `conv3` of the **encoder** | **MISMATCH** (location) | [PAPER] §2.5, Table 1; [CODE] `model.py:14, 19` |
| 15 | **Latent dimension** | Never stated (Phase 2 `[OPEN]`) | `num_filters * 3` → **96** when `num_filters = 32` | **RESOLVED by code** | [CODE] `model.py:26, 30` |
| 16 | **Reparameterization** | `z = μ + σ⊙ε`, `ε ~ N(0, I)` | `std = logvar.mul(0.5).exp_()`; `eps = …normal_(0, 0.1)`; `return eps.mul(std).add_(mean)` → `z = μ + σ⊙ε` with **ε ~ N(0, 0.1²)** | **PARTIAL** | [PAPER] §2.3; [CODE] `model.py:34-38` |
| 17 | **Second encoder head semantics** | Paper writes **σ** | Code treats it as **log-variance** (`logvar.mul(0.5).exp_()`; `logvar.exp()` in the KL). Self-consistent with Eq. (10) if σ is read as log σ² | **PARTIAL** | [PAPER] §2.3; [CODE] `model.py:35`, `run_experiments.py:95` |
| 18 | **Drug decoder** | FC + 3 deconv + FC; filters 32×3/32×2/32×1 | `Linear(nf·3, nf·3·(init_dim−3(k−1)))+ReLU` → `ConvTranspose1d(nf·3→nf·2)`, `(nf·2→nf)`, `(nf→128)`, each +ReLU → `Linear(128, vocab)` | **PARTIAL** | [PAPER] §2.5, Table 1; [CODE] `model.py:58-81` |
| 19 | **Target decoder** | Same structure | Identical class, instantiated with `max_seq_len`, `FILTER_LENGTH2`, `charseqset_size` | **MATCH** | [PAPER] §2.5; [CODE] `model.py:125` |
| 20 | **Reg block topology** (Phase 2 **A6**) | Prose ≈3 FC; Fig. 4 shows 5 FC (2 per branch + 1 merged) | **5 Linear layers**, but arranged 1 per branch + 3 merged: `reg1`(96→96), `reg2`(96→96), then `Linear(192→1024) → Linear(1024→512) → Linear(512→1)` | **PARTIAL** | [PAPER] §2.5; [FIGURE] Fig. 4; [CODE] `model.py:84-113` |
| 21 | **Reg block merge** | "connect them by another FC layer" | `torch.cat((A, B), 1)` then the 3-layer MLP | **MATCH** | [PAPER] §2.5; [CODE] `model.py:111-112` |
| 22 | **Reg block dropout** | 0.2 (Table 1) | `nn.Dropout(0.1)` ×2 | **MISMATCH** | [TABLE] Table 1; [CODE] `model.py:90, 93` |
| 23 | **Reconstruction loss** | Categorical, `x^T log x̂` | `nn.CrossEntropyLoss(reduction='none')` over `recon_x.permute(0,2,1)` vs integer labels, summed over positions | **MATCH** | [EQUATION] Eq. (10); [CODE] `run_experiments.py:93-94` |
| 24 | **KL terms** | Analytic Gaussian, coefficient 1, both branches | `KLD = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)`, added inside `loss_f`, applied to both branches | **MATCH** (form) | [EQUATION] Eq. (10); [CODE] `run_experiments.py:95` |
| 25 | **Co-regularization term** | `−λ Σ(y−ŷ)²`, weight λ on the affinity term | `loss_affinity = nn.MSELoss()(pre_affinity, affinity)` at weight **1**; the VAE terms carry `10**lamda` instead | **PARTIAL** | [EQUATION] Eqs. (9)–(10); [CODE] `run_experiments.py:131, 136` |
| 26 | **λ interpretation** | `λ ∈ {−3, −5}`, sign ambiguous (Phase 2 **A1**) | `10**lamda` → **1e−3 or 1e−5**, positive; applied to the VAE terms | **PARTIAL** (see §4.4) | [TABLE] Table 1; [CODE] `run_experiments.py:136` |
| 27 | **Extra loss factor** | Not in Eq. (10) | `FLAGS.max_smi_len / FLAGS.max_seq_len` multiplies the **target** VAE loss (0.1 at defaults) | **Code-only** | [CODE] `run_experiments.py:136` |
| 28 | **Optimizer** | Adam, lr 0.001 | `optim.Adam(model.parameters())` — **no `lr` argument**; PyTorch default lr is 1e-3 | **MATCH** (by default) | [PAPER] §3.3; [CODE] `run_experiments.py:118` |
| 29 | **Batch size** | 256 | `--batch_size` default **256** | **MATCH** | [TABLE] Table 1; [CODE] `arguments.py:46-51` |
| 30 | **Epochs** | 100 | `--num_epoch` default **100** | **MATCH** | [TABLE] Table 1; [CODE] `arguments.py:40-45` |
| 31 | **Filter-length grid** | drug {5,7}, target {7,11} | `--smi_window_lengths` / `--seq_window_lengths`, `nargs='+'`, **no defaults** (`None`) | **PARTIAL** | [TABLE] Table 1; [CODE] `arguments.py:8-25` |
| 32 | **Number of filters** | 32×1/32×2/32×3 fixed | `--num_windows`, `nargs='+'`, **no default**; swept as a grid axis | **PARTIAL** | [TABLE] Table 1; [CODE] `arguments.py:20-25`, `run_experiments.py:239` |
| 33 | **Fold count (outer)** | six folds, five train + one test | `experiment(FLAGS, foldcount=6)`; `test_set = nfolds[5]`, `outer_train_sets = nfolds[0:5]` | **MATCH** | [PAPER] §3.3; [CODE] `run_experiments.py:470, 166-167` |
| 34 | **Fold generation** | Random, entity-wise | `get_drugwise_folds` (`problem_type=2`, default) / `get_targetwise_folds` (`=3`) / `get_random_folds` (`=1`, pair-wise) | **MATCH** | [PAPER] §3.3; [CODE] `run_experiments.py:508-514` |
| 35 | **Bundled fold files** | not mentioned | `data/*/folds/*.txt` ship but the loading call is **commented out** (`:168`) and `DataSet` has no `read_sets` method | **Code-only artifact** | [CODE]; [VERIFIED] |
| 36 | **Repetitions** | ten random splits, mean ± std | `for i in range(1)` — **one** split, `random.seed(1000)` | **MISMATCH** | [PAPER] §3.3; [CODE] `run_experiments.py:508-509` |
| 37 | **Validation set** (Phase 2 **A2**) | Fig. 6 shows validation (black); prose omits it | `val_sets` built at `:176-181` and **never used**; `test_sets` passed in its place | **MISMATCH** | [FIGURE] Fig. 6; [CODE] `run_experiments.py:188-195` |
| 38 | **Hyperparameter selection** | five-fold CV "based on only training data" | `general_nfold_cv` receives `test_sets` as its evaluation set | **MISMATCH** | [PAPER] §3.3; [CODE] `run_experiments.py:188-191` |
| 39 | **CI** | Eq. in §3.2 | `emetrics.get_cindex` — vectorized global pairwise CI over the whole evaluation array | **MATCH** | [PAPER] §3.2; [CODE] `emetrics.py:4-18` |
| 40 | **MSE** | `Σ(y−ŷ)²` as printed (Phase 2 **A3**) | `nn.MSELoss()` — the **mean**, not the sum | **MATCH with Table 2** | [PAPER] §3.2; [CODE] `run_experiments.py:117, 380` |
| 41 | **MAE** | Reported in Table 2 for all methods | **No implementation anywhere** | **MISSING** | [TABLE] Table 2; [VERIFIED] |
| 42 | **r²m** | `r²(1 − √(r²−r₀²))` | `r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))` — adds `np.absolute` | **MATCH** | [PAPER] §3.2; [CODE] `emetrics.py:54-58` |
| 43 | **AUC threshold sweep** | KIBA only, sweep 10.5→12.5 step 0.1 | **No sweep.** Single hardcoded threshold per call | **MISMATCH** | [PAPER] §3.2; [CODE] `run_experiments.py:163, 394, 399` |
| 44 | **AUC on Davis** | "We only use AUC for the KIBA dataset" | Davis AUC **is** computed, at threshold 7.0 | **MISMATCH** | [PAPER] §3.2; [CODE] `run_experiments.py:392-397` |
| 45 | **Epoch-loop threshold** | n/a | `test()` hardcodes `affinities > 7` for **both** datasets, every epoch | **Code-only inconsistency** | [CODE] `run_experiments.py:163` |
| 46 | **AUPR** | Not among the five paper metrics | `get_aupr` computed and logged (trapezoidal `auc(recall, precision)`) | **Code-only extra** | [CODE] `emetrics.py:61-64`, `run_experiments.py:396` |
| 47 | **Checkpointing** | Not described | `torch.save(model, 'checkpoint.pth')` — fixed filename in CWD, written in both search and final passes, reloaded at `:382` | **Code-only** | [CODE] `run_experiments.py:289, 376, 382` |
| 48 | **`--checkpoint_path`** | n/a | Declared (`arguments.py:71-76`), **never referenced** | **Dead flag** | [VERIFIED] |
| 49 | **Baselines** | KronRLS, DeepDTA, DeepAffinity, GraphDTA, all rerun by the authors | **No implementation, no results import** | **MISSING** | [PAPER] §3.3; [VERIFIED] |
| 50 | **Generative baselines** | GAN, SeqGAN, AAE, VAE (Table 3) | **No implementation** | **MISSING** | [TABLE] Table 3; [VERIFIED] |
| 51 | **Drug generation** | Validity 54.53%, Uniqueness 3.47%; RDKit validation | **No generation, no RDKit, no SMILES decoding, no validity/uniqueness** | **MISSING** | [PAPER] §3.2, §3.4; [VERIFIED] |
| 52 | **SARS-CoV-2 case study** | Top-5 drugs per gene, results in the repo | No prediction script; `result/` holds only a `readme` placeholder | **MISSING** | [PAPER] §3.4; [VERIFIED] |
| 53 | **Results output** | Table 2 / Table 3 / Figs. 7–10 | Text log + two `.txt` affinity dumps. No plotting (`plotLoss` is entirely commented out, `:426-455`) | **PARTIAL** | [CODE] |
| 54 | **Framework** | PyTorch | PyTorch | **MATCH** | [PAPER] §3.3; [CODE] `run_experiments.py:4-9` |

---

## 4. Objective and Variational Formulation

### 4.1 Reconstruction terms

Both are computed and both enter the backward pass. [VERIFIED by tracing `loss.backward()` at `run_experiments.py:137`]

```python
# run_experiments.py:91-96
def loss_f(recon_x, x, mu, logvar):
    cit = nn.CrossEntropyLoss(reduction='none')
    cr_loss = torch.sum(cit(recon_x.permute(0, 2, 1), x), 1)   # sum over sequence positions
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    return torch.mean(cr_loss + KLD)                            # mean over the batch
```

`recon_x` is `(B, L, V)` from the decoder's final `Linear(128, vocab)`; `permute(0,2,1)` gives `(B, V, L)`, matched against integer class labels `x` of shape `(B, L)`. This is exactly the categorical `x^T log x̂` of Eq. (10). **MATCH.** [CODE] `model.py:73, 80`

### 4.2 KL terms

Drug KL and target KL are both present, both inside `loss_f`, both summed with the reconstruction term before `torch.mean`, and both reach `loss.backward()`. Coefficient relative to the reconstruction term is **1**, as in Eq. (10). [CODE] `run_experiments.py:95, 132-137`

However, the KL is **not** applied to a coefficient of 1 in the *total* loss: the whole `loss_f` output (reconstruction **and** KL together) is multiplied by `10**lamda` (§4.4). [CODE] `run_experiments.py:136`

A second, verifiable constraint: both `layer1` and `layer2` terminate in `nn.ReLU()` (`model.py:27, 31`). Therefore `mu ≥ 0` and `logvar ≥ 0` always, i.e. the posterior mean is confined to the non-negative orthant and the posterior variance is confined to `≥ 1`. Neither constraint appears in the paper. [CODE] / [VERIFIED]

### 4.3 Affinity term

```python
# run_experiments.py:117, 131
loss_func = nn.MSELoss()
loss_affinity = loss_func(pre_affinity, affinity)
```
Mean squared error over the batch, weight **1** in the total loss. Eq. (9)'s Normal-likelihood constant `C` is dropped, as expected. [CODE]

### 4.4 λ

**The code exponentiates.** [VERIFIED]

```python
# run_experiments.py:136
loss = loss_affinity + 10**lamda * (loss_drug + FLAGS.max_smi_len / FLAGS.max_seq_len * loss_target)
```

With `--lamda -3 -5`, the coefficient is `10**(-3) = 0.001` or `10**(-5) = 0.00001` — **positive**.

Three findings, stated separately:

1. **Phase 2's A1 sign paradox does not materialize.** Table 1's `[-3,-5]` are exponents, and the runtime coefficient is positive. The paper never writes `10^λ`, so this is resolvable only from code.
2. **The coefficient is attached to the opposite term.** Eq. (9)/(10) put λ on the *affinity* term with the VAE terms at weight 1; the code puts `10^λ` on the *VAE* terms with the affinity term at weight 1. Writing the paper's maximization as a minimization and dividing through by λ gives `MSE + (1/λ)·VAE_loss`, so `10^λ_code` occupies the position of `1/λ_paper` — implying `λ_paper ∈ {10³, 10⁵}`. The two forms share the same minimizer up to a global loss scale, which Adam largely absorbs; this is therefore a **formulation** discrepancy whose behavioural effect is bounded, not a sign error. [INFERRED] for the equivalence argument; [CODE] for the line itself.
3. **An undocumented factor multiplies the target VAE loss.** `FLAGS.max_smi_len / FLAGS.max_seq_len` = 0.1 at the CLI defaults (100/1000), or 85/1200 ≈ 0.0708 for Davis. Eq. (10) weights the two VAE branches equally. [CODE]

### 4.5 Monte-Carlo terms

`L_d = L_t = 1`. A single `reparametrize` call per branch per forward pass (`model.py:54`), one forward pass per batch. There is no sampling loop, so the `Σ_{l_d}Σ_{l_t}` double sum of Eq. (10) reduces to one term and the `1/L_d`, `1/L_t` factors are unity. Phase 2's A13 (asymmetric normalization) is therefore **moot at runtime**. [CODE] / [INFERRED]

### 4.6 Reparameterization

```python
# model.py:34-38
def reparametrize(self, mean, logvar):
    std = logvar.mul(0.5).exp_()                                   # std = exp(logvar/2)  ✓
    eps = torch.cuda.FloatTensor(std.size()).normal_(0, 0.1)       # ε ~ N(0, 0.1²)  ✗
    eps = Variable(eps)
    return eps.mul(std).add_(mean)                                 # z = μ + σ⊙ε     ✓
```

- **Structure matches** `z = μ + σ⊙ε`. [MATCH]
- **The second encoder head is log-variance**, not σ. `exp(logvar/2)` is the standard deviation and `logvar.exp()` is used in the KL, so the code is internally consistent; the paper's §2.3 calls this output σ. Under that reading Eq. (10)'s `log((σ)²)` ≡ `logvar` and `(σ)²` ≡ `exp(logvar)`, so the KL formula agrees. [PARTIAL]
- **ε is drawn from N(0, 0.1²), not N(0, I).** PyTorch's `Tensor.normal_(mean, std)` takes a standard deviation, so `normal_(0, 0.1)` yields variance 0.01. The KL term in `loss_f` is the closed form for `KL(N(μ,σ²) ‖ N(0,I))`, which assumes unit-variance sampling noise. The sampling step and the regularizer therefore assume different noise scales. [CODE] / [VERIFIED]
- `.exp_()` is an in-place op on `logvar`; `logvar` is subsequently returned from `CNN.forward` as `output2` and consumed by `loss_f`. Whether the in-place mutation affects the value reaching the KL term is **not determinable statically** — `reparametrize` is called at `model.py:54` *after* `output2` is bound, and autograd may or may not raise depending on version. **[OPEN]** — a Phase 4 target.

**Reconstructed runtime objective (minimized):**

```
L_code = MSE(ŷ, y)
       + 10^λ · [  mean_B( Σ_pos CE(x̂_d, x_d) + KL(q_d ‖ N(0,I)) )
                 + (max_smi_len/max_seq_len) · mean_B( Σ_pos CE(x̂_t, x_t) + KL(q_t ‖ N(0,I)) ) ]
```
[CODE] `run_experiments.py:131-136`

---

## 5. Architecture Traceability

### Encoders (`model.py:6-55`)

| Item | Code | vs Paper |
|---|---|---|
| Label encoding | `label_smiles` / `label_sequence`, `datahelper.py:71-85` | MATCH |
| Embedding | `nn.Embedding(vocab, 128)`; `permute(0,2,1)` → `(B,128,L)` | MATCH (dim 128) |
| Conv layers | 3 × `Conv1d`, `stride=1`, `padding=k//2` | MATCH (count) |
| Raw out_channels | `2·nf`, `4·nf`, `6·nf` | — |
| Post-gate channels | `nf`, `2·nf`, `3·nf` → 32, 64, 96 at `nf=32` | MATCH Table 1 |
| Filter sizes | `k_size` = `FILTER_LENGTH1` (drug) / `FILTER_LENGTH2` (target), from the CLI | MATCH (grid) |
| Activations after conv | **none** | MISMATCH (§2.5 claims ReLU after each) |
| Dropout | `0.2` before conv2 and conv3 | Code-only location |
| Pooling | `nn.AdaptiveAvgPool1d(1)` → **average** | MISMATCH (max claimed) |
| FC after pooling | `layer1`, `layer2`: `Linear(96,96)+ReLU` each, producing μ and log-var | PARTIAL vs Fig. 4 |
| Latent dim | **96** (`nf·3`, `nf=32`) | Resolves Phase 2 `[OPEN]` |

### GatedCNN (`model.py:41-49`)

```python
x = self.conv1(x)
out, gate = x.split(int(x.size(1) / 2), 1)   # channel-dim split into two equal halves
x = out * torch.sigmoid(gate)                 # A ⊙ sigmoid(B)
```
Implements `A ⊙ σ(B)` exactly as Fig. 3 specifies, with the split on the channel axis. **MATCH.** [CODE]

**Phase 2 A4 resolved from code:** neither 2 nor 3 — there are **zero** ReLU activations following the convolutions. The only nonlinearities in the encoder body are the three sigmoid gates. [VERIFIED]

### Decoders (`model.py:58-81`)

```
Linear(96, 96·(init_dim − 3(k−1))) + ReLU
view(-1, 96, init_dim − 3(k−1))
ConvTranspose1d(96 → 64, k, stride 1, pad 0) + ReLU
ConvTranspose1d(64 → 32, k, stride 1, pad 0) + ReLU
ConvTranspose1d(32 → 128, k, stride 1, pad 0) + ReLU
permute(0, 2, 1)
Linear(128, vocab_size)
```
Length arithmetic: each `ConvTranspose1d` with stride 1 and padding 0 adds `k−1`, so `init_dim − 3(k−1) + 3(k−1) = init_dim`. Output is `(B, init_dim, vocab)`. **Structurally MATCH** the paper's "FC + 3 deconv + FC". Filter counts 96→64→32 match Table 1's `32*3;32*2;32*1` for the first two transitions; the third layer outputs **128** (the embedding width), which Table 1 does not describe. [CODE] / [INFERRED]

### Regression block (`model.py:84-113`)

**Neither the 3-FC prose reading nor the Fig. 4 topology.** Five `Linear` layers arranged as **1 per branch + 3 merged**, where Fig. 4 shows **2 per branch + 1 merged**:

| Layer | Shape (nf = 32) |
|---|---|
| `reg1` | `Linear(96, 96)` + ReLU |
| `reg2` | `Linear(96, 96)` + ReLU |
| `reg[0]` | `Linear(192, 1024)` + ReLU |
| `reg[3]` | `Linear(1024, 512)` + ReLU |
| `reg[6]` | `Linear(512, 1)` |

Dropout `0.1` after the 1024 and 512 activations. Merge is `torch.cat((A, B), 1)` → 192. **Resolves Phase 2's `[OPEN]` on FC widths.** [CODE] / [VERIFIED]

---

## 6. Dataset and Preprocessing Audit

All figures below were obtained by executing the repository's own `DataSet.parse_data` and inspecting the bundled data files. [VERIFIED]

### Davis

| Item | Value |
|---|---|
| Drugs / targets | **68 / 442** — matches the paper |
| Affinity matrix | `(68, 442)`, **0 NaN**, 30,056 observed pairs |
| pKd transform | Applied: `-(np.log10(affinities / 1e9))`, `datahelper.py:148` |
| Runtime Y range | **5.0000 – 10.7959** — exactly the paper's `[0.016, 10000]` nM mapped through the stated formula |
| Length filtering | **None.** The Davis branch has no `get_removelist` call |
| Truncation | `line[:MAX_LEN]` in `label_smiles`/`label_sequence` |
| Bundled folds | 5 × ~5009 + test 5010 = 30,056, max index 30,055 — **exactly compatible**, but never loaded |

### KIBA — critical target

| Item | Value |
|---|---|
| Bundled `ligands_iso.txt` / `proteins.txt` | **2111 / 229** — i.e. the paper's published counts are already baked into the data files |
| Runtime filter applied by code | `get_removelist(ligands, 90)` removes **157**; `get_removelist(proteins, 1365)` removes **12** |
| Filter semantics | `if len(x) >= length: remove` — a **string-length** filter on SMILES / sequences, `datahelper.py:88-94` |
| Runtime matrix | **`(1954, 217)`**, **109,296** observed pairs |
| Runtime Y range | **0.0000 – 17.2002** |
| Paper's claim | 2111 × 229, density 24.4% (≈117,954 pairs), affinities < 10 removed |

**Why the dimensions differ — determinable from code.** The paper's §3.1 reduction (52,498 × 467 → 2111 × 229) is **not performed by any code in this repository**; it is pre-applied to the bundled files. The code then performs a *second*, undocumented reduction — a length filter at thresholds 90 (SMILES) and 1365 (sequences) — giving 1954 × 217. The in-code comment at `datahelper.py:90` reads `# Davis SMILES:85 protein:1200   KIBA SMILES:100   protein:1000`, which matches neither the thresholds actually passed (90 / 1365) nor the paper's stated caps. [CODE] / [VERIFIED]

**The paper's affinity filter is absent.** The runtime KIBA matrix retains values down to 0.0000, so no `< 10` removal occurs either offline or at runtime. This contradicts §3.1's text and Fig. 5's histogram (lower edge ≈9.5). [VERIFIED]

**Bundled KIBA folds are incompatible with the runtime matrix.** `data/kiba/folds/` indexes **118,254** pairs with max index 118,253, against 109,296 observed pairs at runtime — loading them would index out of range. The fold-loading call is commented out (`run_experiments.py:168`) and `DataSet` has no `read_sets` method, so this is inert, but it confirms the fold files were generated against a different matrix than the code builds. [VERIFIED]

### Missing values

`label_row_inds, label_col_inds = np.where(np.isnan(Y) == False)` (`run_experiments.py:505`). Only observed pairs enter folds, training and evaluation; NaN cells are never imputed and never used. This answers a Phase 2 `[OPEN]`. For Davis this is a no-op (0 NaN); for KIBA it excludes 374,123 of 483,419 cells. [VERIFIED]

---

## 7. Split and Cross-Validation Audit

| Question | Finding | Evidence |
|---|---|---|
| Outer folds | **6** (`foldcount=6`); `nfolds[5]` = test, `nfolds[0:5]` = train | [CODE] `:470, 166-167` |
| Inner folds | **5** — leave-one-out over the 5 training folds | [CODE] `:175-186` |
| Generated or loaded | **Generated at runtime.** The `read_sets` call is commented out (`:168`); `DataSet` has no such method | [CODE] / [VERIFIED] |
| Randomization | `random.sample` inside `get_random_folds` | [CODE] `:40` |
| Random seed | **`random.seed(i+1000)` with `i=0` → seed 1000.** Only Python's `random` is seeded; `numpy` and `torch` are not | [CODE] `:509` |
| Repetitions | **1** (`for i in range(1)`), against the paper's ten | [CODE] `:508` |
| Fold rotation | **No.** A single test fold (`nfolds[5]`) is held out; the other five rotate only as the *inner* validation index | [CODE] `:166-186` |
| Drug-wise split | `get_drugwise_folds` — folds drugs, then maps each drug to all its pair indices. `problem_type=2` (**the default**) | [CODE] `:55-70`, `arguments.py:58-63` |
| Target-wise split | `get_targetwise_folds`, `problem_type=3` | [CODE] `:73-87` |
| Pair-wise split | `get_random_folds`, `problem_type=1` — **not described in the paper** | [CODE] `:510-511` |
| Train construction | 4 of the 5 training folds concatenated | [CODE] `:182-184` |
| Validation construction | `val_sets.append(val_fold)` at `:181` — **built and then discarded** | [CODE] |
| Test construction | `test_sets.append(test_set)` — the same fold appended 5× | [CODE] `:185` |

### Model-selection contamination — confirmed

```python
# run_experiments.py:188-195
bestparamind, best_param_list, ... = general_nfold_cv(..., train_sets, test_sets, get_aupr, get_rm2)
best_param, bestperf, ...          = general_nfold_cv_test(..., train_sets, test_sets, get_rm2, best_param_list, i)
```

`val_sets` (built at `:176-181`) is passed to neither call. Consequences, all traced:

- The grid search over `num_windows × smi_window_lengths × seq_window_lengths × lamda` selects `best_param_list` by performance **on the test fold** (`:275-296`).
- Inside both passes, checkpointing is triggered by test-fold CI: `if rperf >= max(rperf_list): torch.save(model, 'checkpoint.pth')` (`:288-289`, `:375-376`).
- Early stopping breaks on test-fold CI: `if rperf < max(rperf_list) - 0.1: break` (`:290-292`, `:377-378`).
- Final metrics are computed by reloading that checkpoint and evaluating on the **same** `test_loader` (`:382-396`).

**Classification:** this is **model-selection and checkpoint-selection contamination** — hyperparameters, stopping epoch and reported weights are all chosen using the evaluation set. It is **not** training-label leakage: `train_dataset` is built from `labeled_sets[foldind]` and `test_dataset` from `val_sets[foldind]`, which are disjoint index sets under an entity-wise split, so no test affinity value enters a gradient step. [VERIFIED] `run_experiments.py:345-357`

---

## 8. Training Procedure

| Item | Finding | Evidence |
|---|---|---|
| Optimizer | `optim.Adam(model.parameters())` — **constructed fresh inside `train()`, i.e. once per epoch**, so Adam moment estimates reset every epoch | [CODE] `:118` |
| Learning rate | Not passed; PyTorch's Adam default is `1e-3`, matching the paper | [CODE] `:118` |
| Batch size | `DataLoader(..., batch_size=batchsz)`, default 256; train loader `shuffle=True`, test loader unshuffled | [CODE] `:271-272` |
| Epochs | `FLAGS.num_epoch`, default 100 | [CODE] `:279` |
| Dropout | 0.2 (encoder), 0.1 (Reg block) | [CODE] `model.py:14,19,90,93` |
| Weight initialization | `weights_init` (`:99-112`) — Xavier normal for `Linear` and `Conv1d`, constant for `BatchNorm1d`, orthogonal for `LSTM`. Applied via `model.apply(weights_init)`. **No initializer for `ConvTranspose1d`**, so the decoders keep PyTorch defaults | [CODE] `:99-112, 277, 362` |
| Random seeds | `random.seed(1000)` only; `os.environ['PYTHONHASHSEED']='0'` at `:14`. **numpy and torch unseeded** | [CODE] |
| Early stopping | Custom: evaluate every 5 epochs (search pass) or every 2 epochs (final pass); break if CI drops >0.1 below the running max | [CODE] `:285-292, 366-378` |
| Patience | None in the usual sense — a single 0.1 CI drop terminates | [CODE] |
| LR schedule | None | [CODE] |
| Checkpoint file | **`'checkpoint.pth'`, fixed name, CWD-relative**, written in both passes | [CODE] `:289, 376` |
| Checkpoint overwrite | **Yes.** The same path is reused across every grid point, every fold and both passes; `general_nfold_cv_test` reloads whatever was last written for the current fold | [CODE] / [INFERRED] |
| `--checkpoint_path` | Declared, never referenced | [VERIFIED] |
| Model used for reported metrics | `model = torch.load('checkpoint.pth')` — the checkpoint selected by **test-fold** CI | [CODE] `:382` |

---

## 9. Evaluation and Metrics

| Metric | Implementation | Notes |
|---|---|---|
| **CI** | `emetrics.get_cindex` — vectorized global pairwise concordance over the full evaluation array | Matches §3.2. Builds three `n × n` float32 matrices; for a KIBA drug-wise test fold (~18k pairs) that is ≈1.3 GB each. [CODE] `emetrics.py:4-18` |
| **MSE** | `nn.MSELoss()` (mean). Computed at `:380` on the reloaded checkpoint's predictions | Resolves Phase 2 **A3**: the code uses the mean, consistent with Table 2's magnitudes, not the sum the paper printed |
| **MAE** | **Absent** | Table 2 reports MAE for all 5 methods × 2 datasets × 2 settings; no implementation exists. [VERIFIED] |
| **r²m** | `emetrics.get_rm2` | Matches Eq. in §3.2, with an added `np.absolute` inside the square root |
| **AUC** | `sklearn.roc_auc_score` | See thresholds below |
| **AUPR** | `emetrics.get_aupr` (`precision_recall_curve` + trapezoidal `auc`) | Computed and logged; **not one of the paper's five metrics** |

### Thresholds

| Location | Threshold | Dataset |
|---|---|---|
| `run_experiments.py:163` — `test()`, called **every evaluation epoch** | `affinities > 7` | **Both** Davis and KIBA |
| `run_experiments.py:394` — final evaluation | `affinities > 7.0` | Davis |
| `run_experiments.py:399` — final evaluation | `affinities > 12.1` | KIBA |

Findings:
- The paper's **sweep over 10.5→12.5 at 0.1 increments is not implemented**; each call uses a single threshold. Fig. 10 cannot be produced from this code path. [VERIFIED]
- The paper states AUC is used **only** for KIBA; the code computes Davis AUC at threshold 7.0. [VERIFIED]
- **Threshold inconsistency within the KIBA run:** the epoch loop's `test()` labels positives at `> 7` while the final metric labels them at `> 12.1`. Since the KIBA runtime range is 0–17.2 with mass concentrated near 11.5–12, these two thresholds produce very different positive-class sizes, and the AUC reported during training is not the AUC reported at the end. [VERIFIED]
- Neither 7 nor 12.1 appears in the paper's description of the sweep. [PAPER] §3.2

### Aggregation

| Question | Finding |
|---|---|
| Per-fold vs global | **Global** per evaluation set — all predictions are accumulated across batches into `pre_affinities`/`affinities` before any metric is computed (`:383-391`) |
| Across folds | Arithmetic mean over the 5 entries of `all_predictions` etc. (`:426`, `nfold_setting_sample:204-234`) |
| Across repetitions | `np.mean`/`np.std` over `perf`, `mseloss`, `auc`, `aupr` — but the outer loop runs **once**, so these reduce to the single value |
| Standard deviation | `np.std(testperfs)` over the **5 CV folds**, which all evaluate the *same* test set. The paper's std is over **10 independent random splits** |
| Checkpoint/epoch used | The checkpoint whose test-fold CI was highest |

---

## 10. Baseline and Generation Traceability

| Item | Status | Evidence |
|---|---|---|
| KronRLS | **MISSING** | No file, class, or import. [VERIFIED] |
| DeepDTA | **MISSING** — appears only in a header comment: `## This program is based on the DeepDTA model(https://github.com/hkmztrk/DeepDTA)` (`run_experiments.py:1`) | [VERIFIED] |
| DeepAffinity | **MISSING** | [VERIFIED] |
| GraphDTA | **MISSING** | [VERIFIED] |
| GAN / SeqGAN / AAE / VAE | **MISSING** | [VERIFIED] |
| External results import | **None.** No CSV/JSON result loader, no hardcoded baseline numbers | [VERIFIED] |
| Drug generation | **MISSING.** The decoder produces logits over the vocabulary, but nothing decodes them to SMILES strings — no `argmax`, no index-to-character map, no string assembly | [VERIFIED] |
| Validity / Uniqueness | **MISSING.** No RDKit import anywhere | [VERIFIED] |
| SARS-CoV-2 prediction | **MISSING.** No script, no NCBI sequence loading | [VERIFIED] |
| Bundled result files | **None.** `result/`, `figures/`, `logs/` contain only 2-byte `readme` placeholders | [VERIFIED] |

**Conclusion:** only Co-VAE itself is implemented. Every number in Table 2 attributed to a baseline, all of Table 3, all of Tables 4–5, and Figs. 7–10 originate outside this repository. The paper's claim that "the same settings were used for all the comparison methods" [PAPER] §3.3 cannot be traced to any code here. [VERIFIED]

---

## 11. Runtime / Execution Blockers

Each item was checked against the source rather than assumed.

| # | Blocker | Verification | Severity |
|---|---|---|---|
| **B1** | **Python ≥ 3.11 incompatibility.** `get_random_folds` calls `random.sample(indices, int(sample_size))` where `indices` is a `set` (`:33, 40`). Executed on Python 3.13: `TypeError: Population must be a sequence. For dicts or sets, use sorted(d).` On the mandatory path for all three `problem_type` values. | [VERIFIED] by execution | **CRITICAL** |
| **B2** | **`--lamda` omitted causes a crash.** `arguments.py:83-88` declares `nargs='+'` with `default=-5` (a bare int). Omitted → `FLAGS.lamda == -5`; `general_nfold_cv:242,248` calls `len(lamda_set)` → `TypeError: object of type 'int' has no len()`. Passing `--lamda -5` yields `[-5]` and works. The repository ships **no README and no example command**, so no documented invocation avoids this. | [VERIFIED] by reproducing the argparse behaviour | **CRITICAL** |
| **B3** | **CUDA required, no CPU fallback.** `torch.cuda.FloatTensor` (`model.py:36`); `.cuda()` at `model.py:128,131` and `run_experiments.py:127,276,361`. No `device` argument, no `torch.cuda.is_available()` guard anywhere. | [VERIFIED] by grep over all `.py` | **CRITICAL** |
| **B4** | **Required flags have no defaults.** `--num_windows`, `--smi_window_lengths`, `--seq_window_lengths` default to `None` (`arguments.py:8-25`); `len(None)` fails at `:239-241`. Combined with B2, four flags must be supplied with no documentation. | [VERIFIED] | **HIGH** |
| **B5** | **Fixed checkpoint filename.** `'checkpoint.pth'` in the CWD, written by both passes and every grid point (`:289, 376`), reloaded at `:382`. Concurrent runs or repeated runs in one directory clobber each other. | [CODE] | **HIGH** |
| **B6** | **Davis needs non-default lengths.** `--max_smi_len` and `--max_seq_len` default to 100/1000 (the KIBA values); Davis requires 85/1200 per the paper. These also feed the loss weight (§4.4), so wrong values silently change the objective. | [VERIFIED] `arguments.py:27-38` | **HIGH** |
| **B7** | **Windows path separator hardcoded.** `FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "\\"` (`:542`). Benign on Windows; produces a literal-backslash directory name on POSIX. | [CODE] | **MEDIUM** |
| **B8** | **`roc_auc_score` single-class failure.** `:163` and `:395/:400` call it without a class-count guard. A test fold whose affinities all fall on one side of the threshold raises `ValueError`. For KIBA at `> 7` (epoch loop) essentially all pairs are positive, since the runtime range is 0–17.2 with mass at 11.5–12 — so this is a realistic path. | [CODE] / [INFERRED] | **MEDIUM** |
| **B9** | **`get_cindex` memory.** Builds three `n × n` float32 arrays. For a KIBA drug-wise test fold (~18k pairs) that is ≈1.3 GB each, ≈4 GB peak, re-allocated at every evaluation epoch. | [CODE] / [INFERRED] | **MEDIUM** |
| **B10** | **`squeeze()` on batch size 1.** `CNN.forward:51` and `net.forward:136` call `.squeeze()`; `DataLoader` has no `drop_last`, so a trailing batch of size 1 collapses a dimension and breaks the downstream `torch.cat(..., 1)`. Whether this occurs depends on the fold sizes. | [CODE] / [INFERRED] | **LOW** |
| **B11** | **In-place `.exp_()` on `logvar`.** `model.py:35` mutates the tensor that is also returned as `output2` and consumed by the KL term. Whether autograd raises or the KL silently receives `exp(logvar/2)` instead of `logvar` is version-dependent. | [OPEN] | **HIGH if confirmed** |
| **B12** | **Ragged `np.savetxt` — checked, does NOT apply.** `all_affinities` collects one array per fold (`:416`), and because `test_sets` holds the *same* fold five times, all five arrays have identical length. `np.array(all_affinities)` is therefore a clean `(5, n_test)` array. | [VERIFIED] | **none** |
| **B13** | **Embedding index overflow — checked, does NOT apply.** `nn.Embedding(64, …)` / `nn.Embedding(25, …)` admit indices 0–63 / 0–24, while `CHARISOSMISET`/`CHARPROTSET` assign values up to 64/25. On the bundled data the observed maxima are 49/54 (SMILES) and 24/23 (proteins), all in range, and no character falls outside either dictionary. | [VERIFIED] | **none** |

---

## 12. Critical Discrepancy Audit

### C1 — KIBA runtime matrix is not the published matrix

**Paper:** KIBA reduced to **2111 drugs × 229 targets**, density 24.4%, by removing "the drug-target pairs which have known affinities less than 10". `[PAPER]` §3.1
**Code:** the bundled files already hold 2111/229; the code then applies a **string-length** filter (`get_removelist(ligands, 90)`, `get_removelist(proteins, 1365)`), yielding **1954 × 217** with **109,296** observed pairs. No affinity filter exists; the runtime range is **0.0000–17.2002**. `[CODE]` `datahelper.py:88-94, 154-159`
**Verification:** executed `DataSet.parse_data` on the bundled KIBA data and inspected `ligands_iso.txt`/`proteins.txt` directly. `[VERIFIED]`
**Impact:** **CRITICAL**
**Why it matters:** every KIBA number in Table 2 refers to a dataset the released code does not build — 157 drugs and 12 targets smaller, ~8,700 pairs fewer, and retaining affinities the paper says were removed.

### C2 — Hyperparameter selection, early stopping and reporting all use the test fold

**Paper:** hyperparameters chosen by "five-fold cross-validation **based on only training data**"; Fig. 6 depicts a distinct validation partition. `[PAPER]` §3.3, Fig. 6
**Code:** `nfold_setting_sample` builds `val_sets` (`:176-181`) and passes **`test_sets`** to both `general_nfold_cv` and `general_nfold_cv_test` (`:188-195`). `val_sets` is never read. Inside both passes, `EarlyStopping`-equivalent logic, `torch.save('checkpoint.pth')` and `rperf = max(...)` all key off the test fold. `[CODE]`
**Verification:** traced every use of `val_sets` and `test_sets` through both call sites. `[VERIFIED]`
**Impact:** **CRITICAL**
**Why it matters:** the grid point, the stopping epoch and the reported weights are all selected on the set the metrics are reported on. Reproduction must replicate this exact protocol to land on the published values. (This is selection contamination, not training-label leakage — the train and test index sets are disjoint.)

### C3 — One repetition, not ten

**Paper:** "we randomly split the drugs or targets for **ten times**, and the means and standard deviations for each evaluation metrics were reported." `[PAPER]` §3.3
**Code:** `for i in range(1): random.seed(i+1000)` (`:508-509`). The reported std is `np.std` over the 5 CV folds (`:224`), which all evaluate the same test set. `[CODE]`
**Verification:** direct read. `[VERIFIED]`
**Impact:** **HIGH**
**Why it matters:** every `(std)` value in Table 2 has a different meaning than the code produces — across-split variability versus across-fold variability on one shared test set.

### C4 — MAE is reported but not implemented

**Paper:** Table 2 reports MAE for 5 methods × 2 datasets × 2 settings — 20 values. `[PAPER]` Table 2
**Code:** no MAE anywhere; `test()` and `general_nfold_cv_test` return CI, MSE, r²m, AUC (+AUPR). `[CODE]`
**Verification:** grep for `mae`, `abs`, `L1Loss` across all `.py`; only `np.abs` inside `get_rm2`. `[VERIFIED]`
**Impact:** **HIGH**
**Why it matters:** a quarter of the main results table has no code path.

### C5 — ε sampled at the wrong scale

**Paper:** `ε ~ N(0, I)`. `[PAPER]` §2.3
**Code:** `torch.cuda.FloatTensor(std.size()).normal_(0, 0.1)` → `ε ~ N(0, 0.01)`, since PyTorch's second argument is the standard deviation. `[CODE]` `model.py:36`
**Verification:** direct read plus PyTorch's `normal_(mean, std)` signature. `[VERIFIED]`
**Impact:** **HIGH**
**Why it matters:** the KL term in `loss_f` is the closed form for `KL(N(μ,σ²) ‖ N(0,I))`, which presumes unit-variance sampling noise. The sampling step and the regularizer assume different scales, so the model is not the variational objective Eq. (10) describes.

### C6 — λ weights the opposite term

**Paper:** Eq. (9)/(10) place λ on the affinity term, VAE terms at weight 1. `[PAPER]` Eqs. (9)–(10)
**Code:** `loss = loss_affinity + 10**lamda * (loss_drug + (max_smi_len/max_seq_len) * loss_target)` — `10^λ` on the VAE terms, affinity at weight 1, plus an undocumented `max_smi_len/max_seq_len` factor on the target branch. `[CODE]` `:136`
**Verification:** direct read. `[VERIFIED]`
**Impact:** **MEDIUM**
**Why it matters:** the exponentiation resolves Phase 2's A1 favourably (`10^λ` is positive, so the sign paradox is notational). The two weightings share a minimiser up to a global loss scale that Adam largely absorbs, so the behavioural effect is bounded — but the undocumented 0.1× on the target VAE loss is a genuine asymmetry absent from Eq. (10), and the effective λ implied by the code is `10³`/`10⁵`, not `10⁻³`/`10⁻⁵`.

### C7 — Average pooling, not max pooling

**Paper:** "a max-pooling layer"; Fig. 4 labels the box "max-pooling". `[PAPER]` §2.5, Fig. 4
**Code:** `self.out = nn.AdaptiveAvgPool1d(1)`. `[CODE]` `model.py:24`
**Verification:** direct read. `[VERIFIED]`
**Impact:** **MEDIUM**
**Why it matters:** changes what the latent summarises — mean activation across the sequence rather than peak motif response.

### C8 — Encoder has no ReLU activations

**Paper:** "There is a ReLU activation function after each 1D-convolutional layer" (Phase 2 **A4** asked 2 vs 3). `[PAPER]` §2.5
**Code:** zero post-convolution activations; the only encoder-body nonlinearity is the three sigmoid gates. `[CODE]` `model.py:9-22, 41-49`
**Verification:** direct read. `[VERIFIED]`
**Impact:** **MEDIUM**
**Why it matters:** resolves A4 in a direction neither paper reading anticipated.

### C9 — AUC protocol differs on every axis

**Paper:** AUC on KIBA only, swept over 10.5→12.5 at 0.1 increments (21 points), producing Fig. 10. `[PAPER]` §3.2, Fig. 10
**Code:** no sweep; a single threshold per call — `> 7` in the epoch loop for **both** datasets (`:163`), `> 7.0` for Davis and `> 12.1` for KIBA at final evaluation (`:394, 399`). `[CODE]`
**Verification:** direct read of all three sites. `[VERIFIED]`
**Impact:** **MEDIUM**
**Why it matters:** Fig. 10 has no code path, Davis AUC is computed against an explicit statement that it should not be, and the KIBA AUC printed during training uses a different positive class than the one finally reported.

### C10 — Reg-block topology matches neither prose nor figure

**Paper:** prose reads as 3 FC layers; Fig. 4 shows 5 arranged 2-per-branch + 1 merged. `[PAPER]` §2.5, Fig. 4
**Code:** 5 `Linear` layers arranged **1-per-branch + 3 merged**: `Linear(96,96)` ×2, then `192→1024→512→1`. Dropout **0.1**, against Table 1's 0.2. `[CODE]` `model.py:84-113`
**Verification:** direct read. `[VERIFIED]`
**Impact:** **MEDIUM**
**Why it matters:** resolves Phase 2's A6 and the `[OPEN]` on FC widths, but agrees with neither paper source, and the dropout rate contradicts Table 1.

### C11 — Only Co-VAE is implemented

**Paper:** four affinity baselines rerun under shared splits; four generative baselines; a drug-generation experiment with RDKit validity/uniqueness; a SARS-CoV-2 case study. `[PAPER]` §3.3, §3.4, Tables 2–5
**Code:** none of it. No baseline code, no results import, no RDKit, no SMILES decoding, no generation. `[CODE]`
**Verification:** exhaustive grep across all `.py`; `result/`/`figures/`/`logs/` hold only placeholders. `[VERIFIED]`
**Impact:** **HIGH**
**Why it matters:** Tables 2 (baseline rows), 3, 4, 5 and Figs. 7–10 cannot be traced to this repository at all.

### C12 — Execution blockers prevent any run as shipped

**Paper:** n/a — a reproducibility property of the release.
**Code:** three independent hard stops — `random.sample` on a `set` (Python ≥ 3.11), `--lamda` omitted (`len()` on an int), and unconditional CUDA. Four flags have no defaults and the repository ships no README. `[CODE]`
**Verification:** B1 and B2 reproduced by execution; B3/B4 by grep and argparse inspection. `[VERIFIED]`
**Impact:** **CRITICAL**
**Why it matters:** the released code cannot execute a single training step without source modification on any modern Python, and no documented invocation exists to derive the correct flags from.

---

## 13. Reproducibility Status

| Component | Status | Reason |
|---|---|---|
| Dataset | **PARTIAL** | Davis matches exactly (68×442, 30,056, range 5.0–10.796). KIBA does not: runtime 1954×217 / 109,296 vs published 2111×229 / ≈117,954 (C1) |
| Preprocessing | **PARTIAL** | Davis pKd transform matches. The paper's KIBA affinity filter is absent; an undocumented length filter is present instead; no Davis length filter; silent head-truncation (C1) |
| Input representation | **COMPLETE** | Label encoding, vocabularies 64/25, embedding dim 128 all match the paper |
| Architecture | **PARTIAL** | Gated CNN, layer counts, filter progression, decoder shape and merge-by-concat all match. Average vs max pooling (C7), zero ReLUs (C8), Reg-block topology and dropout rate (C10) do not |
| Loss/objective | **PARTIAL** | All three terms present and in the backward pass. λ weights the opposite term with an extra undocumented factor (C6); ε is mis-scaled (C5); `.exp_()` in-place effect unresolved (B11) |
| Optimizer | **COMPLETE** | Adam at PyTorch's default 1e-3, matching the paper — though re-instantiated every epoch |
| Hyperparameters | **PARTIAL** | Batch 256, epochs 100, filter counts and filter-length grid all reachable; dropout 0.1 vs 0.2; four flags have no defaults; the paper never reports which grid point was selected |
| Splits | **PARTIAL** | 6 folds and entity-wise granularity match. Validation built then discarded; no fold rotation; bundled folds incompatible with the runtime KIBA matrix (C2) |
| Randomness | **BLOCKED** | Only `random` is seeded (1000); numpy and torch unseeded; one repetition instead of ten (C3) |
| Training | **PARTIAL** | Runs a genuine joint optimisation, but early stopping and checkpointing key off the test fold (C2); fixed checkpoint filename overwritten across configurations (B5) |
| Evaluation | **PARTIAL** | CI, MSE, r²m implemented and consistent with §3.2. MAE missing (C4); AUC protocol differs on dataset, threshold and sweep (C9) |
| Baselines | **BLOCKED** | No implementation and no results import (C11) |
| Generation experiment | **BLOCKED** | No generation, no RDKit, no validity/uniqueness (C11) |
| Runtime environment | **BLOCKED** | Three confirmed hard stops; no README, no requirements file, no documented invocation (C12) |

---

## 14. Phase 4 Targets

Eight items, ordered. Each is small and diagnostic.

1. **Make the code run at all, minimally.** Establish the smallest patch set that clears B1 (`random.sample` on a set), B2 (`--lamda` scalar) and B3 (CUDA), and record each as an explicit deviation. Nothing below is testable until this is done.
2. **Confirm the runtime objective numerically.** Instrument one batch: print `loss_affinity`, `loss_drug`, `loss_target` and the `10**lamda` coefficient, and confirm both KL terms carry non-zero gradient. Settles §4.2 and C6 empirically.
3. **Resolve B11.** Check whether `logvar.mul(0.5).exp_()` mutates the tensor reaching `loss_f`'s KL term — inspect `output2` before and after `reparametrize`, or check for an autograd in-place error. This determines whether the KL is computed on `logvar` or on `exp(logvar/2)`.
4. **Quantify the ε mis-scaling (C5).** Compare posterior statistics and affinity CI under `normal_(0, 0.1)` versus `normal_(0, 1)` for one fold, to size the effect of the sampling/KL mismatch.
5. **Pin down the KIBA construction (C1).** Confirm 1954×217 / 109,296 under the exact flags a user would supply, and determine whether any flag combination recovers 2111×229. Check whether the affinities `< 10` that survive are concentrated in particular drugs or targets.
6. **Measure the contamination effect (C2).** Run one fold twice — once as shipped (test fold as the selection set) and once with `val_sets` substituted — and compare the reported CI/MSE. This bounds how much of the published margin depends on the protocol.
7. **Check metric behaviour.** Verify `get_cindex` memory on a KIBA drug-wise test fold (B9), and whether `roc_auc_score` at `> 7` raises on a single-class KIBA fold (B8).
8. **One minimal end-to-end run.** Davis, `--problem_type 2`, a single grid point (one `num_windows`, one filter length per branch, one λ), reduced epochs — enough to produce a `log.txt` and a `result/iter0*.txt`, confirming the pipeline completes and establishing the cost envelope.

**Do not attempt full-paper reproduction before items 1–5 are resolved.** Baselines, Table 3, Tables 4–5 and Figs. 7–10 have no code path (C11) and are out of reach regardless of what Phase 4 establishes.

---

## 15. Executive Verdict

### Overall Traceability

**🔴 RED**

### Paper-to-Code Fidelity

The repository contains a real, coherent implementation of a co-regularized two-VAE model: two gated-CNN encoders, two deconvolutional decoders, a joint regression head, and a three-term objective in which all terms genuinely participate in backpropagation. To that extent the paper's core method is traceable, and the code resolves several Phase 2 unknowns — latent dimension 96, FC widths 96/192/1024/512/1, and λ used as `10^λ`. But fidelity breaks on the things reproduction depends on. The KIBA dataset the code builds is not the one the paper reports; the evaluation protocol selects hyperparameters, stopping epoch and checkpoint on the test fold rather than the validation partition Fig. 6 depicts; one of the four reported metrics is not implemented; the AUC protocol differs on dataset, threshold and sweep; and four of the five experiment families in the paper — all baselines, the generative comparison, the generated-drug tables, and the SARS-CoV-2 case study — have no code path at all.

### Top 5 Problems

- **KIBA runtime matrix is 1954 × 217 / 109,296 pairs, not the published 2111 × 229 / ≈117,954**, because an undocumented length filter runs on top of the already-filtered bundled files, while the paper's stated affinity filter never runs (C1).
- **Hyperparameter search, early stopping, checkpoint saving and final reporting all evaluate on the test fold**; the `val_sets` the code builds is discarded (C2).
- **Only Co-VAE is implemented** — no KronRLS, DeepDTA, DeepAffinity, GraphDTA, GAN, SeqGAN, AAE, VAE, no drug generation, no RDKit, no results import (C11).
- **Three independent hard stops prevent execution as shipped**: `random.sample` on a `set` under Python ≥ 3.11, `--lamda` omitted, and unconditional CUDA — with no README to supply the four flags that have no defaults (C12).
- **The variational formulation is mis-scaled**: `ε ~ N(0, 0.1²)` against a KL term derived for `N(0, I)` (C5), compounded by λ weighting the VAE terms rather than the affinity term with an extra undocumented 0.1× on the target branch (C6).

### Execution Blockers

- **CRITICAL** — `random.sample(set, k)` fails on Python ≥ 3.11 (B1)
- **CRITICAL** — `--lamda` omitted → `len()` on an int (B2)
- **CRITICAL** — CUDA required, no CPU fallback (B3)
- **HIGH** — `--num_windows`, `--smi_window_lengths`, `--seq_window_lengths` have no defaults, and no README documents them (B4)
- **HIGH** — fixed `checkpoint.pth` overwritten across grid points, folds and passes (B5)
- **HIGH** — Davis requires non-default `--max_smi_len 85 --max_seq_len 1200`, which also silently change the loss weight (B6)
- **HIGH (unresolved)** — in-place `.exp_()` on `logvar` may alter the KL input (B11)
- **MEDIUM** — POSIX path separator (B7), single-class `roc_auc_score` (B8), `get_cindex` memory (B9)
- **Checked and cleared** — ragged `np.savetxt` (B12) and embedding index overflow (B13) do **not** occur on the bundled data

### Phase 4 Recommendation

Treat Phase 4 as a narrow diagnostic pass, not a reproduction. Clear the three hard stops with the smallest possible patch set, logging each as an explicit deviation, then run the five cheap instrumented checks above — objective composition, the `.exp_()` question, the ε mis-scaling, the KIBA matrix construction, and a single-fold comparison of test-fold versus validation-fold selection — before spending compute on anything resembling Table 2. The baseline rows, Table 3, Tables 4–5 and Figs. 7–10 should be treated as outside this repository's reach; recovering them needs author contact or independent reimplementation, which is a scoping decision rather than an audit finding.

---

*End of Phase 3. Read-only: no repository source file was modified, no dependency installed, no training run or reproduction attempted.*

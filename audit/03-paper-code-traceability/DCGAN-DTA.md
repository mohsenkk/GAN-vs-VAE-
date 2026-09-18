# Stage 3 — Paper ↔ Code Traceability Audit: DCGAN-DTA

**Specification:** `audit/02-paper-forensic/DCGAN-DTA.md` (Stage 2)
**Implementation:** `DCGAN-DTA/DCGAN-DTA/` (upstream `github.com/mojtabaze7/DCGAN-DTA`, HEAD `453fe16`)
**Mode:** read-only. No files modified, no dependencies installed, no execution.

Tags: `[PAPER]` `[CODE]` `[VERIFIED]` `[INFERRED]` `[OPEN]`.
`[VERIFIED]` is used only where I traced the construct end-to-end in source or confirmed a data file's contents by direct inspection.

---

## A. Executive Verdict

**Overall traceability status: 🔴 RED**

The released code implements one experiment family (warm-start CV for variants A/B/C) and does so with several concrete departures from the paper. Three of the paper's five experiment families have no implementation at all, and the method's namesake mechanism — DCGAN→DTA transfer — is dimensionally inconsistent on the drug branch of all three variants.

### Confirmed mismatches (established from code)

1. **The transferred discriminator layer is dimensionally incompatible with the drug branch in all three variants.** `gan_smiles.layers[-3]` is the discriminator's `Conv1D(64, k=3)`, built on a **1-channel** input (`Reshape((200,1))`, `run_experiments.py:76,90`). It is then applied to a **32-channel** `Embedding(output_dim=32)` tensor (`:365-366`, `:410-411`, `:452-453`). The same mismatch applies to the protein branch of variant A (`:376-377`). Variant B's protein transfer is the only dimensionally consistent one (20 → 20). [VERIFIED shapes; [INFERRED] that Keras rejects this]

2. **BatchNormalization is present in all three generators**, contradicting the paper's explicit and justified claim that it was removed. `run_experiments.py:62, 66` (drug), `:164, 168` (target), `:268, 272` (BLOSUM). [VERIFIED]

3. **BLOSUM encoding is 20-dimensional, not 25.** `protein_feature_vecblsm.json` holds 22 keys × **20**-dim vectors; `dataset.py:12` hardcodes `self.pro_emb = 20`; `build_GAN_B/C` declare `XTinput = Input(shape=(FLAGS.max_seq_len, 20,))` (`:406`, `:449`). [VERIFIED]

4. **The reported CI is not the paper's Eq. (1).** `run_experiments.py:20` imports only `get_rm2` from `emetrics`; the exact global CI in `emetrics.get_cindex` is **never called**. Reported CI comes from `cindex_score` (`:870`) used as a Keras `metrics=` entry, so it is computed per-batch and averaged across batches — a batch-size-dependent approximation of Eq. (1). [VERIFIED]

5. **The FC-width conflict resolves in favour of Figure 1.** The code uses **1024 → 1024 → 512** (`:389-393`, `:434-438`, `:477-481`), matching Fig. 1 and contradicting the prose (p. 6) and Supplementary Table 2, which both say 1024/512/512. [VERIFIED]

6. **Three experiment families have no implementation.** No cold-start/logP splitting, no adversarial-control shuffling, no baseline models (DeepDTA, GraphDTA, FusionDTA, DGDTA, TEFDTA, G-K BertDTA, FC net, KNN), and no concatenation-merge ablation. Greps for `logp|xlogp|babel`, `shuffle|permut|straw`, `Concatenate`, `KNeighbors` return nothing on the relevant paths. This covers Figs. 5, 6, 7, 8 and all comparison bars in Figs. 2–3. [VERIFIED]

7. **No UniProt/ChEMBL retrieval exists.** GAN pretraining consumes bundled `ligands_train.txt` (50,068 entries) and `proteins_train.txt` (50,202 entries) (`datahelper.py:149-150`). No code, script, URL or manifest links these to UniProt or ChEMBL. [VERIFIED that the files are what is used; [OPEN] as to provenance]

8. **Variant B's protein GAN is pretrained on data that includes every evaluation protein.** `run_experiments.py:603` builds `new_XT = np.concatenate((XT, XT_t))` and takes `range(10000)`; `XT` is the full evaluation protein set (1606 PDBbind / 1088 BindingDB), so it is entirely contained in the pretraining batch. [VERIFIED]

9. **In the pass that produces the reported numbers, early stopping and epoch selection run on the evaluation set.** `nfold_1_2_3_setting_sample` (`:507-540`) calls the CV routine twice, discarding the `val_sets` predictions and keeping the `test_sets` pass; inside that pass `EarlyStopping(monitor='val_cindex_score', restore_best_weights=True)` and `rperf = max(history['val_cindex_score'])` both act on `test_set` (`:787-800`). [VERIFIED]

### Unresolved (not mismatches)

- Whether Keras raises on finding #1 or silently rebuilds the layer — needs execution.
- Provenance of `ligands_train.txt` / `proteins_train.txt`.
- Whether the paper's published numbers came from this code path at all, given #6.
- Whether the transferred layer is frozen: `dis_v.trainable = False` (`:142, 246, 344`) propagates recursively to sublayers in Keras, which would make the spliced layer non-trainable, but the code never sets trainability on the transferred layer explicitly. [INFERRED]

---

## B. Paper → Code Traceability Table

| ID | Paper claim | Paper evidence | Code location | Status | Code behavior | Notes |
|---|---|---|---|---|---|---|
| **G1** | Generator: 3 × Conv1DTranspose, filters 128/64/1, kernel 3 | p. 4, p. 6; SupT2 | `run_experiments.py:55-71` (drug), `:157-173` (target) | **MATCH** | 3 Conv1DTranspose, filters 128/64/1, `kernel_size=3` | Preceded by `Dense(256*5)`+`Reshape` (drug) / `Dense(256*10)`+`Reshape` (target), not described in paper |
| **G2** | Generator activations ReLU/ReLU/tanh | p. 4 | `:60,64` (`activation='relu'`), `:68-70` (`Activation('tanh')`) | **MATCH** | As stated | |
| **G3** | **BatchNorm omitted from generator and discriminator** | p. 4 (explicit, with justification) | `:62,66`; `:164,168`; `:268,272` | **MISMATCH** | BatchNorm present after generator conv 1 and 2 in **all three** GANs. Discriminators have none. | Claim holds for D, fails for G |
| **G4** | Discriminator: 5 × Conv1D, filters 4/8/16/32/64, kernel 3, all ReLU | p. 4, p. 6; SupT2 | `:73-95` (drug), `:175-197` (target) | **MATCH** | Exactly 4/8/16/32/64, `kernel_size=3`, `strides=1`, `padding='same'`, `activation='relu'` | Strides/padding are code-only (paper [OPEN]) |
| **G5** | Discriminator head: Flatten → Dense(1) with **tanh** | p. 4 | `:93-94` | **MATCH** | `Flatten()` → `Dense(1, activation='tanh')` | |
| **G6** | BLOSUM-protein discriminator also 4/8/16/32/64 | p. 6; SupT2 | `:280-300` | **MISMATCH** | `ganForBlosumTarget` uses **4, 8, 16, 20, 40** | Variant-specific; paper reports one filter list for all |
| **G7** | Generator final filter count = 1 | p. 6; SupT2 | `:274` | **MISMATCH** (BLOSUM GAN) | `ganForBlosumTarget` generator ends `Conv1DTranspose(20, ...)` | Resolves paper ambiguity A8 — code uses 20 channels to match 20-dim BLOSUM |
| **G8** | Adversarial training occurs | p. 4 | `:103-132`, `:136-152` | **MATCH** | Real adversarial loop: `dis_v.train_on_batch(imgs, real)`, `dis_v.train_on_batch(gen_imgs, fake)`, then `gan_v.train_on_batch(z, real)` | Genuine alternating optimisation, not a stub |
| **G9** | GAN objective / loss | **[OPEN]** in paper | `:137-139`, `:144-146` | **MISSING in paper; code defines it** | `loss='binary_crossentropy'`, `optimizer=Adam()` for both D and combined GAN | Code establishes it de novo |
| **G10** | Real/fake labels | **[OPEN]** | `:107-108` | — | `real = np.ones((bs,1))`, `fake = np.zeros((bs,1))` | Paired with a **tanh** output head (range −1..1) under BCE |
| **G11** | Noise dimension / distribution | **[OPEN]** | `:134`, `:114` | — | `zdim = 100`; `np.random.normal(0, 1, (bs, 100))` | |
| **G12** | GAN iterations / batch size | **[OPEN]** | `:152`, `:256`, `:354` | — | `train(5000, 5, 500, XD)`; `train(5000, 10, 500, XT)`; `train(5000, 10, 100, XT)` → **5000 iterations, batch 5 (drug) / 10 (protein)** | Paper's batch 256 is the DTA stage only |
| **G13** | D:G update ratio | **[OPEN]** | `:118-123` | — | 2 D updates (real, fake) per 1 G update | |
| **G14** | Input scaling to GAN | not described | `:105`, `:209` | **Code-only** | drug `X/32.5 − 1.0`; protein `X/12.5 − 1.0`; BLOSUM **unscaled** | Maps label codes (1..65 / 1..25) to ≈[−1,1] |
| **T1** | Learned DCGAN reused for feature extraction | p. 4 | `:366, 377, 411, 420, 453` | **PARTIAL** | One layer spliced: `gan.layers[-3]` | See T2–T4 |
| **T2** | **Which** network is reused | Fig. 1 implies discriminator; prose says "the learned models" | `:132, 236, 344` return `dis_v` | **MATCH** (Fig. 1 reading) | **Discriminator only.** Generator is discarded after pretraining. | Resolves paper ambiguity |
| **T3** | **Which layer** is transferred | **[OPEN]** | `:366` etc. | — | `layers[-3]` = the **last Conv1D** of the discriminator: `Conv1D(64)` for drug/target GANs, `Conv1D(40)` for the BLOSUM GAN | Sequential indices: `[Reshape, C4, C8, C16, C32, C64, Flatten, Dense]` |
| **T4** | Transferred layer is dimensionally valid in the DTA model | implied | `:76` vs `:365-366`; `:180` vs `:375-377` | **MISMATCH** | Layer built with **in_channels = 1**; applied to **32-channel** Embedding output. Affects drug branch of A, B **and** C, and protein branch of A. | Variant B protein path (20→20) is the only consistent transfer |
| **T5** | Frozen vs trainable | **[OPEN]** | `:142, 246, 344` | **UNCLEAR** | `dis_v.trainable = False` set before `build_gan`; Keras propagates recursively, so the spliced layer would be non-trainable. Never set explicitly on the transferred layer. | [INFERRED] |
| **T6** | Pretraining once vs repeatedly | **[OPEN]** | `:781` / `:654` inside the `param3ind` loop | — | `build_GAN_*` (which calls `ganForDrug`/`ganForTarget`) is invoked **per grid point per fold per CV pass** → GAN retrained every iteration | With the README grid (2×3×3 × 5 folds × 2 passes) = 180 pretrainings |
| **P1** | CNN block: 3 conv layers | p. 4 | `:367-372`, `:378-383` | **MATCH** | 3 stacked `Conv1D` per branch | |
| **P2** | Filters 128, 256, 384 | p. 6; SupT2 | `:367-372` | **MATCH** (conditional) | `NUM_FILTERS`, `×2`, `×3` — yields 128/256/384 **iff** `--num_windows 128` | README example passes `--num_windows 128 32`, which also sweeps 32/64/96 |
| **P3** | Filter length 4 (drug) / 8 (protein) | p. 6; SupT2 | `:367` (`FILTER_LENGTH1`), `:378` (`FILTER_LENGTH2`) | **MATCH** (conditional) | Bound to `--smi_window_lengths` / `--seq_window_lengths`; README sweeps `4 8 16` for both | Paper's single values are one point in that sweep |
| **P4** | One max-pooling layer | p. 4 (prose); absent from Fig. 1 | `:373`, `:384` | **MATCH** | `GlobalMaxPooling1D()` per branch | Resolves paper ambiguity A2 in favour of the prose |
| **P5** | Add merge (not concatenation) | p. 4; Fig. 1 | `:386`, `:429`, `:472` | **MATCH** | `keras.layers.Add()([encode_smiles, encode_protein])` | |
| **P6** | FC widths **1024/512/512** (prose, SupT2) vs **1024/1024/512** (Fig. 1) | p. 6; SupT2; Fig. 1 | `:389-393` | **MISMATCH vs prose/SupT2; MATCH vs Fig. 1** | `Dense(1024) → Dropout → Dense(1024) → Dropout → Dense(512)` | Conflict A1 resolved: Fig. 1 is right |
| **P7** | Dropout 0.25 | p. 6; SupT2 | `:390, 392` | **MATCH** | After FC1 and FC2 | Placement is code-only |
| **P8** | Single output node | Fig. 1 | `:395` | **MATCH** | `Dense(1, kernel_initializer='normal')`, no activation | |
| **P9** | Prediction loss | **[OPEN]** | `:397` | **MISSING in paper; code defines it** | `loss='mean_squared_error'`, `optimizer='adam'`, `metrics=[cindex_score]` | Keras `'adam'` default lr = 0.001, coincides with paper |
| **R1** | SMILES label encoding, embedding | p. 4 | `datahelper.py:64-68`, `run_exp:365` | **MATCH** | `label_smiles` over `CHARISOSMISET` (65 symbols); `Embedding(charsmiset_size+1, output_dim=32)` | Vocabulary is **isomeric**-SMILES (paper [OPEN]); embedding dim 32 is code-only |
| **R2** | Max SMILES length 200, protein 2000 | p. 6; SupT2 | CLI `--max_smi_len/--max_seq_len`; **hardcoded** in `dataset.py:27-33` | **PARTIAL** | `DataGenerator` hardcodes `(bs,200)` and `(bs,2000,20)`, ignoring the flags on the B/C path | Agrees with paper only when flags = 200/2000 |
| **R3** | Padding to fixed length | p. 4 | `datahelper.py:64-77` | **MATCH** | `np.zeros(MAX_LEN)` pre-allocated, filled left-to-right → **post-padding with 0** | Pad value/side is code-only |
| **R4** | Truncation of over-length inputs | not mentioned in paper | `datahelper.py:66, 74` | **Code-only** | `line[:MAX_LEN]` — silent head truncation | Confirms Stage 2 inference A15 |
| **R5** | BLOSUM = **25**-dim per residue | p. 4 (stated twice) | `protein_feature_vecblsm.json`; `dataset.py:12`; `run_exp:406, 449` | **MISMATCH** | **20**-dim, 22 symbols | Paper ambiguity A7 resolves against the paper |
| **R6** | Variant A = label+embedding both branches | SupT3 | `build_GAN_A:361-377` | **MATCH** | Both branches `Embedding(…, 32)` then transferred layer | |
| **R7** | Variant B = BLOSUM protein + DCGAN; label drug + DCGAN | SupT3 | `build_GAN_B:406-420` | **MATCH** | `XTinput (L,20)` → `gan_protein.layers[-3]`; drug via Embedding → `gan_smiles.layers[-3]` | |
| **R8** | Variant C = BLOSUM protein, **no** protein DCGAN; drug keeps DCGAN | SupT3 | `build_GAN_C:446-462` | **MATCH** | `ganForDrug` still called (`:446`); protein goes straight into `Conv1D` | Confirms SupT3's merged-cell reading |
| **D1** | BindingDB 9864 × 1088, 42,203 interactions | p. 3; SupT1 | `data/bindingdb/` | **MATCH** | ligands 9864, proteins 1088, `Y` (9864,1088) with **42,203** non-NaN | [VERIFIED] by loading the files |
| **D2** | PDBbind 4231 × 1606, 5,014 interactions | p. 3; SupT1 | `data/pdb/` | **MATCH** | ligands 4231, proteins 1606, `Y` (4231,1606) with **5,014** non-NaN | [VERIFIED] |
| **D3** | Log-transform of affinity | p. 3 (formula [OPEN]) | `datahelper.py:152-156` | **PARTIAL** | `is_log==1` → `−log10(Y/1e9)`; `is_log==2` → `−log10((Y+1)/1e9)`; `is_log==0` → none | PDBbind `Y` is **already** pK (2.0–11.92) and README uses `--is_log 0`; BindingDB `Y` is raw (0–1e7) with `--is_log 2` |
| **D4** | Missing-value handling | **[OPEN]** | `run_exp:949-950` | — | `np.where(np.isnan(Y) == False)` → only observed pairs enter training/eval | NaN = missing, never imputed |
| **D5** | "Data harmonization" / redundancy removal | p. 3 | — | **MISSING** | No preprocessing, dedup or filtering code exists; `Y`, `ligands.txt`, `proteins.txt` ship pre-built | Preprocessing is unauditable from the repo |
| **D6** | Drug SMILES file named `ligands_can.txt` | README (repo) | `datahelper.py:143,146` | **MISMATCH** (README ↔ code) | Code reads `ligands.txt`; no `ligands_can.txt` exists in either dataset dir | Repo-level, not paper-level |
| **S1** | Fivefold cross-validation | p. 6 | `datahelper.py:124-134`; `run_exp:488-506` | **MATCH** | Loads `folds/train_fold_setting{N}.txt` (5 folds) + `folds/test_fold_setting{N}.txt`; LOO over the 5 train folds | Folds are **predefined files**, not generated |
| **S2** | Fold index integrity | — | `data/*/folds/` | **MATCH** | pdb 5×836+834 = 5014 = non-NaN count, max idx 5013; bindingdb 5×7034+7033 = 42,203, max idx 42,202 | [VERIFIED] |
| **S3** | "five nearly equal-sized training and validation sets"; test set undefined | p. 6 (ambiguity A4) | `run_exp:498-506, 507-540` | **MISMATCH vs paper description** | A **separate test fold exists** on disk and is used. The CV routine is run twice: once on `val_sets` (predictions discarded → only `bestparamind` kept) and once on `test_sets` (predictions reported). | Paper never describes the test fold or the two-pass structure |
| **S4** | Hyperparameter selection on validation | p. 6 | `:507-518` vs `:551` | **PARTIAL** | `bestparamind` does come from the `val_sets` pass. But the reported per-fold values come from the `test_sets` pass, in which early stopping and epoch selection also use the test set. | |
| **S5** | Early stopping | p. 6 (details [OPEN]) | `:787-788` | — | `EarlyStopping(monitor='val_cindex_score', mode='max', patience=75, restore_best_weights=True)` | Monitors whichever set was passed as `validation_data` |
| **S6** | Fold aggregation | **[OPEN]** | `:551-566` | — | Per fold: `rperf = max(history['val_cindex_score'])` over epochs; `loss = val_loss` at that epoch. Then **mean over the 5 test folds**, plus `np.std` | Reported CI = mean over folds of (max over epochs) |
| **S7** | `problem_type` selects cold-start mode | README (repo): 2 = Open Babel logP, 3 = XLOGP3 | `datahelper.py:131-132` | **MISMATCH** | `problem_type` only interpolates into the fold **filename**. No logP logic exists anywhere. | `data/bindingdb/folds/` has only `setting1`, so `--problem_type 2` there is a `FileNotFoundError` |
| **E1** | CI per Eq. (1) | Eq. (1)–(2), p. 6 | `emetrics.py:27-41` (unused); `run_exp:870-880` (used) | **MISMATCH** | Exact global CI (`get_cindex`) exists but is **never imported**. Reported CI is the TF `cindex_score` Keras metric → averaged over batches. | `run_exp:20` imports only `get_rm2` |
| **E2** | MSE per Eq. (3) | Eq. (3), p. 6 | `:397`, `:801-803` | **MATCH** | Keras `mean_squared_error`; reported value is `val_loss` at the max-CI epoch | |
| **E3** | `r²_m` per Eq. (4) | Eq. (4), p. 6 | `emetrics.py:76-80` | **MATCH** | `r2 * (1 − sqrt(|r2² − r02²|))` — the code adds `np.absolute`, which the paper's Eq. (4) omits | Minor formula hardening |
| **E4** | AUPR with threshold 7 | p. 6 | `:810-820` | **MATCH** | `thresh = 7`; `temp = 1 if val_Y[i] > thresh else 0`; `average_precision_score(temp, labels)` | Resolves ambiguity A11: positive class = affinity **> 7** (strict), same threshold for both datasets |
| **E5** | AUPR uses `auc.jar` | implied by bundled jar | `emetrics.py:5-24` vs `run_exp:21` | **Dead code** | `emetrics.get_aupr` (Java `auc.jar`) is never imported; sklearn `average_precision_score` is used instead | AP ≠ trapezoidal PR-AUC |
| **E6** | Four metrics reported per experiment | p. 6 | `:816-823`, `:551-566` | **PARTIAL** | Only **CI and MSE** are aggregated into `all_predictions`/`all_losses`. **AUPR and rm2 are logged per (param, fold) to `log.txt` and never aggregated.** | Paper's AUPR/RM2 bars must be reconstructed by hand from logs |
| **X1** | Cold-start drug splitting by logP | p. 9; Figs. 5–6 | — | **MISSING** | No `logp`/`xlogp`/`babel` reference in any `.py` | Figs. 5 and 6 not reproducible from this code |
| **X2** | Adversarial controls (3 shuffling settings) | p. 10; Fig. 7 | — | **MISSING** | No affinity-shuffling code; only `shuffle=False` kwargs (`:640, 642, 793`) | Fig. 7 not reproducible |
| **X3** | Merging-layer ablation (Add vs concatenation) | p. 12; Fig. 8 | — | **MISSING** | No `Concatenate` anywhere in the model builders | |
| **X4** | Baselines (7 comparison + FC net + KNN + DeepDTA) | p. 6, p. 12; Figs. 2,3,5,6,8 | — | **MISSING** | Only `build_GAN_A/B/C` exist; no `KNeighbors`, no baseline modules, no result-import path | Baseline provenance (A12) remains [OPEN] |
| **U1** | DCGAN pretrained on UniProt + ChEMBL | p. 4 | `datahelper.py:149-150`; `run_exp:781` | **PARTIAL / UNCLEAR** | Pretraining is real and uses bundled `ligands_train.txt` (50,068) / `proteins_train.txt` (50,202). **No code references UniProt or ChEMBL**; the files are byte-identical across `data/pdb/` and `data/bindingdb/`. | Provenance unverifiable from the repo |
| **U2** | Pretraining corpus is disjoint from evaluation data | implied by "unlabeled data" | `:601-612` | **MISMATCH** (variant B) | Variant B builds `blosum_data` from `np.concatenate((XT, XT_t))[:10000]` — `XT` is the full evaluation protein set (1606/1088), entirely inside that slice | Variants A and C pretrain on `XT_t`/`XD_t` only |
| **H1** | 300 epochs, batch 256, Adam, lr 0.001 | p. 6; SupT2 | `arguments.py:59-77`; `:790-791` | **MATCH** (DTA stage) | CLI defaults `num_epoch=100`, `batch_size=256`, but README passes `--num_epoch 300`; optimizer `'adam'` (Keras default lr 0.001) | Resolves ambiguity A10: these govern the **DTA** stage; the GAN stage uses batch 5/10 and 5000 iterations (G12) |
| **H2** | `--learning_rate` flag honoured | — | `arguments.py:52-57` | **Dead flag** | Declared, never referenced; both stages use bare `Adam()` / `'adam'` | |
| **H3** | Checkpointing | not claimed | — | **MISSING** | No `model.save`, `save_weights`, or `ModelCheckpoint` anywhere | Only `log.txt`, `figures/*.png|tiff` persist |
| **H4** | Random seed | **[OPEN]** in paper | `:26-30` | **PARTIAL** | `np.random.seed(1)`, `rn.seed(1)`, `tf2.set_random_seed(0)` set at import. No `PYTHONHASHSEED`, no per-fold reseeding, no GPU determinism flags | Code provides more than the paper states |

---

## C. Critical Discrepancies

### C1 — Transferred discriminator layer has incompatible input channels — **CRITICAL**

- **Paper says:** the learned DCGAN model is reused for feature extraction downstream of the embedding layer (p. 4; Fig. 1, p. 5).
- **Code does:** `ganForDrug` builds its discriminator on `Reshape((200, 1))` (`run_experiments.py:76`), so `Conv1D(64)` at `:90` acquires a kernel of shape `(3, 1, 64)`. `build_GAN_A/B/C` then call that same built layer on `Embedding(..., output_dim=32)` output of shape `(None, 200, 32)` (`:365-366`, `:410-411`, `:452-453`). Variant A's protein branch repeats this (`:180` builds on 1 channel; `:376-377` applies to 32).
- **Why it matters:** this is the method's namesake mechanism. In TF2 Keras a built `Conv1D` carries `InputSpec(axes={-1: input_channel})`, so re-invocation on a different channel count raises `ValueError`. If so, variants A, B and C cannot construct at all; if some environment tolerates it, the transferred weights are not the pretrained ones. Either way the published warm-start numbers cannot be traced to this code path as written.
- **Evidence:** [VERIFIED] shapes above. [INFERRED] that Keras rejects it (not executed, per audit scope).
- **Severity:** CRITICAL

### C2 — Three of five experiment families are absent — **CRITICAL** (for reproduction scope)

- **Paper says:** cold-start logP splitting (Figs. 5–6), adversarial controls on shuffled affinities (Fig. 7), lightweight baselines and the Add-vs-concatenation ablation (Fig. 8), plus seven comparison methods (Figs. 2–3).
- **Code does:** implements only warm-start CV for `build_GAN_A/B/C`. Exhaustive greps find no logP logic, no shuffling, no `Concatenate`, no baseline models, and no mechanism for importing external baseline results.
- **Why it matters:** 4 of the paper's 8 figures and every non-DCGAN-DTA bar in the other 4 have no code path. Baseline provenance (paper ambiguity A12) cannot be resolved from the repository.
- **Evidence:** [VERIFIED]
- **Severity:** CRITICAL

### C3 — Reported CI is a per-batch average, not Eq. (1) — **HIGH**

- **Paper says:** CI per Eq. (1), a global pairwise concordance over the evaluation set.
- **Code does:** `run_experiments.py:20` imports only `get_rm2`. The exact `emetrics.get_cindex` is never called. The reported CI is `max(history['val_cindex_score'])`, where `cindex_score` (`:870-880`) is a Keras `metrics=` entry evaluated per batch and averaged — with `batch_size=256`, this is a mean of ~165 (PDBbind) or ~165 (BindingDB test fold) within-batch CIs.
- **Why it matters:** the headline metric in every figure is a batch-size-dependent approximation. Reproduced values will not match a global-CI recomputation, and CI comparisons against baselines computed globally are not on the same scale.
- **Evidence:** [VERIFIED]
- **Severity:** HIGH

### C4 — Evaluation-set-informed epoch selection in the reported pass — **HIGH**

- **Paper says:** "The model was trained and hyperparameter tuning was performed using these [validation] sets" (p. 6); a test set is never described.
- **Code does:** `nfold_1_2_3_setting_sample:507-540` runs the CV twice. The first pass uses `val_sets` and its predictions are bound to `all_predictions_not_need` — discarded except for `bestparamind`. The second pass passes `test_sets` as `validation_data`, so `EarlyStopping(monitor='val_cindex_score', restore_best_weights=True)` and `rperf = max(history['val_cindex_score'])` both operate on the test fold. All reported numbers come from this second pass.
- **Why it matters:** the reported per-fold CI is the maximum over 300 epochs of the test-set metric, with weights restored to that epoch. Any reproduction must replicate this exact protocol to land on the published values.
- **Evidence:** [VERIFIED]
- **Severity:** HIGH

### C5 — BatchNorm present despite an explicit claim of removal — **HIGH**

- **Paper says:** "while the original DCGAN integrates Batch Normalization … our implementation forgoes this approach due to observed performance degradation" (p. 4).
- **Code does:** `BatchNormalization()` after generator conv layers 1 and 2 in all three GANs (`:62, 66`; `:164, 168`; `:268, 272`). Discriminators contain none.
- **Why it matters:** the claim is specific and justified in the paper, which makes it a load-bearing design statement. It holds for the discriminator and fails for the generator.
- **Evidence:** [VERIFIED]
- **Severity:** HIGH

### C6 — BLOSUM is 20-dimensional, not 25 — **HIGH**

- **Paper says:** each amino acid → a 25-dimensional vector; protein = 25 × L matrix (p. 4, stated twice, repeated in the Fig. 1 caption).
- **Code does:** `protein_feature_vecblsm.json` = 22 symbols × 20 dims; `dataset.py:12` `self.pro_emb = 20`; `dataset.py:28` `np.empty((bs, 2000, 20))`; `build_GAN_B/C` declare `Input(shape=(max_seq_len, 20,))`. Three amino-acid codes in `CHARPROTSET` (`B`, `O`, `U`) are absent from the BLOSUM table; `general_nfold_cv:605-612` catches the resulting `KeyError` and substitutes a 20-zero vector, but `DataGenerator.get_pro_vec` (`dataset.py:45-55`) has no such guard.
- **Why it matters:** fixes the protein feature dimensionality for any reimplementation and resolves paper ambiguities A7 and A8 (the BLOSUM generator's final filter count is 20, matching 20 channels — not the 1 the paper states).
- **Evidence:** [VERIFIED]
- **Severity:** HIGH

### C7 — Variant B pretrains its protein GAN on the evaluation proteins — **HIGH**

- **Paper says:** the DCGAN is trained on unlabeled data from external databases (p. 4), framed throughout as separate from the labelled DTA data.
- **Code does:** `run_experiments.py:601-612` sets `new_XT = np.concatenate((XT, XT_t), axis=0)` and iterates `for i in range(10000)`. `XT` is the complete evaluation protein set (1606 PDBbind / 1088 BindingDB), so every evaluation protein is inside the pretraining slice. `ganForBlosumTarget(blosum_data)` then trains on it.
- **Why it matters:** for variant B the GAN feature extractor has seen all test-fold proteins. Variants A and C do not have this property (they use `XT_t`/`XD_t` only). The `10000` bound is also hardcoded and unrelated to either array's size.
- **Evidence:** [VERIFIED]
- **Severity:** HIGH

### C8 — FC widths contradict the paper text and Supplementary Table 2 — **MEDIUM**

- **Paper says:** 1024, 512, 512 (p. 6 and Sup Table 2) — but Fig. 1 draws 1024, 1024, 512.
- **Code does:** `Dense(1024) → Dropout(0.25) → Dense(1024) → Dropout(0.25) → Dense(512)` in all three builders (`:389-393`, `:434-438`, `:477-481`).
- **Why it matters:** resolves Stage 2 ambiguity A1 — the figure is correct and the prose and supplementary table are both wrong. A reimplementation following Sup Table 2 would build a different network.
- **Evidence:** [VERIFIED]
- **Severity:** MEDIUM

### C9 — AUPR and RM2 are computed but never aggregated — **MEDIUM**

- **Paper says:** four metrics reported per experiment (Figs. 2–3 show AUPR and RM2 bars).
- **Code does:** `aupr` and `rm2` are computed per (param, fold) and written to `log.txt` (`:816-823`), but only CI and MSE enter `all_predictions`/`all_losses` and the `avgperf`/`avgloss`/`teststd` aggregation (`:551-566`).
- **Why it matters:** the published AUPR and RM2 values must have been extracted from log text by hand; there is no code path that produces the aggregated numbers in the figures, so the aggregation rule for those two metrics remains unestablished.
- **Evidence:** [VERIFIED]
- **Severity:** MEDIUM

### C10 — `problem_type` does not implement cold-start — **MEDIUM**

- **Paper says (via the repo README):** `--problem_type 2` selects Open Babel logP cold-start, `3` selects XLOGP3.
- **Code does:** `datahelper.py:131-132` interpolates `problem_type` into `folds/test_fold_setting{N}.txt` / `train_fold_setting{N}.txt` and nothing else. `data/pdb/folds/` ships settings 1–3; `data/bindingdb/folds/` ships only setting 1.
- **Why it matters:** setting 2 and 3 on PDBbind silently run the ordinary pipeline on different fold files whose construction rule is undocumented; on BindingDB they raise `FileNotFoundError`. The logP threshold (Stage 2 item #10) cannot be recovered from code — only from the fold files themselves.
- **Evidence:** [VERIFIED]
- **Severity:** MEDIUM

---

## D. Reproduction-Critical Unknowns (still open after code inspection)

| # | Unknown | Why unresolved |
|---|---|---|
| **U1** | Whether variants A/B/C construct at all under any TF/Keras version, given C1 | Requires execution, excluded from this stage |
| **U2** | Provenance of `ligands_train.txt` (50,068) and `proteins_train.txt` (50,202) | No code, script, URL or manifest links them to UniProt/ChEMBL; files are identical across both dataset dirs |
| **U3** | Whether the published numbers were produced by this code | Given C2, most figures have no code path; no run scripts, configs, logs or result files ship with the repo |
| **U4** | Baseline provenance — rerun vs transcribed | No baseline code and no result-import path exists |
| **U5** | Whether the transferred layer is frozen in practice | `dis_v.trainable = False` propagation is standard Keras semantics but never set explicitly on the spliced layer (T5) |
| **U6** | How `test_fold_setting2/3.txt` were constructed (the logP rule) | The fold files exist; the generating code does not |
| **U7** | Aggregation rule behind the published AUPR and RM2 bars | C9 — computed per fold, logged, never aggregated |
| **U8** | Preprocessing that produced `Y`, `ligands.txt`, `proteins.txt` | D5 — no preprocessing code ships; the harmonization and redundancy-removal claims are unauditable |
| **U9** | Which `--num_windows` / window-length point produced the published numbers | P2/P3 — code sweeps a grid; the paper reports a single configuration |
| **U10** | Whether `DataGenerator.get_pro_vec` ever hits `B`/`O`/`U` in these datasets | C6 — the guard is absent there but present in the `blosum_data` builder |

---

## E. Confirmed Implementation Facts

Established directly from source or from inspecting the bundled data files:

1. **Entry point** `run_experiments.py:978` → `argparser()` → `run_regression()` (`:970`) → `experiment()` (`:928`) → `nfold_1_2_3_setting_sample()` (`:486`) → `general_nfold_cv2` (model A) or `general_nfold_cv` (models B/C).
2. **GAN training is genuine adversarial optimisation** — 5000 iterations of {D on real, D on fake, G via frozen-D combined model}, BCE loss, `Adam()` defaults, `zdim=100`, N(0,1) noise, batch 5 (drug) / 10 (protein).
3. **Only the discriminator survives pretraining**; `ganForDrug/Target/BlosumTarget` all `return dis_v`. The generator is discarded.
4. **Exactly one layer is transferred** — `layers[-3]`, the final Conv1D of the discriminator: `Conv1D(64)` for the drug and label-encoded-protein GANs, `Conv1D(40)` for the BLOSUM GAN.
5. **GAN pretraining is re-run inside the innermost grid/fold loop**, not once — `build_GAN_*` is `runmethod` at `:781` / `:654`.
6. **DTA head:** `Add()` merge → `Dense(1024, relu)` → `Dropout(0.25)` → `Dense(1024, relu)` → `Dropout(0.25)` → `Dense(512, relu)` → `Dense(1)`; compiled with `'adam'` + `mean_squared_error` + `cindex_score`.
7. **Feature extractor:** 3 × `Conv1D(NUM_FILTERS × {1,2,3}, padding='valid', strides=1)` → `GlobalMaxPooling1D()` per branch.
8. **Folds are predefined JSON files**, loaded by `DataSet.read_sets` (`datahelper.py:124-134`); index sets exactly cover the non-NaN entries of `Y` in both datasets.
9. **The CV routine runs twice**; reported numbers come from the `test_sets` pass, in which early stopping and epoch selection also use the test fold.
10. **Reported CI** = mean over 5 test folds of (max over epochs of the per-batch-averaged Keras `cindex_score`). **Reported MSE** = `val_loss` at that epoch. `np.std` over folds is computed and logged though the paper reports none.
11. **AUPR** = `sklearn.average_precision_score` with positive class `affinity > 7` (strict), identical threshold for both datasets. `emetrics.get_aupr` + `auc.jar` are dead code.
12. **BLOSUM features are 20-dimensional** over 22 symbols; `B`, `O`, `U` are absent from the table.
13. **Truncation is silent head-truncation** (`line[:MAX_LEN]`); padding is post-padding with 0.
14. **SMILES vocabulary is `CHARISOSMISET`** (65 symbols, isomeric); protein vocabulary is `CHARPROTSET` (25 symbols). Both embeddings use `output_dim=32`.
15. **Dataset counts match Supplementary Table 1 exactly** — BindingDB 9864/1088/42,203; PDBbind 4231/1606/5,014.
16. **Seeds are set** (`np.random.seed(1)`, `rn.seed(1)`, `tf2.set_random_seed(0)`) — more than the paper documents, but with no per-fold reseeding or GPU-determinism configuration.
17. **Nothing is persisted but logs and figures** — no checkpoint, no weights, no predictions file.
18. **`--learning_rate`, `--checkpoint_path`, `--num_hidden`, `--num_classes`, `--binary_th` are declared but never read.**

---

## F. Recommended Next Step (Stage 4 scope — not performed here)

Stage 4 should be a **minimal, instrumented construction-and-smoke-execution study**, not a full reproduction. Specifically, it should determine:

1. **Whether `build_GAN_A/B/C` construct at all** (C1). Build each variant in isolation under one or more candidate TF/Keras versions and record the exact exception or the resulting layer shapes. This single result determines whether anything downstream is worth pursuing.
2. **The viable TF/Keras version window.** `Conv1DTranspose` requires TF ≥ 2.3; `fit_generator`/`predict_generator` (`:656, 659`, models B/C) are removed in Keras 3. Pin the window empirically and record it.
3. **The actual trainability of the transferred layer** (T5/U5) — inspect `model.trainable_weights` after construction rather than reasoning from Keras semantics.
4. **The magnitude of the CI discrepancy** (C3) — recompute global `emetrics.get_cindex` alongside the per-batch `cindex_score` on identical predictions at batch 256, and quantify the gap on both datasets.
5. **Whether `data/pdb/folds/test_fold_setting{2,3}.txt` are drug-disjoint** (U6/C10) — check the fold index sets against `label_row_inds` to see whether setting 2/3 partition by drug, and whether the implied logP grouping can be recovered by correlating the split with computed logP values.
6. **Whether `DataGenerator.get_pro_vec` can raise** on `B`/`O`/`U` (U10) — scan `proteins.txt` and `proteins_train.txt` for those codes; this is a static data check, not an execution.
7. **The BindingDB pKd ≈ 5 mass** (Stage 2 item #7) — quantify exactly how many of the 42,203 entries sit at that value and how they distribute across the 6 fold files, since roughly half the dataset at one value governs what CI and MSE mean there.
8. **The cost envelope of one honest run** — with GAN pretraining re-executed per grid point per fold per pass (T6), estimate the iteration count implied by the README grid before committing to any reproduction.

Stage 4 should **not** attempt to reproduce Figs. 5–8 or any baseline bar: per C2 there is no code path for them, so those targets need either author contact or independent reimplementation, which is a separate decision for you to make.

---

*End of Stage 3. Read-only: no repository file was modified, no dependency installed, no experiment executed.*

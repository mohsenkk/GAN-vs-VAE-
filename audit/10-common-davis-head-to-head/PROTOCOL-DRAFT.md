# Stage 10A — Common Davis head-to-head protocol draft

**Date:** 2026-09-28. **Status: protocol design IN PROGRESS; DRAFT, NOT FROZEN. READY TO FREEZE: NO.**

Stage 09A and 09B remain CLOSED / PASS and frozen historical evidence. No Stage 10 training, benchmark, runner, evaluator, final split artifact, or representation implementation is authorized or created by this document. Stage 11+ has not started. The methodological decisions in Section 16 require review before freezing this protocol.

Evidence labels: **DATA** = inspected files; **CODE** = static source; **PRIOR RUNTIME** = accepted earlier execution; **PROPOSAL** = a decision for review; **ESTIMATE** = conditional arithmetic, not a new measurement. Paths resolve from this document. Static file diagnostics used standard-library parsing and hashing only; neither baseline nor a training framework was imported.

## 1. Scientific objective and primary setting

**PROPOSAL:** Davis **new-drug / cold-drug**, with one shared drug-level train/validation/test partition. Ask: “When DCGAN-DTA and Co-VAE are trained and evaluated on exactly the same Davis prediction task, under one predeclared split and one common evaluation protocol, how do their predictive results compare?” There is no numerical answer yet. This is a **head-to-head comparison of two predeclared baseline configurations**, not paper reproduction or best achievable performance.

Preserve architecture, GAN/VAE mechanism, objective and native optimization behavior. Share entity/pair identities, labels, supervised membership, final held-out examples and external scoring. Drug coldness must distinguish supervised identity separation from unsupervised pretraining exposure; the native DCGAN corpus violates the latter (Section 11). Do not silently switch to pair-wise splitting.

## 2. Evidence, repository history and implementation feasibility

Relevant prior findings were traced across the [paper audits](../02-paper-forensic/), [paper/code audits](../03-paper-code-traceability/), [dataset audits](../04-dataset-fold/), [feasibility](../05-reproduction-feasibility/), [smoke](../06-smoke-execution/), [training feasibility](../07-training-feasibility/), [controlled validation](../08-controlled-training-validation/), [Kaggle validation](../08-5-kaggle-gpu-validation/), [Stage 09 plan](../09-baseline-reproduction/PLAN.md) and accepted [09A](../09-baseline-reproduction/DCGAN-DTA.md)/[09B](../09-baseline-reproduction/Co-VAE.md) closures. Earlier environment limitations and the withdrawn DCGAN transfer-channel allegation are not current architectural blockers: Stage 05 Section 5.3 corrected that allegation, and Stage 09A executed the 128/4/8 path.

**CODE / DATA:** DCGAN ships only PDBbind and BindingDB task directories, not a labeled Davis task. Its `DataSet.parse_data` expects `ligands.txt`, `proteins.txt`, `Y`, auxiliary feature files and pretraining corpora; `read_sets` expects shipped fold files ([datahelper.py](../../DCGAN-DTA/DCGAN-DTA/datahelper.py), lines 124–156). Co-VAE ships Davis and generates drug folds at runtime ([run_experiments.py](../../Co-VAE/CoVAE/run_experiments.py), lines 31–72). Neither released top-level driver can simply be run unchanged to enforce this common protocol.

**PROPOSAL:** later external adapters must feed the frozen canonical roles directly into native pair preparation/DataLoaders or DCGAN `DataGenerator`, then call native `build_GAN_C`/training or Co-VAE `net`/`train`/`test`. Do not independently regenerate folds, invoke the released grids/test-informed selection drivers, repurpose PDBbind folds, or patch either baseline. DCGAN's [pair helper](../../DCGAN-DTA/DCGAN-DTA/run_experiments.py), lines 909–926, and [DataGenerator](../../DCGAN-DTA/DCGAN-DTA/dataset.py), lines 9–57, consume provided arrays and labels without dataset-name-dependent model architecture. This establishes **static feasibility with an external adapter**, not an executed Davis DCGAN loader/forward pass. Adapter implementation and a separately authorized bounded preflight remain future prerequisites.

GitNexus symbol context corroborated the drug-GAN call and native fold path. Keyword query reported missing FTS indexes, so source inspection was authoritative; no repair, installation or reindexing occurred. Graph results are static relationships, not evidence of a new Davis execution.

**Anti-cherry-picking search:** current documentation/scripts, the Stage 09 plan, and relevant tracked history contain a common-Davis intention but **no predeclared Stage 10 seed, partition or role mapping**. Root HEAD is `d79b452e38b164cf8bb2a254861efe5ce934bc93`; its Stage 09 plan was committed at **2026-09-28 12:00:40 +03:30**, before Stage 09B's first resource sample at **2026-09-28 14:17:18 UTC**, and names no Stage 10 split. The seed 1000 in that plan is specifically Stage 09B, not a prior Stage 10 decision. This is limited to available local files/refs, not unrecorded discussions. Existing working-tree changes are distinct from that commit.

## 3. Canonical Davis interaction universe

**DATA:** use the unchanged Co-VAE Davis files as the sole labeled universe. JSON insertion order is the matrix axis order; do not numerically sort IDs, canonicalize strings, merge duplicate targets or apply KIBA filtering.

| Canonical input | SHA-256 |
|---|---|
| `Co-VAE/CoVAE/data/davis/ligands_iso.txt` | `9789900f1e723226187215235b379998e5735a3ffec0e9487bad27c1c6a64161` |
| `Co-VAE/CoVAE/data/davis/proteins.txt` | `187cedeee10cd3b58af915a2861a5ed0c4ff42a2d728f4dea5e0b47e3d75b291` |
| `Co-VAE/CoVAE/data/davis/drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt` | `31b3f3efea9afe53239fa9fe735aedd6195586b99261319917b55b1d1c9e8392` |

| Property | Canonical / Co-VAE Davis | DCGAN evidence and proposed consumption |
|---|---|---|
| Entities / ordering | **68 drugs / 442 target IDs** in file order | Pretraining corpus prefixes reproduce the same 68 drug and 442 target IDs **and strings in order**; this is not a shipped labeled Davis dataset |
| Interaction universe | **68 × 442 = 30,056 observed pairs**, no missing entries | External adapter must consume this exact universe; existing PDBbind/BindingDB `Y` matrices cannot substitute |
| Raw labels | Kd in **nM**, range **0.016–10,000**, all finite and positive | No DCGAN Davis-specific matrix exists to cross-check; transform equivalence is established from source formulas, with runtime equality to be checked later |
| Matrix axes | Row = drug index; column = target index | Resolve native tensors by canonical entity IDs/indices, never by coincidental independent order |
| Filtering / missingness | No Davis length filter; 0 NaN; 20,931 values at Kd 10,000 | Do not drop censored values, invent negatives, impute or remove truncated entities; a changed hash/shape/missingness stops preparation |

**PROPOSAL:** `drug_index=0..67` and `target_index=0..441` refer permanently to those original ordered files. Define `canonical_pair_id = davis:v1:d{drug_index:03d}:t{target_index:03d}` and canonical order by the two numeric indices. Entity maps also retain original strings, lengths and hashes. The future full pair manifest must contain `canonical_pair_id, drug_id, target_id, drug_index, target_index, raw_Kd_nM, canonical_pKd, split_role, drug_fold`. Version/schema/source hashes must accompany it. Use UTF-8 without BOM, LF, a fixed column order and round-trippable numeric serialization; hash the actual bytes. Export train/validation/test role manifests from this single authority, not separate baseline split functions.

## 4. Proposed split algorithm and seed policy

**PROPOSAL:** split seed **20260928**, the Stage 10A design date written as YYYYMMDD. It is new, model-independent, and chosen by this rule rather than model performance. Only this one candidate was inspected, solely for deterministic membership/count/exposure checks; no model predictions, label-stratified seed search or alternative seed sweep was performed. Do not change it because a partition looks inconvenient. The seed, algorithm, role mapping, manifest hashes and approved protocol must be frozen in a dated Git commit **before either model trains**. This task does not commit or freeze them.

Version-stable algorithm `sha256-drug-rank-v1`:

1. Read each original drug ID as its unchanged JSON string. For seed `20260928`, encode exactly `stage10-davis-cold-drug-v1\nseed=20260928\ndrug_id=` followed by that ID in UTF-8; `\n` means LF and there is no trailing LF.
2. Calculate SHA-256 and sort by digest bytes ascending, breaking a digest tie by the drug-ID UTF-8 bytes ascending. Do not use Python's `hash`, sets, RNG libraries, or `random.sample`.
3. Divide this ranked list into six **contiguous** folds with sizes **[12,12,11,11,11,11]**, giving the remainder to the first folds. Fold indices are zero-based.
4. Folds **0–3 → train**, fold **4 → validation**, fold **5 → final held-out test**. Every pair inherits its drug's role. Canonical output order remains matrix row/column order, not hash rank.

| Role | Drugs | Pairs | Batches at 256 | Final batch |
|---|---:|---:|---:|---:|
| Train | **46** | **20,332** | **80** | **108** |
| Validation | **11** | **4,862** | **19** | **254** |
| Test | **11** | **4,862** | **19** | **254** |

All three drug sets must be pairwise disjoint and cover all 68 IDs. All 442 targets appear in each role; pair sets are disjoint and cover all 30,056 pairs. Validation is drug-cold, matching the final prediction question, and has the same size as test. Reserving a validation fold reduces Co-VAE's training pool relative to Stage 09B; both Stage 10 models receive the same reduced training pool. These sizes are design consequences, not selected from performance.

The proposed test drug set is **not Stage 09B's seed-1000 test set**; it shares only ID `6450551` with that set. A new split cannot erase previous knowledge of the benchmark or prevent some entity overlap. Do not reuse Stage 09 weights, inspect per-drug Stage 09 errors to revise this seed, or describe the new set as wholly previously unseen evidence. No final entity/pair/split files have been generated.

## 5. Label harmonization

**CODE:** Co-VAE's Davis branch always computes `-log10(Kd/1e9)` ([datahelper.py](../../Co-VAE/CoVAE/datahelper.py), lines 136–149), independent of `is_log`. DCGAN applies the same formula only for `is_log=1`; `is_log=0` passes its `Y` through, and `is_log=2` adds 1 nM before the log ([datahelper.py](../../DCGAN-DTA/DCGAN-DTA/datahelper.py), lines 152–156). PDBbind's existing pKd matrix and BindingDB's `+1` handling are not Davis labels. There is no need for clipping or a `+1` here: all Davis Kd values are positive. Keep the censor floor pKd 5 intact.

**PROPOSAL:** compute scientific pKd once in float64: `pKd64 = -log10(raw_Kd_nM / 1e9)`. Freeze the operational **canonical_pKd = IEEE-754 float32(pKd64)**, serialized losslessly (e.g. `.17g` after conversion to a Python float). Retain raw Kd and the transform/dtype version as provenance. Both models train against that exact float32 value; the evaluator promotes it to float64 for accumulation. This avoids framework-specific double-rounding differences while retaining the baseline training precision.

Native Co-VAE loader arrays are float64 before the training tensors become float32. DCGAN `DataGenerator` creates NumPy float64 buffers but the neural model/loss uses float32. Require bit-identical float32 labels at each model boundary. If a native loader is checked later, compare its single transform with `pKd64` within absolute **1e-12** and require equality after float32 conversion. Never log-transform already canonical pKd. An external DCGAN array adapter passes canonical labels directly; a hypothetical raw-Kd loader route would need `is_log=1`, while a pretransformed `Y` route would need `0`. Native Co-VAE's Davis path must receive raw Kd if invoked, not a pKd file mislabeled as raw Davis.

## 6. Fixed model configurations and structural validity

| Policy | DCGAN-DTA | Co-VAE |
|---|---|---|
| Fixed configuration | **Variant C, 128 filters, drug kernel 4, protein kernel 8** | **32 filters, drug kernel 5, target kernel 7, lambda exponent -5** |
| Batch / lengths | **256; 200 / 2000** | **256; 85 / 1200** |
| Representation | Drug label embedding, transferred drug-GAN discriminator layer; 20-dimensional BLOSUM protein input; native additive regression path | 128-dimensional token embeddings, gated variational CNNs, latent sampling, reconstruction/KL terms and native regression |
| Optimization | Native Adam default .001; the CLI learning-rate flag is not consumed | Native Adam default .001; recreated by every native `RE.train` call |
| Initial state | Fresh GAN and fresh DTA fit; no Stage 09 checkpoint/GAN reuse | Fresh model and native initialization; no Stage 09 checkpoint reuse |

**CODE / PRIOR RUNTIME:** DCGAN `build_GAN_C` ([run_experiments.py](../../DCGAN-DTA/DCGAN-DTA/run_experiments.py), lines 446–483) has no drug-count/target-count-sized prediction layer. Retaining 200/2000 keeps its native generator and GAN dimensions intact. Its three valid drug convolutions leave **191** positions before pooling; the protein path leaves **1979**. The transferred 64-channel Conv1D consumes 32 embedding channels; the earlier channel-mismatch allegation was withdrawn. These shapes were executed on PDBbind in Stage 09A, not yet on the adapted Davis inputs.

Co-VAE decoder start lengths are **85−3(5−1)=73** and **1200−3(7−1)=1182**; its odd-kernel padded encoders preserve sequence lengths and decoders restore them ([model.py](../../Co-VAE/CoVAE/model.py), lines 6–81). Davis retained drug/protein code maxima are **49/24**, within embedding sizes **64/25**. The 32/5/7/-5 Davis path was executed in Stage 09B. Proposed trailing batch sizes are greater than 1, avoiding the known `.squeeze()` size-1 concern on this design.

No architectural blocker is identified statically; the missing DCGAN Davis adapter and its runtime verification are integration prerequisites. These configurations are **predeclared references**, not Co-VAE's undisclosed paper-selected winner or either method's optimum. Comparable training-only tuning could answer a stronger question, but would require equal declared search budgets, an untouched outer test, fresh fits and a separately approved protocol. Prior Co-VAE paper-aligned tuning involved about 40 search fits plus a final fit for one outer split; do not spend that budget or change these configurations in response to Stage 10 results.

## 7. Common task with model-native training behavior

**PROPOSAL:** share training/validation/test role manifests and evaluation; preserve these explicit training asymmetries:

- **DCGAN:** one fresh native drug GAN, **5,000 iterations, batch 5**, normalization `token/32.5−1`, then Variant C DTA training for **at most 300 epochs**, 80 updates per complete epoch. Keep native generator ordering, default Adam and validation `val_cindex_score` selection, **patience 75, mode max, min_delta 0**. Save a checkpoint only on strict improvement; ties retain the earliest best epoch. Restore that best checkpoint explicitly before the sole final test traversal, including a run that reaches the epoch cap. Test metrics never influence patience, stopping, configuration or selection. No protein GAN is used for Variant C.
- **Co-VAE:** **100 complete native epochs, 80 updates per epoch = 8,000 updates, 100 Adam constructions**, preserving per-epoch optimizer recreation, native initialization/loss/dropout and training DataLoader shuffle. Save and evaluate **epoch 100**, without early stopping or validation/test checkpoint selection. Withhold the shared validation pairs from training even though this fixed schedule does not use them for selection. Do not call native `RE.test` for periodic validation: its latent sampling and DataLoader iteration can advance RNG and alter later training. No extra validation forwards are needed for the primary fit.
- Freeze native validation pair order/batching for DCGAN; its selection CI remains a native, order/batch-dependent training-policy value. Common global CI is the **reporting** metric, not an undisclosed replacement of the stopping rule. Co-VAE's validation non-use and DCGAN's access to validation labels are material, declared policy differences; the comparison evaluates the two fixed training policies, not equal tuning or equal compute.
- Preserve/record framework RNG controls: DCGAN's source NumPy/Python seed 1 and TensorFlow seed 0; Co-VAE's declared Python/NumPy/Torch CPU/CUDA seed 1000, with launch hash seed 0. Split seed 20260928 is independent of these. No bitwise determinism claim. Native Co-VAE noise remains stochastic during the single final `eval/no_grad` pass; do not average repeated test draws or select an inference seed after looking at results.

Changing Co-VAE to validation-selected checkpoints, or changing both stopping rules to common global CI, would define a different benchmark. Either needs a pre-training amendment. This proposal does not make that change. Future DCGAN full-vector validation diagnostics, if added, must be proven observational with respect to weights, optimizer, callbacks, RNG and generators, and must never open test data.

DCGAN's `DataGenerator` retains ordered indices and a no-op `on_epoch_end`; its constructor's `shuffle=False` argument does not implement shuffling. The native driver and Stage 09A wrapper omit a fit-level shuffle argument. Preserve that native call semantics and record the installed Keras resolved default during the later preflight; do not infer unshuffled training from the generator argument. Validation/test prediction order must remain canonical regardless of training batch scheduling.

## 8. Common external metric definitions

**PROPOSAL:** one versioned evaluator accepts the two frozen, aligned `canonical_pair_id, y_true, y_pred` vectors; all calculations use float64 and the same implementation. It makes no model calls. Record evaluator source hash, library versions, counts, thresholds and metric-definition IDs. Native values remain separately labeled diagnostics.

**Global CI:** for every unordered pair of examples with unequal labels, give 1 for matching label/prediction order, 0 for reversal and **0.5 for exactly tied predictions**. Exclude exactly tied labels. Formally, with `U = {(i,j): i<j, y_i != y_j}`, average `1[(y_i−y_j)(p_i−p_j)>0] + 0.5[p_i=p_j]` over U. Do not use approximate tie thresholds. Every unequal-label unordered pair appears once, irrespective of array permutation; a future efficient implementation must agree with a small all-pairs reference under permutations and label/prediction ties. If U is empty, report undefined, not an invented zero.

Both native `emetrics.get_cindex` functions include only `j<i` with `y_i>y_j`; Co-VAE's vectorized lower triangle and DCGAN's loop therefore share an **order-sensitive** omission, while DCGAN's compiled `cindex_score` also aggregates by batch ([DCGAN metrics](../../DCGAN-DTA/DCGAN-DTA/emetrics.py), lines 27–41; [Co-VAE metrics](../../Co-VAE/CoVAE/emetrics.py), lines 4–18; DCGAN driver lines 871–881). Do not call either the common CI, even after canonical-ID sorting. Sorting by true label could recover full comparable pairs, but arbitrary canonical pair order does not. For example, y=(2,1,3), p=(3,1,2) yields native CI 1/2 and common CI 2/3; reversing those vectors changes native CI to 1 while common CI stays 2/3. This is a toy mathematical demonstration, not a benchmark result. This proposal does not relabel or repair frozen Stage 09 CI values.

**MSE / MAE:** `mean((y−p)^2)` and `mean(abs(y−p))` on the entire identical held-out set, with no batch averaging, rescaling or clipping.

**RM2 — explicit shared repository convention:** let centered sums be `S_y=Σ(y−mean(y))²`, `S_p=Σ(p−mean(p))²`, and `C=Σ(y−mean(y))(p−mean(p))`. Define `q=C²/(S_y*S_p)`, `k=Σ(y*p)/Σ(p²)`, and `s=1−Σ(y−k*p)²/S_y`. The proposed common value is **`q * (1 − sqrt(abs(q² − s²)))`**, recorded as `RM2_repo_squared_difference_v1`. Both inspected repositories use this same squared-difference formula ([DCGAN metrics](../../DCGAN-DTA/DCGAN-DTA/emetrics.py), lines 44–82; [Co-VAE metrics](../../Co-VAE/CoVAE/emetrics.py), lines 20–58). DCGAN additionally concatenates its prediction input; that shape convention must not enter the common evaluator. Flatten once to aligned 1-D vectors. Do not silently replace this by `q*(1−sqrt(abs(q−s)))` or claim equivalence to every metric named RM2 in the literature. Formula choice requires approval at freeze time. Report undefined on zero denominators, and do not clip a negative finite value.

**Optional secondary classification metrics:** define positive as **canonical pKd > 7**, scores as continuous predictions. Report `average_precision_score` as **average precision / AP**, not trapezoidal PR area, and `roc_auc_score` as ROC AUC, using one pinned scikit-learn version for both. Co-VAE's unused `get_aupr` uses trapezoidal `auc(recall,precision)`; DCGAN's active driver uses AP, while its legacy Java helper writes files and is not the active metric path. Do not invoke those helpers or substitute ROC AUC for AUPR. Single-class classification metrics are undefined and explicitly labeled; they are not primary metrics. Undefined primary CI/RM2 prevents a complete primary metric comparison rather than being silently coerced to a favorable value.

## 9. Prediction alignment and held-out access gates

Each final prediction file must include **`canonical_pair_id, drug_id, target_id, drug_index, target_index, y_true, y_pred, split_role`**, plus run/checkpoint identity in its metadata. Before any comparative metric is calculated, the evaluator must require:

1. Exactly **4,862 rows** for each model and the canonical test manifest, with unique IDs and no missing/extra predictions.
2. Identical canonical pair-ID sets; sort by canonical drug/target indices and require identical order and entity coordinates/IDs.
3. Each `y_true` equals the frozen canonical float32 label promoted to float64; require exact equality after the declared conversion, not an arbitrary tolerance that hides different labels. Round-trip serialization must preserve those bits. Both full y vectors then match exactly.
4. All predictions finite, with no dropping NaN rows, averaging duplicate predictions, filling missing values or evaluating partial sets.

For future workers, provide only train labels to optimization and validation labels to the permitted DCGAN selection path. Fixed encodings, entity metadata and raw file inspection are not learned test-label access. Open the test pair loader/prediction path only after stopping, checkpoint selection/restoration or fixed final-state saving is complete. Execute one final test prediction traversal per model; derive common and native diagnostics from that same saved vector where possible. Never call the model twice just to compute another metric. Co-VAE's single native test call can retain its existing capture mechanism. An integrity or count failure stops scoring; it does not authorize a retry, new seed, changed corpus or silent patch.

## 10. Model-specific preprocessing and measured consequences

**DATA / CODE:** retain native character maps, head truncation and zero padding. DCGAN uses 65 SMILES codes (an extra `p` symbol absent from Co-VAE's 64-code vocabulary) and a 32-wide drug embedding; Co-VAE uses 128-wide drug/target embeddings. Davis retained SMILES characters fit both maps. DCGAN's protein input uses the static **22-symbol, 20-dimensional** BLOSUM JSON through `DataGenerator`; Co-VAE uses learned token embeddings. No unsupported retained Davis drug/protein character or embedding index was found, and all retained DCGAN protein characters have BLOSUM entries. This is static data coverage, not a model forward test.

| Input consequence, full canonical universe | DCGAN-DTA | Co-VAE |
|---|---:|---:|
| Drug cap / truncated drugs | 200 / **0 of 68 (0%)** | 85 / **3 of 68 (4.41%)** |
| Target cap / truncated targets | 2000 / **6 of 442 (1.36%)** | 1200 / **61 of 442 (13.80%)** |
| Maximum raw drug / target length | 92 / 2549 | Same raw strings |
| Unique retained drug strings / target sequences | 68 / 379 | 68 / 379 |

Drugs truncated **only by Co-VAE**: `51004351`, `10461815`, `11984591`. Targets truncated by **both**: `CIT`, `LRRK2`, `LRRK2(G2019S)`, `MTOR`, `ROS1`, `TRPM6`. No entity is truncated only by DCGAN.

The **55 targets truncated only by Co-VAE** are:

```text
ALK, ASK1, ASK2, DAPK1, DMPK2, EGFR, EGFR(E746A750del), EGFR(G719C), EGFR(G719S),
EGFR(L747E749del), EGFR(L747S752del), EGFR(L747T751del), EGFR(L858R),
EGFR(L858RT790M), EGFR(L861Q), EGFR(S752I759del), EGFR(T790M), ERBB2, ERBB3,
ERBB4, FLT1, FLT4, GAK, GCN2(KinDom2S808G), HIPK1, HIPK3, IGF1R, INSR, INSRR,
MAP3K1, MAP3K15, MAP3K4, MAP4K4, MAST1, MET, MET(M1250T), MET(Y1235D), MINK,
MRCKA, MRCKB, MST1R, MYLK, MYO3A, MYO3B, NEK1, PIK3C2B, PIK3C2G, QSK,
ROCK1, SLK, STK36, TAOK2, TNIK, VEGFR2, YSK4
```

These are **model-specific information differences**, not missing canonical examples; keep every pair and disclose them when interpreting results. Do not harmonize caps to erase architecture differences. There are 442 unique target IDs but only **379 full sequences**, with **18 duplicate-sequence groups / 81 IDs**; both capped inputs still have 379 distinct sequences. Since every target is shared by design, this does not defeat new-drug separation, but the benchmark is not cold-target or mutation-specific generalization.

## 11. Leakage and GAN-pretraining policy — blocking review decision

**DATA / CODE:** native `ligands_train.txt` has **50,068 rows** and includes all **68 Davis drug IDs and exact SMILES as its prefix**, in the same order. The proposed validation and test drugs therefore have **11/11 exact-input exposure each** in the native drug-GAN corpus. Protein pretraining data contains all 442 Davis IDs / 379 sequences, but **Variant C calls only the drug GAN**, so do not attribute protein-GAN exposure to this chosen variant. Stage 04 Sections 7.3–7.6 document the origin/overlap; the current static check confirms it.

| Option | Scientific interpretation | Recommendation |
|---|---|---|
| A: preserve native corpus and disclose overlap | Supervised drug-disjoint head-to-head with **unsupervised evaluation-drug exposure** on DCGAN's side; not strictly unseen-drug representation learning | Retain as an explicitly authorized native-corpus sensitivity, not the preferred strict-cold primary |
| B: exclude Davis drug exposure from the GAN input | Drug-cold supervised benchmark with no exact canonical Davis drug input in DCGAN pretraining; **declared corpus adaptation**, while retaining GAN architecture/objective/schedule | **Preferred primary**, pending review/approval of this deviation |

For B, propose a derived, separately hashed corpus that removes entries whose **ID is a canonical Davis drug ID OR raw SMILES equals a Davis SMILES OR the native encoded length-200 input equals one of the Davis inputs**. Exclude **all 68 Davis drugs**, not just the chosen validation/test roles: the rule is task-wide, independent of split/model results, and makes the retained corpus external to the canonical drug set. The current ID/raw/prefix matches cover exactly **68 rows**, so **50,000 rows** would remain under this exact-input rule. Keep order and all other corpus records; notably **650 empty SMILES** and other historical quality defects are not permission for additional cleaning. No corpus was altered or derived file created here. Record inclusion/exclusion rules, counts, ordered IDs, raw/encoded overlap checks and both parent/derived hashes before any GAN training.

No duplicate raw or retained Davis drug strings were found. **Chemical graph/canonical-equivalent SMILES identity remains UNVERIFIED**: RDKit is unavailable in the inspected local interpreter, and no canonicalization was run or installed. Proposed coldness claims are bounded to IDs and actual token inputs, not chemically disjoint scaffolds or every alternative SMILES serialization. If stricter molecular equivalence is required, approve a pinned canonicalization policy and audit before freezing the exclusion claim; do not normalize the baseline inputs silently. Fixed vocabularies, static BLOSUM tables and deterministic caps are not fitted to test labels. Co-VAE's VAE must see only training pairs during optimization, unlike the external GAN corpus.

**Sensitivity decision:** with B as primary, a separately authorized DCGAN run with A's original corpus is recommended to assess native-exposure sensitivity and is required before claiming a measured effect of removing exposure or generalizing to the unmodified native pipeline. It is **not required to pass the narrowly declared B-primary benchmark**. If A is instead chosen as primary, an exclusion sensitivity is required before making strict unseen-drug claims. A sensitivity uses the same frozen split, configuration and selection rule; it is not a seed/config search, must be declared before test access, and cannot replace the primary based on performance. Neither option removes DCGAN's additional unlabeled-data advantage over Co-VAE; that remains a model-resource asymmetry, not a measured causal explanation.

## 12. Future artifact and provenance contract

This is a proposed layout only; the draft is the sole Stage 10 artifact created now:

```text
audit/10-common-davis-head-to-head/
  PROTOCOL-DRAFT.md                 # current design, not frozen
  PROTOCOL.md / freeze.json          # later approved spec, commit, version and hashes
  inputs/                           # entity maps, pair universe, role manifests,
                                    # split specification, corpus inclusion/exclusion
  runs/dcgan-primary/               # separate immutable run identity
  runs/covae-primary/               # separate immutable run identity
  comparison/                       # generated later from both saved vectors
```

Both runs must record environment, source commit where verifiable, source/data/feature-map/corpus hashes, canonical entity/split/pair-manifest hashes, resolved configs, complete command and RNG controls, history/events, actual epochs/optimizer updates, native metrics, selected/final checkpoint with hash, predictions with canonical keys, finite/count checks, resources/stopping reason, output paths and final artifact manifest. DCGAN selection state/history and restoration evidence must be saved; Co-VAE must record 100 Adam constructions and fixed epoch-100 identity. Preserve failed runs rather than overwrite/retry invisibly. No cross-model or Stage 09 checkpoint reuse.

One later comparison artifact must verify frozen protocol/hash identity, align both prediction files, report the four common metrics and descriptive paired differences, and retain native metrics separately. It must link the exact two vectors/checkpoints and evaluator version. Entity/pair/fold maps, original strings and encoded-input hashes must permit later Stage 11 alignment without reconstructing ambiguous order; **no representation extraction is implemented now**.

## 13. Compute and resource planning

**PRIOR RUNTIME:** 09A's native drug GAN took **580.931 s**; DTA fit including validation took **2,213.522 s / 150 epochs = 14.757 s per epoch** on 3,344 train + 836 validation pairs. 09B averaged **24.929 s per training epoch / 99 batches**, using the same proposed Co-VAE architecture and batch size. These are previous measurements, not Stage 10 timings.

**ESTIMATE:** DCGAN linear pair-volume scaling gives `14.757 * (20,332+4,862)/(3,344+836) = 88.943 s/epoch`, plus approximately 581 s for one GAN. This assumes unchanged per-example cost across datasets, validation batching and hardware; the new corpus, startup, diagnostic overhead and memory behavior are unmeasured.

| One Stage 10 fit on a comparable T4 | Conditional estimate |
|---|---|
| DCGAN, earliest ordinary patience stop at epoch 76 | **2.04 T4 hours** |
| DCGAN, illustrative 150 epochs | **3.87 T4 hours** |
| DCGAN, full 300-epoch cap | **7.57 T4 hours**; plan roughly **8–10 hours** including allowance, not a guaranteed runtime |
| Co-VAE, 100 epochs / 80 batches | `24.929*80/99*100` ≈ **33.6 minutes** of training; plan roughly **40–60 minutes** including setup/evaluation |

The actual DCGAN stopping epoch on this split is unknown. Do not carry over Stage 09A's 150 epochs or its 5-hour supervisor wall cap as if sufficient for a 300-epoch Davis run. A full-budget run needs separately authorized runtime/session capacity, selected before training; no installation or claim about current Kaggle limits is made here. A native-corpus sensitivity would add another DCGAN fit, not another primary Co-VAE fit.

Prior sampled peaks were **13,883 MiB GPU / 3.77 GiB RSS** for DCGAN and **2,463 MiB / 1.69 GiB** for Co-VAE. Same caps/batch imply static tensor-shape compatibility, not a new memory guarantee. Future preflight must check the actual device and derived safety cap (e.g. 15,360−512 = **14,848 MiB** on the previously measured T4), host **12 GiB** boundary, finite values and batch tails. Preserve native per-batch BLOSUM conversion instead of allocating the full expanded pair corpus. Threshold/OOM/numerical failures stop and are reported; do not silently reduce batch/caps or change the architecture to obtain a comparable-looking result.

## 14. Statistical claim scope

One fixed common split is defensible as the primary controlled comparison under the stated compute budget. It provides paired test examples, **not across-split uncertainty, robustness across training seeds, or a universal method ranking**. No paper-reported standard deviation substitutes for Stage 10 uncertainty. Report metric differences descriptively, disclose the 11 test-drug clusters and heavy censoring, and do not claim significance from 4,862 supposedly independent pairs. Multi-split/training-seed robustness belongs in Stage 14 unless a later explicit methodological decision moves it earlier. Dataset, preprocessing, corpus and training-policy differences bound every conclusion.

## 15. Proposed Stage 10 PASS criteria

PASS establishes a completed, auditable comparison under the frozen protocol, not which model wins:

1. Approved, committed protocol/seed/algorithm/role/corpus policy predates training; both runs name the same frozen entity and full/role pair-manifest hashes.
2. Exactly 20,332/4,862/4,862 train/validation/test pairs, 46/11/11 drugs, full unique coverage, and **zero pairwise drug overlap**; preserved canonical labels/IDs and valid encodings.
3. The approved pretraining exposure rule passes its declared ID/input checks and hashes; no test-driven configuration, inference-seed, stopping or checkpoint decision.
4. Both complete their declared schedules without numerical/resource failure: DCGAN restores strict-best validation-CI checkpoint after native patience/cap stopping and records actual full epochs/updates; Co-VAE completes 100 epochs, 8,000 updates and 100 Adam constructions and saves final state.
5. Each makes one final held-out prediction traversal with exactly the same 4,862 pair IDs and finite predictions. Model-boundary label equality and evaluator alignment assertions pass before comparative scoring.
6. All four primary metrics are computed by one hashed common evaluator under the exact definitions; undefined primary metrics are disclosed and prevent complete metric-comparison PASS, not silently replaced.
7. Source/data/split/corpus/checkpoint/prediction/artifact hashes, environment, exact commands, seeds, histories, resource/stopping evidence and deviations are recorded; integrity checks pass.
8. The final interpretation remains a comparison of these two fixed configurations and declared native training policies on **one common split**, without paper reproduction, generalized superiority or hybrid claims.

## 16. Unresolved decisions and prerequisites

**Methodological blockers to freeze:** approve **B (all-Davis exact-input exclusion) as primary** versus A's explicitly transductive native-corpus alternative; approve the limited ID/token-input coldness claim versus requiring molecular canonicalization; accept the **DCGAN validation-selected / Co-VAE fixed epoch-100** asymmetry; approve the **explicit shared repository-convention RM2 formula** and proposed single-split/seed policy. These are concrete choices, not unknown source behavior. No protocol freeze or final manifest is implied by this recommendation.

**Later implementation/execution prerequisites:** an external Davis DCGAN adapter and shared manifest/evaluator implementation do not exist; neither has passed a new bounded preflight. Verify identical model-boundary labels, BLOSUM batches/shapes, corpus exclusions, metrics/permutation fixtures, checkpoint/test gates and memory without source changes under separate authorization. Approve a budget/session plan covering the potential full DCGAN cap; the Stage 09 monitor budget cannot be reused blindly. These prerequisites do not establish a structural impossibility, and they do not authorize runner implementation or training now.

## 17. Static forensic review and freeze recommendation

| Question | Static answer and limit |
|---|---|
| 1. Can both consume exact Davis new-drug manifests? | **Conditionally yes:** native array/pair/loader interfaces support them; DCGAN has no shipped Davis dataset and requires an external adapter. No new adapter execution has been performed. |
| 2. Can transforms be equivalent? | **Yes by the explicit canonical float32 target contract:** both formulas agree for raw positive Kd when DCGAN uses the appropriate transform route; future boundary assertions must verify implementation and prevent double logs. |
| 3. Can test remain isolated? | **Conditionally yes:** controlled late test gates can exclude supervised labels/model selection; native GAN corpus currently exposes test inputs, so strict representation isolation requires the approved exclusion policy. Raw labels exist in the shared dataset; do not claim physical non-loading. |
| 4. Is validation needed and how? | **Yes for DCGAN:** disjoint drug fold 4, 11 drugs/4,862 pairs; Co-VAE withholds the same fold and follows its fixed schedule without periodic validation. |
| 5. Are fixed configurations structurally valid? | **Yes statically:** dimensions/vocab/BLOSUM coverage and non-singleton tails are valid; prior 09A/09B validate the native architectures, not a new DCGAN Davis adapter. |
| 6. Are truncations materially different? | **Yes in input coverage:** 0 vs 3 drugs and 6 vs 61 targets are truncated. The effect on prediction quality is unmeasured. |
| 7. Is common CI independent of native CI? | **Yes by definition:** all unequal-label unordered pairs, tie credit 0.5; invariant to vector permutation. Neither native estimator is substituted. Implementation verification is future work. |
| 8. Is common RM2 explicit? | **Yes:** Section 8 defines centered Pearson-squared q, through-origin s and `q*(1−sqrt(abs(q²−s²)))`, with degenerate handling and a versioned convention. Approval remains pending. |
| 9. Does GAN pretraining expose evaluation drugs? | **Yes, confirmed:** all 68 Davis drugs, including all 11 validation and 11 proposed test drugs; input exposure, not affinity-label training. No protein GAN in Variant C. |
| 10. Is a leakage-clean sensitivity needed? | **B is recommended as primary.** Native-corpus sensitivity is recommended, mandatory for exposure-effect/unmodified-pipeline claims; A-primary would need exclusion sensitivity before strict-cold claims. |
| 11. Is one split defensible? | **Yes for the bounded question:** no across-split/seed robustness or significance claim; defer robustness to Stage 14. |
| 12. Ready to freeze? | **NO:** pretraining/claim boundary and training/metric policy approvals are outstanding; no final split artifact, runner or benchmark may be produced yet. |

**Exact next step:** review and approve/amend the named methodological choices in Section 16, then explicitly authorize protocol freeze and immutable manifest creation before any runner work. A later separately authorized adapter/evaluator implementation and bounded preflight must precede benchmark execution. Do not start Stage 11, representations, fusion or optimization from this design task.

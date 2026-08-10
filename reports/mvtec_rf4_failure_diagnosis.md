# MVTec RF4 Longitudinal Failure Diagnosis

> Classification: **RF4 FAIL / BLOCKED**
> Analyzer status: `FAIL_SEMANTIC_REVIEW_REQUIRED`
> Scope: **NON-BENCHMARK / MECHANISM VALIDATION**

## Decision

The frozen 15-epoch E2 run is complete and the run-level Integrity Gate passes, but the shared Sample Dynamics correctness contract is not ready for active intervention. E3, E4 and E5 remain blocked.

## Evidence that passed

- 15 epochs × 128 train samples, batch 4, final optimizer step 479.
- Manifest SHA-256 and Seg Small checkpoint SHA-256 match the frozen contract.
- `num_classes=1`, resolution 384, augmentation/EMA/multi-scale/scale-jitter disabled, `resume=null`.
- All persisted metrics and trajectories are finite; all 15 epoch checkpoints exist.
- Per-sample history/probe lengths are 15, sample IDs match the manifest, and replayed final state equals exported state.
- Offline outputs include Instant-Loss and Dynamics trajectories, FN-reduction correlations, per-image transition distributions, corruption smoke, frozen reference subsets, and a completed 50-sample visual/protocol audit.

Evidence: `reports/mvtec_rf4_run_integrity.md`, `reports/mvtec_rf4_longitudinal_validation.md`, `reports/mvtec_reference_subsets.json`, `reports/mvtec_rf4_manual_audit.csv`.

## Blocking finding

The frozen core treats `gt_recall=0` as a probe conflict without first requiring `gt_count > 0`. The E2 evidence contains **930 frozen-core conflicts on empty-GT observations**. The analysis-side report correctly excludes these undefined-recall cases, but the replayed state still uses the frozen predicate. In the 50-sample audit, 22 visually normal `train/good` samples have zero analysis conflicts but frozen conflicts in all 15 epochs.

This is a shared correctness/semantics issue, not an intervention result. It can contaminate `SUSPECT`, difficulty, and future sampler/weight decisions, so the longitudinal Gate cannot authorize active intervention.

## Other findings (non-blocking after the decision)

- The controlled-corruption smoke is mixed: shifted masks produce persistent analysis conflict, drop-mask samples are intentionally empty-target cases, and drop-component produces transient FP/conflict. These remain smoke evidence only.
- After excluding the three controlled-corruption samples, the natural `HARD_LEARNABLE` analysis contains 383 events across 41 samples; first-entry outcomes are reported by 1/2/3-epoch horizon. These are still high-loss candidate observations, not a learnability claim.
- The 50-sample audit is targeted and Codex-reviewed rather than blinded/statistical. It is evidence of the semantic issue and protocol cases, not an RF8 precision/recall estimate.

## Diagnosis by category

| Category | Diagnosis | Status |
| --- | --- | --- |
| Run/data integrity | No hash, NaN/Inf, missing-trajectory, step, or replay failure | PASS |
| Loss semantics | Per-image normalized loss trajectory is present and replayable | PASS WITH CAVEATS |
| Probe semantics | Empty-GT conflict predicate is semantically invalid for normal images | BLOCKING |
| State transition | Replay is deterministic, but state can consume the contaminated frozen conflict signal | BLOCKED BY PROBE |
| Corruption protocol | Five records are present and tracked; no RF8 claim | SMOKE ONLY |
| Horizon | 15 epochs provide the requested longitudinal windows | PASS |
| Segmentation metric | Mask-IoU undefined cases are preserved as undefined | PASS WITH CAVEATS |

## Required next action

Do not run E3/E4/E5. Correct or explicitly contract the empty-GT probe semantics, add focused tests, rerun the RF4 Longitudinal Gate with the same frozen data/checkpoint contract, and obtain a fresh independent Verify before reconsidering any intervention.

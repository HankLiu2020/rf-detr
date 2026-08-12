# MVTec RF4 Longitudinal Failure Diagnosis

> Classification: **HISTORICAL FAILURE DIAGNOSIS — RESOLVED BY MINIMAL FIX**
> Analyzer status: `PASS_WITH_CAVEATS` after corrected rerun
> Scope: **NON-BENCHMARK / MECHANISM VALIDATION**

## Decision

The original frozen 15-epoch E2 run exposed an empty-GT Probe conflict semantics defect. The defect was corrected without changing StatePolicy thresholds, the same-contract E2 rerun completed, and the corrected RF4 Gate is now `PASS_WITH_CAVEATS`. The original failure evidence below remains useful as an audit trail; it no longer blocks E5 execution. E3 and E4 are still pending their own real intervention runs.

## Evidence that passed

- 15 epochs × 128 train samples, batch 4, final optimizer step 479.
- Manifest SHA-256 and Seg Small checkpoint SHA-256 match the frozen contract.
- `num_classes=1`, resolution 384, augmentation/EMA/multi-scale/scale-jitter disabled, `resume=null`.
- All persisted metrics and trajectories are finite; all 15 epoch checkpoints exist.
- Per-sample history/probe lengths are 15, sample IDs match the manifest, and replayed final state equals exported state.
- Offline outputs include Instant-Loss and Dynamics trajectories, FN-reduction correlations, per-image transition distributions, corruption smoke, frozen reference subsets, and a completed 50-sample visual/protocol audit.

Evidence: `reports/mvtec_rf4_run_integrity.md`, `reports/mvtec_rf4_longitudinal_validation.md`, `reports/mvtec_reference_subsets.json`, `reports/mvtec_rf4_manual_audit.csv`.

## Original blocking finding

The original frozen core treated `gt_recall=0` as a probe conflict without first requiring `gt_count > 0`. The original E2 evidence contained **930 frozen-core conflicts on empty-GT observations**. The analysis-side report excluded these undefined-recall cases, but the replayed state still used the old predicate. In the 50-sample audit, 22 visually normal `train/good` samples had zero analysis conflicts but frozen conflicts in all 15 epochs.

This was a shared correctness/semantics issue, not an intervention result. It could contaminate `SUSPECT`, difficulty, and future sampler/weight decisions, so the original longitudinal Gate correctly blocked active intervention.

## Resolution and corrected-gate evidence

- Empty-GT `gt_recall=0` is now treated as undefined rather than conflict.
- Empty-GT unmatched mask-IoU placeholders are ignored; real FP still contributes to difficulty.
- Nonempty-GT FN/class/mask conflict behavior is unchanged.
- Focused state tests pass, and corrected E2 replay reports `empty_gt_frozen_conflict_observations=0` with `replay_final_state_match=true`.
- Corrected evidence: `reports/mvtec_rf4_longitudinal_validation.md`, `reports/mvtec_rf4_longitudinal_analysis.json`, and `reports/mvtec_reference_subsets.json`.

## Other findings (non-blocking after the decision)

- The controlled-corruption smoke is mixed: shifted masks produce persistent analysis conflict, drop-mask samples are intentionally empty-target cases, and drop-component produces transient FP/conflict. These remain smoke evidence only.
- After excluding controlled-corruption events, the natural `HARD_LEARNABLE` analysis contains 386 events across 42 samples; first-entry outcomes are reported by 1/2/3-epoch horizon. These are still high-loss candidate observations, not a learnability claim.
- The 50-sample audit is targeted and Codex-reviewed rather than blinded/statistical. It is evidence of the semantic issue and protocol cases, not an RF8 precision/recall estimate.

## Diagnosis by category

| Category | Diagnosis | Status |
| --- | --- | --- |
| Run/data integrity | No hash, NaN/Inf, missing-trajectory, step, or replay failure | PASS |
| Loss semantics | Per-image normalized loss trajectory is present and replayable | PASS WITH CAVEATS |
| Probe semantics | Original empty-GT conflict predicate was semantically invalid; corrected rerun excludes undefined placeholders | RESOLVED / PASS WITH CAVEATS |
| State transition | Corrected replay is deterministic and final state matches exported state | PASS WITH CAVEATS |
| Corruption protocol | Five records are present and tracked; no RF8 claim | SMOKE ONLY |
| Horizon | 15 epochs provide the requested longitudinal windows | PASS |
| Segmentation metric | Mask-IoU undefined cases are preserved as undefined | PASS WITH CAVEATS |

## Historical required action / current next action

The historical required action was to correct the empty-GT semantics, add focused tests, rerun RF4 under the same data/checkpoint contract, and obtain independent verification; that action is complete. The current next action is to run the frozen-contract E3 and E4 intervention comparisons, keeping E0 and E5 as the existing Seed-1 endpoints and preserving the complete E0/E3/E4/E5 matrix for later replication.

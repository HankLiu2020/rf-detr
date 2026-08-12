OVERALL:
  rf4_gate: PASS_WITH_CAVEATS
  e0_15: COMPLETED
  e3: COMPLETED_SEED1_MECHANISM_PASS_MIXED_SIGNAL
  e4: COMPLETED_SEED1_MECHANISM_PASS_MIXED_SIGNAL
  e5: COMPLETED_PASS_WITH_CAVEATS
  completed_seeds: [20260810]
  correctness_failures: []
  algorithm_signal: MIXED_SINGLE_SEED_NOT_VALIDATED
  next_gate: RUN_COMPLETE_E0_E3_E4_E5_FOR_SEEDS_20260811_AND_20260812

# Overnight Sample Dynamics Status

## Current checkpoint

- Repository: `/home/liujiyuan/rf-detr-sample-dynamics`
- Branch: `agent/dynamic-scheduling-rfdetr`
- Evidence checkpoint before final commit: `c28e066eeaf5635cde074ca89eb6600caa90be44`
- Remote branch: `HankLiu2020/rf-detr:agent/dynamic-scheduling-rfdetr`
- Scope: `NON-BENCHMARK / MECHANISM VALIDATION`
- Current rule: RF4 must pass before intervention; RF4 passes with caveats. The full Seed1 E0/E3/E4/E5 matrix is now complete and awaits two full replication seeds.

## Completed before overnight execution

- RF4 preparation passed final independent Verify with no P0/P1 findings.
- `run_pilot.py` fairness snapshot reads CLI epochs, batch size and seed.
- Analyzer contract locks 15 epochs, batch 4, seed `20260810`, fixed manifest/checkpoint hashes, 128 train IDs, frozen StatePolicy and five corruption records.
- Analyzer preserves frozen StateStore replay while separately reporting empty-GT/undefined mask-IoU analysis semantics.
- AST, diff-check and RF4 helper tests passed.
- No Sample Dynamics core modification was made for the RF4 preparation.

## Runs and artifacts

- RF4 15-epoch E2 corrected: **completed** at `/home/liujiyuan/mvtec-sample-dynamics-runs/e2-rf4-15-corrected`; analyzer status `PASS_WITH_CAVEATS`.
- E0-15 baseline: **completed** at `/home/liujiyuan/mvtec-sample-dynamics-runs/e0-15`.
- E5-15 combined: **completed** at `/home/liujiyuan/mvtec-sample-dynamics-runs/e5-15`.
- E3-15 loss-weight-only: **completed** at `/home/liujiyuan/mvtec-sample-dynamics-runs/e3-15`.
- E4-15 dynamic-sampler-only: **completed** at `/home/liujiyuan/mvtec-sample-dynamics-runs/e4-15`.
- `reports/mvtec_seed1_ablation_validation.md`: generated; correctness and realized-resource gates pass, while algorithm signal remains mixed and single-seed.
- `reports/mvtec_e5_combined_validation.md`: generated; E5 mechanism execution passes with caveats, algorithm benefit not validated.
- `mvtec_reference_subsets.json`: frozen from E2 before any intervention.
- `mvtec_rf4_run_integrity.json/md`: present; Integrity Gate `PASS`.
- `mvtec_rf4_manual_audit.csv` and decisions: 50/50 completed (`23 reasonable`, `5 questionable`, `22 wrong`); the `wrong` labels identify empty-GT frozen-core conflict semantics, not model mAP.
- `mvtec_rf4_failure_diagnosis.md`: historical diagnosis retained; corrected RF4 is now `PASS_WITH_CAVEATS`; active E3/E4 intervention remains pending its complete same-contract matrix.
- `intervention_contract.json`: not yet present; migration contract remains draft pending E0/E3/E4/E5 replication.

## Runtime status

- `/var/run/docker.sock` is currently `root:docker`, mode `660`.
- The Codex worker process still has only its original supplementary groups, but `sg docker -c` provides the authorized `docker` group for the run.
- Existing `rfdetr-dynamic-scheduling` container is up with image `train-env-rfdetr-claude:20260806`.
- RF4 15-epoch E2 evidence and the Seed1 E0/E3/E4/E5 matrix are complete.

## Current result and next action

- Final mask mAP50:95 is E0=`0.1390`, E3=`0.1479`, E4=`0.1020`, E5=`0.1079`; box/F1 move in different directions, so the signal is mixed rather than a benefit claim.
- E3 gave `REFERENCE_HARD` actual applied loss weight `1.1222` versus mastered `0.9276`, with equal exposure; E4 gave hard exposure `1.4990` versus mastered `0.8341`, with no loss weighting. Both isolated mechanisms therefore changed the intended resource.
- Do not tune StatePolicy from this result. Next run the complete E0/E3/E4/E5 matrix for seeds `20260811` and `20260812`, then report mean ± std without winner selection.

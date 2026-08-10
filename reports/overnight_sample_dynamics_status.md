OVERALL:
  rf4_gate: FAIL_BLOCKED_EMPTY_GT_PROBE_SEMANTICS
  e0_15: NOT_STARTED
  e3: BLOCKED_BY_RF4
  e4: BLOCKED_BY_RF4
  e5: BLOCKED_BY_RF4
  completed_seeds: []
  correctness_failures:
    - EMPTY_GT_FROZEN_CORE_PROBE_CONFLICT
  algorithm_signal: NOT_AVAILABLE
  next_gate: DIAGNOSE_EMPTY_GT_PROBE_SEMANTICS

# Overnight Sample Dynamics Status

## Current checkpoint

- Repository: `/home/liujiyuan/rf-detr-sample-dynamics`
- Branch: `agent/dynamic-scheduling-rfdetr`
- Commit: `e0264223422362964373c0cf5d21752537bee78c`
- Remote branch: `HankLiu2020/rf-detr:agent/dynamic-scheduling-rfdetr`
- Scope: `NON-BENCHMARK / MECHANISM VALIDATION`
- Current rule: RF4 Longitudinal must complete before any E3/E4/E5 intervention.

## Completed before overnight execution

- RF4 preparation passed final independent Verify with no P0/P1 findings.
- `run_pilot.py` fairness snapshot reads CLI epochs, batch size and seed.
- Analyzer contract locks 15 epochs, batch 4, seed `20260810`, fixed manifest/checkpoint hashes, 128 train IDs, frozen StatePolicy and five corruption records.
- Analyzer preserves frozen StateStore replay while separately reporting empty-GT/undefined mask-IoU analysis semantics.
- AST, diff-check and RF4 helper tests passed.
- No Sample Dynamics core modification was made for the RF4 preparation.

## Runs and artifacts

- RF4 15-epoch E2: **completed** at `/home/liujiyuan/mvtec-sample-dynamics-runs/e2-rf4-15`.
- E0-15, E3-15, E4-15, E5-15: **not started and blocked by RF4**.
- `mvtec_rf4_longitudinal_validation.md`: present; analyzer status `FAIL_SEMANTIC_REVIEW_REQUIRED`.
- `mvtec_reference_subsets.json`: frozen from E2 before any intervention.
- `mvtec_rf4_run_integrity.json/md`: present; Integrity Gate `PASS`.
- `mvtec_rf4_manual_audit.csv` and decisions: 50/50 completed (`23 reasonable`, `5 questionable`, `22 wrong`); the `wrong` labels identify empty-GT frozen-core conflict semantics, not model mAP.
- `mvtec_rf4_failure_diagnosis.md`: present; active intervention blocked.
- `intervention_contract.json`: not yet present; must be created only after RF4 PASS.

## Runtime status

- `/var/run/docker.sock` is currently `root:docker`, mode `660`.
- The Codex worker process still has only its original supplementary groups, but `sg docker -c` provides the authorized `docker` group for the run.
- Existing `rfdetr-dynamic-scheduling` container is up with image `train-env-rfdetr-claude:20260806`.
- RF4 15-epoch E2 evidence is complete; no intervention run was started.

## Only next action

Do not start E3/E4/E5. Diagnose and correct the empty-GT probe semantics, then rerun the same RF4 Longitudinal Gate and independent Verify before requesting any intervention.

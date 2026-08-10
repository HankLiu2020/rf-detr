OVERALL:
  rf4_gate: WAIT_RF4_LONGITUDINAL_DOCKER_RUN
  e0_15: NOT_STARTED
  e3: BLOCKED_BY_RF4
  e4: BLOCKED_BY_RF4
  e5: BLOCKED_BY_RF4
  completed_seeds: []
  correctness_failures: []
  algorithm_signal: NOT_AVAILABLE
  next_gate: RUN_RF4_LONGITUDINAL_15_EPOCH

# Overnight Sample Dynamics Status

## Current checkpoint

- Repository: `/home/liujiyuan/rf-detr-sample-dynamics`
- Branch: `agent/dynamic-scheduling-rfdetr`
- Commit: `3ac75219fe28dcec9e2077bc7060e41378876fa7`
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

- RF4 15-epoch E2: **not started**.
- E0-15, E3-15, E4-15, E5-15: **not started and blocked by RF4**.
- `mvtec_rf4_longitudinal_validation.md`: not yet present.
- `mvtec_reference_subsets.json`: not yet present; must be frozen only after valid E2-15 evidence.
- `intervention_contract.json`: not yet present; must be created only after RF4 PASS.

## Runtime status

- `/var/run/docker.sock` is currently `root:docker`, mode `660`.
- The Codex worker process still has only its original supplementary groups, but `sg docker -c` provides the authorized `docker` group for the run.
- Existing `rfdetr-dynamic-scheduling` container is up with image `train-env-rfdetr-claude:20260806`.
- RF4 15-epoch E2 has not yet produced evidence; the run is now ready to start through `sg docker -c`.

## Only next action

Run the already frozen `e2-rf4-15` command against `rfdetr-dynamic-scheduling`. After the run completes, perform Run Integrity Gate, offline E1/longitudinal analysis, independent Verify and RF4 classification. Do not start E3/E4/E5 before that classification.

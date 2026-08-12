OVERALL:
  rf4_gate: PASS_WITH_CAVEATS
  e0_15: COMPLETED
  e3: THREE_SEEDS_MECHANISM_PASS_NO_BENEFIT_VALIDATED
  e4: THREE_SEEDS_MECHANISM_PASS_GLOBAL_REGRESSION
  e5: THREE_SEEDS_MECHANISM_PASS_TARGETED_RECOVERY_GLOBAL_REGRESSION
  completed_seeds: [20260810, 20260811, 20260812]
  correctness_failures: []
  algorithm_signal: TARGETED_HARD_LOSS_RECOVERY_WITH_GLOBAL_REGRESSION_NO_DEPLOYMENT
  next_gate: REDESIGN_INTERVENTION_POLICY_WITHOUT_TUNING_ON_THIS_PILOT

# Overnight Sample Dynamics Status

## Current checkpoint

- Repository: `/home/liujiyuan/rf-detr-sample-dynamics`
- Branch: `agent/dynamic-scheduling-rfdetr`
- Milestone parent commit: `a52ba6740684f1c526f8a0431f6beff4a39e5936`
- Remote branch: `HankLiu2020/rf-detr:agent/dynamic-scheduling-rfdetr`
- Scope: `NON-BENCHMARK / MECHANISM VALIDATION`
- Current rule: RF4 passed with caveats. The full E0/E3/E4/E5 matrix is complete for three frozen seeds; correctness and realized-resource gates pass, while the algorithm-benefit gate does not.

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
- Replication seeds `20260811` and `20260812`: **completed** for all four runs under `/home/liujiyuan/mvtec-sample-dynamics-runs/seed-<seed>/`.
- `reports/mvtec_three_seed_ablation_validation.md`: generated; paired mean ± sample std, frozen hard-subset recovery, direction consistency, and resource ratios are recorded.
- `reports/mvtec_seed1_ablation_validation.md`: generated; correctness and realized-resource gates pass, while algorithm signal remains mixed and single-seed.
- `reports/mvtec_e5_combined_validation.md`: generated; E5 mechanism execution passes with caveats, algorithm benefit not validated.
- `mvtec_reference_subsets.json`: frozen from E2 before any intervention.
- `mvtec_rf4_run_integrity.json/md`: present; Integrity Gate `PASS`.
- `mvtec_rf4_manual_audit.csv` and decisions: 50/50 completed (`23 reasonable`, `5 questionable`, `22 wrong`); the `wrong` labels identify empty-GT frozen-core conflict semantics, not model mAP.
- `mvtec_rf4_failure_diagnosis.md`: historical diagnosis retained; corrected RF4 is now `PASS_WITH_CAVEATS`; the complete same-contract matrix is finished, its mechanism gate passes, and its algorithm-benefit gate fails.
- `intervention_contract.json`: intentionally not frozen; the replicated intervention policy does not pass the benefit gate.

## Runtime status

- `/var/run/docker.sock` is currently `root:docker`, mode `660`.
- The Codex worker process still has only its original supplementary groups, but `sg docker -c` provides the authorized `docker` group for the run.
- Existing `rfdetr-dynamic-scheduling` container is up with image `train-env-rfdetr-claude:20260806`.
- RF4 15-epoch E2 evidence and all three E0/E3/E4/E5 matrices are complete. Final/best checkpoints and audit ledgers are retained; intermediate Seed2/Seed3 checkpoints were deleted only after trajectory extraction and SHA-256-verified archival.

## Current result and next action

- Three-seed final mask mAP50:95 is E0=`0.1347±0.0065`, E3=`0.1333±0.0131`, E4=`0.1012±0.0050`, E5=`0.0995±0.0078`. Paired Δmask is E3=`-0.0015±0.0092`, E4=`-0.0336±0.0030`, E5=`-0.0352±0.0088`.
- Realized hard/mastered resource ratios are repeatable: E3 applied loss weight=`1.2074±0.0043` at equal exposure; E4 appearance=`1.7385±0.0605`; E5 appearance=`1.7080±0.0424` and effective contribution=`2.4360±0.1255`.
- E5 improves frozen hard-subset mean loss reduction relative to E0 in 3/3 seeds (`+28.73±11.31`) but does not improve hard-subset mask IoU consistently, while global mask mAP and F1 regress in 3/3 seeds. The mechanism works; the frozen allocation policy over-focuses and is not accepted for deployment or RF9 freezing.

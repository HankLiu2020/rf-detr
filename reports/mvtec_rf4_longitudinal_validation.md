# MVTec RF4 Longitudinal Validation

> Gate: **FAIL_SEMANTIC_REVIEW_REQUIRED**
> Scope: **NON-BENCHMARK / RF4 MECHANISM VALIDATION**
> Sample Dynamics core and frozen StatePolicy were not changed for this run.

## Run contract

- Run: `/workspace/runs/e2-rf4-15`
- Epochs / samples per epoch: `15 / 128`
- Batch / seed: `4 / 20260810`
- Resolution: `384`
- Manifest SHA-256: `90673182af0c409e99778f30545f550e91b6845e48f6894a7ebe029c53172528`
- Checkpoint SHA-256: `6de3da31b2572cac214a1c76cce4a92a13966d56390ac2b3a3de9a8dc2b2bca3`
- State evidence SHA-256: `f1f0b36c5943e5bcfddf1df13673837174d718f108d73cbe5bd184d3d7411b62`
- Replay final state match: **True**
- Frozen StatePolicy verified: **True**
- Empty-GT observations counted as frozen-core conflicts: `930`; analysis conflict trajectory excludes this undefined recall case.
- Augmentation, multi-scale, scale jitter and EMA remained disabled; no policy parameter was tuned.
- Frozen reference subsets: `/workspace/rfdetr/reports/mvtec_reference_subsets.json`.

## What was reconstructed

The analyzer reads only the persisted E2 `sample_state.json`. It reconstructs the frozen policy one epoch at a time and emits:

- per-sample weighted per-image normalized loss and instant percentile trajectories;
- instant hard/easy/middle buckets and the frozen four-state trajectory;
- state transition counts, loss EMA, slope, forgetting count and difficulty;
- matched mask IoU, FN, FP, class error and probe-conflict streaks;
- t→t+1/t+2 loss and mask-IoU improvement correlations;
- controlled-corruption trajectories and persistent-conflict smoke signals.

## Epoch trajectory summary

| Epoch | Instant easy | Instant middle | Instant hard | LEARNING | MASTERED | HARD_LEARNABLE | SUSPECT | State transitions | Analysis conflict samples |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 32 | 64 | 32 | 96 | 0 | 32 | 0 | 0 | 66 |
| 1 | 32 | 64 | 32 | 96 | 0 | 32 | 0 | 12 | 66 |
| 2 | 32 | 64 | 32 | 86 | 10 | 29 | 3 | 17 | 66 |
| 3 | 32 | 64 | 32 | 74 | 22 | 30 | 2 | 23 | 66 |
| 4 | 32 | 64 | 32 | 69 | 27 | 31 | 1 | 20 | 66 |
| 5 | 32 | 64 | 32 | 66 | 30 | 32 | 0 | 20 | 66 |
| 6 | 32 | 64 | 32 | 65 | 31 | 31 | 1 | 14 | 66 |
| 7 | 32 | 64 | 32 | 64 | 32 | 32 | 0 | 16 | 66 |
| 8 | 0 | 96 | 32 | 96 | 0 | 30 | 2 | 36 | 66 |
| 9 | 0 | 96 | 32 | 96 | 0 | 27 | 5 | 5 | 66 |
| 10 | 0 | 96 | 32 | 96 | 0 | 27 | 5 | 2 | 62 |
| 11 | 32 | 64 | 32 | 68 | 28 | 28 | 4 | 33 | 53 |
| 12 | 32 | 64 | 32 | 77 | 19 | 29 | 3 | 16 | 31 |
| 13 | 32 | 64 | 32 | 83 | 13 | 29 | 3 | 14 | 17 |
| 14 | 32 | 64 | 32 | 88 | 8 | 31 | 1 | 15 | 12 |

## State stability

| Trajectory | Total transitions | Mean | Median | P90 | Unchanged sample fraction | Mean unique labels/sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Instant Loss buckets | 236 | 1.8438 | 1.5000 | 4.0000 | 0.3516 | 1.6484 |
| Training Dynamics states | 243 | 1.8984 | 2.0000 | 4.0000 | 0.2969 | 1.7266 |

The instant baseline is a three-bucket current-loss view (`EASY/MIDDLE/HARD`), while Dynamics is the frozen four-state view. Their churn is therefore compared as stability evidence, not as an accuracy ranking.

The frozen core currently reports `gt_recall=0` for empty-GT normal images; the analysis-side conflict trajectory therefore excludes empty-GT recall from conflict, while the replayed Dynamics state still uses the exact frozen predicate. This semantic discrepancy is reported rather than hidden.

## Longitudinal diagnostic means

| Epoch | Mean loss EMA | Mean slope | Mean forgetting | Mean mask IoU | Mean FN | Mean FP | Mean class error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 191.3282 | 0.0000 | 0.0000 | 0.6825 | 0.4531 | 0.6250 | 0.0625 |
| 1 | 179.4333 | -39.6498 | 0.0000 | 0.7510 | 0.4531 | 0.5938 | 0.0625 |
| 2 | 165.3709 | -29.3848 | 0.0000 | 0.7499 | 0.4219 | 0.5703 | 0.0938 |
| 3 | 151.5245 | -23.5456 | 0.0000 | 0.8079 | 0.4453 | 0.3906 | 0.0703 |
| 4 | 138.9678 | -19.5781 | 0.0000 | 0.8524 | 0.4453 | 0.3750 | 0.0703 |
| 5 | 128.3294 | -11.9234 | 0.0000 | 0.8252 | 0.4375 | 0.3047 | 0.0781 |
| 6 | 119.0946 | -8.5733 | 0.0000 | 0.8864 | 0.4531 | 0.1328 | 0.0625 |
| 7 | 111.5243 | -6.2834 | 0.0000 | 0.8721 | 0.4766 | 0.1172 | 0.0391 |
| 8 | 105.9783 | -4.2909 | 0.0000 | 0.8767 | 0.4922 | 0.1406 | 0.0234 |
| 9 | 101.8483 | -2.7099 | 0.0000 | n/a | 0.5156 | 0.1094 | 0.0000 |
| 10 | 98.2544 | -1.7005 | 0.0000 | 0.9618 | 0.4766 | 0.0781 | 0.0078 |
| 11 | 95.4814 | -1.2867 | 0.0000 | 0.8658 | 0.4141 | 0.0859 | 0.0000 |
| 12 | 92.8538 | -1.5830 | 0.0000 | 0.8510 | 0.2344 | 0.0625 | 0.0000 |
| 13 | 90.3515 | -1.8544 | 0.0000 | 0.8372 | 0.1172 | 0.2344 | 0.0000 |
| 14 | 88.2024 | -1.7860 | 0.0000 | 0.8292 | 0.0781 | 0.3047 | 0.0000 |

## Difficulty versus later improvement

Positive loss improvement means `loss[t] - loss[t+h]`; positive mask-IoU improvement means `IoU[t+h] - IoU[t]`. Pearson and Spearman values are sample-wise correlations within each epoch window; `n/a` means the target or signal was constant.

| Signal | Target | Horizon | Defined windows | Mean Pearson | Median Pearson | Mean Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Instant-Loss percentile | loss improvement: loss[t] - loss[t+h] | 1 | 14 | 0.3542 | 0.4133 | 0.4552 |
| Instant-Loss percentile | loss improvement: loss[t] - loss[t+h] | 2 | 13 | 0.4695 | 0.4714 | 0.5473 |
| Instant-Loss percentile | mask IoU improvement: IoU[t+h] - IoU[t] | 1 | 12 | 0.0933 | 0.1205 | 0.0813 |
| Instant-Loss percentile | mask IoU improvement: IoU[t+h] - IoU[t] | 2 | 10 | 0.3392 | 0.3403 | 0.2611 |
| Instant-Loss percentile | FN reduction: FN[t] - FN[t+h] | 1 | 14 | 0.0611 | 0.0177 | 0.0614 |
| Instant-Loss percentile | FN reduction: FN[t] - FN[t+h] | 2 | 13 | 0.0922 | 0.0016 | 0.0942 |
| Training-Dynamics difficulty | loss improvement: loss[t] - loss[t+h] | 1 | 14 | 0.3293 | 0.4072 | 0.4344 |
| Training-Dynamics difficulty | loss improvement: loss[t] - loss[t+h] | 2 | 13 | 0.4318 | 0.4223 | 0.5315 |
| Training-Dynamics difficulty | mask IoU improvement: IoU[t+h] - IoU[t] | 1 | 12 | -0.0192 | 0.1448 | -0.0801 |
| Training-Dynamics difficulty | mask IoU improvement: IoU[t+h] - IoU[t] | 2 | 10 | 0.2337 | 0.2048 | 0.0565 |
| Training-Dynamics difficulty | FN reduction: FN[t] - FN[t+h] | 1 | 14 | 0.1122 | 0.0271 | 0.1007 |
| Training-Dynamics difficulty | FN reduction: FN[t] - FN[t+h] | 2 | 13 | 0.1493 | 0.0395 | 0.1357 |

## HARD_LEARNABLE natural improvement

- Final-epoch `HARD_LEARNABLE` count: `28`.
- Earlier `HARD_LEARNABLE` events with a later epoch available: `383` across `41` samples.
- Events with next-epoch loss improvement: `297`; with any later loss improvement: `366`.
- Events with next-epoch mask-IoU improvement: `22`; with any later mask-IoU improvement: `32`.
- First HARD_LEARNABLE entries with a future epoch: `41`; per-entry horizon summaries include loss, mask-IoU, FN reduction and leaving the instant HARD bucket.
- Controlled-corruption samples are excluded from the natural-recovery estimate; excluded HARD events: `36`.

`HARD_LEARNABLE` remains a high-loss candidate label in this report. These counts do not establish that the sample is learnable, and the final epoch is excluded from future-improvement counts because it has no later observation.

### HARD_LEARNABLE first-entry outcomes

| Horizon | Available entries | Loss improved | Mask IoU improved | FN reduced | Left instant HARD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41 | 36 | 1 | 1 | 8 |
| 2 | 41 | 37 | 1 | 3 | 7 |
| 3 | 40 | 38 | 1 | 3 | 8 |

The per-event and per-sample distributions, including all transition counts and first-entry IDs, remain in the analysis JSON; no `HARD_LEARNABLE` label is interpreted as proof of learnability.

## Frozen reference subsets

These sets are derived only from E2 and must be reused unchanged by any later intervention run.

| Subset | Count | Rule |
| --- | ---: | --- |
| REFERENCE_HARD | 34 | non-corruption train samples with at least 3 epochs in instant HARD or at least 3 epochs in HARD_LEARNABLE |
| REFERENCE_MASTERED | 33 | non-corruption train samples with at least 3 MASTERED state epochs |
| CLEAN_NATURAL_HARD | 34 | REFERENCE_HARD samples that are not controlled corruptions; this is a frozen natural-hard comparison subset, not an intervention-derived label |
| CONTROLLED_CORRUPTION | 5 | the five manifest corruption records, by stable_sample_id |
| ALL_VALID | 58 | all validation stable_sample_id values from the frozen manifest |

## Manual audit

- Audited samples: `50`; pending labels: `0`.
- Labels: `{'questionable': 5, 'reasonable': 23, 'wrong': 22}`.
- Packet CSV: `reports/mvtec_rf4_manual_audit.csv`; decisions: `reports/mvtec_rf4_manual_audit_decisions.json`.
- Labels are a visual/protocol review of probe/state interpretation, not a model mAP estimate. `wrong` specifically marks the observed empty-GT frozen-core conflict semantic on visually normal samples.

## Controlled-corruption smoke

The five corruptions are tracked as trajectories only. This section does not promote their precision/recall to an RF8 statistic.

| Corruption | Final state | Max analysis conflict streak | Final analysis conflict streak | Any SUSPECT | Final FN | Final FP | Final class error | Final mask IoU |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| mask_shift (`train:1023827126115464097:grid/test/thread/006.png`) | HARD_LEARNABLE | 15 | 15 | False | 1 | 0 | 0 | n/a |
| drop_mask (`train:1718245752427016899:pill/test/faulty_imprint/014.png`) | LEARNING | 0 | 0 | False | 0 | 0 | 0 | n/a |
| mask_shift (`train:4832004734997902173:capsule/test/crack/020.png`) | HARD_LEARNABLE | 15 | 15 | True | 1 | 0 | 0 | n/a |
| drop_mask (`train:5899465832141176488:grid/test/broken/003.png`) | LEARNING | 0 | 0 | False | 0 | 0 | 0 | n/a |
| drop_component (`train:6016708638749308861:grid/test/broken/010.png`) | HARD_LEARNABLE | 11 | 0 | False | 0 | 3 | 0 | 0.7042 |

## RF4 Gate conclusion

**FAIL_SEMANTIC_REVIEW_REQUIRED**: the 15-epoch E2 evidence is complete and replayable, but the frozen core still records empty-GT `gt_recall=0` as probe conflict (`930` observations). This semantic correctness issue blocks the RF4 Gate; no E3/E4/E5 benefit, RF8 precision, or stage-aware policy claim is made.

Analysis JSON: `/workspace/rfdetr/reports/mvtec_rf4_longitudinal_analysis.json`

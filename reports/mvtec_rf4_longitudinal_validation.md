# MVTec RF4 Longitudinal Validation

> Gate: **PASS_WITH_CAVEATS**
> Scope: **NON-BENCHMARK / RF4 MECHANISM VALIDATION**
> The frozen StatePolicy and thresholds were unchanged; only the minimal empty-GT probe-conflict correctness fix was applied before this rerun.

## Run contract

- Run: `/workspace/runs/e2-rf4-15-corrected`
- Epochs / samples per epoch: `15 / 128`
- Batch / seed: `4 / 20260810`
- Resolution: `384`
- Manifest SHA-256: `90673182af0c409e99778f30545f550e91b6845e48f6894a7ebe029c53172528`
- Checkpoint SHA-256: `6de3da31b2572cac214a1c76cce4a92a13966d56390ac2b3a3de9a8dc2b2bca3`
- State evidence SHA-256: `e3c247567006bdbb61a742a4de74f745d5a6014d05651279d28dc74d7203162f`
- Replay final state match: **True**
- Frozen StatePolicy verified: **True**
- Empty-GT observations counted as probe conflicts after the correctness fix: `0`; undefined empty-GT recall/mask placeholders are ignored, while real FP remains an error signal.
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
| 1 | 32 | 64 | 32 | 96 | 0 | 32 | 0 | 10 | 66 |
| 2 | 32 | 64 | 32 | 68 | 28 | 32 | 0 | 32 | 66 |
| 3 | 32 | 64 | 32 | 65 | 31 | 31 | 1 | 20 | 66 |
| 4 | 32 | 64 | 32 | 64 | 32 | 30 | 2 | 18 | 66 |
| 5 | 32 | 64 | 32 | 69 | 27 | 30 | 2 | 21 | 66 |
| 6 | 32 | 64 | 32 | 71 | 25 | 30 | 2 | 18 | 66 |
| 7 | 0 | 96 | 32 | 96 | 0 | 28 | 4 | 33 | 66 |
| 8 | 0 | 96 | 32 | 96 | 0 | 29 | 3 | 11 | 66 |
| 9 | 0 | 96 | 32 | 96 | 0 | 32 | 0 | 7 | 66 |
| 10 | 0 | 96 | 32 | 96 | 0 | 30 | 2 | 4 | 63 |
| 11 | 32 | 64 | 32 | 65 | 31 | 28 | 4 | 37 | 55 |
| 12 | 32 | 64 | 32 | 77 | 19 | 31 | 1 | 24 | 29 |
| 13 | 32 | 64 | 32 | 80 | 16 | 31 | 1 | 13 | 15 |
| 14 | 32 | 64 | 32 | 87 | 9 | 31 | 1 | 15 | 12 |

## State stability

| Trajectory | Total transitions | Mean | Median | P90 | Unchanged sample fraction | Mean unique labels/sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Instant Loss buckets | 208 | 1.6250 | 1.0000 | 4.0000 | 0.3516 | 1.6484 |
| Training Dynamics states | 263 | 2.0547 | 2.0000 | 4.0000 | 0.3125 | 1.7344 |

The instant baseline is a three-bucket current-loss view (`EASY/MIDDLE/HARD`), while Dynamics is the frozen four-state view. Their churn is therefore compared as stability evidence, not as an accuracy ranking.

The corrected core and replay now share the same empty-GT rule: serialized `gt_recall=0` and no-match mask placeholders are undefined when `gt_count=0` and do not create probe conflict. A real FP on an empty-GT image still contributes difficulty; nonempty-GT FN/class/mask rules are unchanged.

## Longitudinal diagnostic means

| Epoch | Mean loss EMA | Mean slope | Mean forgetting | Mean mask IoU | Mean FN | Mean FP | Mean class error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 193.7768 | 0.0000 | 0.0000 | 0.7062 | 0.4453 | 0.8828 | 0.0703 |
| 1 | 180.6000 | -43.9226 | 0.0000 | 0.7399 | 0.4141 | 0.7500 | 0.1016 |
| 2 | 166.2084 | -30.5744 | 0.0000 | 0.7624 | 0.4297 | 0.5391 | 0.0859 |
| 3 | 152.0793 | -24.1222 | 0.0000 | 0.8238 | 0.4453 | 0.4062 | 0.0703 |
| 4 | 139.6626 | -19.6916 | 0.0000 | 0.7940 | 0.4531 | 0.2578 | 0.0625 |
| 5 | 129.4429 | -11.0452 | 0.0000 | 0.8638 | 0.4531 | 0.2422 | 0.0625 |
| 6 | 121.1447 | -7.5206 | 0.0000 | 0.8618 | 0.4609 | 0.1328 | 0.0547 |
| 7 | 114.3909 | -4.9867 | 0.0000 | 0.8623 | 0.4688 | 0.0781 | 0.0469 |
| 8 | 108.4554 | -3.9134 | 0.0000 | 0.8758 | 0.4922 | 0.0781 | 0.0234 |
| 9 | 103.7121 | -3.3081 | 0.0000 | 0.8521 | 0.5078 | 0.0625 | 0.0078 |
| 10 | 100.1578 | -2.5823 | 0.0000 | 0.9469 | 0.4766 | 0.1094 | 0.0156 |
| 11 | 96.9732 | -2.0921 | 0.0000 | 0.8802 | 0.4219 | 0.0859 | 0.0078 |
| 12 | 94.2428 | -1.6570 | 0.0000 | 0.8550 | 0.2266 | 0.0625 | 0.0000 |
| 13 | 91.5974 | -1.8432 | 0.0000 | 0.8454 | 0.1172 | 0.1953 | 0.0000 |
| 14 | 88.9935 | -2.2011 | 0.0000 | 0.8360 | 0.0859 | 0.1641 | 0.0000 |

## Difficulty versus later improvement

Positive loss improvement means `loss[t] - loss[t+h]`; positive mask-IoU improvement means `IoU[t+h] - IoU[t]`. Pearson and Spearman values are sample-wise correlations within each epoch window; `n/a` means the target or signal was constant.

| Signal | Target | Horizon | Defined windows | Mean Pearson | Median Pearson | Mean Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Instant-Loss percentile | loss improvement: loss[t] - loss[t+h] | 1 | 14 | 0.3973 | 0.3819 | 0.4479 |
| Instant-Loss percentile | loss improvement: loss[t] - loss[t+h] | 2 | 13 | 0.5245 | 0.5326 | 0.5780 |
| Instant-Loss percentile | mask IoU improvement: IoU[t+h] - IoU[t] | 1 | 12 | 0.1143 | 0.2276 | 0.1116 |
| Instant-Loss percentile | mask IoU improvement: IoU[t+h] - IoU[t] | 2 | 11 | -0.0158 | 0.2213 | -0.0731 |
| Instant-Loss percentile | FN reduction: FN[t] - FN[t+h] | 1 | 14 | 0.0595 | 0.0072 | 0.0598 |
| Instant-Loss percentile | FN reduction: FN[t] - FN[t+h] | 2 | 13 | 0.0875 | -0.0087 | 0.0898 |
| Training-Dynamics difficulty | loss improvement: loss[t] - loss[t+h] | 1 | 14 | 0.3698 | 0.3345 | 0.4345 |
| Training-Dynamics difficulty | loss improvement: loss[t] - loss[t+h] | 2 | 13 | 0.4735 | 0.4805 | 0.5624 |
| Training-Dynamics difficulty | mask IoU improvement: IoU[t+h] - IoU[t] | 1 | 12 | 0.2627 | 0.2982 | 0.2386 |
| Training-Dynamics difficulty | mask IoU improvement: IoU[t+h] - IoU[t] | 2 | 11 | -0.0426 | 0.0956 | -0.0790 |
| Training-Dynamics difficulty | FN reduction: FN[t] - FN[t+h] | 1 | 14 | 0.1014 | -0.0029 | 0.0979 |
| Training-Dynamics difficulty | FN reduction: FN[t] - FN[t+h] | 2 | 13 | 0.1374 | -0.0120 | 0.1319 |

## HARD_LEARNABLE natural improvement

- Final-epoch `HARD_LEARNABLE` count: `28`.
- Earlier `HARD_LEARNABLE` events with a later epoch available: `386` across `42` samples.
- Events with next-epoch loss improvement: `304`; with any later loss improvement: `377`.
- Events with next-epoch mask-IoU improvement: `19`; with any later mask-IoU improvement: `28`.
- First HARD_LEARNABLE entries with a future epoch: `42`; per-entry horizon summaries include loss, mask-IoU, FN reduction and leaving the instant HARD bucket.
- Controlled-corruption samples are excluded from the natural-recovery estimate; excluded HARD events: `40`.

`HARD_LEARNABLE` remains a high-loss candidate label in this report. These counts do not establish that the sample is learnable, and the final epoch is excluded from future-improvement counts because it has no later observation.

### HARD_LEARNABLE first-entry outcomes

| Horizon | Available entries | Loss improved | Mask IoU improved | FN reduced | Left instant HARD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 42 | 37 | 0 | 4 | 9 |
| 2 | 42 | 41 | 1 | 4 | 9 |
| 3 | 41 | 38 | 0 | 4 | 10 |

The per-event and per-sample distributions, including all transition counts and first-entry IDs, remain in the analysis JSON; no `HARD_LEARNABLE` label is interpreted as proof of learnability.

## Frozen reference subsets

These sets are derived only from E2 and must be reused unchanged by any later intervention run.

| Subset | Count | Rule |
| --- | ---: | --- |
| REFERENCE_HARD | 33 | non-corruption train samples with at least 3 epochs in instant HARD or at least 3 epochs in HARD_LEARNABLE |
| REFERENCE_MASTERED | 43 | non-corruption train samples with at least 3 MASTERED state epochs |
| CLEAN_NATURAL_HARD | 33 | REFERENCE_HARD samples that are not controlled corruptions; this is a frozen natural-hard comparison subset, not an intervention-derived label |
| CONTROLLED_CORRUPTION | 5 | the five manifest corruption records, by stable_sample_id |
| ALL_VALID | 58 | all validation stable_sample_id values from the frozen manifest |

## Controlled-corruption smoke

The five corruptions are tracked as trajectories only. This section does not promote their precision/recall to an RF8 statistic.

| Corruption | Final state | Max analysis conflict streak | Final analysis conflict streak | Any SUSPECT | Final FN | Final FP | Final class error | Final mask IoU |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| mask_shift (`train:1023827126115464097:grid/test/thread/006.png`) | HARD_LEARNABLE | 15 | 15 | True | 1 | 0 | 0 | n/a |
| drop_mask (`train:1718245752427016899:pill/test/faulty_imprint/014.png`) | LEARNING | 0 | 0 | False | 0 | 0 | 0 | n/a |
| mask_shift (`train:4832004734997902173:capsule/test/crack/020.png`) | HARD_LEARNABLE | 15 | 15 | False | 1 | 0 | 0 | n/a |
| drop_mask (`train:5899465832141176488:grid/test/broken/003.png`) | LEARNING | 0 | 0 | False | 0 | 0 | 0 | n/a |
| drop_component (`train:6016708638749308861:grid/test/broken/010.png`) | HARD_LEARNABLE | 12 | 0 | False | 0 | 1 | 0 | 0.7033 |

## RF4 Gate conclusion

**PASS_WITH_CAVEATS**: the 15-epoch E2 evidence is complete and replayable under the corrected empty-GT semantics. The RF4 evidence is suitable for the next E5 intervention gate, with the controlled-corruption five-sample result retained as smoke only; no RF8 precision/recall claim is made.

Analysis JSON: `/workspace/rfdetr/reports/mvtec_rf4_longitudinal_analysis.json`

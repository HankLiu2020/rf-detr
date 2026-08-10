# State Policy Specification

> Draft status: correctness-verified, not frozen pending real-data E0–E5.

The policy consumes one aggregated record per sample per epoch plus deterministic-probe records. Loss rank uses `weighted_per_image_normalized_loss`, not the global-normalized scalar reconstruction view. The default classifier uses `MASTERED` only after at least three history windows with low rank and non-positive trend, `HARD_LEARNABLE` as the current high-loss candidate bucket, `LEARNING` for the middle, and `SUSPECT` only after cross-epoch hard patience, probe-conflict patience, and a stalled/non-improving slope.

The component scores are bounded to `[0, 1]`: loss percentile, probe error severity, stalled trend score, and forgetting score. Detection probe severity uses FN, class error, and FP without double-counting a match. Segmentation probes additionally record box-matched mask IoU; low mask IoU is a probe conflict and contributes a configurable mask-error term. The component defaults `0.50/0.30/0.15/0.05` and mask-error multiplier `0.5` are policy parameters, not assumptions that a downstream framework must hard-code.

The current weights are a state prototype rather than a validated stage policy. In particular, `HARD_LEARNABLE` means high-loss candidate until replay-response evidence defines “learnable.” Early/mid/late schedules and localization emphasis must be selected from real-data ablations before this document is frozen.

# State Policy Specification

The policy consumes detached observer records and optional deterministic-probe records. Loss rank is an empirical percentile within the current observation set. The default classifier uses `MASTERED` for low rank with non-positive trend, `HARD_LEARNABLE` for high rank, `LEARNING` for the middle, and `SUSPECT` only after hard patience, probe-conflict patience, and a stalled/non-improving slope.

The component scores are bounded to `[0, 1]`: loss percentile, probe error severity, stalled trend score, and forgetting score. Their default weights are `0.50/0.30/0.15/0.05`; these are policy parameters, not assumptions that a downstream framework must hard-code.

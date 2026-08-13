# Policy V2 Status

| Stage | Status | Evidence |
|---|---|---|
| V1 immutable baseline | PASS | `4cc9fc5e0392d3be31235d9ce93632a7f34a35ba` + four recorded SHA256 values |
| Convergence diagnosis | PASS | `USE_INDEPENDENT_30_EPOCH_CONTRACT` |
| V1 coverage/exposure diagnosis | PASS | `SUPPORTED` |
| Hard-definition offline analysis | PASS_WITH_CAVEATS | predictive, not causal; frontier frozen before discovery |
| Loss-component diagnosis | PASS_WITH_CAVEATS | `COMPLETE_SEED1_LONGITUDINAL` |
| Policy V2 discovery matrix | PASS | frozen seed 20260813, unified 30-epoch contract |
| C0/D0 baseline | PASS | final mask mAP `0.1552`; best `0.1758` at epoch 28 |
| C3 V1 loss weight | PASS_WITH_CAVEATS | final Δmask vs C0 `-0.0188` |
| C4 V1 sampler | PASS_WITH_CAVEATS | final Δmask vs C0 `-0.0288` |
| C5 V1 combined | PASS_WITH_CAVEATS | final Δmask vs C0 `-0.0221` |
| First D1 attempts | FAIL | interrupted and excluded: two sampler length boundary defects found by independent Verify |
| D1 coverage + warmup + 5% frontier bonus | PASS | final mask `0.1827` (`+0.0275` vs C0), F1 `+0.0456`; 100% coverage, 200 bonus appearances |
| Frozen Hard recovery | PASS_WITH_CAVEATS | D1−C0: loss `+17.64`, mask IoU `+0.0985`, FN reduction `+0.1212`; one discovery seed |
| Independent Verify | PASS_WITH_CAVEATS | no remaining P0/P1; final provenance and full 30-epoch artifacts present |

Next gate: D1 qualifies for untouched confirmatory seeds `20260814–20260816`, but `ALGORITHM_BENEFIT` remains unpassed until paired replication. D2–D4 remain unstarted and the discovery policy is frozen.

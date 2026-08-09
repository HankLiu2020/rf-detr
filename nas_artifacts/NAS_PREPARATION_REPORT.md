# RF-DETR-Seg-Small Native-Bounded NAS Preparation Report

Date: 2026-08-09 (Asia/Shanghai)

## Status

`NAS_PREPARATION_STATUS=PASS`
`NAS_PREPARATION_HARDENED=PASS`

This report covers the implementation and bounded CPU/GPU verification for the
native-bounded RF-DETR-Seg-Small NAS preparation. The earlier container checks
remain read-only/no-privileged; the current bounded target-GPU regression ran in
the existing host CUDA environment because the Docker socket is not accessible
from this session.

Formal supernet training, full subnet sweep, Pareto search, and TensorRT
benchmarking were not executed.

The 2026-08-09 review hardening is recorded in
`nas_artifacts/hardening_regression.json`. It removes dynamic `num_select`
from `ArchitectureSpec`, keeps the native PostProcess policy fixed, adds the
Controller native-state snapshot/reset lifecycle, validates the Group-DETR
train/eval width invariant, and retains the transformer runtime proposal
tensor guard. The bounded CPU regression and the fixed four-case target-GPU
regression both passed. Formal search remains disabled and target data remains
unready.

## Hardening addendum

| Area | Result | Evidence |
|---|---|---|
| Native PostProcess policy | PASS | `ArchitectureSpec` has no dynamic `num_select`; SearchSpace metadata records `native_fixed` and Controller never rewrites PostProcess top-k. |
| Controller lifecycle | PASS | Immutable `NativeStateSnapshot`, `reset_to_native()`, value/type restoration test, and real-model A-B-C-A allclose. |
| Group-DETR invariant | PASS | Training `Q_total=active_q*group_detr`, evaluation `Q_total=active_q`; `active_q=50`, `group_detr=13` is accepted. |
| Proposal estimator | PASS | Renamed `estimate_encoder_proposal_pool_size()` and checked against a real encoder tensor width; transformer runtime guard retained. |
| Decoder 0 evaluator path | PASS | Encoder-only output shape plus real `PostProcess` regression. |
| Target GPU hardening regression | PASS | Fixed native, 480/p20/w2, 576/p12/w1, decoder=0/q50 cases; 3 backward checks and A-B-C-A all passed on RTX 3090. |
| Formal search safety | PASS | `full_search.enabled=false`; no formal search, Pareto search, or full subnet sweep executed. |

The final focused pytest command passed 35 tests with 1 expected environment-
dependent skip; the skip is the existing CUDA-only PostProcess check on this
CPU-visible session.

## Native architecture

The runtime/checkpoint inspection, rather than the paper table, is authoritative:

| Field | Runtime value |
|---|---:|
| model variant | `RFDETRSegSmall` |
| resolution | 384 |
| patch size | 12 |
| windows | 2 |
| decoder layers | 4 |
| per-group active queries | 100 |
| `group_detr` | 13 |
| `num_select` | 100 |
| query/refpoint capacity | 1300 / 1300 |
| encoder proposal count | 1024 |
| mask feature resolution | 96 x 96 |
| normalization | 65 LayerNorm, 0 BatchNorm |

The checkpoint omits explicit `args.num_queries` and `args.group_detr`; this
discrepancy is preserved in the native artifact. The runtime configuration and
embedding shapes are used as the source of truth and are independently checked
by the Group DETR audit. The audit demonstrates that per-group slicing is
correct and flat slicing is not.

## Validation matrix

| ID | Result | Evidence |
|---|---|---|
| V-1 | PASS | `gpu_gate/runtime_environment.json`; current checkout imports in CUDA container |
| V0/V1 | PASS | `native_architecture.json`, `native_vs_paper_report.md` |
| V2/V3/V4 | PASS (CPU) | `baseline/` and `baseline/native_equivalence.json` |
| V5/V6/V7 | PASS (CPU) | `query_group_detr.json`, bounded query checks |
| V8/V9 | PASS (CPU) | decoder 0..4 and multi-resolution random probes |
| V10/V11/V12 | PASS (CPU) | `patch_window_controls.json`, pure NAS tests, random probes |
| V13 | PASS (CPU) | `optimizer_group_audit.json` and native dump layer-decay metadata |
| V14 | PASS | `norm_audit.json`: LayerNorm only, no recalibration required |
| V15 | PASS (CPU) | `loss_scale_audit.json`: 384/480/576, mask point count recorded |
| V16 | PASS | `hardware_feasibility.jsonl`: 576/12/1/native corner passed on RTX 3090; peak 1558436864 bytes |
| V17/V18 | PASS (GPU) | A-B-C-A state check and 10 target-GPU architecture records |
| V19 | PASS (GPU) | `smoke_training.jsonl`: 6 finite optimizer steps; required patch/window coverage assertion passed |
| V20 | PASS (GPU) | fresh-process `resume_verification.json`; EMA (544 params), native architecture, git commit, RF-DETR version, and base checkpoint SHA-256 all restored |
| V21 | PASS (CPU) | native/reduced TorchScript export and reload verification |
| V22/V23 | PASS | `FULL_SEARCH_MANIFEST.json`, `safety_lock_test.json` |

The CPU checks remain bounded functional checks; the target-GPU hardening gate
is covered by `tools/hardening_gpu_regression.py` and the summary in
`nas_artifacts/hardening_regression.json`.

The fixed ten-sample target-GPU probe covers patch 12/16/20, windows 1/2,
query 50/100, and decoder 0..4. Its pool is recorded as
`WORST_CORNER_PASS_TARGET_GPU_POOL`.

## Search and safety state

The native-derived search space contains 360 candidate records, 220 geometry-
valid records, 22 encoder tuples, decoder depth 0..4, and query candidates
50/100. All query and decoder choices are bounded by native capacity.

```text
HARDENING_TARGET_GPU_REGRESSION=PASS
FULL_SEARCH_EXECUTED=NO
PARETO_SEARCH_EXECUTED=NO
FULL_SUBNET_SWEEP_EXECUTED=NO
NAS_FULL_SEARCH_APPROVED=ABSENT
TARGET_DATASET_READY=NO
FORMAL_SEARCH_ENGINE_READY=NO
```

Both formal configurations keep `full_search.enabled: false`.
The dry-run manifest records all eight future search stages as
`executed: false` and reports the four-lock state as
`formal_search_allowed: false`; the additional target-data lock requires a
dataset id, split manifest, domain adapter, and explicit ready marker.

The seccomp A/B report shows that the same GPU/image/mount/capability setup
passes with `seccomp=unconfined` and the minimal custom profile, while Docker
default seccomp reproduces CUDA Error 304. It does not claim that RF-DETR or
CUDA requires disabling seccomp.

## Required next action

Commit the execution-preparation code and bounded regression artifacts that are
in scope, retain the four-lock safety state, and stop. No formal search, Pareto
search, full subnet sweep, or TensorRT batch benchmark is authorized in this
phase.

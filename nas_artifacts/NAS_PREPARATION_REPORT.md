# RF-DETR-Seg-Small Native-Bounded NAS Preparation Report

Date: 2026-08-07 (Asia/Shanghai)

## Status

`NAS_PREPARATION_STATUS=PASS`

This report covers the implementation and bounded CPU/GPU verification for the
native-bounded RF-DETR-Seg-Small NAS preparation. The current checkout was
verified in `train-env-rfdetr-claude:20260806` with project/model/data mounts
read-only, rootfs read-only, network disabled, no privileged flag, and no extra
capabilities.

Formal supernet training, full subnet sweep, Pareto search, and TensorRT
benchmarking were not executed.

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

The CPU checks remain bounded functional checks; the target-GPU gate is now
covered by the artifacts under `nas_artifacts/gpu_gate/` and
`nas_artifacts/gpu_smoke/`.

The fixed ten-sample target-GPU probe covers patch 12/16/20, windows 1/2,
query 50/100, and decoder 0..4. Its pool is recorded as
`WORST_CORNER_PASS_TARGET_GPU_POOL`.

## Search and safety state

The native-derived search space contains 360 candidate records, 220 geometry-
valid records, 22 encoder tuples, decoder depth 0..4, and query candidates
50/100. All query and decoder choices are bounded by native capacity.

```text
FULL_SEARCH_EXECUTED=NO
PARETO_SEARCH_EXECUTED=NO
FULL_SUBNET_SWEEP_EXECUTED=NO
NAS_FULL_SEARCH_APPROVED=ABSENT
WAITING_FOR_GPU_RESOURCE=YES
```

Both formal configurations keep `full_search.enabled: false`.
The dry-run manifest also records all eight future search stages as
`executed: false` and reports the three-lock state as
`formal_search_allowed: false`.

The seccomp A/B report shows that the same GPU/image/mount/capability setup
passes with `seccomp=unconfined` and the minimal custom profile, while Docker
default seccomp reproduces CUDA Error 304. It does not claim that RF-DETR or
CUDA requires disabling seccomp.

## Required next action

Commit the preparation code and artifacts that are in scope, retain the
three-lock safety state, and stop. No formal search, Pareto search, full subnet
sweep, or TensorRT batch benchmark is authorized in this phase.

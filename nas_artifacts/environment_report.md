# RF-DETR Seg-Small NAS Environment Gate

Status: `PASS`

The version-controlled target checkout is `/home/liujiyuan/rf-detr` at commit
`cc538cea510c24d6d7bc64332f0bf29875a5b2d6`. GPU execution was verified by
mounting that checkout read-only into `train-env-rfdetr-claude:20260806`.

The container reports PyTorch `2.7.1+cu128`, CUDA runtime `12.8.61`, driver
`570.211.01`, six RTX 3090 devices, and imports RF-DETR from
`/workspace/rf-detr/src/rfdetr/__init__.py`. The Small-Seg checkpoint is
readable and has SHA-256
`6de3da31b2572cac214a1c76cce4a92a13966d56390ac2b3a3de9a8dc2b2bca3`.

The selected dataset is `/home/inspur1/data/MVTec-AD/pill`. Its image and
pixel-mask files are readable, but it is an MVTec anomaly-segmentation layout,
not COCO/RF-DETR annotations. NAS smoke training must therefore use an explicit
MVTec adapter; this is recorded rather than treating the directory as a native
RF-DETR dataset.

The host Miniforge environment is CPU-only and is not used for GPU execution.
The container has PyTorch Lightning `2.6.5`; Kornia is not installed, so the
default augmentation backend for this preparation is torchvision-native.

Formal full search, Pareto search, full subnet sweep, and TensorRT batch
benchmarking are disabled by scope.

The gate snapshot above predates the implementation work. The current
preparation worktree changes are listed in `environment_report.json` under
`post_gate_worktree`.

## Current GPU gate — 2026-08-07 18:40 +08:00

The existing docker-group membership was refreshed for the verification
command with `sg docker`; no group or credential change was made. The current
checkout was mounted read-only into the approved image with rootfs read-only,
network disabled, no privileged flag, and no extra capabilities. CUDA import,
the 576/12/1/native worst corner, ten random architectures, three backward
checks, GPU smoke training, and fresh-process resume all passed. The detailed
seccomp A/B evidence is in `nas_artifacts/seccomp_ab_report.json`.

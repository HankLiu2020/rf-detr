# RF-DETR Sample Dynamics 验证报告

## 结论

本分支在官方 `develop` 基线 `c107a748bbf369ad455cd41d85b9c91a5a0ecbad` 上实现了 RF1–RF9 的可选路径，并为 RF-DETR Seg Small 提供了可复现配置和 smoke 脚本。默认配置保持关闭，因此默认 loss、matcher、optimizer、DataLoader 行为不变。

验证使用 `rf-detr-claude` Docker 镜像中的 Python 3.10.12、PyTorch 2.7.1+cu128、Lightning 2.6.5 和 RTX 3090；没有安装或修改 Conda。真实数据集未随仓库提供，因此没有伪造 mAP 或 ablation 数字。

## 已验证

- RF1 stable sample ID 贯穿 COCO/YOLO dataset、transform、collate 和 device transfer。
- RF2 observer 复用 matcher，记录 cls/bbox/GIoU/mask/keypoint per-image numerator，并验证默认 scalar 与 gradient 不变；aux/enc 前缀显式保留。
- RF3 fixed transform、sequential、eval、no-grad train probe，重复运行结果一致且恢复 train state。
- RF4 EMA、window slope、variance、hard patience、forgetting、difficulty 和四态 state store 可序列化恢复。
- RF5 previous-policy loss weights 做均值归一化、clip，SUSPECT 不提升权重。
- RF6 global epoch index → rank slice，覆盖 small dataset、GradAccumAlignedDataset original index、seed+epoch 和动态 DataModule sampler。
- RF7 combined path 对 `loss_multiplier * sampling_multiplier` 施加默认 `effective_cap=2.5`。
- RF8 只导出 SUSPECT review queue，不删图、不改标签。

## Docker 验证命令

```bash
export RFDETR_SKIP_DINOV2_PREWARM=1
python -m pytest tests/sample_dynamics tests/models/test_criterion.py tests/training/test_module_data.py -n 1 -m 'not gpu' --timeout=240
pre-commit run --all-files
```

`tests/sample_dynamics`、criterion 和 DataModule targeted tests 已通过；pre-commit 的非-mypy hooks 已通过。仓库原有 mypy 基线仍有若干类型错误（见 `reports/failure_cases.md`），不是本分支新增的 runtime 失败。

最新 targeted suite 实际结果为 `397 passed, 3 skipped`；非-mypy pre-commit hooks 全部通过。mypy 仅剩仓库既有的 8 个基线错误，集中在 tensors、COCO eval、matcher、YOLO/PyYAML、LW-DETR 和 detr 配置类型声明。

## Seg Small smoke

不下载预训练权重的架构 smoke：

```bash
python experiments/seg_small_sample_dynamics_smoke.py --device cuda
```

该命令实际构造 `RFDETRSegSmallConfig` 的 segmentation head 并执行一次 GPU inference；训练数据集、权重和 mAP 需要由使用者提供后再执行完整 E0–E5 矩阵。

本次 Docker 实测输出为 `variant=rfdetr-seg-small detections=30 has_masks=True`。两 rank GPU smoke `torchrun --standalone --nproc_per_node=2 experiments/ddp_sampler_smoke.py` 也通过，报告 `world_size=2` 且正常完成 barrier。

Lightning 集成 smoke：

```bash
python experiments/lightning_dynamic_sampler_smoke.py
```

该命令在两张 GPU、`ddp_spawn`、真实 `RFDETRDataModule` 动态 sampler 和 `use_distributed_sampler=False` 下完成一个短 fit，实际输出 `lightning_dynamic_sampler_smoke=PASS world_size=2`。它覆盖了 `GradAccumAlignedDataset` 对齐路径；另一个 `ddp_sampler_smoke.py` 覆盖了 global-list/rank-slice 语义。

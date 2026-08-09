# RF-DETR Sample Dynamics 验证报告

## 结论

本分支在官方 `develop` 基线 `c107a748bbf369ad455cd41d85b9c91a5a0ecbad` 上实现了 Sample Dynamics Reference Implementation，并完成 Round 2 correctness 修正。它不是算法效果验收：RF9 migration contract 仍是 draft，必须等真实 E0–E5 后才能冻结。默认配置保持关闭，因此默认 loss、matcher、optimizer、DataLoader 行为不变。

验证使用 `rf-detr-claude` Docker 镜像中的 Python 3.10.12、PyTorch 2.7.1+cu128、Lightning 2.6.5 和 RTX 3090；没有安装或修改 Conda。真实数据集未随仓库提供，因此没有伪造 mAP 或 ablation 数字。

## 已验证

- RF1 stable sample ID 贯穿 COCO/YOLO dataset、transform、collate 和 device transfer。
- RF2 observer 复用 matcher，记录 cls/bbox/GIoU/mask/keypoint per-image numerator，并验证默认 scalar 与 gradient 不变；aux/enc 前缀显式保留。
- RF3 fixed transform、sequential、eval、no-grad train probe，重复运行结果一致且恢复 train state。
- RF3→RF4 已接入 epoch lifecycle：rank0 按 interval 执行 probe，结果与全局 observations 一起更新 state。
- RF4 先按 epoch/sample 聚合 replay，再更新一次 history/patience；difficulty 优先使用 per-image normalized loss。
- RF5 classification 使用 image-mean 权重；box/mask/keypoint 使用 global matched-instance-mass 归一，SUSPECT 不提升权重。
- RF6 各 rank observations gather 到 rank0，唯一 state/policy snapshot broadcast；Sampler 的 global plan 也由 rank0 broadcast 后再切片。
- RF7 combined path 对 `loss_multiplier * sampling_multiplier` 施加默认 `effective_cap=2.5`。
- RF8 只导出 SUSPECT review queue，不删图、不改标签。
- Observation buffer 在 epoch policy boundary drain；有限容量超限会显式失败，不再静默裁剪或依赖位置 cursor。
- Probe 先做 score threshold，再按 IoU 一对一匹配；wrong-class 只计 class error，不重复计 FN/FP。

## Docker 验证命令

```bash
export RFDETR_SKIP_DINOV2_PREWARM=1
python -m pytest tests/sample_dynamics tests/models/test_criterion.py tests/training/test_module_data.py -n 1 -m 'not gpu' --timeout=240
pre-commit run --all-files
```

`tests/sample_dynamics`、criterion 和 DataModule targeted tests 已通过；pre-commit 的非-mypy hooks 已通过。仓库原有 mypy 基线仍有若干类型错误（见 `reports/failure_cases.md`），不是本分支新增的 runtime 失败。

Round 2 targeted suite 实际结果为 `414 passed, 3 skipped`；非-mypy pre-commit hooks 全部通过。mypy 仅剩仓库既有的 8 个基线错误，集中在 tensors、COCO eval、matcher、YOLO/PyYAML、LW-DETR 和 detr 配置类型声明。

## Seg Small smoke

不下载预训练权重的架构 smoke：

```bash
python experiments/seg_small_sample_dynamics_smoke.py --device cuda
```

该命令实际构造 `RFDETRSegSmallConfig` 的 segmentation head 并执行一次 GPU inference；训练数据集、权重和 mAP 需要由使用者提供后再执行完整 E0–E5 矩阵。

本次 Docker 实测输出为 `variant=rfdetr-seg-small detections=30 has_masks=True`。两 rank GPU smoke `torchrun --standalone --nproc_per_node=2 experiments/ddp_sampler_smoke.py` 使用不同 rank-local observations，验证 gather、唯一 state broadcast、故意污染 rank1 后的 rank0 sampler-plan broadcast，输出 `ddp_state_sampler_smoke=PASS world_size=2`。

Lightning 集成 smoke：

```bash
python experiments/lightning_dynamic_sampler_smoke.py
```

该命令在两张 GPU、`ddp_spawn`、真实 `RFDETRDataModule` 动态 sampler、生产 `RFDETRModelModule.on_train_epoch_end()` 和 `use_distributed_sampler=False` 下完成两个 epoch，两个 rank 每个 epoch 都断言 state snapshot 相同，实际输出 `lightning_dynamic_sampler_smoke=PASS world_size=2`。

## 当前 Gate

- Reference Implementation correctness：`PASS WITH FIXES`。
- Algorithm Validation：`NOT VALIDATED`。
- RF9 Migration Contract：`DRAFT / NOT FROZEN`。
- 下一 Gate：真实 Seg Small 数据集上的 E0/E1/E2/E3/E4/E5、长期内存/开销、hard subset、SUSPECT precision 和 stage-aware policy。

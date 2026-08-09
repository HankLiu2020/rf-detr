# Failure Cases and Limits

- 仓库没有随任务提供 COCO/Roboflow 数据、类别定义或 checkpoint，因此不能诚实地报告完整 Seg Small mAP、review hit rate 或 E0–E5 训练收益；`ablation_results.csv` 对此明确标记。
- 首次完整 pre-commit 运行中，既有 mypy 错误位于 `src/rfdetr/utilities/tensors.py`、`src/rfdetr/evaluation/coco_eval.py`、`src/rfdetr/models/matcher.py`、`src/rfdetr/datasets/yolo.py`、`src/rfdetr/models/lwdetr.py` 和 `src/rfdetr/detr.py`。非-mypy hooks 通过；没有为本任务扩大无关修复范围。
- DDP sampler 的 state provider 要求各 rank 在 epoch 开始看到一致的 state snapshot；当前实现保证全局 index 生成语义和等长切片，但没有擅自引入跨 rank state broadcast。生产 DDP 应在 checkpoint/state 更新后广播或只由 rank 0 生成并同步 snapshot。
- `SUSPECT` 是隔离与复核状态，不是自动标签修正器。review queue 为空时仍写出空文件，不删除任何原始样本。
- point-mask loss 的 per-image observer 使用实际 sampled points；它是诊断 numerator，不应被解释成像素级全图损失。

# Failure Cases and Limits

- 仓库没有随任务提供 COCO/Roboflow 数据、类别定义或 checkpoint，因此不能诚实地报告完整 Seg Small mAP、review hit rate 或 E0–E5 训练收益；`ablation_results.csv` 对此明确标记。
- 首次完整 pre-commit 运行中，既有 mypy 错误位于 `src/rfdetr/utilities/tensors.py`、`src/rfdetr/evaluation/coco_eval.py`、`src/rfdetr/models/matcher.py`、`src/rfdetr/datasets/yolo.py`、`src/rfdetr/models/lwdetr.py` 和 `src/rfdetr/detr.py`。非-mypy hooks 通过；没有为本任务扩大无关修复范围。
- Round 2 已加入 rank0 authoritative state/policy 和 sampler-plan broadcast，并通过两 GPU divergent-state smoke；但尚未在长周期真实 Seg Small 训练中量化 Python object collective 的通信成本。
- Stage-aware policy 尚未冻结；当前固定 difficulty 权重仍属于 Training Dynamics State Prototype，matched IoU 的阶段性权重需要 E0–E5 数据支持后再确定。
- Probe threshold 默认 `0.05` 是可配置协议起点，不是经过目标数据集校准的最优值。
- `SUSPECT` 是隔离与复核状态，不是自动标签修正器。review queue 为空时仍写出空文件，不删除任何原始样本。
- point-mask loss 的 per-image observer 使用实际 sampled points；它是诊断 numerator，不应被解释成像素级全图损失。

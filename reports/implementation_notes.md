# Implementation Notes

## 插入点

`CocoDetection` 与 `YoloDetection` 在原始 metadata 层生成 stable ID；`RFDETRDataModule.transfer_batch_to_device` 只移动 tensor，保留字符串 ID。`SetCriterion.forward(..., return_per_sample=True)` 是显式 opt-in，主 matcher 的 indices 同时服务 scalar loss 和 observer packet。

`RFDETRModelModule` 只在 `sample_dynamics_enabled=True` 时建立 buffer/state/policy。`observe` 只记录，`loss_weight` 只改变显式 active reduction，`sampler` 只改变 DataLoader sampler，`combined` 才启用二者和 RF7 cap。

## State 与时序

epoch 结束时 `drain()` detached buffer，并按 `sample_id` 聚合为每图每 epoch 一条记录；replacement、padding 和 replay 不会在同一 epoch 重复推进 history/patience。rank0 更新 state 后生成下一版本 policy；当前 epoch 的 loss 不会反过来改变同一个 batch 的 weight。checkpoint 保存 state/policy，resume 后版本继续递增。

状态排序优先读取 `weighted_per_image_normalized_loss`。global-normalized loss 继续保留，用于重构官方 scalar、debug 和 audit，不再作为跨图片 difficulty 的首选信号。

`MASTERED` 默认至少需要三条跨 epoch history，避免首轮因 slope 缺省为零而提前降权。当前 `HARD_LEARNABLE` 仍是“高 loss 候选桶”；是否真的可通过 replay 改善，必须由 E4/E5 的 replay-response 轨迹验证。

## Sampler 与 DDP

各 rank 在 epoch boundary 先 gather 本地 observations；rank0 聚合、更新唯一 StateStore/policy，再向所有 rank broadcast checkpoint-safe snapshot。Sampler 的 global index plan 也只由 rank0 生成并 broadcast，所有 rank 最后执行 `global[rank::world_size]`。大数据集的 padding index 通过 `GradAccumAlignedDataset.original_index()` 映射回原始 stable ID；base coverage 无放回抽取并按需循环，hard/mastered/exploration replay 保留 replacement 语义。

RF5 的 image weight 也使用 Tensor `all_reduce` 在 global image batch 上归一；clip/effective-cap 后的 residual redistribution 同样按全局 eligible image 数计算。因此即使 rank0 全是 HARD、rank1 全是 MASTERED，也不会分别归一成两组全 1。box/mask/keypoint 分支再使用已有的 global matched-instance-mass denominator 保持 branch 尺度。

## Probe 与 review

`train_probe_dataloader()` 复用 train annotations，但替换成 val-style fixed resize transform；`run_deterministic_probe()` 保证 eval/no-grad 和原 train state 恢复。生产 epoch hook 按 `sample_dynamics_probe_interval` 只在 rank0 执行 Probe，并把结果正式传给 `SampleStateStore.update(..., probe_records=...)`。

Probe 把 `sample_dynamics_probe_score_threshold` 传入支持该参数的 PostProcess，使 Seg Small 在 full-resolution mask resize 前过滤 Top-K；匹配层会再次执行同一阈值契约。未匹配 GT 是 FN，未匹配 prediction 是 FP，IoU 匹配但类别错误只计 class error，不再重复计 FN/FP。Seg Small 在相同 box match 上额外计算 `matched_mask_iou`；低 mask IoU 进入 probe conflict 与 difficulty。最后由 `ReviewExporter` 输出 JSONL/JSON SUSPECT queue。

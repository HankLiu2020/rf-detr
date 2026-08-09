# Implementation Notes

## 插入点

`CocoDetection` 与 `YoloDetection` 在原始 metadata 层生成 stable ID；`RFDETRDataModule.transfer_batch_to_device` 只移动 tensor，保留字符串 ID。`SetCriterion.forward(..., return_per_sample=True)` 是显式 opt-in，主 matcher 的 indices 同时服务 scalar loss 和 observer packet。

`RFDETRModelModule` 只在 `sample_dynamics_enabled=True` 时建立 buffer/state/policy。`observe` 只记录，`loss_weight` 只改变显式 active reduction，`sampler` 只改变 DataLoader sampler，`combined` 才启用二者和 RF7 cap。

## State 与时序

epoch 结束时消费 detached buffer，先更新 state，再生成下一版本 policy；当前 epoch 的 loss 不会反过来改变同一个 batch 的 weight。checkpoint 保存 state/policy，resume 后版本继续递增。

## Sampler 与 DDP

每个 epoch 使用 `seed + epoch` 生成一个全局 index list，再以 `global[rank::world_size]` 切片。大数据集的 padding index 通过 `GradAccumAlignedDataset.original_index()` 映射回原始 stable ID；小数据集保留 replacement 语义。Lightning module epoch hook 会推进 sampler epoch。

## Probe 与 review

`train_probe_dataloader()` 复用 train annotations，但替换成 val-style fixed resize transform；`run_deterministic_probe()` 保证 eval/no-grad 和原 train state 恢复。probe 结果可传入 `SampleStateStore.update(..., probe_records=...)`，最后由 `ReviewExporter` 输出 JSONL/JSON SUSPECT queue。

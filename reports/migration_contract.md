# Sample Dynamics Migration Contract

> Status: **DRAFT / NOT FROZEN**. Round 3 correctness is implemented, but numerical policy choices and effect claims require real-data E0–E5 validation.

1. **逐图片 Loss 定义**：记录同一 batch/global normalizer 下的 raw numerator、global-normalized loss，并另存以 `max(gt_count_i, 1)` 为分母的 per-image 诊断值。
2. **cls / bbox / giou 拆分**：cls 保留 query 的正负/background contribution 后按 image 聚合；bbox L1 和 GIoU 通过 matcher 的 batch index scatter 回 `[B]`。
3. **aux / enc**：按 `loss_name_0`、`loss_name_enc` 显式保存；它们继续进入官方 total loss，是否进入 difficulty 由 policy 配置决定，默认 observer 组件可见但不重复计权。
4. **empty GT**：保留 background classification signal；bbox/GIoU/mask 对该图为零，`gt_count=0`、`matched_count=0`，per-image denominator 使用 clamp 保护。
5. **normalizer**：global `num_boxes` 是官方 batch/DDP loss 语义；per-image `gt_count`/`matched_count` 不冒充官方 normalizer。跨图片 difficulty 优先使用 `weighted_per_image_normalized_loss`，global-normalized loss 只用于 scalar reconstruction/debug/audit。
6. **stable sample_id**：必须是 `split:image_id:relative_path`，split 归一化、路径 POSIX 化；同一数据版本中不可因 shuffle、rank 或 transform 改变。
7. **Probe 指标**：固定 train split、resize、eval/no-grad/sequential，先按可配置 score threshold 过滤，再做一对一 IoU matching；unmatched GT=FN、unmatched prediction=FP、matched wrong class=class error，三者不重复计数。Segmentation 在 box match 上另算 matched mask IoU；Probe 结束后恢复 model train state。
8. **趋势算法**：默认 EMA、短窗口 least-squares slope、窗口 variance；算法放在可替换的 `StatePolicy`/纯函数中。
9. **state 规则**：至少 3 条跨 epoch history、低 percentile 且 slope 不上升才是 MASTERED；高 percentile 暂称 HARD_LEARNABLE candidate；中间为 LEARNING；高难度、probe 冲突、patience 达标且无改善为 SUSPECT。真正 learnable 必须由 replay-response 实验验证。
10. **forgetting**：previous state 为 MASTERED 后在后续 epoch 再次进入 hard percentile 的次数；同一 epoch 的 replay/padding/replacement 必须先按 sample 聚合，不能重复推进 patience/history/forgetting。
11. **Loss Weight 时序**：epoch `t` 的 batch 使用 epoch `t-1` 的 policy；epoch 末 `drain → group by sample_id → global gather → update once → next policy`。旧 batch records 不跨 policy boundary 常驻内存。
12. **推荐 Weight 范围**：初始 state weights `MASTERED=.8, LEARNING=1.0, HARD_LEARNABLE=1.3, SUSPECT=.8`，默认 clip `[.7, 1.3]`。classification 在 DDP global image batch 上做 mean normalization，clip/cap 后 residual 也全局重分配；box/mask/keypoint 再做 global matched-instance-mass normalization。
13. **Dynamic Sampler 语义**：base coverage 使用无放回 permutation/cycle；hard/mastered/exploration replay 使用 replacement draw。每 epoch 的 global list 只由 rank0 生成并 broadcast，再按 rank slice。
14. **DDP 必要条件**：各 rank observations gather 到 rank0，rank0 生成唯一 StateStore/policy snapshot 并 broadcast；Sampler global plan 同样由 rank0 broadcast；RF5 loss weights 用 Tensor collective 计算全局 image mean。所有 rank 仍需相同 dataset stable-ID 顺序、quota、epoch 和每-rank `num_samples`，且不能被 Lightning 第二个 sampler 覆盖。
15. **effective contribution cap**：近似 `loss_multiplier * sampling_multiplier`，combined 默认上限 `2.5`；若 cap 与 minimum weight 不可同时满足，应显式报错。
16. **SUSPECT 判据**：高难度 + 跨 epoch 连续观测 + 无改善 + deterministic probe 稳定冲突 + patience；Probe 按 interval 在 rank0 执行并正式进入 epoch state update。SUSPECT 限制 replay/weight，不增加 loss weight。
17. **不建议移植的失败方案**：复制 batch scalar 为每张图、为 observer 再跑一次 matcher、用 rank-local sampler、在 DDP 中按 rank-local batch 归一 loss weight、把 global normalizer 当 per-image normalizer、在当前 batch 立刻反馈 weight。
18. **RF-DETR 特有实现**：`SetCriterion` 的 Hungarian matcher、aux/enc loss key 命名、point-mask sampled loss、`NestedTensor`/`GradAccumAlignedDataset` 和 Lightning transfer hook 属于 RF-DETR 适配层；其他框架只移植数据契约、状态机、policy 和 global-sampler 语义，不照搬这些模块。

# MVTec-AD Sample Dynamics Validation (Initial Runtime Gate)

> Historical scope: **initial 3-epoch runtime/observer gate**
> Current status: **superseded for longitudinal conclusions by RF4/E5 reports**
> Overall Gate: **E0/E2 COMPLETED — OBSERVER PASS WITH CUDA NON-DETERMINISM CAVEAT**
> Scope: **NON-BENCHMARK / MECHANISM VALIDATION**
> Source: `/home/inspur1/data/MVTec-AD`
> Pilot dataset: `/home/liujiyuan/mvtec-sample-dynamics-pilot-v2`

本报告记录最初的 3-epoch Runtime Contract、E0 baseline 和 E2 observe-only 门；它们均在现有 `rf-detr-claude` Docker 镜像中完成。后续 15-epoch RF4/E0/E5 结果以 [RF4 Longitudinal Validation](mvtec_rf4_longitudinal_validation.md) 和 [E5 Combined Validation](mvtec_e5_combined_validation.md) 为准。整个流程没有使用 Conda，也没有修改原始 MVTec 数据。

## Gate 状态

| Gate | Status | Evidence |
| --- | --- | --- |
| Explore actual source | PASS | 5354 source images；1258/1258 abnormal masks 完整 |
| Core/experiment adapter | PASS WITH FIXES | Runtime Contract 暴露并修复 CUDA matcher index 对齐；Probe 恢复 Python/NumPy/Torch/CUDA RNG；训练写入 sample-order provenance |
| Fixed Pilot manifest | PASS | seed 20260810；category/defect stratified；train/valid source disjoint |
| Controlled corruption | PASS | 5 train anomalies；drop-mask 2、mask-shift 2、drop-component 1 |
| COCO/RLE contract | PASS | pycocotools 解码 100 annotations，area 全一致 |
| RF-DETR Runtime Contract | PASS | 1 normal + 1 pristine abnormal + 1 controlled corruption；forward → criterion → backward 完成 |
| E0 baseline | PASS | 3 epochs、128 samples/epoch、final Lightning step 95 |
| E2 observe-only | PASS WITH CAVEAT | observer 生命周期接通；不改 sampler/order/optimizer step；跨独立 CUDA 进程的 loss/metric 非 bitwise identical |
| E3/E4 | NOT RUN | 仍按执行门禁等待 E0/E3/E4/E5 同合同矩阵 |
| E5 combined | COMPLETED SEPARATELY | 15 epochs；单 seed 机制执行 PASS WITH CAVEATS，算法收益未验证；见当前报告 |

## 固定 Pilot v2

Pilot 使用 `pill / capsule / grid`，统一一个 COCO `anomaly` category。每个 defect type 固定抽取 4 train + 2 valid；每个 category 另取 20 个官方 train normal 和 8 个官方 test normal。

```json
{
  "train": 128,
  "train_abnormal": 68,
  "train_normal": 60,
  "valid": 58,
  "valid_abnormal": 34,
  "valid_normal": 24,
  "controlled_corruption": 5
}
```

异常图保留 MVTec semantic union mask 作为一个 annotation，不把 connected components 冒充 instance identity。component 数仍写入 manifest，并用于构造 `drop_component` 漏标。

固定产物：

- `split_manifest.json` / `split_manifest.jsonl`
- `corruption_ground_truth.json`
- `dataset_summary.json`
- `train/_annotations.coco.json`
- `valid/_annotations.coco.json`

所有图片都是指向原始 MVTec 文件的只读 symlink；源数据未修改。

## Runtime Contract 与公平性快照

权威证据：[mvtec_runtime_contract_v3.json](mvtec_runtime_contract_v3.json)。运行使用现有 `train-env-rfdetr-claude:20260806` 镜像、CUDA，未安装 Conda 或额外 Python 环境。

Runtime Contract 实际检查了：

- `RFDETRSegSmallConfig(num_classes=1, resolution=384)`；分类 head 的所有 `class_embed` weight/bias 第一维均为 `2`（RF-DETR 二分类 logits）；
- 1 张 normal、1 张 pristine abnormal、1 张 non-empty controlled corruption；collate 后保留 `boxes / labels / masks / image_id / sample_id`；
- `forward → loss_cls/bbox/giou/mask_ce/mask_dice → backward`；所有 loss 有限且梯度非零；
- 关闭 observer 与开启 `return_per_sample=True` 的同进程比较。

最终 batch 为 `[3, 3, 384, 384]`，loss 与梯度摘要如下：

| Value | Result |
| --- | ---: |
| `loss_cls` | 0.3090896606 |
| `bbox` | 0.1752459854 |
| `giou` | 0.8807296157 |
| `mask_ce` | 0.0511718094 |
| `mask_dice` | 0.5982190371 |
| gradient norm | 1215.2677001953 |
| observer loss max abs diff | 0.0000000000 |
| observer gradient max abs diff | 0.0291938782 |
| scalar baseline repeat gradient max abs diff | 0.0503807068 |

CUDA backward 在独立重复中存在非确定性；本检查以“observer 差异不超过同条件 baseline repeat 差异”为准，因此 Runtime Contract PASS。期间真实暴露的 CUDA 错误是 matcher 返回的 CPU `batch_idx` 与 CUDA scatter 的设备不一致，已在 `criterion.py` 修复。

E0/E2 公平性快照：

| Item | Value |
| --- | --- |
| seed | `20260810` |
| manifest SHA-256 | `90673182af0c409e99778f30545f550e91b6845e48f6894a7ebe029c53172528` |
| checkpoint SHA-256 | `6de3da31b2572cac214a1c76cce4a92a13966d56390ac2b3a3de9a8dc2b2bca3` |
| initial model parameter SHA-256 | `af6b20ee3c19dbfe2d538efacdff02641a3baa920ebaff86cacc3fef08cf798f`（E0/E2 相同） |
| resolution / batch / accumulation | `384 / 4 / 1` |
| epochs / augmentation / EMA | `3 / disabled / disabled` |
| optimizer steps | E0/E2 均为 final step `95` |

## E0/E2 执行结果

结果目录：

- [E0 run](/home/liujiyuan/mvtec-sample-dynamics-runs/e0)
- [E2 run](/home/liujiyuan/mvtec-sample-dynamics-runs/e2)

| Run | final train loss | final val loss | val det mAP50:95 | val seg mAP50:95 | F1 | precision | recall | elapsed | peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 baseline | 19.496027 | 23.447380 | 0.015818 | 0.013521 | 0.067568 | 0.043860 | 0.147059 | 97.07 s | 3156018176 B |
| E2 observe-only | 19.635305 | 23.935588 | 0.016011 | 0.014199 | 0.057554 | 0.038095 | 0.117647 | 106.25 s | 3156028416 B |

两组每个 epoch 都处理 128 个样本；三轮 `sample_order_hash` 完全相同：

```text
epoch 0  c6474aedab5b0e62fafccefa07dd37cedb8aec06424710bcc36490cf6af4a952
epoch 1  4c9e522b4f8c456ca8841973556aceab91e6d991ec85f92383a76b8a34969bbf
epoch 2  0a5899705bb764f979486d20ecf5567d232239c0d2a7479c33cc2aca321079e5
```

E2 的最终观察状态为 `HARD_LEARNABLE=30`、`LEARNING=72`、`MASTERED=24`、`SUSPECT=2`。5 个 controlled corruption 中本轮预测 `SUSPECT` 的 true positive 为 0，因此 corruption retrieval 的 precision/recall 均为 0.0；这不是算法成功结果，也不应被解释成脏数据检测已经验证。

## 本轮结论与边界

1. Runtime Contract PASS：Seg Small、单类别 head、MVTec target contract、mask 分支和 backward 路径均可运行。
2. Observer 基础设施在真实训练入口中接通：E2 多出了 sample-level loss、Probe、trajectory/state 和导出文件，但没有改变 sample order、optimizer step 数或 sampler/loss/augmentation policy。
3. E0/E2 使用独立 CUDA 训练进程，因此训练曲线不是 bitwise identical；Runtime Contract 的同进程 baseline control 证明 GPU backward 本身有可观测非确定性。E2 仅能记为 **PASS WITH CAVEAT**，不能据此宣称算法收益或严格无扰动。
4. E2 比 E0 约多 9.5% wall time，peak VRAM 基本相同；该数字只适用于本次小 Pilot，不是长期 overhead 结论。
5. 本历史报告完成时 E3/E4 尚未运行；后续已完成 15-epoch E5 combined 的单 seed 机制执行，但尚未证明算法收益，也没有完成 E3/E4 对照及多 seed 复制。因此 migration contract 仍不应冻结，SUSPECT/corruption precision 也不能作为 RF8 结论。

## 可复核命令与环境说明

本轮采用已有 Docker 运行环境，不使用 Conda。主机 Docker socket 当前为 `root:docker`、mode `660`；Docker Server API、GPU 和容器内 CUDA 均正常。镜像内没有 `pytest` 命令，因此本轮没有安装新测试依赖或宣称完整 pytest suite 重跑；Runtime Contract、真实 E0/E2 和 Docker 内编译检查已完成。

```bash
python experiments/mvtec_sample_dynamics/runtime_contract.py \
  --dataset /workspace/mvtec-pilot \
  --weights /workspace/models/rf-detr-seg-small.pt \
  --output /workspace/rfdetr/reports/mvtec_runtime_contract_v3.json \
  --device cuda

python experiments/mvtec_sample_dynamics/run_pilot.py \
  --mode baseline --dataset /workspace/mvtec-pilot \
  --output /workspace/runs/e0 --epochs 3 --batch-size 4 --device cuda

python experiments/mvtec_sample_dynamics/run_pilot.py \
  --mode observe --dataset /workspace/mvtec-pilot \
  --output /workspace/runs/e2 --epochs 3 --batch-size 4 --device cuda
```

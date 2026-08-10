# MVTec-AD Sample Dynamics Exploration

> Gate: **PASS**
> Scope: **NON-BENCHMARK / MECHANISM VALIDATION**
> Source: `/home/inspur1/data/MVTec-AD` (read-only)
> Date: 2026-08-10

## 结论

服务器数据可以用于 RF-DETR Seg Small 的 Sample Dynamics 机制验证，但不能把官方 normal-only train split 的结果宣称为 MVTec-AD 标准算法效果。推荐先使用 `pill + capsule + grid` 构造固定 derived split：三类分别覆盖较多 defect type、极小单组件缺陷和小面积多组件缺陷。转换可完全放在 `experiments/`，生成 Roboflow-style COCO segmentation 数据和 manifest，不需要修改 core dataset adapter，也不修改原始 MVTec 文件。

## 实际数据结构与完整性

实际根目录包含 15 个 category。每类结构均为：

```text
<category>/train/good/<id>.png
<category>/test/good/<id>.png
<category>/test/<defect_type>/<id>.png
<category>/ground_truth/<defect_type>/<id>_mask.png
```

所有异常原图都能按 `<id>.png -> <id>_mask.png` 找到唯一 mask；无 missing mask、无 orphan mask、无图像/mask 尺寸不一致。1258 张 mask 均为单通道 `L`、像素值严格为 `{0, 255}`。

| category   | train good | test good | abnormal | masks | defect types | median mask area | mean components |
| ---------- | ---------: | --------: | -------: | ----: | -----------: | ---------------: | --------------: |
| bottle     |        209 |        20 |       63 |    63 |            3 |          5.7178% |            1.08 |
| cable      |        224 |        58 |       92 |    92 |            8 |          3.6808% |            1.64 |
| capsule    |        219 |        23 |      109 |   109 |            5 |          0.4295% |            1.05 |
| carpet     |        280 |        28 |       89 |    89 |            5 |          1.4757% |            1.09 |
| grid       |        264 |        21 |       57 |    57 |            5 |          0.6015% |            2.98 |
| hazelnut   |        391 |        40 |       70 |    70 |            4 |          1.6158% |            1.94 |
| leather    |        245 |        32 |       92 |    92 |            5 |          0.4615% |            1.08 |
| metal_nut  |        220 |        22 |       93 |    93 |            4 |          3.6714% |            1.42 |
| pill       |        267 |        26 |      141 |   141 |            7 |          1.0327% |            1.74 |
| screw      |        320 |        41 |      119 |   119 |            5 |          0.2879% |            1.13 |
| tile       |        230 |        33 |       84 |    84 |            5 |          7.9303% |            1.02 |
| toothbrush |         60 |        12 |       30 |    30 |            1 |          1.0778% |            2.20 |
| transistor |        213 |        60 |       40 |    40 |            4 |          2.0284% |            1.10 |
| wood       |        247 |        19 |       60 |    60 |            5 |          2.8955% |            2.80 |
| zipper     |        240 |        32 |      119 |   119 |            7 |          2.3544% |            1.49 |
| **total**  |   **3629** |   **467** | **1258** | **1258** |       **78** |                — |               — |

可直接使用的 source images 为 5354 张：4096 normal、1258 abnormal。mask 文件是标签资产，不重复计入 image 数。

## Pilot 选择

第一轮固定选择三个 category：

- `pill`：141 个异常、7 个 defect type，数量最多且 26.2% 的 mask 有多个 component。
- `capsule`：109 个异常、5 个 defect type，mask area median 仅 0.4295%，提供极小、主要为单 component 的缺陷。
- `grid`：57 个异常、5 个 defect type，mask area median 0.6015%，component median/P90 为 3/6，提供小面积多组件缺陷。

扩展候选为 `zipper` 和 `metal_nut`；前者提供稳定多组件结构，后者 mask area 的 10/50/90 percentile 约为 1.16%/3.67%/48.22%，提供大异常区域对照。`pill/pill_type` 自身也是大面积子群，因此 split 必须继续按 defect type 分层。

## RF-DETR adapter 与 target 契约

现有 `build_roboflow_from_coco()` 已支持如下 experiment-side 目录，无需新增 MVTec core adapter：

```text
pilot_dataset/
  train/_annotations.coco.json
  valid/_annotations.coco.json
  test/_annotations.coco.json
```

`CocoDetection(include_masks=True)` 会把 COCO annotation 转为：

- `boxes`: absolute `xyxy`，transform 后变成 normalized `cxcywh`；
- `labels`: contiguous model labels；Pilot 统一为一个 `anomaly` class；
- `masks`: `[N,H,W]` bool instance masks；
- `orig_size` / `size` / `image_id`；
- `sample_id`: `split:image_id:file_name`。

转换时每张异常图的 binary anomaly union mask 生成一个 COCO RLE annotation、union bbox 和 area。MVTec 不提供 instance identity，因此第一版不把 connected components 冒充真实实例；component 数只作为 hardness 特征和 controlled corruption 的依据。normal image 保留在 COCO `images` 中但无 annotation，正好覆盖 empty-GT 路径。

## 推荐固定协议

- 固定 seed；按 `category/defect_type` 分层抽取，原图不得跨 split。
- anomaly train/val 来自官方 test 的互斥子集；normal train 来自官方 train/good，normal val 来自官方 test/good。
- 小规模 smoke 每个 defect stratum 最多取 4 train + 2 val；每类另取 20 train normal + 8 val normal。
- train 中约 8% anomaly 做 deterministic controlled corruption；val 永远使用 pristine mask。
- 原图使用只读 symlink；所有 pristine/corrupted 路径、corruption 参数、mask hash 与 stable image ID 写入 manifest。
- E0/E2 使用完全相同的 split、corruption、checkpoint、augmentation、epoch、batch 和 seed，只改变 Sample Dynamics observe 开关。

## 本地可用 checkpoint

已发现与目标模型匹配的本地权重：

```text
/home/liujiyuan/rf-detr-models/rf-detr-seg-small.pt
```

因此 Pilot 不需要联网下载权重。正式启动前仍需在 `rf-detr-claude` Docker 内验证 checkpoint 加载与类别头重建。

## 风险与边界

- derived split 使用了官方 test 异常，结果只能称 mechanism validation。
- `HARD_LEARNABLE` 当前仍是高 loss candidate；只有增加 weight/exposure 后恢复才能后验支持 “learnable”。
- connected components 是工程实例定义，不等于 MVTec 原生 instance 标注。
- 后续 Runtime Contract 与 E0/E2 已在现有 `rf-detr-claude` Docker 中完成；本文件保留为数据探索记录，算法效果仍未验证。

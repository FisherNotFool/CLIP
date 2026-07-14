# CLIP 图像分类模块优化记录

## 当前目标

为 Ceramics 文献解析链路中的图片提供类型标签，并优先拦截不属于目标类型的 `other` 图片。

当前已知类别为：

| 标签 | 含义 |
|---|---|
| `bar_chart` | 柱状图 |
| `line_chart` | 折线图 / 曲线图 |
| `sem` | SEM 扫描电子显微图 |
| `tem` | TEM 透射电子显微图 |
| `xrd` | XRD 衍射图谱 |
| `other` | 其他图片，如示意图、装置图、照片、表格等 |

`other` 不参与五分类主分类器，而是由独立 gate 拦截。

---

## 当前架构

```text
图片
  → CLIP ViT 图像编码器（冻结）
  → known / other 二分类 gate
      ├─ other → 返回 other
      └─ known → 五分类 Linear Probe
                    → bar_chart / line_chart / sem / tem / xrd
```

接口可携带 caption。当视觉五分类置信度低于 0.75 时，caption 规则可对明确、单一类别信号做保守纠偏；gate 已拦截为 `other` 的结果不会被 caption 推翻。

---

## 数据集与人工审查

待审查图片按“文献目录 / 图片”的结构保存。预标注脚本生成 `review_manifest.csv`，人工审查规则为：

```text
review_label 为空  → 接受 predicted_label
review_label 非空  → 使用 review_label 作为最终标签
```

导入脚本将图片组织为：

```text
samples/
├── sem/reviewed/<document_id>/...
├── tem/reviewed/<document_id>/...
├── xrd/reviewed/<document_id>/...
└── other/reviewed/<document_id>/...
```

该结构保留文献归属，供后续按文献分组评估使用。

本轮已审查并导入 2517 张图片；五个已知类的训练图片数量为：

| 类别 | 数量 |
|---|---:|
| bar_chart | 121 |
| line_chart | 711 |
| sem | 315 |
| tem | 80 |
| xrd | 186 |
| other | 1129 |

新增的 `samples/tem/reviewed/new_tem` 图片会被训练脚本递归读取，支持 JPG、JPEG、PNG、BMP、TIFF、WEBP。

---

## 当前训练结果（图片级分层留出集）

### 五分类 Linear Probe

| 指标 | 结果 |
|---|---:|
| Accuracy | 92.6% |
| Macro-F1 | 0.89 |
| bar_chart F1 | 0.84 |
| line_chart F1 | 0.96 |
| SEM F1 | 0.93 |
| TEM F1 | 0.80 |
| XRD F1 | 0.90 |

TEM 样本从 55 张增至 80 张后，TEM recall 提升到 0.88；仍建议持续补充易与 SEM 混淆的 TEM 样本。

### Other gate

| 指标 | 结果 |
|---|---:|
| other recall | 95.6% |
| known false rejection | 1.8% |
| 保存阈值 | 0.4701 |

以上 gate 指标来自训练脚本的留出验证。

---

## 启用 gate

训练生成以下产物：

```text
model_cache/linear_probe.pt
model_cache/label_map.json
model_cache/other_gate.pt
```

gate 是唯一的 other 拦截器，`.env` 只需保留产物路径：

```ini
OTHER_GATE_PATH=./model_cache/other_gate.pt
```

---

## Caption 规则

规则只在视觉置信度 `< 0.75` 时使用，且 caption 只能命中一个类别；复合图或同时命中多个类别时不覆盖视觉结果。

| 类别 | 强信号示例 |
|---|---|
| TEM | `TEM`、`HRTEM`、`HAADF`、`STEM`、`SAED`、`transmission electron microscopy` |
| SEM | `SEM`、`scanning electron microscopy` |
| XRD | `XRD`、`x-ray diffraction`、`2θ` |
| bar_chart | `bar chart`、`bar graph`、`histogram` |
| line_chart | `line chart`、`line plot`、`line graph`、`curve` |

---

## API 请求

当前推荐请求体：

```json
{
  "document_id": "paper_001",
  "images": [
    {
      "image_path": "/doc_001/fig1.jpg",
      "caption": "Figure 1. SEM morphology of the sample."
    },
    {
      "image_path": "/doc_001/fig2.jpg",
      "caption": "Figure 2. XRD patterns."
    }
  ]
}
```

旧的 `image_paths` 请求字段仍暂时兼容，便于上游渐进迁移。

---

## 已知限制与下一步

当前训练/测试是按图片随机分层切分。同一篇文献中的相似图片可能进入训练与测试两侧，指标可能偏乐观。

下一步应使用 `reviewed/<document_id>` 目录做按文献分组的训练、标定和测试切分，并重新评估：

- 五分类 Macro-F1 是否达到 0.90；
- TEM 与 SEM 的混淆；
- gate 的 other recall 与已知类误拒；
- caption 规则在真实低置信度样本上的净增益。

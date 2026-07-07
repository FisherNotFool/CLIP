# CLIP 图像分类模块优化记录

## 背景

为 Ceramics（陶瓷材料科学论文处理）平台构建图像分类模块，将学术 PDF 中提取的图片分为 5 类：

| 类别 | 说明 |
|------|------|
| bar_chart | 柱状图 |
| line_chart | 折线图 |
| sem | SEM 显微图 |
| xrd | XRD 衍射图谱 |
| other | 其他（照片、示意图等） |

部署环境：离线（本地），开发用 CPU，生产有 GPU。

---

## Phase 1: 零样本 CLIP（失败）

### 方案

使用 `openai/clip-vit-base-patch32`，为每类编写自然语言 prompt，通过图像特征与文本特征的余弦相似度进行分类。

### 结果

多轮 prompt 调优后准确率稳定在 **~50%**。

### 失败原因

- **XRD vs line_chart 无法区分**：两者视觉特征在 CLIP 空间中高度重叠（都是线条 + 坐标轴），自然语言 prompt 无法捕捉 X 轴标注 "2θ" 这类细粒度差异
- **"other" 类别无统一特征**：照片、示意图、表格等外观差异极大，无法用有限 prompt 覆盖

### 尝试的 prompt 优化

- 为 XRD 强调"横坐标 2θ、纵坐标 Intensity、尖锐衍射峰" → 仍被分类为 line_chart
- 移除 other 的 prompt，改为"四类都低置信度时兜底" → 无改善
- 多轮 prompt 精调后准确率反而下降到 48%

结论：**零样本 CLIP 不适合同类图表区分任务。**

---

## Phase 2: Linear Probe 微调（成功）

### 方案

冻结 CLIP 视觉编码器，在 512 维特征之上训练一个 `nn.Linear(512, 4)` 分类头。

**核心设计**：
- 只训练 4 类（bar_chart / line_chart / sem / xrd）
- "other" **不参与训练**（无统一特征，强训只会过拟合到训练集中的具体样本）
- "other" 通过置信度阈值兜底：softmax 最高分 < 阈值 → other

### 训练流程

```
samples/{class}/ 下所有图片
  → 冻结的 CLIP ViT 提取 512 维特征
  → 缓存到 scripts/features.pt
  → 分层 80/20 划分
  → 训练 nn.Linear(512, 4) + CrossEntropyLoss(weight=class_weights) + Adam
  → 保存 model_cache/linear_probe.pt + model_cache/label_map.json
```

### 结果

| 指标 | 零样本 | Linear Probe |
|------|--------|-------------|
| 准确率 | ~50% | **93.1%** |
| 训练时间 | — | < 1 分钟 (CPU) |

### 遗留问题

Softmax 阈值兜底基本无效——104 张 other 图只有 7 张（6.7%）被拦截。原因是 softmax 强制 4 类概率之和为 1，对不相关的图也必须"选一个最像的"，置信度常高达 0.90+。

```
softmax([0.3, 4.8, -1.5, 0.8]) → line_chart 0.95
                                    ↑ 看似很确定，但其实什么都不是
```

---

## Phase 3: Centroid 距离检测（当前方案）

### 方案

把 "other" 的判断从分类问题改为**离群检测**：

```
训练: 计算每个类的 CLIP 特征中心点 (centroid, 512 维 L2 归一化向量)

推理: 图像 → CLIP 特征 → 余弦距离到 4 个中心
      → 最短距离 > 阈值 → "other"（离所有已知类都太远）
      → 最短距离 ≤ 阈值 → Linear Probe 分类（判断属于哪个类）
```

**关键**："是不是已知类"和"是哪个已知类"是两个独立判断，互不污染。

### 距离分布

samples/other/ 下 104 张图片到最近类中心的余弦距离：

```
范围: [0.095, 0.479]
25th: 0.172    50th: 0.229    75th: 0.270    85th: 0.313    95th: 0.355
```

### 阈值选择

| 阈值 | Other 拦截率 | 训练类准确率 | 误杀（false positive） |
|------|-------------|-------------|----------------------|
| 0.17 | 76.9% | 89.3% | 25 张 (4.6%) |
| **0.27** | **25.0%** | **93.1%** | **2 张 (0.4%)** |
| 0.31 | 17.3% | 93.3% | 1 张 (0.2%) |

**选定 0.27** — 在不牺牲训练类准确率的前提下，最大化 other 拦截。

### 局限性

Other 拦截率上限受 CLIP 视觉编码器本身约束。部分 other 图（材料照片等）在 CLIP 的 512 维空间中与图表类的距离最近仅 0.095，与类内距离无法区分。这不是分类器设计问题，而是 CLIP 对"图表 vs 非图表"的底层判别力有限。

---

## 最终架构

```
图像 → CLIP ViT (冻结)
         │
         ├─→ L2 归一化 → 余弦距离到 4 个类中心
         │                    │
         │               min_dist > 0.27 → "other"
         │               min_dist ≤ 0.27 ↓
         │
         └─→ nn.Linear(512, 4) → softmax → bar_chart / line_chart / sem / xrd
```

---

## 文件结构

```
CLIP/
├── app/
│   ├── services/classifier.py    # ClipClassifier（推理核心）
│   ├── config.py                 # 配置（阈值、路径）
│   ├── lifespan.py               # FastAPI 生命周期
│   └── api/router.py             # POST /api/clip/classify
├── scripts/
│   └── train.py                  # 训练脚本（特征提取 + 训练 + centroids）
├── model_cache/
│   ├── linear_probe.pt           # 训练的线性分类头
│   ├── label_map.json            # 标签映射
│   ├── centroids.pt              # 类中心点（离群检测）
│   └── models--openai--...       # CLIP 模型缓存
├── samples/                      # 训练/评估样本
│   ├── bar_chart/  line_chart/  sem/  xrd/
│   └── other/                    # 仅用于阈值标定
├── tests/
│   ├── test_classifier.py        # 单元测试（mock 模型）
│   ├── test_api.py               # API 测试
│   └── test_integration.py       # 集成测试（真实模型）
└── .env                          # CENTROID_DISTANCE_THRESHOLD=0.27
```

---

## 命令速查

```bash
# 训练
python scripts/train.py

# 单元测试
pytest tests/test_classifier.py tests/test_api.py -v

# 集成测试（含阈值效果评估）
pytest tests/test_integration.py -m integration -v -s

# 启动服务
uvicorn app.main:app --port 8011

# API 调用
curl -X POST http://localhost:8011/api/clip/classify \
  -H "Content-Type: application/json" \
  -d '{"document_id": "test", "image_paths": ["/path/to/image.jpg"]}'
```

# CLIP 图像分类模块 — 上游系统交接说明

> **目标读者**：CLIP 文档图片分类项目的开发 Agent
> **上游项目**：Ceramics — 陶瓷材料科研文献解析与知识图谱平台

---

## 1. 上游系统是什么

**Ceramics** 是一个面向陶瓷材料领域的科研文献智能处理平台。一句话概括：

> 上传 PDF 论文 → 自动解析出文本/表格/图片/公式 → 构建知识图谱 → 支持智能问答

技术栈：FastAPI + MongoDB + Elasticsearch + Neo4j + MySQL + Dify Agent

---

## 2. 核心数据流（CLIP 的上游）

```
用户上传 PDF 论文
      ↓
MinerU API 云端解析（OCR + 版面分析 + 公式识别）
      ↓
┌─────────────────────────────────────────┐
│  从 PDF 中提取出的多模态内容：            │
│  - 纯文本（按页）                         │
│  - 表格（HTML 格式 + caption）            │
│  - 图片（JPG/PNG，含 caption）  ← CLIP 的输入  │
│  - 行内公式 & 行间公式（LaTeX）           │
└─────────────────────────────────────────┘
      ↓
存入 MongoDB（结构化 JSON）+ Elasticsearch（全文索引）
      ↓
下游消费：
  - 知识图谱构建（DeepSeek 大模型抽取实体/关系 → Neo4j）
  - 智能问答（Dify Agent + RAG + KG 联合检索）
```

---

## 3. CLIP 要插入的位置

当前系统对图片的处理是**只提取、不分类**。每篇论文解析后会得到一批图片（例如 `outputs/{doc_id}/images/fig1.jpg`），但这些图片的**类型标签是缺失的**——系统不知道某张图是柱状图、折线图、SEM 显微照片还是 XRD 衍射图谱。

**CLIP 模块的插入点**：在 MinerU 解析完成、图片解压落盘之后，在 MongoDB 存储之前（或之后触发更新），对这批发出的图片做分类标注。

```
MinerU 解析 → 图片落盘 → [CLIP 图像分类] → MongoDB（带图片类型标签）
                              ↑
                        你的模块在这里
```

---

## 4. 图片分类的业务价值（下游如何消费 CLIP 的输出）

CLIP 输出的图片类型标签，会被以下模块消费：

| 下游场景 | 如何使用图片类型标签 |
|---------|-------------------|
| **知识图谱** | 图片类型作为实体的属性（如 "Fig.3 的 chart_type = SEM"），与材料、性能、工艺等实体关联 |
| **全文检索** | 用户可以搜 "SEM 图像" 或 "XRD 图谱"，ES 按图片类型过滤命中页面 |
| **智能问答** | Dify Agent 回答 "这篇文献用的是什么表征手段？" 时，可以引用图片类型信息增强回答 |
| **前端展示** | 解析结果页按图片类型分组展示，或标注图标（📊柱状图 / 📈折线图 / 🔬SEM / 📐XRD） |

---

## 5. 图片数据的具体形态

每篇论文解析后的图片存储在：

```
outputs/{document_id}/images/
├── 1f2a3b4c5d6e7f8g9h0i1j2k3l4m5n6o.jpg  # 随机 hash 文件名
├── 2a3b4c5d6e7f8g9h0i1j2k3l4m5n6o7p.jpg
└── ...
```

MongoDB 中对应的图片元数据（当前结构）：

```json
{
  "pages": [{
    "page_idx": 5,
    "images": [{
      "path": "/outputs/doc_xxx/images/1f2a3b4c5d.jpg",
      "caption": "Fig. 3 SEM micrograph of the sintered sample"
    }]
  }]
}
```

**CLIP 需要做的**：为每张图片输出一个 `image_type` 字段，例如 `"bar_chart"` / `"line_chart"` / `"sem"` / `"xrd"`，追加到图片元数据中。

---

## 6. 目标分类类别

根据材料科学文献的实际图片类型，你的 CLIP 模型需要区分的类别至少包括：

| 类别 | 英文 label | 说明 |
|------|-----------|------|
| 柱状图 | bar_chart | 分组柱状图、堆叠柱状图等 |
| 折线图 | line_chart | 单/多线趋势图 |
| SEM 图像 | sem | 扫描电子显微镜照片（灰度、纹理丰富） |
| XRD 图谱 | xrd | X 射线衍射图谱（有特征峰的曲线） |
| 其他图片 | other | 示意图、照片、流程图等不属上述类的图片 |

后续可能扩展：TEM 图像、EDS 能谱、TG-DSC 热分析曲线、FTIR 光谱等。

---

## 7. 与上游系统的接口约定

CLIP 模块对外暴露的接口预计是：

```
POST /api/clip/classify
输入：
{
  "document_id": "doc_xxx",         // 文献 ID
  "image_paths": [                  // 图片路径列表（相对于 outputs/ 目录）
    "/outputs/doc_xxx/images/1f2a.jpg",
    "/outputs/doc_xxx/images/2a3b.jpg"
  ]
}

输出：
{
  "document_id": "doc_xxx",
  "classifications": [
    {
      "image_path": "/outputs/doc_xxx/images/1f2a.jpg",
      "image_type": "sem",
      "confidence": 0.94
    },
    {
      "image_path": "/outputs/doc_xxx/images/2a3b.jpg",
      "image_type": "bar_chart",
      "confidence": 0.87
    }
  ]
}
```

或者设计为回调模式：CLIP 分类完成后，调用上游的某个 webhook 回写分类结果。具体接口形式由你决定，上述仅为参考。

---

## 8. 上游系统关键信息速查

| 项目 | 说明 |
|------|------|
| 项目名称 | Ceramics（ceramics） |
| 仓库地址 | github.com/ding0meng-cloud/ceramics |
| 后端框架 | FastAPI（端口 8000） |
| 解析引擎 | MinerU v4 API（云端 OCR + 版面分析） |
| 图片存储 | 本地磁盘 `outputs/{doc_id}/images/` |
| 元数据存储 | MongoDB（`literature_kg` 库，`parsed_documents` 集合） |
| 全文索引 | Elasticsearch（`literature_text` 索引） |
| 知识图谱 | Neo4j（bolt://localhost:7689） |
| 前端框架 | Vue 3 + Vite（端口 5173） |

---

## 9. 开发建议

1. **数据集构建**：从上游系统已解析的论文图片中采样，人工标注一批 bar_chart / line_chart / sem / xrd 样本作为训练/测试集。`outputs/` 目录下已有不少现成的图片可直接使用。

2. **模型选型**：使用 CLIP（如 `openai/clip-vit-base-patch32` 或 `ViT-B/32`）做零样本或少样本微调，利用材料科学图片的 caption 文本（如 "Fig. 3 SEM micrograph..."）作为自然语言监督信号。

3. **部署方式**：可以用 FastAPI 单独起一个服务（如端口 8011），或者作为 Ceramics 后端的一个子模块直接集成到解析流水线中。

4. **与上游联调**：确保能访问 `outputs/` 目录读取图片，能访问 MongoDB 写入分类结果。

---

*此文档由 Ceramics 项目组整理，供 CLIP 子项目开发参考。如有疑问请联系上游。*

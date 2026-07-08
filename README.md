# CLIP 图像分类服务

为 Ceramics 平台提供的学术论文图片分类模块。给定一组图片路径，返回每张图的类别标签和置信度。

## 分类类别

| 标签 | 含义 |
|------|------|
| `bar_chart` | 柱状图 |
| `line_chart` | 折线图 |
| `sem` | SEM 显微图 |
| `xrd` | XRD 衍射图谱 |
| `other` | 其他（照片、示意图等，距所有已知类太远时返回） |

## 环境要求

- Python ≥ 3.11
- PyTorch ≥ 2.0
- 磁盘 ≥ 2 GB（CLIP 模型 + 权重文件）

支持 CPU 和 GPU（CUDA）。GPU 推理更快，部署方式相同。

## 快速部署

### 1. 获取代码

```bash
git clone <repo-url>
cd CLIP
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

建议使用 conda 创建独立环境：

```bash
conda create -n clip_env python=3.11
conda activate clip_env
pip install -r requirements.txt
```

### 3. 下载 CLIP 基础模型（仅首次，需联网）

仓库只包含本项目训练的轻量文件（分类头 + 中心点，共 ~20 KB）。
CLIP 预训练权重（~600 MB）需要单独下载一次：

```bash
# 正常网络
python scripts/download_model.py

# 国内网络（使用镜像）
python scripts/download_model.py --mirror https://hf-mirror.com
```

下载完成后 `model_cache/` 目录结构：

```
model_cache/
├── models--openai--clip-vit-base-patch32/   # CLIP 预训练权重（~600 MB）
├── linear_probe.pt                          # 线性分类头（~10 KB，仓库已含）
├── label_map.json                           # 标签映射（仓库已含）
└── centroids.pt                             # 类中心点（~10 KB，仓库已含）
```

### 4. 配置

```bash
cp .env.example .env
```

按需编辑 `.env`：

```ini
# 必须修改
IMAGE_BASE_PATH=./outputs           # 上游系统传入图片的根目录

# 可选调整
DEVICE=cpu                          # 有 GPU 改为 cuda
CENTROID_DISTANCE_THRESHOLD=0.27    # other 判定阈值（越小越严格）
BATCH_SIZE=8                        # GPU 可调大，如 32
```

| 配置项 | 说明 |
|--------|------|
| `IMAGE_BASE_PATH` | **必改**。API 中 `image_paths` 相对于此路径拼接 |
| `CENTROID_DISTANCE_THRESHOLD` | other 灵敏度（0~2），0.27 经验最优 |
| `DEVICE` | `cpu` 或 `cuda` |
| `TRANSFORMERS_OFFLINE` | 默认 `1`（离线），下载模型时临时改为 `0` |
| `BATCH_SIZE` | GPU 显存充足可调大，CPU 用 8 |

### 5. 启动

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8011
```

健康检查：

```bash
curl http://localhost:8011/api/clip/health
# → {"status": "ok", "model_loaded": true}
```

### 离线部署

在能联网的机器上完成步骤 1-3，然后将整个项目目录（含 `model_cache/`）拷贝到离线服务器。`.env` 中 `TRANSFORMERS_OFFLINE=1` 确保服务不访问外网。

---

## API 文档

### POST /api/clip/classify

分类一组图片。

**Query 参数**：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `debug` | bool | `false` | 为 `true` 时返回每类的完整概率分布 |

**Request Body**（JSON）：

```json
{
  "document_id": "paper_001",
  "image_paths": [
    "/doc_001/fig1.jpg",
    "/doc_001/fig2.jpg",
    "/doc_001/fig3.jpg"
  ]
}
```

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `document_id` | string | 1-256 字符 | 论文 ID |
| `image_paths` | string[] | 1-100 个 | 图片路径，相对于 `IMAGE_BASE_PATH` |

**Response**（200）：

```json
{
  "document_id": "paper_001",
  "classifications": [
    {
      "image_path": "outputs/doc_001/fig1.jpg",
      "image_type": "sem",
      "confidence": 0.9999,
      "all_scores": {
        "bar_chart": 0.0,
        "line_chart": 0.0001,
        "sem": 0.9999,
        "xrd": 0.0
      },
      "error": null
    },
    {
      "image_path": "outputs/doc_001/fig2.jpg",
      "image_type": "xrd",
      "confidence": 0.9883,
      "all_scores": null,
      "error": null
    },
    {
      "image_path": "outputs/doc_001/fig3.jpg",
      "image_type": "other",
      "confidence": 0.7300,
      "all_scores": null,
      "error": null
    }
  ],
  "model_name": "openai/clip-vit-base-patch32",
  "model_device": "cpu"
}
```

| 字段 | 说明 |
|------|------|
| `image_type` | 分类结果标签 |
| `confidence` | 置信度（0~1），`other` 时为 1 - 最小余弦距离 |
| `all_scores` | 仅 `debug=true` 时返回，4 类 softmax 概率 |
| `error` | 非 `null` 表示该图片处理失败 |

**错误响应**：

| 状态码 | 含义 |
|--------|------|
| 422 | 请求参数不合法（`document_id` 为空、`image_paths` 超 100 条等） |
| 500 | 模型推理异常 |

### GET /api/clip/health

健康检查。

---

## 调用示例

```bash
curl -X POST "http://localhost:8011/api/clip/classify?debug=true" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "test_001",
    "image_paths": ["/paper_001/fig1.jpg", "/paper_001/fig2.jpg"]
  }'
```

---

## 模型更新

当收集到新的训练样本后，重新训练：

```bash
# 将新图片放入 samples/{class_name}/ 对应目录
# 然后运行：
python scripts/train.py --force-extract

# 查看 other 距离分布，按需调整 .env 中的 CENTROID_DISTANCE_THRESHOLD
# 重启服务
```

新样本加入后重新训练和评估，流程约 2-3 分钟（CPU）。

---

## 目录结构

```
CLIP/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置（pydantic-settings）
│   ├── lifespan.py          # 启动/关闭生命周期
│   ├── services/
│   │   └── classifier.py    # 推理核心
│   ├── api/
│   │   ├── router.py        # API 路由
│   │   └── deps.py          # 依赖注入
│   ├── schemas/             # Pydantic 模型
│   └── errors/              # 错误处理
├── scripts/
│   ├── train.py             # 训练脚本
│   └── download_model.py    # 模型下载（仅首次）
├── model_cache/             # 模型权重（需下载）+ 分类头 + 中心点
├── .env.example             # 配置模板
├── requirements.txt
└── README.md
```

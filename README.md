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

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

建议使用 conda 创建独立环境：

```bash
conda create -n clip_env python=3.11
conda activate clip_env
pip install -r requirements.txt
```

### 2. 准备模型文件

确保 `model_cache/` 目录下有以下文件（由训练脚本生成）：

```
model_cache/
├── models--openai--clip-vit-base-patch32/   # CLIP 预训练权重（~1.7 GB）
├── linear_probe.pt                          # 线性分类头（~1 MB）
├── label_map.json                           # 标签映射
└── centroids.pt                             # 类中心点（离群检测用）
```

如果是从开发环境迁移，直接拷贝整个 `model_cache/` 目录即可。

**离线部署注意**：`.env` 中设置 `TRANSFORMERS_OFFLINE=1`，服务不会访问外网。

### 3. 配置

编辑 `.env` 文件：

```ini
# 模型
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
MODEL_CACHE_DIR=./model_cache
DEVICE=cpu                          # 改为 cuda 使用 GPU
TRANSFORMERS_OFFLINE=1              # 离线模式，只读本地缓存

# 服务
HOST=0.0.0.0
PORT=8011

# 图片处理
IMAGE_BASE_PATH=./outputs           # 上游图片存放的根目录
CENTROID_DISTANCE_THRESHOLD=0.27    # other 判定阈值
MAX_IMAGE_SIZE=1920                 # 超过此尺寸自动缩放
BATCH_SIZE=8                        # 每批处理图片数
```

| 配置项 | 说明 |
|--------|------|
| `IMAGE_BASE_PATH` | 图片根目录，API 中的 `image_paths` 相对于此路径 |
| `CENTROID_DISTANCE_THRESHOLD` | other 判定灵敏度（0~2），越小越严格。当前 0.27 是最优平衡 |
| `DEVICE` | `cpu` 或 `cuda` |
| `BATCH_SIZE` | 根据 GPU 显存调整，CPU 用 8 即可 |

### 4. 启动

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8011
```

启动后访问 `http://localhost:8011/api/clip/health` 确认状态：

```json
{"status": "ok", "model_loaded": true}
```

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
├── model_cache/             # 模型权重 + 分类头 + 中心点
├── .env                     # 配置文件
├── requirements.txt
└── README.md
```

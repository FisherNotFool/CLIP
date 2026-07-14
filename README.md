# CLIP 文献图片分类 API

面向材料文献图片的本地 FastAPI 服务。运行时使用冻结的 CLIP 编码器、`other` gate 与五分类线性分类头。

## 类别

`bar_chart`、`line_chart`、`sem`、`tem`、`xrd`、`other`。

## 运行

需要本地模型缓存与以下项目产物：

```text
model_cache/linear_probe.pt
model_cache/label_map.json
model_cache/other_gate.pt
```

```powershell
conda activate clip_env
uvicorn app.main:app --host 0.0.0.0 --port 8011 --log-level info
```

## 请求

```json
{
  "document_id": "paper_001",
  "images": [
    {
      "image_path": "/doc_001/fig1.jpg",
      "caption": "Figure 1. TEM image of the sample."
    }
  ]
}
```

`caption` 可选。视觉五分类置信度低于 0.75 时，明确的单类别 caption 信号可做保守纠偏；gate 判为 `other` 的图片不会被纠偏。

## 响应

```json
{
  "document_id": "paper_001",
  "classifications": [
    {
      "image_path": "outputs/doc_001/fig1.jpg",
      "image_type": "tem",
      "confidence": 0.91,
      "all_scores": null,
      "error": null
    }
  ],
  "model_name": "openai/clip-vit-base-patch32",
  "model_device": "cpu"
}
```

传入 `?debug=true` 可返回五分类分数。服务日志在 `INFO` 级别输出请求、耗时和标签统计；`DEBUG` 级别输出单图预测与 caption 覆盖记录。

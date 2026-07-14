# Ceramics 接入 CLIP 服务交接文档

## 目标与边界

CLIP 是独立的图像分类模型服务，输出 `bar_chart`、`line_chart`、`sem`、`tem`、`xrd` 或 `other`。Ceramics 负责在文献解析后提交图片路径与 caption，并把分类结果写回 MongoDB；CLIP 不访问 Ceramics 数据库，也不修改上游文件。

不要把 CLIP 的 PyTorch / Transformers 依赖加入 Ceramics 主后端环境，也不要把模型实现复制到 `backend/app`。它应以独立容器运行，作为 `backend/model_services` 下的一个模型服务。

## 需要引入的文件

将 CLIP 仓库作为 Git submodule 放在：

```text
material-kg/backend/model_services/clip-service/
```

```powershell
cd D:\Project\ceramics\material-kg\backend
git submodule add git@github.com:FisherNotFool/CLIP.git model_services/clip-service
git submodule update --init --recursive
```

该目录包含 Dockerfile、`compose.yaml` 和三个必须随镜像分发的模型产物：

```text
model_cache/linear_probe.pt
model_cache/label_map.json
model_cache/other_gate.pt
```

## Docker Compose 接入

生产部署应引用已发布的版本化镜像，不能在 Ceramics 主机上使用 `build:`。镜像发布和拉取步骤见 [container-release.md](container-release.md)。

在 `backend/model_services/docker-compose.yml` 的 `services:` 下增加：

```yaml
  clip-model:
    image: ${CLIP_IMAGE}
    container_name: ceramics_clip_model
    restart: unless-stopped
    environment:
      DEVICE: cpu
      IMAGE_BASE_PATH: /data/outputs
      MAX_IMAGE_SIZE: 1920
      BATCH_SIZE: 8
    ports:
      - "8011:8011"
    volumes:
      - ../outputs:/data/outputs:ro
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8011/api/clip/health', timeout=5)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 45s
```

`../outputs` 的相对路径以 `backend/model_services/` 为工作目录，对应 `backend/outputs`。容器只读挂载，这是强制约束。

Docker 构建不访问 Hugging Face，而是直接复制 `model_cache/` 中已准备好的 CLIP 基座缓存。基座权重不纳入 Git；首次部署前需将现有 CLIP 工作目录的 `model_cache/models--openai--clip-vit-base-patch32/` 一并带入 submodule 工作树（或在可联网的宿主机先启动一次 CLIP API 生成该缓存）。容器运行时始终使用离线模式。

启动与检查：

```powershell
cd D:\Project\ceramics\material-kg\backend\model_services
docker compose up -d --build clip-model
Invoke-RestMethod http://127.0.0.1:8011/api/clip/health
```

如果 Ceramics 主后端也改为容器运行，配置 `CLIP_SERVICE_URL=http://clip-model:8011`，并确保两个容器在同一个 Compose network。若主后端继续直接在宿主机运行，使用 `CLIP_SERVICE_URL=http://127.0.0.1:8011`。

## 后端改动清单

### 1. 配置

文件：`backend/app/config/config.py`

保留既有的 `clip_service_url`，在 `backend/.env` 中显式设置：

```dotenv
CLIP_SERVICE_URL=http://127.0.0.1:8011
CLIP_ENABLED=true
```

### 2. 协议升级并传递 caption

文件：`backend/app/utils/parse/clip_client.py`

当前客户端只发送旧字段 `image_paths`，不会使用 CLIP 的低置信度 caption 规则。将 `classify_images` 改为接收图片对象列表，向 CLIP 发送：

```json
{
  "document_id": "doc_xxx",
  "images": [
    {
      "image_path": "doc_xxx/images/figure-1.jpg",
      "caption": "Figure 1. HRTEM image of the sample."
    }
  ]
}
```

约束：

- `image_path` 必须是相对于 `backend/outputs` 的路径；上游数据库中的 `/outputs/...` 继续由现有 `_to_relative_path` 转换。
- `caption` 可为空字符串或 `null`；不要拼接整页正文，也不要上传图片 base64。
- 更新 `VALID_IMAGE_TYPES` 为：`bar_chart`、`line_chart`、`sem`、`tem`、`xrd`、`other`。

### 3. 调用处保留 caption

文件：`backend/app/routes/parse/parse_routes.py`

在 `classify_document_images` 构建 `image_map` 时加入：

```python
"caption": img.get("caption") or "",
```

后台函数 `_classify_and_persist` 应把整个 `image_map` 传给更新后的 `classify_images`，而不是只传 `paths`。写回 MongoDB 的 `image_type`、`confidence`、`all_scores` 与 `error` 逻辑可保持不变。

已有手动入口可继续使用：

```text
POST /api/parse/result/{document_id}/classify-images
GET  /api/parse/clip/health
```

第一阶段建议保持“解析完成后由前端或任务编排显式触发”，便于上线核验。稳定后再由 Ceramics 的任务队列在 `run_parse_pipeline` 完成 MongoDB 写入后异步触发；CLIP 故障不得把 PDF 解析任务标记为失败。

## 验收

1. `GET http://127.0.0.1:8011/api/clip/health` 返回成功。
2. 对一篇已有解析结果执行分类接口，所有结果能按图片路径匹配并写回 MongoDB。
3. 含 `HRTEM` 或 `XRD` caption 的低置信度图片能随请求到达 CLIP 日志（日志不应输出 caption 正文）。
4. `tem` 能通过 Ceramics 的标签校验和前端展示；`other` 不触发后续柱状图、曲线图或 XRD 解析。
5. 暂停 CLIP 容器后，Ceramics 的 PDF 解析仍应完成，仅分类任务返回可追踪的失败状态。

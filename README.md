# CLIP 文献图片分类服务

面向材料论文图片的分类 API。服务返回 `bar_chart`、`line_chart`、`sem`、`tem`、`xrd` 或 `other`，并使用 gate 拦截非目标图片。

## Docker 快速部署

部署者只需 Docker，不需要 conda、PyTorch、Hugging Face 网络或本地模型文件。镜像已包含 CLIP 基座模型和训练得到的分类器、gate。

### 1. 获取部署文件

```powershell
git clone https://github.com/FisherNotFool/CLIP.git
cd CLIP
```

### 2. 如镜像为私有，先登录 GHCR

公开镜像可跳过此步骤。私有镜像需要具备读取包权限的 GitHub PAT：

```powershell
docker login ghcr.io -u FisherNotFool
```

### 3. 配置上游图片目录并启动

`IMAGE_OUTPUTS_PATH` 必须指向 Ceramics 的 `backend/outputs` 目录。该目录以只读方式挂载到容器，不会被 CLIP 修改。

```powershell
$env:CLIP_IMAGE = "ghcr.io/fishernotfool/ceramics-clip:0.2.0"
$env:IMAGE_OUTPUTS_PATH = "D:/Project/ceramics/material-kg/backend/outputs"
docker compose -f compose.deploy.yaml pull
docker compose -f compose.deploy.yaml up -d
```

### 4. 验证服务

```powershell
Invoke-RestMethod http://127.0.0.1:8011/api/clip/health
```

预期响应：

```json
{"status":"ok","model_loaded":true}
```

停止服务：

```powershell
docker compose -f compose.deploy.yaml down
```

## 调用分类接口

图片路径相对于 `IMAGE_OUTPUTS_PATH`。`caption` 可选；视觉分类置信度较低时，明确的 TEM、SEM、XRD 等 caption 会参与保守纠偏。被 gate 判为 `other` 的结果不会被 caption 覆盖。

```powershell
$body = @{
  document_id = "paper_001"
  images = @(
    @{
      image_path = "/doc_001/images/fig1.jpg"
      caption = "Figure 1. HRTEM image of the sample."
    }
  )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8011/api/clip/classify" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

调试时附加 `?debug=true` 可返回各已知类别分数：

```text
POST /api/clip/classify?debug=true
```

## 更新版本

使用新版本时只修改镜像标签，然后重新拉取并启动：

```powershell
$env:CLIP_IMAGE = "ghcr.io/fishernotfool/ceramics-clip:0.2.1"
docker compose -f compose.deploy.yaml pull
docker compose -f compose.deploy.yaml up -d
```

生产环境应使用明确版本号，不要使用 `latest`。

## Ceramics 集成

Ceramics 后端通过 HTTP 调用本服务，并把 `image_type`、`confidence`、`all_scores` 和错误信息写回 MongoDB。具体改动清单见：[Ceramics 接入交接文档](docs/ceramics-clip-integration-handoff.md)。

镜像发布流程见：[容器发布与他人部署](docs/container-release.md)。

## 本地开发与构建镜像

仅模型维护者需要本地构建。构建依赖 `model_cache/` 中预先准备好的 CLIP 基座缓存；普通部署者请使用上面的预构建镜像流程。

```powershell
$env:IMAGE_OUTPUTS_PATH = "D:/Project/ceramics/material-kg/backend/outputs"
docker compose up -d --build
```

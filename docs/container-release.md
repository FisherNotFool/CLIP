# CLIP 容器发布与他人部署

## 原则

Git 仓库不包含 CLIP 基座权重。发布者在已验证、且本地 `model_cache/` 已完整的工作机上构建一次镜像；部署者只拉取带版本号的镜像，绝不执行 `docker compose build`。

不要使用 `latest` 作为生产版本。每次模型、gate 或应用代码变化都使用新的不可变标签，例如 `0.2.0`、`0.2.1`。

## 发布镜像（发布者执行）

先在本项目目录完成本地验证：

```powershell
docker compose build clip-model
docker compose up -d clip-model
Invoke-RestMethod http://127.0.0.1:8011/api/clip/health
```

以 GitHub Container Registry 为例，准备一个具有 `write:packages` 权限的 GitHub PAT 并登录。请将镜像地址与实际 GitHub owner 保持一致：

```powershell
docker login ghcr.io -u FisherNotFool
$env:CLIP_IMAGE = "ghcr.io/fishernotfool/ceramics-clip:0.2.0"
docker tag ceramics-clip:latest $env:CLIP_IMAGE
docker push $env:CLIP_IMAGE
```

首次推送后，在 GitHub Packages 中将包设为与 Ceramics 部署环境相匹配的可见性；私有包的部署主机必须先执行 `docker login ghcr.io`。

## 部署者执行

部署者只需要 Docker 和上游图片目录的读权限：

```powershell
cd D:\Project\CLIP
$env:CLIP_IMAGE = "ghcr.io/fishernotfool/ceramics-clip:0.2.0"
$env:IMAGE_OUTPUTS_PATH = "D:/Project/ceramics/material-kg/backend/outputs"
docker compose -f compose.deploy.yaml pull
docker compose -f compose.deploy.yaml up -d
Invoke-RestMethod http://127.0.0.1:8011/api/clip/health
```

`compose.deploy.yaml` 不含 `build:`，因此不会请求 Hugging Face，也不依赖本地模型缓存。它只读挂载 `backend/outputs` 到容器的 `/data/outputs`。

## Ceramics 使用

在 Ceramics 的 `backend/model_services/docker-compose.yml` 中，生产环境使用：

```yaml
  clip-model:
    image: ${CLIP_IMAGE}
```

其余 `environment`、`ports`、`volumes` 和 `healthcheck` 可直接采用 `compose.deploy.yaml` 中同名服务的配置。不要保留 `build:`；否则部署主机会重新尝试构建模型镜像。

后端地址仍遵循：主后端在宿主机运行时使用 `http://127.0.0.1:8011`；主后端与 CLIP 同一 Compose network 时使用 `http://clip-model:8011`。

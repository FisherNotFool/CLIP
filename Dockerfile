# CPU production image. The CLIP base model is downloaded during image build,
# so a running container never needs access to Hugging Face.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

# PyTorch's CPU index prevents CUDA runtime packages being installed into a
# service that is intentionally deployed as a CPU container.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY model_cache/ /app/model_cache/
COPY app/ /app/app/

ENV HF_HOME=/opt/huggingface \
    MODEL_CACHE_DIR=/app/model_cache \
    DEVICE=cpu \
    IMAGE_BASE_PATH=/data/outputs \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1

# The build is the only stage allowed to access Hugging Face. Download both
# model and processor before the offline runtime flags above take effect.
RUN TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 python -c "from transformers import CLIPModel, CLIPProcessor; model = 'openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(model); CLIPProcessor.from_pretrained(model)"

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /opt/huggingface
USER appuser

EXPOSE 8011

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8011"]

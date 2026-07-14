# CPU production image. The CLIP base model cache is copied from model_cache/
# during the build, so Docker itself never needs access to Hugging Face.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

# PyTorch's CPU index prevents CUDA runtime packages being installed into a
# service that is intentionally deployed as a CPU container.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY model_cache/ /app/model_cache/
COPY app/ /app/app/

ENV MODEL_CACHE_DIR=/app/model_cache \
    DEVICE=cpu \
    IMAGE_BASE_PATH=/data/outputs \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8011

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8011"]

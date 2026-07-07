# --- CPU-only production image ---
FROM python:3.11-slim

# Install torch CPU-only to keep image small (~200 MB vs ~2 GB for CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-downloaded model cache (build host must have run download_model.py first)
COPY model_cache/ /app/model_cache/

# Copy application code
COPY app/ /app/app/

# Copy environment config
COPY .env /app/.env

# Force offline — no network requests to HuggingFace Hub at runtime
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV MODEL_CACHE_DIR=/app/model_cache
ENV DEVICE=cpu

EXPOSE 8011

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8011"]

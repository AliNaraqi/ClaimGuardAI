# ClaimGuard AI — Streamlit dashboard image.
#
# NOT yet built/verified in this environment (Docker is not installed on the dev
# machine this was authored on) — verify with `docker build .` before relying on it.
#
# This image does NOT bundle the licensed Kaggle dataset or trained model artifacts.
# Mount them at runtime, e.g.:
#   docker run -p 8501:8501 \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/reports:/app/reports \
#     claimguard-ai

FROM python:3.11-slim

WORKDIR /app

# libgomp1 provides the OpenMP runtime XGBoost needs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM python:3.12-slim

# System deps — only what's needed for subprocess sandbox
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer if requirements don't change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agents/       ./agents/
COPY store/        ./store/
COPY concept_packets/ ./concept_packets/
COPY models.py router.py orchestrator.py app.py ./

# Data directory — mount a volume here in production for persistence
# docker run -v /host/data:/app/data ...
RUN mkdir -p /app/data

# Gradio listens on 7860 by default
EXPOSE 7860

# DASHSCOPE_API_KEY must be passed at runtime — never bake it in
# docker run -e DASHSCOPE_API_KEY=sk-... icap-tutor
ENV ICAP_DATA_DIR=/app/data

CMD ["python", "app.py"]

# ── Base image ─────────────────────────────────────────────
# Python 3.11 slim — smaller image than full Python
# reduces container size from ~1GB to ~200MB
FROM python:3.11-slim

# ── Why these system packages? ─────────────────────────────
# build-essential: compiles Python packages with C extensions
#                  (faiss-cpu, tiktoken need this)
# curl: health check in production deployments
# git: some langchain packages fetch from git at install time
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ────────────────────────────
# Copy requirements first — Docker layer caching means
# if requirements.txt hasn't changed, this layer is cached
# and pip install is skipped on subsequent builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Download spaCy model ───────────────────────────────────
# Must be done after pip install, inside the container
RUN python -m spacy download en_core_web_lg

# ── Copy application code ──────────────────────────────────
COPY . .

# ── Create required directories ────────────────────────────
RUN mkdir -p data vectorstore logs

# ── Expose ports ───────────────────────────────────────────
# 8000: FastAPI backend
# 8501: Streamlit frontend
EXPOSE 8000
EXPOSE 8501

# ── Default command runs FastAPI ───────────────────────────
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
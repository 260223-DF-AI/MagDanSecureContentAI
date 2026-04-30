# -----------------------------
# Base Image
# -----------------------------
FROM python:3.11-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -----------------------------
# System Dependencies
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Create App Directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Install Python Dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# -----------------------------
# Default Command
# (You can override this at runtime)
# -----------------------------
# CMD ["python", "scripts/ingest.py"]

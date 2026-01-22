# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Ensure unbuffered output for logs
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with visible progress
COPY requirements.txt .
RUN pip install --no-cache-dir --progress-bar on -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set default port (Railway overrides via PORT env var)
ENV PORT=8000

# Default command - use shell form to expand $PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT

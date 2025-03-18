# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Set environment variables to optimize Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Create necessary directories
RUN mkdir -p logs data/tmp reports

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default environment variables
# These will be overridden by Docker Compose or Kubernetes configs
ENV SOCRATA_APP_TOKEN="" \
    SOCRATA_API_KEY_ID="" \
    SOCRATA_API_KEY_SECRET="" \
    MINIO_ENDPOINT="minio:9000" \
    MINIO_ACCESS_KEY="admin" \
    MINIO_SECRET_KEY="password" \
    TRINO_HOST="trino" \
    TRINO_PORT="8080" \
    TRINO_USER="admin" \
    PORT=5000 \
    LOG_LEVEL="INFO"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Expose the port the app runs on
EXPOSE 5000

# Start the Flask application
CMD ["python", "app.py"] 
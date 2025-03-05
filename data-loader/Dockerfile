# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Ensure logs directory exists for debugging and monitoring
RUN mkdir -p logs

# Create directories
RUN mkdir -p data/tmp

# Copy only the dependencies first to leverage build cache
COPY requirements.txt .

# Upgrade pip and install any required packages
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Environment variables with default values (will be overridden by .env file)
ENV SOCRATA_APP_TOKEN=""
ENV SOCRATA_API_KEY_ID=""
ENV SOCRATA_API_KEY_SECRET=""
ENV MINIO_ENDPOINT="minio:9000"
ENV MINIO_ACCESS_KEY="admin"
ENV MINIO_SECRET_KEY="password"
ENV TRINO_HOST="trino"
ENV TRINO_PORT="8080"
ENV TRINO_USER="admin"

# Start the data loader service with logging
CMD ["python", "main.py", "--dataset", "vx8i-nprf"] 
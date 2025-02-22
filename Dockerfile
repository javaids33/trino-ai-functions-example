FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY llm-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY llm-service/ .

# Create directories for persistent data
RUN mkdir -p data .chromadb

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5001

# Run the application
CMD ["python", "app.py"] 
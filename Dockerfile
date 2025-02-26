FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including wget and java for Trino CLI
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    default-jre \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Trino CLI
RUN wget -O /usr/local/bin/trino https://repo1.maven.org/maven2/io/trino/trino-cli/471/trino-cli-471-executable.jar \
    && chmod +x /usr/local/bin/trino

# Create directories for persistent data and logs
RUN mkdir -p data .chromadb logs

# Copy requirements first to leverage Docker cache
COPY llm-service/requirements.txt .

# Install PyTorch with CUDA support if available, otherwise CPU version
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY llm-service/ .

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5001

# Run the application
CMD ["python", "app.py"] 
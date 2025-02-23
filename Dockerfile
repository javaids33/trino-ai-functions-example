FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including wget and java for Trino CLI
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Install Trino CLI
RUN wget -O /usr/local/bin/trino https://repo1.maven.org/maven2/io/trino/trino-cli/471/trino-cli-471-executable.jar \
    && chmod +x /usr/local/bin/trino

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
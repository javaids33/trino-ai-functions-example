FROM python:3.9-slim

WORKDIR /usr/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dbt project
COPY . .

# Set environment variables
ENV DBT_PROFILES_DIR=/usr/app

# Set entrypoint
ENTRYPOINT ["dbt"] 
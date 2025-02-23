# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Ensure logs directory exists for debugging and monitoring
RUN mkdir logs

# Create data directory
RUN mkdir data

# Copy only the dependencies first to leverage build cache
COPY requirements.txt .

# Upgrade pip and install any required packages
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the data files
COPY data/*.parquet data/

# Copy the rest of the application code
COPY . .

# Expose the port where data loader might be listening (adjust if needed)
EXPOSE 5002

# Start the data loader service with logging
CMD ["python", "data_loader.py"] 
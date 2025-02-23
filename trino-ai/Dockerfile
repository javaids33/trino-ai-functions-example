# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Create a directory for logs
RUN mkdir logs

# Copy only requirements to take advantage of Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port used by the Trino-AI service
EXPOSE 5001

# Start the Trino-AI service
CMD ["python", "app.py"] 
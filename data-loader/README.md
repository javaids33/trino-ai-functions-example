# NYC Data Loader

A streamlined ETL system for loading NYC Open Data into Trino with Iceberg catalog for advanced analytics and AI-driven processing.

## 🚀 Features

- **Simplified ETL Pipeline**: Direct loading from Socrata API to Trino with Iceberg catalog
- **Connection Pooling**: Efficient management of database connections
- **Intelligent Caching**: Smart caching system that minimizes API calls
- **Stateless Operation**: Containerized application with minimal dependencies
- **RESTful API**: API endpoints for managing data loading and querying
- **Swagger Documentation**: Comprehensive API documentation

## 📋 Architecture

The system follows a simplified ETL pipeline:

```
Socrata API → Parquet Files → MinIO Object Storage → Trino with Iceberg Catalog
```

### Components:

- **Cache Manager**: Tracks dataset metadata and caches data locally
- **Connection Manager**: Handles connections to external systems with connection pooling
- **MinIO Manager**: Manages interactions with MinIO object storage
- **Data Loader**: Core ETL process for loading datasets

## 🔧 Setup

### Prerequisites

- Docker and Docker Compose
- Socrata API credentials (optional but recommended)
- MinIO instance
- Trino instance with Iceberg catalog

### Environment Variables

Create a `.env` file with the following variables:

```
# Socrata API credentials
SOCRATA_APP_TOKEN=your_app_token_here
SOCRATA_API_KEY_ID=your_api_key_id_here
SOCRATA_API_KEY_SECRET=your_api_key_secret_here

# MinIO configuration
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=password
MINIO_SECURE=False
MINIO_BUCKET=iceberg

# Trino configuration
TRINO_HOST=trino
TRINO_PORT=8080
TRINO_USER=admin
TRINO_CATALOG=iceberg
TRINO_SCHEMA=iceberg

# Application settings
PORT=5000
DEBUG=False
LOG_LEVEL=INFO
CACHE_DIR=data_cache
TEMP_DIR=temp
DATA_DIR=data
LOGS_DIR=logs
```

### Docker Setup

Build and run the Docker container:

```bash
docker-compose build trino-ai
docker-compose up -d trino-ai
```

## 📊 Usage

### Command Line Interface

Load datasets via the command line:

```bash
# Load a specific dataset
python data_loader.py 5694-9szk

# Load multiple datasets
python data_loader.py 5694-9szk kz4z-fdn2 sqcr-6vxa

# Load datasets from a file
python data_loader.py --file dataset_ids.txt

# Force reload even if cached
python data_loader.py --force 5694-9szk

# Load all cached datasets
python data_loader.py --all
```

### API Endpoints

The system provides RESTful API endpoints:

- `GET /api/datasets`: List available datasets
- `POST /api/datasets/load`: Load a dataset
- `GET /api/popular`: Get popular datasets
- `GET /api/metadata/{dataset_id}`: Get dataset metadata
- `GET /health`: Check system health
- `GET /system-status`: Get detailed system status

API documentation is available at `/swagger`.

## 🧪 Testing

Run the tests using pytest:

```bash
pytest tests/
```

## 🗃️ Folder Structure

```
data-loader/
├── api/                 # API endpoints
├── app.py               # Main Flask application
├── app_config.py        # Configuration management
├── cache_manager.py     # Dataset cache management
├── data_loader.py       # Main data loading logic
├── logger_config.py     # Logging configuration
├── minio_helper.py      # MinIO interactions
├── socrata_loader.py    # Socrata API interactions
├── trino_connector.py   # Trino connection management
├── Dockerfile           # Docker configuration
└── requirements.txt     # Python dependencies
```

## 📝 Improvements

Recent improvements include:

1. **Consolidated Configuration**: Single configuration system via `AppConfig`
2. **Simplified Pipeline**: Removed DuckDB intermediate step
3. **Connection Management**: Implemented connection pooling for Trino
4. **Simplified Cache**: More efficient dataset tracking with SQLite
5. **Removed Dead Code**: Deleted unused features and files

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

# NYC Open Data Loader

## Overview
This service provides an API for loading, managing, and querying datasets from the NYC Open Data portal.
It handles the ETL process from Socrata to Trino via MinIO storage using the Iceberg format.

## Setup Instructions

### Environment Variables
Create a `.env` file based on the `.env.example` template: 
@echo off
REM Script to load a specific NYC dataset
REM Usage: load-dataset.bat [dataset_id]
REM Example: load-dataset.bat vx8i-nprf

echo Starting dataset loading process...

REM Check if dataset ID is provided
if "%1"=="" (
    echo Using default dataset ID: vx8i-nprf
    set DATASET_ID=vx8i-nprf
) else (
    echo Using provided dataset ID: %1
    set DATASET_ID=%1
)

REM Build the data-loader container
echo Building data-loader container...
docker-compose build data-loader

REM Run the data-loader with the specified dataset
echo Loading dataset %DATASET_ID%...
set DATASET_ID=%DATASET_ID%
docker-compose run --rm data-loader

echo Dataset loading process completed.
echo.
echo To query the imported data in Trino, use the following SQL:
echo.
echo -- List all available schemas
echo SHOW SCHEMAS FROM iceberg;
echo.
echo -- List all tables in a schema
echo SHOW TABLES FROM iceberg.general;
echo.
echo -- Query the metadata registry
echo SELECT * FROM iceberg.nycdata.dataset_registry WHERE dataset_id = '%DATASET_ID%';
echo. 
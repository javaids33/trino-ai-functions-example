@echo off
REM NYC Open Data Manager - Windows Batch Script
REM Usage: nyc-data-manager.bat [command] [options]

echo NYC Open Data Manager

REM Check if Docker is running
docker info > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker is not running. Please start Docker first.
    exit /b 1
)

REM Check if a command is provided
if "%1"=="" (
    echo Usage: nyc-data-manager.bat [command] [options]
    echo.
    echo Commands:
    echo   list-popular   List popular datasets
    echo   load-popular   Load popular datasets
    echo   load-dataset   Load a specific dataset
    echo   load-pool      Load a diverse pool of datasets
    echo   report         Generate a report on all datasets
    echo.
    echo Examples:
    echo   nyc-data-manager.bat list-popular --limit 10
    echo   nyc-data-manager.bat load-popular --limit 5 --concurrency 3
    echo   nyc-data-manager.bat load-dataset --dataset-id vx8i-nprf
    echo   nyc-data-manager.bat load-pool --pool-size 5
    echo   nyc-data-manager.bat report
    exit /b 0
)

REM Check if Trino is running
echo Checking if Trino is running...
docker-compose ps trino | findstr "(healthy)" > nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Trino is not running or not healthy. Please start Trino first.
    exit /b 1
)

REM Check if MinIO is running
echo Checking if MinIO is running...
docker-compose ps minio | findstr "Up" > nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: MinIO is not running. Please start MinIO first.
    exit /b 1
)

REM Build the command string
set CMD=python nyc_data_manager.py %*

REM Run the command in the data-loader container
echo Running command: %CMD%
docker-compose run --rm data-loader %CMD%

echo.
echo Command completed.
exit /b 0 
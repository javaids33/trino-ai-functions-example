#!/bin/bash

# Socrata ETL Manager - Main script to manage the Socrata ETL process
# Usage: ./scripts/socrata_etl_manager.sh [command]

# Set up logging
LOG_FILE="logs/socrata_etl_manager.log"
mkdir -p logs

echo "$(date) - Starting Socrata ETL Manager" | tee -a $LOG_FILE

# Check if scripts are executable
if [ ! -x "scripts/start_services.sh" ] || [ ! -x "scripts/run_socrata_etl.sh" ] || [ ! -x "scripts/load_nyc_dataset.sh" ] || [ ! -x "scripts/query_trino.sh" ]; then
    echo "Making scripts executable..." | tee -a $LOG_FILE
    chmod +x scripts/start_services.sh scripts/run_socrata_etl.sh scripts/load_nyc_dataset.sh scripts/query_trino.sh
fi

# Display help message
show_help() {
    echo "Socrata ETL Manager - Manage the Socrata ETL process"
    echo ""
    echo "Usage: ./scripts/socrata_etl_manager.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start       Start all necessary services (Trino, MinIO)"
    echo "  run         Run the full Socrata ETL process"
    echo "  load        Load a specific dataset from Socrata"
    echo "  query       Query data in Trino"
    echo "  status      Check the status of all services"
    echo "  stop        Stop all services"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/socrata_etl_manager.sh start"
    echo "  ./scripts/socrata_etl_manager.sh run data.cityofnewyork.us 10"
    echo "  ./scripts/socrata_etl_manager.sh load vx8i-nprf"
    echo "  ./scripts/socrata_etl_manager.sh query \"SELECT * FROM iceberg.metadata.dataset_registry\""
    echo ""
}

# Check if a command is provided
if [ -z "$1" ]; then
    show_help
    exit 0
fi

# Process commands
case "$1" in
    start)
        echo "Starting services..." | tee -a $LOG_FILE
        ./scripts/start_services.sh
        ;;
    run)
        echo "Running Socrata ETL process..." | tee -a $LOG_FILE
        if [ -n "$2" ]; then
            DOMAIN="$2"
            if [ -n "$3" ]; then
                LIMIT="$3"
                ./scripts/run_socrata_etl.sh "$DOMAIN" "$LIMIT"
            else
                ./scripts/run_socrata_etl.sh "$DOMAIN"
            fi
        else
            ./scripts/run_socrata_etl.sh
        fi
        ;;
    load)
        echo "Loading dataset..." | tee -a $LOG_FILE
        if [ -n "$2" ]; then
            ./scripts/load_nyc_dataset.sh "$2"
        else
            echo "ERROR: No dataset ID provided. Usage: ./scripts/socrata_etl_manager.sh load <dataset_id>" | tee -a $LOG_FILE
            echo "Example: ./scripts/socrata_etl_manager.sh load vx8i-nprf (NYC Civil Service List)" | tee -a $LOG_FILE
            exit 1
        fi
        ;;
    query)
        echo "Querying Trino..." | tee -a $LOG_FILE
        if [ -n "$2" ]; then
            ./scripts/query_trino.sh "$2"
        else
            ./scripts/query_trino.sh
        fi
        ;;
    status)
        echo "Checking service status..." | tee -a $LOG_FILE
        docker-compose ps
        ;;
    stop)
        echo "Stopping all services..." | tee -a $LOG_FILE
        docker-compose down
        echo "All services stopped" | tee -a $LOG_FILE
        ;;
    help)
        show_help
        ;;
    *)
        echo "ERROR: Unknown command: $1" | tee -a $LOG_FILE
        show_help
        exit 1
        ;;
esac

echo "$(date) - Socrata ETL Manager completed" | tee -a $LOG_FILE

exit 0 
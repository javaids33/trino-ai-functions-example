# Trino AI Translate Integration

This document explains how the Trino AI multi-agent system has been integrated with Trino's `ai_translate` function.

## Overview

The integration allows Trino's `ai_translate` function to use our sophisticated multi-agent system to translate natural language queries to SQL. When a user calls `SELECT "ai-functions".ai.ai_translate('query', 'sql')` in Trino, the request is routed to our multi-agent system, which processes the query and returns the generated SQL.

## Components

The integration consists of the following components:

1. **AI Translate Handler** (`ai_translate_handler.py`): Intercepts and processes `ai_translate` function calls from Trino.
2. **Conversation Logger** (`conversation_logger.py`): Enhanced to track the detailed workflow of the multi-agent system.
3. **API Endpoint** (`app.py`): Added a new endpoint `/api/ai_translate` to handle requests from Trino.
4. **Workflow Viewer** (`static/workflow-viewer.html`): A web interface to view the detailed workflow of the multi-agent system.
5. **UDF Wrapper** (`trino-ai-functions/ai_translate_udf.sql`): A SQL script to create the UDF wrapper in Trino.
6. **Application Script** (`scripts/apply_ai_translate_udf.sh`): A script to apply the UDF to Trino.

## How It Works

1. User executes a query in Trino: `SELECT "ai-functions".ai.ai_translate('find top customers', 'sql')`
2. Trino routes the request to our API endpoint: `http://trino-ai:5001/api/ai_translate`
3. The AI Translate Handler processes the request using the multi-agent system:
   - For data queries, it returns the generated SQL
   - For knowledge queries, it returns the answer as a SQL comment
4. The response is returned to Trino and displayed to the user
5. The detailed workflow is logged and can be viewed in the Workflow Viewer

## Workflow Viewer

The Workflow Viewer provides a detailed view of the multi-agent system's workflow. It shows:

- Each step in the workflow
- The agent responsible for each step
- The action performed at each step
- The details of each step

To access the Workflow Viewer, visit: `http://localhost:5001/workflow-viewer`

## Setup Instructions

1. Ensure all services are running:
   ```bash
   docker-compose up -d minio nessie trino data-loader ollama trino-ai
   ```

2. Apply the UDF to Trino:
   ```bash
   ./scripts/apply_ai_translate_udf.sh
   ```

3. Test the integration:
   ```sql
   SELECT "ai-functions".ai.ai_translate('find top customers we sold products to', 'sql')
   ```

4. View the workflow:
   - Open a browser and navigate to `http://localhost:5001/workflow-viewer`

## Troubleshooting

If you encounter issues with the integration:

1. Check the logs:
   ```bash
   docker-compose logs -f trino-ai
   ```

2. Ensure all services are running:
   ```bash
   docker-compose ps
   ```

3. Check the Trino logs:
   ```bash
   docker-compose logs -f trino
   ```

4. Verify the UDF is properly installed:
   ```sql
   SHOW FUNCTIONS IN "ai-functions".ai LIKE 'ai_translate'
   ```

## Example Queries

Here are some example queries to test the integration:

```sql
-- Data query
SELECT "ai-functions".ai.ai_translate('find top 5 customers by total purchase amount', 'sql')

-- Knowledge query
SELECT "ai-functions".ai.ai_translate('what is the capital of France?', 'sql')
``` 
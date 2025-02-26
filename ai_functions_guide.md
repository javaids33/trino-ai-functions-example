# Trino AI Functions Guide

## Overview

This guide explains how to use the AI functions available in the Trino AI connector. These functions allow you to leverage large language models (LLMs) directly from SQL queries.

## Configuration

The AI functions are configured in the `trino/etc/catalog/ai-functions.properties` file. The current configuration is:

```properties
connector.name=ai
ai.provider=openai
ai.model=llama3.2
ai.openai.endpoint=http://trino-ai:5001
ai.openai.api-key=none
```

This configuration connects to the `trino-ai` service, which provides an OpenAI-compatible API that forwards requests to the Ollama service.

## Available AI Functions

The following AI functions are available:

| Function | Description | Example |
|----------|-------------|---------|
| `ai_gen` | Convert natural language to SQL or generate text based on a prompt | `SELECT "ai-functions".ai.ai_gen('find top 3 products by cost')` |
| `ai_analyze_sentiment` | Perform sentiment analysis on text | `SELECT "ai-functions".ai.ai_analyze_sentiment('This product is fantastic!')` |
| `ai_translate` | Translate text to the specified language | `SELECT "ai-functions".ai.ai_translate('Hello, how are you?', 'Spanish')` |
| `ai_classify` | Classify text with the provided labels | `SELECT "ai-functions".ai.ai_classify('The weather is sunny', ARRAY['weather', 'food', 'sports'])` |
| `ai_extract` | Extract values for the provided labels from text | `SELECT "ai-functions".ai.ai_extract('My name is John and I am 30 years old', ARRAY['name', 'age'])` |
| `ai_fix_grammar` | Correct grammatical errors in text | `SELECT "ai-functions".ai.ai_fix_grammar('I is going to the store')` |
| `ai_mask` | Mask values for the provided labels in text | `SELECT "ai-functions".ai.ai_mask('My credit card number is 1234-5678-9012-3456', ARRAY['credit card'])` |

## Usage Examples

### Natural Language to SQL Conversion (NEW)

The `ai_gen` function now automatically converts natural language queries to SQL using a multi-agent architecture. Simply provide your question about the data:

```sql
SELECT "ai-functions".ai.ai_gen('find customers who spent more than $1000 last month')
```

This will return a formatted response containing:
- The SQL query that answers your question
- An explanation of how the query works
- Information about the database schema used

### Text Generation

For general text generation that's not related to SQL queries:

```sql
SELECT "ai-functions".ai.ai_gen('Write a short poem about data')
```

### Sentiment Analysis

```sql
SELECT 
  product_name,
  review_text,
  "ai-functions".ai.ai_analyze_sentiment(review_text) AS sentiment
FROM 
  iceberg.iceberg.product_reviews
LIMIT 10
```

### Translation

```sql
SELECT 
  original_text,
  "ai-functions".ai.ai_translate(original_text, 'French') AS french_translation,
  "ai-functions".ai.ai_translate(original_text, 'Spanish') AS spanish_translation
FROM 
  iceberg.iceberg.messages
LIMIT 5
```

### Classification

```sql
SELECT 
  article_title,
  "ai-functions".ai.ai_classify(
    article_content, 
    ARRAY['politics', 'sports', 'technology', 'entertainment']
  ) AS category
FROM 
  iceberg.iceberg.articles
LIMIT 10
```

## Multi-Agent NL2SQL Architecture

The `ai_gen` function now uses a sophisticated multi-agent architecture to convert natural language to SQL:

### Agents

1. **DBA Agent**: Analyzes your natural language query to identify:
   - Relevant tables and columns
   - Required joins between tables
   - Necessary filters and conditions
   - Appropriate aggregations

2. **SQL Agent**: Generates optimized SQL based on the DBA Agent's analysis:
   - Creates syntactically correct Trino SQL
   - Applies proper table qualifications
   - Implements efficient joins and filters
   - Validates the SQL before returning it

### Tools

The agents use specialized tools to perform their tasks:

- **Schema Context Tool**: Retrieves relevant schema information based on your query
- **SQL Validation Tool**: Checks if generated SQL is valid without executing it
- **SQL Execution Tool**: Executes SQL queries and returns results (when needed)
- **Metadata Refresh Tool**: Updates the schema information cache

### Workflow

When you use the `ai_gen` function with a natural language query:

1. The system detects that you're asking about data
2. The Schema Context Tool retrieves relevant tables and columns
3. The DBA Agent analyzes your query and identifies needed database objects
4. The SQL Agent generates a SQL query based on the DBA analysis
5. The SQL Validation Tool checks if the query is valid
6. If needed, the SQL Agent refines the query to fix any issues
7. The system returns the SQL query with an explanation

This multi-step process ensures more accurate and reliable SQL generation compared to single-step approaches.

## Troubleshooting

If you encounter issues with the AI functions:

1. Check that the Trino service is running: `docker-compose ps trino`
2. Check that the trino-ai service is running: `docker-compose ps trino-ai`
3. Check that the Ollama service is running: `docker-compose ps ollama`
4. Check the logs for errors: `docker-compose logs trino` or `docker-compose logs trino-ai`
5. Ensure the `ai-functions.properties` file is correctly configured

## Architecture

The AI functions work by:

1. Trino receives a SQL query with an AI function
2. The AI connector processes the function call
3. The connector sends a request to the trino-ai service (OpenAI-compatible API)
4. The trino-ai service detects the function type (e.g., `ai_gen` is treated as NL2SQL)
5. For NL2SQL queries, the service activates the multi-agent system:
   - The Agent Orchestrator coordinates the workflow
   - The DBA Agent analyzes the query
   - The SQL Agent generates and refines SQL
6. The response is returned through the chain back to Trino
7. Trino includes the result in the SQL query result 
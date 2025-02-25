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
| `ai_gen` | Generate text based on a prompt | `SELECT "ai-functions".ai.ai_gen('Tell me a short joke')` |
| `ai_analyze_sentiment` | Perform sentiment analysis on text | `SELECT "ai-functions".ai.ai_analyze_sentiment('This product is fantastic!')` |
| `ai_translate` | Translate text to the specified language | `SELECT "ai-functions".ai.ai_translate('Hello, how are you?', 'Spanish')` |
| `ai_classify` | Classify text with the provided labels | `SELECT "ai-functions".ai.ai_classify('The weather is sunny', ARRAY['weather', 'food', 'sports'])` |
| `ai_extract` | Extract values for the provided labels from text | `SELECT "ai-functions".ai.ai_extract('My name is John and I am 30 years old', ARRAY['name', 'age'])` |
| `ai_fix_grammar` | Correct grammatical errors in text | `SELECT "ai-functions".ai.ai_fix_grammar('I is going to the store')` |
| `ai_mask` | Mask values for the provided labels in text | `SELECT "ai-functions".ai.ai_mask('My credit card number is 1234-5678-9012-3456', ARRAY['credit card'])` |

## Usage Examples

### Text Generation

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
4. The trino-ai service forwards the request to Ollama
5. Ollama processes the request using the specified model (llama3.2)
6. The response is returned through the chain back to Trino
7. Trino includes the result in the SQL query result 
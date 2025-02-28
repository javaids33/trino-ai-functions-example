# Trino AI

Current Status: 
    -Able to run NLQ to SQL translation and return results, need refine the agentic flow and usage of metadata. 
    -Have not tested the system with complex queries/ large datasets yet.
    - Able to run the ai-agent from different trino, looking into  https://github.com/SwanseaUniversityMedical/trino2trino or https://github.com/sajjoseph/trino/tree/add-trino-to-trino-connector for ai-agent flow to return data results over trino-ai-only.
    -Need to add more tests and documentation. 

## Overview

Trino AI bridges the gap between natural language and SQL, allowing business users, analysts, and anyone unfamiliar with SQL to effectively query their Trino data warehouse. By leveraging large language models and a multi-agent architecture, Trino AI translates natural language questions into optimized SQL queries, executes them against your Trino instance, and presents the results in an easy-to-understand format.

## Key Features

- **Natural Language to SQL Translation**: Translate English questions to valid SQL queries
- **Multi-Agent Architecture**: Uses specialized agents (DBA, SQL) to analyze, generate, and optimize queries
- **Schema Awareness**: Automatically identifies and utilizes relevant tables and fields
- **Interactive UI**: Simple web interface for querying and viewing results
- **Context Retention**: Remembers previous queries for follow-up questions
- **Query Validation**: Ensures generated SQL is valid before execution
- **Trino Integration**: Seamlessly connects to your existing Trino infrastructure

## Architecture

Trino AI consists of several components:

- **Web Interface**: Flask-based UI for inputting queries and displaying results
- **Agent Orchestrator**: Coordinates specialized AI agents to analyze and process queries
- **LLM Integration**: Connects to Ollama for language model capabilities
- **Vector Database**: Uses Chroma for semantic search and schema understanding
- **Trino Client**: Executes generated SQL against Trino and retrieves results

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Trino instance (included in Docker setup)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trino-ai.git
   cd trino-ai
   ```

2. Start the Docker containers:
   ```bash
   docker-compose up -d
   ```

3. Access the UI:
   Open your browser and navigate to `http://localhost:5001`

### Configuration

Configure your Trino connection and other settings in the `.env` file:

```
TRINO_HOST=trino
TRINO_PORT=8080
TRINO_USER=admin
TRINO_CATALOG=iceberg
TRINO_SCHEMA=iceberg
OLLAMA_HOST=ollama
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3.2
```

## Usage

1. Enter a natural language query in the input box:
   - "Show me the top 5 selling products"
   - "What's the total revenue by category last month?"
   - "Find customers who made more than 3 purchases"

2. View the generated SQL and results in the interface

3. Ask follow-up questions, as the system maintains conversation context

## Development

### Project Structure

```
trino-ai/
├── agents/                 # AI agents for different tasks
│   ├── dba_agent.py        # Database analysis agent
│   ├── sql_agent.py        # SQL generation agent
├── tools/                  # Tools used by agents
├── static/                 # Web UI files
├── templates/              # HTML templates
├── app.py                  # Main Flask application
├── agent_orchestrator.py   # Coordinates agent activities
├── conversation_logger.py  # Logs all interactions
├── ollama_client.py        # Interface to Ollama LLM
├── trino_client.py         # Interface to Trino database
├── docker-compose.yml      # Docker compose configuration
└── README.md               # Project documentation
```

### Adding New Capabilities

1. Create new agent types in the `agents/` directory
2. Add new tools in the `tools/` directory
3. Register them in the `agent_orchestrator.py` file

## Troubleshooting

- **Connection Issues**: Ensure Trino is running and accessible
- **Model Loading**: Verify Ollama is running and the specified model is available
- **SQL Errors**: Check the logs for details on SQL validation failures


## Acknowledgments

- Trino Project
- Ollama
- HuggingFace for Sentence Transformers


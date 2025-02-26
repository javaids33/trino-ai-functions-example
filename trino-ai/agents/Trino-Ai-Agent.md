# Trino AI Multi-Agent Architecture

This directory contains the implementation of the multi-agent architecture for natural language to SQL conversion in Trino AI.

## Overview

The multi-agent architecture is designed to improve the accuracy and reliability of natural language to SQL conversion by breaking down the process into specialized steps performed by different agents. Each agent has a specific role and expertise, and they work together to produce high-quality SQL queries.

## Components

### Agents

- **Base Agent (`base_agent.py`)**: Abstract base class that defines the common interface for all agents.
- **DBA Agent (`dba_agent.py`)**: Analyzes natural language queries to identify relevant tables, columns, joins, filters, and aggregations.
- **SQL Agent (`sql_agent.py`)**: Generates and refines SQL queries based on the DBA analysis.

### Tools

Tools are located in the `../tools` directory and provide specific functionalities that agents can use:

- **Schema Context Tool**: Retrieves relevant schema information for a query.
- **SQL Validation Tool**: Validates SQL queries without executing them.
- **SQL Execution Tool**: Executes SQL queries and returns results.
- **Metadata Refresh Tool**: Refreshes the database metadata cache.

### Orchestration

The `../agent_orchestrator.py` file contains the `AgentOrchestrator` class that coordinates the workflow between agents and tools.

## Workflow

1. The orchestrator receives a natural language query.
2. It uses the Schema Context Tool to retrieve relevant schema information.
3. It passes the query and schema context to the DBA Agent for analysis.
4. The DBA Agent identifies the necessary database objects and operations.
5. The orchestrator passes the DBA analysis to the SQL Agent.
6. The SQL Agent generates a SQL query based on the analysis.
7. The SQL query is validated using the SQL Validation Tool.
8. If needed, the SQL Agent refines the query to fix any issues.
9. The orchestrator returns the final SQL query with an explanation.

## Extending the Architecture

### Adding a New Agent

1. Create a new Python file in the `agents` directory.
2. Implement a class that inherits from `Agent`.
3. Override the `execute` method to implement the agent's functionality.
4. Update the `agent_orchestrator.py` file to include the new agent.

### Adding a New Tool

1. Create a new Python file in the `tools` directory.
2. Implement a class that inherits from `Tool`.
3. Override the `execute` method to implement the tool's functionality.
4. Update the `agent_orchestrator.py` file to include the new tool.

## Example

```python
# Example of using the agent orchestrator
from agent_orchestrator import AgentOrchestrator
from ollama_client import OllamaClient

# Initialize the Ollama client
ollama_client = OllamaClient(model="llama3")

# Initialize the agent orchestrator
orchestrator = AgentOrchestrator(ollama_client=ollama_client)

# Process a natural language query
result = orchestrator.process_natural_language_query(
    "Find customers who spent more than $1000 last month"
)

# Print the SQL query
print(result["sql_query"])
``` 
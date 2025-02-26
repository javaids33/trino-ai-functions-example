# Trino AI Tools

This directory contains the implementation of tools used by agents in the Trino AI multi-agent architecture.

## Overview

Tools are specialized components that provide specific functionalities to agents. Each tool has a well-defined purpose and interface, making it easy for agents to use them to accomplish their tasks.

## Components

### Base Tool

- **Base Tool (`base_tool.py`)**: Abstract base class that defines the common interface for all tools.

### Metadata Tools

- **Get Schema Context Tool (`metadata_tools.py`)**: Retrieves relevant schema information for a natural language query.
- **Refresh Metadata Tool (`metadata_tools.py`)**: Refreshes the database metadata cache.

### SQL Tools

- **Validate SQL Tool (`sql_tools.py`)**: Validates SQL queries without executing them.
- **Execute SQL Tool (`sql_tools.py`)**: Executes SQL queries and returns results.

## Tool Interface

Each tool implements the following interface:

- **`__init__(name, description)`**: Initializes the tool with a name and description.
- **`execute(inputs)`**: Executes the tool's functionality with the provided inputs.
- **`get_schema()`**: Returns the schema for the tool, including its parameters.

## Using Tools

Tools are typically used by agents through the agent orchestrator. The orchestrator initializes the tools and provides them to the agents.

```python
# Example of using a tool directly
from tools.metadata_tools import GetSchemaContextTool

# Initialize the tool
schema_tool = GetSchemaContextTool()

# Execute the tool
result = schema_tool.execute({
    "query": "Find customers who spent more than $1000 last month",
    "max_tables": 5
})

# Print the result
print(f"Retrieved schema context with {result['table_count']} tables")
```

## Extending Tools

To add a new tool:

1. Create a new Python file in the `tools` directory or add to an existing file.
2. Implement a class that inherits from `Tool`.
3. Override the `execute` method to implement the tool's functionality.
4. Override the `get_schema` method to define the tool's parameters.
5. Update the `agent_orchestrator.py` file to include the new tool.

## Tool Parameters

Each tool defines its parameters in the `get_schema` method. The parameters are defined using a JSON Schema format:

```python
def get_schema(self) -> Dict[str, Any]:
    schema = super().get_schema()
    schema["parameters"] = {
        "type": "object",
        "properties": {
            "param_name": {
                "type": "string",
                "description": "Description of the parameter"
            },
            "another_param": {
                "type": "integer",
                "description": "Description of another parameter",
                "default": 10
            }
        },
        "required": ["param_name"]
    }
    return schema
```

This schema is used to validate inputs to the tool and to provide documentation for users. 
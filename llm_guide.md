
    LLM Service
    A[User Question] --> B{VectorDB Lookup}
    B --> C[Relevant Schema Elements]
    B --> D[Similar Past Queries(IF AVAILABLE)]
    C --> E[LangChain Agent]
    D --> E
    E --> F[Generated SQL]
    F --> G[Trino Sql Query Validation via sqlglot trino dialect]
    G --> H[Trino Execution]
    H --> I[Result + Explanation]


sequenceDiagram
    User->>AI Service: "Top 5 customers in West region"
    AI Service->>Ollama: Generate SQL
    Ollama-->>AI Service: SELECT customer_id...
    AI Service->>Trino: Execute query
    Trino->>Nessie: Get table version
    Trino->>MinIO: Read Parquet files
    MinIO-->>Trino: Data
    Trino-->>AI Service: Query results
    AI Service->>User: JSON response
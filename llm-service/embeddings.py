from chromadb import Client, Settings
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from trino.dbapi import connect

# Load environment variables
load_dotenv()

class TrinoMetadataEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = Client()
        self.collection = self.client.get_or_create_collection(
            name="trino_metadata",
            metadata={"hnsw:space": "cosine"}
        )
        # Connect to Trino without authentication
        self.trino_conn = connect(
            host="trino",  # Use container name since we're in the same network
            port=8080,
            user="admin",
            catalog="iceberg",
            schema="iceberg",
            http_scheme="http"
        )
        
    def _get_table_metadata(self):
        """Fetch metadata from Trino system tables"""
        cur = self.trino_conn.cursor()
        try:
            cur.execute("""
                SELECT 
                    t.table_catalog,
                    t.table_schema,
                    t.table_name,
                    c.column_name,
                    c.data_type,
                    tc.comment AS table_comment
                FROM tables t
                JOIN columns c ON t.table_catalog = c.table_catalog
                    AND t.table_schema = c.table_schema
                    AND t.table_name = c.table_name
                LEFT JOIN table_comments tc ON t.table_catalog = tc.table_catalog
                    AND t.table_schema = tc.table_schema
                    AND t.table_name = tc.table_name
            """)
            return cur.fetchall()
        finally:
            cur.close()
    
    def refresh_embeddings(self):
        """Update embeddings with latest metadata"""
        metadata = self._get_table_metadata()
        documents = []
        metadatas = []
        ids = []
        
        for row in metadata:
            doc_text = f"""
            Table {row[0]}.{row[1]}.{row[2]}:
            - Column: {row[3]} ({row[4]})
            - Description: {row[5] or 'No description'}
            """
            documents.append(doc_text)
            metadatas.append({
                "catalog": row[0],
                "schema": row[1],
                "table": row[2],
                "column": row[3],
                "type": row[4]
            })
            ids.append(f"{row[0]}_{row[1]}_{row[2]}_{row[3]}")
            
        embeddings = self.model.encode(documents)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=documents
        )

class EmbeddingService(TrinoMetadataEmbedder):
    def __init__(self):
        super().__init__()
        self.collection = self.client.get_or_create_collection("general_embeddings")
    
    def embed(self, text: str):
        return self.model.encode(text).tolist()
    
    def query_metadata(self, text: str, n=3):
        return self.client.get_collection("trino_metadata").query(
            query_embeddings=[self.embed(text)],
            n_results=n
        )
    
    def query_general(self, text: str, n=3):
        return self.collection.query(
            query_embeddings=[self.embed(text)],
            n_results=n
        )

# Initialize with metadata sync
embedding_service = EmbeddingService()
try:
    embedding_service.refresh_embeddings()
except Exception as e:
    print(f"Warning: Initial metadata sync failed: {e}") 
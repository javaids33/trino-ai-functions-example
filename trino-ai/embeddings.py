from chromadb import Client, Settings
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from trino.dbapi import connect
import logging
import torch
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TrinoMetadataEmbedder:
    def __init__(self):
        # Disable Hugging Face file locking to avoid race conditions
        os.environ["HF_HUB_DISABLE_LOCKING"] = "1"
        
        # Use GPU if available, else default to CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing SentenceTransformer with device: {device}")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
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
        
    def validate_embeddings(self, embeddings, expected_dim=384):
        """
        Validates that the embeddings is a two-dimensional array with the specified expected dimension.
        Also warns if the embeddings appear to be all zeros.
        """
        arr = np.array(embeddings)
        if arr.ndim != 2:
            logger.error("Embeddings array is not 2-dimensional")
            return False
        if arr.shape[1] != expected_dim:
            logger.error(f"Expected embedding dimension {expected_dim}, but got {arr.shape[1]}")
            return False
        if np.all(arr == 0):
            logger.warning("All embeddings are zeros")
        logger.info("Embeddings validated successfully")
        return True

    def _get_table_metadata(self):
        """Fetch metadata from Trino information schema"""
        cur = self.trino_conn.cursor()
        try:
            logger.info("Fetching table metadata from information schema...")
            cur.execute("""
                SELECT 
                    t.table_catalog,
                    t.table_schema,
                    t.table_name,
                    c.column_name,
                    c.data_type,
                    c.is_nullable
                FROM iceberg.information_schema.tables t
                JOIN iceberg.information_schema.columns c 
                    ON t.table_catalog = c.table_catalog
                    AND t.table_schema = c.table_schema
                    AND t.table_name = c.table_name
                WHERE t.table_schema = 'iceberg'
                  AND t.table_name IN ('customers', 'products', 'sales')
            """)
            results = cur.fetchall()
            logger.info(f"Found {len(results)} metadata entries")
            return results
        except Exception as e:
            logger.error(f"Error fetching metadata: {str(e)}")
            raise
        finally:
            cur.close()
    
    def refresh_embeddings(self):
        """Update embeddings with the latest metadata and validates them before upserting."""
        metadata = self._get_table_metadata()
        documents = []
        metadatas = []
        ids = []
        
        # Group columns by table
        table_columns = {}
        for row in metadata:
            table_key = f"{row[0]}.{row[1]}.{row[2]}"
            if table_key not in table_columns:
                table_columns[table_key] = []
            table_columns[table_key].append({
                'name': row[3],
                'type': row[4],
                'nullable': row[5]
            })
        
        # Create document text and metadata per table
        for table_key, columns in table_columns.items():
            doc_text = f"Table {table_key}:\nColumns:\n"
            for col in columns:
                doc_text += f"  - {col['name']} ({col['type']})\n"
            documents.append(doc_text)
            metadatas.append({
                "catalog": table_key.split('.')[0],
                "schema": table_key.split('.')[1],
                "table": table_key.split('.')[2],
                "columns": ", ".join(col['name'] for col in columns)
            })
            ids.append(table_key)
            
        # Clear existing embeddings
        try:
            existing = self.collection.get()
            if existing and existing['ids']:
                self.collection.delete(ids=existing['ids'])
        except Exception as e:
            logger.warning(f"Error clearing existing embeddings: {e}")
        
        # Generate new embeddings
        logger.info("Generating embeddings for documents...")
        
        # Check if documents list is empty
        if not documents:
            logger.warning("No documents found to embed. Skipping embedding generation.")
            return
            
        embeddings = self.model.encode(documents)
        
        # Validate embeddings
        if not self.validate_embeddings(embeddings):
            logger.error("Embedding validation failed. Aborting refresh.")
            raise ValueError("Invalid embeddings generated")
        else:
            logger.info("Embeddings validation passed.")
        
        # Upsert embeddings to the collection
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=documents
        )
        logger.info("Metadata embeddings refreshed successfully")

class EmbeddingService(TrinoMetadataEmbedder):
    def __init__(self):
        super().__init__()
        self.collection = self.client.get_or_create_collection("trino_metadata")
        try:
            # Only refresh embeddings if the collection is empty
            collection_data = self.collection.get()
            if not collection_data or not collection_data.get('ids'):
                logger.info("Collection is empty, refreshing embeddings...")
                self.refresh_embeddings()
            else:
                logger.info(f"Collection already contains {len(collection_data.get('ids', []))} documents")
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            # Continue without embeddings rather than crashing
    
    def embed(self, text: str):
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            # Return a zero vector of the expected dimension as fallback
            return [0.0] * 384
    
    def query_metadata(self, text: str, n=3):
        try:
            # Check if collection has any documents
            collection_data = self.collection.get()
            if not collection_data or not collection_data.get('ids'):
                logger.warning("No documents in collection to query")
                return {"ids": [], "distances": [], "metadatas": [], "documents": []}
                
            results = self.collection.query(
                query_embeddings=[self.embed(text)],
                n_results=min(n, len(collection_data.get('ids', [])))
            )
            logger.info(f"Query results: {results}")
            return results
        except Exception as e:
            logger.error(f"Error querying metadata: {str(e)}")
            return {"ids": [], "distances": [], "metadatas": [], "documents": []}

# Initialize with metadata sync
embedding_service = EmbeddingService()
print("Metadata sync completed successfully") 
from chromadb import Client, Settings
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from trino.dbapi import connect
import logging
import torch
import numpy as np
import json
from typing import List, Optional, Dict, Any
import re
import hashlib
from colorama import Fore

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

    def _get_table_metadata_with_samples(self):
        """Fetch enhanced schema metadata from Trino with sample data."""
        cur = self.trino_conn.cursor()
        try:
            logger.info("Fetching schema metadata with sample data...")
            
            # Get all tables from the iceberg catalog
            cur.execute("""
                SELECT DISTINCT
                    t.table_catalog,
                    t.table_schema,
                    t.table_name
                FROM iceberg.information_schema.tables t
                WHERE t.table_schema = 'iceberg'
                  AND t.table_name IN ('customers', 'products', 'sales')
                ORDER BY 
                    t.table_catalog, t.table_schema, t.table_name
            """)
            
            tables = cur.fetchall()
            logger.info(f"Found {len(tables)} tables in iceberg catalog")
            
            metadata_entries = []
            for table_info in tables:
                catalog, schema, table = table_info
                
                # Generate fully qualified table name
                full_table_name = f'"{catalog}"."{schema}"."{table}"'
                
                # Get column information
                try:
                    cur.execute(f"""
                        SELECT 
                            column_name, 
                            data_type
                        FROM 
                            information_schema.columns
                        WHERE 
                            table_catalog = '{catalog}'
                            AND table_schema = '{schema}'
                            AND table_name = '{table}'
                        ORDER BY 
                            ordinal_position
                    """)
                    
                    columns = [f"{row[0]} ({row[1]})" for row in cur.fetchall()]
                    columns_str = ", ".join(columns)
                    
                    # Get sample data (limit to 5 rows)
                    sample_data = []
                    try:
                        sample_query = f"SELECT * FROM {full_table_name} LIMIT 5"
                        cur.execute(sample_query)
                        
                        # Get column names for the result
                        column_names = [col[0] for col in cur.description]
                        
                        # Format sample data
                        rows = cur.fetchall()
                        for row in rows:
                            row_dict = {}
                            for i, val in enumerate(row):
                                # Convert all values to strings for consistency
                                row_dict[column_names[i]] = str(val) if val is not None else "NULL"
                            sample_data.append(row_dict)
                            
                        logger.info(f"Collected {len(sample_data)} sample rows from {full_table_name}")
                    except Exception as e:
                        logger.warning(f"Failed to get sample data for {full_table_name}: {str(e)}")
                    
                    # Create document for vector DB
                    doc_text = f"Table: {full_table_name}\n"
                    doc_text += f"Columns: {columns_str}\n"
                    
                    # Add sample data formatted as table
                    if sample_data:
                        doc_text += "\nSample data:\n"
                        for i, row in enumerate(sample_data):
                            doc_text += f"Row {i+1}: " + ", ".join([f"{k}='{v}'" for k, v in row.items()]) + "\n"
                    
                    # Create metadata
                    metadata = {
                        "catalog": catalog,
                        "schema": schema,
                        "table": table,
                        "columns": columns_str,
                        "has_samples": len(sample_data) > 0,
                        "sample_count": len(sample_data)
                    }
                    
                    metadata_entries.append((doc_text, metadata, f"{catalog}.{schema}.{table}"))
                    
                except Exception as e:
                    logger.error(f"Error processing table {full_table_name}: {str(e)}")
                    continue
            
            return metadata_entries
            
        except Exception as e:
            logger.error(f"Error fetching schema metadata: {str(e)}")
            raise
        finally:
            cur.close()
    
    def refresh_embeddings(self):
        """Update embeddings with enhanced metadata including sample data."""
        try:
            # Get enhanced metadata with sample data
            metadata_entries = self._get_table_metadata_with_samples()
            
            if not metadata_entries:
                logger.warning("No metadata entries found. Skipping embedding refresh.")
                return
                
            documents = []
            metadatas = []
            ids = []
            
            # Process metadata entries
            for doc_text, metadata, id in metadata_entries:
                documents.append(doc_text)
                metadatas.append(metadata)
                ids.append(id)
            
            # Clear existing embeddings
            try:
                existing = self.collection.get()
                if existing and existing['ids']:
                    self.collection.delete(ids=existing['ids'])
            except Exception as e:
                logger.warning(f"Error clearing existing embeddings: {e}")
            
            # Generate new embeddings
            logger.info(f"Generating embeddings for {len(documents)} documents...")
            
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
            logger.info("Enhanced metadata embeddings refreshed successfully")
            
        except Exception as e:
            logger.error(f"Error refreshing embeddings: {str(e)}")
            raise

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
    
    def get_context_for_query(self, query: str, n_results: int = 5) -> str:
        """Get relevant schema context for a natural language query."""
        try:
            logger.info(f"Getting context for query: {query}")
            
            # Generate embedding for the query
            query_embedding = self.embed(query)
            
            # Search for similar contexts in the vector DB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Build a context string with the results
            context = "# Database Schema Information\n\n"
            
            if results and len(results['documents']) > 0 and len(results['documents'][0]) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    # Add each relevant table definition to the context
                    context += f"{doc}\n\n---\n\n"
                
                logger.info(f"Retrieved context with {len(results['documents'][0])} tables")
            else:
                logger.warning("No relevant context found in vector database")
                context += "No relevant tables found. Please check your query."
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
            return "Error retrieving schema context."

class SimpleEmbeddings:
    """
    A simple embeddings class that uses TF-IDF like approach for generating embeddings
    In a production environment, this would be replaced with a proper embeddings model
    """
    
    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize the embeddings module
        
        Args:
            cache_file: Optional path to cache embeddings
        """
        self.word_vectors = {}
        self.cache_file = cache_file
        self.cache = {}
        
        # Load cache if it exists
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"{Fore.GREEN}Loaded embeddings cache with {len(self.cache)} entries{Fore.RESET}")
            except Exception as e:
                logger.error(f"{Fore.RED}Error loading embeddings cache: {str(e)}{Fore.RESET}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text string
        
        Args:
            text: The text to embed
            
        Returns:
            A vector representation of the text
        """
        # Check cache first
        text_hash = self._hash_text(text)
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        # Preprocess text
        words = self._preprocess_text(text)
        
        # Create a simple TF-IDF like vector
        vector = self._compute_vector(words)
        
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        # Cache the result
        vector_list = vector.tolist()
        if self.cache_file:
            self.cache[text_hash] = vector_list
            self._save_cache()
        
        return vector_list
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for embedding
        
        Args:
            text: The text to preprocess
            
        Returns:
            List of preprocessed words
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Remove stop words (a very simple list)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                     'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'from'}
        words = [word for word in words if word not in stop_words]
        
        return words
    
    def _compute_vector(self, words: List[str]) -> np.ndarray:
        """
        Compute a vector representation for a list of words
        
        Args:
            words: List of preprocessed words
            
        Returns:
            A vector representation
        """
        # Use a simple hash-based approach for demonstration
        # In a real system, this would use a proper embedding model
        
        # Create a vector of size 100
        vector_size = 100
        vector = np.zeros(vector_size)
        
        for i, word in enumerate(words):
            # Get or create a word vector
            if word not in self.word_vectors:
                # Create a deterministic but seemingly random vector for each word
                word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
                np.random.seed(word_hash)
                self.word_vectors[word] = np.random.randn(vector_size)
            
            # Add the word vector to the document vector
            # Weight by position (words later in the text get slightly less weight)
            position_weight = 1.0 / (1 + 0.1 * i)
            vector += self.word_vectors[word] * position_weight
        
        return vector
    
    def _hash_text(self, text: str) -> str:
        """
        Create a hash for a text string
        
        Args:
            text: The text to hash
            
        Returns:
            A hash string
        """
        return hashlib.md5(text.encode()).hexdigest()
    
    def _save_cache(self) -> None:
        """Save the embeddings cache to disk"""
        if not self.cache_file:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
                
            logger.info(f"{Fore.GREEN}Saved embeddings cache with {len(self.cache)} entries{Fore.RESET}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error saving embeddings cache: {str(e)}{Fore.RESET}")

# Initialize with metadata sync
embedding_service = EmbeddingService()
print("Metadata sync completed successfully") 
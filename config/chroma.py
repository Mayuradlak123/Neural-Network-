from dotenv import load_dotenv
import os
import chromadb
from config.logger import logger

# Load .env
load_dotenv()
CHROMA_API_KEY=os.getenv("CHROMA_API_KEY")
CHROMA_TENANT=os.getenv("CHROMA_TENANT")
CHROMA_DATABASE=os.getenv("CHROMA_DATABASE")
class ChromaDB:
    client: chromadb.CloudClient = None
    collection = None

db = ChromaDB()

def connect_to_chromadb(collection_name: str = None):
    """
    Connects to ChromaDB Cloud and gets or creates the collection.
    """
    try:
        logger.info("Attempting to connect to ChromaDB...", {"collection": collection_name})
        
        if not CHROMA_API_KEY:
            logger.error("CHROMA_DB_KEY not set in .env")
            raise ValueError("CHROMA_DB_KEY not set in .env")
        
        # Use collection name from .env if not passed
        if collection_name is None:
            collection_name = os.getenv("CHROMA_COLLECTION_NAME", "quickstart")
        
        logger.debug(f"Initializing ChromaDB client with collection: {collection_name}")
        
        # Connect to ChromaDB Cloud
        db.client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE
        )
        
        # Get or create collection
        logger.debug(f"Getting or creating collection '{collection_name}'...")
        db.collection = db.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity like Pinecone
        )
        
        logger.info(f"Successfully connected to ChromaDB collection '{collection_name}'")
        
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        raise
    finally:
        logger.debug("ChromaDB connection attempt completed")

def close_chromadb_connection():
    """
    Clears the ChromaDB client/collection.
    """
    try:
        logger.info("Closing ChromaDB connection...")
        if db.collection or db.client:
            db.collection = None
            db.client = None
            logger.info("ChromaDB connection closed successfully")
        else:
            logger.warning("No ChromaDB connection to close")
    except Exception as e:
        logger.error(f"Error closing ChromaDB connection: {e}")
        raise
    finally:
        logger.debug("ChromaDB connection closure operation completed")

def get_chromadb_collection():
    """
    Returns the active ChromaDB collection.
    """
    try:
        logger.debug("Getting ChromaDB collection...")
        if not db.collection:
            logger.error("ChromaDB collection not initialized")
            raise ValueError("ChromaDB collection not initialized. Call connect_to_chromadb() first.")
        
        logger.debug("ChromaDB collection retrieved successfully")
        return db.collection
        
    except Exception as e:
        logger.error(f"Error getting ChromaDB collection: {e}")
        raise
    finally:
        logger.debug("Get ChromaDB collection operation completed")

def insert_vector(id: str, vector: list, metadata: dict = None):
    """
    Insert a single vector with optional metadata into ChromaDB.
    """
    try:
        logger.debug(f"Inserting vector with ID: {id}")
        
        if not db.collection:
            logger.error("ChromaDB collection not initialized")
            raise ValueError("ChromaDB collection not initialized. Call connect_to_chromadb() first.")
        
        if not id or not vector:
            logger.error("Vector ID and values are required")
            raise ValueError("Vector ID and values are required")
        
        # ChromaDB requires documents (text) even if using embeddings
        # Use empty string if no document text available
        db.collection.add(
            ids=[id],
            embeddings=[vector],
            metadatas=[metadata] if metadata else None
        )
        
        logger.info(f"Successfully inserted vector ID '{id}' with metadata: {metadata}")
        
    except Exception as e:
        logger.error(f"Error inserting vector '{id}': {e}")
        raise
    finally:
        logger.debug(f"Insert vector '{id}' operation completed")

def query_vector(vector: list, top_k: int = 5, filter_metadata: dict = None):
    """
    Query ChromaDB for similar vectors and return a list of matches.
    """
    try:
        logger.debug(f"Querying ChromaDB with top_k={top_k}")
        
        if not db.collection:
            logger.error("ChromaDB collection not initialized")
            raise ValueError("ChromaDB collection not initialized. Call connect_to_chromadb() first.")
        
        results = db.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filter_metadata if filter_metadata else None
        )
        
        # Transform ChromaDB results to Pinecone-like format
        matches = []
        if results and results['ids'] and len(results['ids']) > 0:
            for i, id in enumerate(results['ids'][0]):
                match = {
                    'id': id,
                    'score': 1 - results['distances'][0][i] if results['distances'] else None,  # Convert distance to similarity
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                }
                matches.append(match)
        
        logger.debug(f"Query returned {len(matches)} matches")
        return matches
        
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        raise
    finally:
        logger.debug("Query vector operation completed")

def batch_insert_vectors(ids: list, vectors: list, metadatas: list = None):
    """
    Insert multiple vectors with optional metadata into ChromaDB.
    """
    try:
        logger.debug(f"Batch inserting {len(ids)} vectors")
        
        if not db.collection:
            logger.error("ChromaDB collection not initialized")
            raise ValueError("ChromaDB collection not initialized. Call connect_to_chromadb() first.")
        
        if not ids or not vectors or len(ids) != len(vectors):
            logger.error("IDs and vectors must be non-empty and of equal length")
            raise ValueError("IDs and vectors must be non-empty and of equal length")
        
        db.collection.add(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas if metadatas else None
        )
        
        logger.info(f"Successfully batch inserted {len(ids)} vectors")
        
    except Exception as e:
        logger.error(f"Error batch inserting vectors: {e}")
        raise
    finally:
        logger.debug("Batch insert operation completed")
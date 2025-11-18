from pymongo import MongoClient
from typing import Dict, List, Any, Optional
from config.logger import logger
import json
from datetime import datetime


class MongoDBAgentService:
    """
    Service to connect to user's MongoDB and analyze collections
    """
    
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db = None
        self.collections_schema = {}
    
    def connect(self, connection_url: str, database_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Connect to MongoDB instance and extract metadata
        """
        try:
            logger.info("Attempting to connect to MongoDB...")
            
            # Connect to MongoDB
            self.client = MongoClient(connection_url, serverSelectionTimeoutMS=5000)
            
            # Test connection
            self.client.admin.command('ping')
            
            # Get database
            if database_name:
                self.db = self.client[database_name]
            else:
                # Try to extract database from URL or use first available
                db_list = self.client.list_database_names()
                # Filter out system databases
                db_list = [db for db in db_list if db not in ['admin', 'local', 'config']]
                if db_list:
                    self.db = self.client[db_list[0]]
                else:
                    raise ValueError("No databases found")
            
            logger.info(f"Successfully connected to database: {self.db.name}")
            
            return {
                "success": True,
                "database": self.db.name,
                "available_databases": self.client.list_database_names()
            }
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise ValueError(f"Connection failed: {str(e)}")
    
    def extract_schema(self) -> Dict[str, Any]:
        """
        Extract all collections, sample documents, and inferred schema
        """
        try:
            if self.db is None:
                raise ValueError("Not connected to database")
            
            logger.info("Extracting collections schema...")
            
            collections = self.db.list_collection_names()
            schema_data = {}
            
            for collection_name in collections:
                logger.debug(f"Analyzing collection: {collection_name}")
                
                collection = self.db[collection_name]
                
                # Get collection stats
                stats = self.db.command("collStats", collection_name)
                doc_count = stats.get("count", 0)
                
                # Get sample documents (up to 5)
                samples = list(collection.find().limit(5))
                
                # Serialize ObjectId and datetime for JSON
                samples = self._serialize_documents(samples)
                
                # Infer schema from samples
                schema = self._infer_schema(samples) if samples else {}
                
                schema_data[collection_name] = {
                    "document_count": doc_count,
                    "sample_documents": samples,
                    "schema": schema,
                    "indexes": list(collection.list_indexes())
                }
            
            self.collections_schema = schema_data
            logger.info(f"Extracted schema for {len(collections)} collections")
            
            return {
                "success": True,
                "database": self.db.name,
                "collections": schema_data,
                "total_collections": len(collections)
            }
            
        except Exception as e:
            logger.error(f"Failed to extract schema: {e}")
            raise
    
    def _serialize_documents(self, documents: List[Dict]) -> List[Dict]:
        """Convert MongoDB documents to JSON-serializable format"""
        serialized = []
        for doc in documents:
            serialized_doc = {}
            for key, value in doc.items():
                # ObjectId / datetime etc.
                if str(type(value)) in ("<class 'bson.objectid.ObjectId'>", "<class 'datetime.datetime'>"):
                    serialized_doc[key] = str(value)
                else:
                    serialized_doc[key] = value
            serialized.append(serialized_doc)
        return serialized
    
    def _infer_schema(self, samples: List[Dict]) -> Dict[str, str]:
        """
        Infer schema structure from sample documents
        """
        schema = {}
        
        for doc in samples:
            for field, value in doc.items():
                field_type = type(value).__name__
                if field not in schema:
                    schema[field] = field_type
                elif schema[field] != field_type:
                    schema[field] = "mixed"
        
        return schema
    
    def query_collection(
        self, 
        collection_name: str, 
        query: Dict = None, 
        limit: int = 100,
        projection: Dict = None,
        sort: List = None
    ) -> List[Dict]:
        """
        Execute a query on a collection
        """
        try:
            if self.db is None:
                raise ValueError("Not connected to database")
            
            collection = self.db[collection_name]
            
            cursor = collection.find(query or {}, projection or {})
            
            if sort:
                cursor = cursor.sort(sort)
            
            cursor = cursor.limit(limit)
            
            results = list(cursor)
            results = self._serialize_documents(results)
            
            logger.info(f"Query returned {len(results)} documents from {collection_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def aggregate_collection(
        self, 
        collection_name: str, 
        pipeline: List[Dict]
    ) -> List[Dict]:
        """
        Execute an aggregation pipeline
        """
        try:
            if self.db is None:
                raise ValueError("Not connected to database")
            
            collection = self.db[collection_name]
            results = list(collection.aggregate(pipeline))
            results = self._serialize_documents(results)
            
            logger.info(f"Aggregation returned {len(results)} results from {collection_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            raise
    
    def get_collections_info(self) -> Dict[str, Any]:
        """Get stored collections schema"""
        return self.collections_schema
    
    def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collections_schema = {}
            logger.info("Disconnected from MongoDB")


# Session storage for multiple users (Temp — use Redis in production)
_sessions = {}


def get_session(session_id: str) -> MongoDBAgentService:
    """Get or create a MongoDB agent session"""
    if session_id not in _sessions:
        _sessions[session_id] = MongoDBAgentService()
    return _sessions[session_id]


def cleanup_session(session_id: str):
    """Clean up a session"""
    if session_id in _sessions:
        _sessions[session_id].disconnect()
        del _sessions[session_id]

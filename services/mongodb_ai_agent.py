from services.groq_service import GroqService
from services.mongodb_agent import MongoDBAgentService

from config.logger import logger
from typing import Dict, Any, List
import json
import re


class MongoDBChatAgent:
    """
    AI Agent that understands natural language queries and executes MongoDB operations
    """
    
    def __init__(self, mongo_service: MongoDBAgentService):
        self.mongo_service = mongo_service
        self.groq_service = GroqService()
        self.conversation_history = []
    
    def chat(self, user_query: str) -> Dict[str, Any]:
        """
        Process user's natural language query and return results
        """
        try:
            logger.info(f"Processing chat query: {user_query}")
            
            # Get collections schema
            collections_info = self.mongo_service.get_collections_info()
            
            # FIXED ❌ if not collections_info → ✔ if collections_info is None
            if collections_info is None:
                return {
                    "success": False,
                    "error": "No collections schema available. Please connect to database first."
                }
            
            # Step 1: Identify relevant collections and generate MongoDB query
            query_plan = self._generate_query_plan(user_query, collections_info)
            
            # Step 2: Execute the query
            query_results = self._execute_query_plan(query_plan)
            
            # Step 3: Generate human-friendly answer
            final_answer = self._generate_answer(user_query, query_plan, query_results, collections_info)
            
            # Store in conversation history
            self.conversation_history.append({
                "user_query": user_query,
                "query_plan": query_plan,
                "results_count": len(query_results) if isinstance(query_results, list) else 1,
                "answer": final_answer
            })
            
            return {
                "success": True,
                "query": user_query,
                "answer": final_answer,
                "data": query_results,
                "query_plan": query_plan
            }
            
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": user_query
            }
    
    def _generate_query_plan(self, user_query: str, collections_info: Dict) -> Dict[str, Any]:
        """
        Use AI to understand the query and generate a MongoDB query plan
        """
        try:
            schema_summary = self._format_schema_for_prompt(collections_info)
            
            prompt = f"""You are a MongoDB query expert. Analyze the user's question and generate a query plan.

Available Collections and Schema:
{schema_summary}

User Question: {user_query}

Return ONLY JSON:
{{
    "collection": "",
    "operation": "",
    "query": {{}},
    "pipeline": [],
    "reasoning": ""
}}
"""

            response = self.groq_service.process_prompt(prompt).strip()

            # Clean JSON blocks
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "").strip()
            elif response.startswith("```"):
                response = response.replace("```", "").strip()
            
            query_plan = json.loads(response)
            
            logger.info(f"Generated query plan: {query_plan.get('reasoning', 'N/A')}")
            
            return query_plan
            
        except Exception as e:
            logger.error(f"Failed to generate query plan: {e}")
            raise ValueError(f"Could not understand the query: {str(e)}")
    
    def _execute_query_plan(self, query_plan: Dict) -> Any:
        """
        Execute the generated MongoDB query plan
        """
        try:
            collection = query_plan.get("collection")
            operation = query_plan.get("operation", "find")
            
            # FIX: check collection is not None
            if collection is None:
                raise ValueError("Query plan missing target collection")
            
            if operation == "find":
                query = query_plan.get("query", {})
                results = self.mongo_service.query_collection(collection, query)

            elif operation == "aggregate":
                pipeline = query_plan.get("pipeline", [])
                results = self.mongo_service.aggregate_collection(collection, pipeline)

            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            return results
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def _generate_answer(
        self, 
        user_query: str, 
        query_plan: Dict, 
        results: Any,
        collections_info: Dict
    ) -> str:
        """
        Use AI to generate a human-friendly answer from the query results
        """
        try:
            results_summary = self._summarize_results(results)
            
            prompt = f"""You are a helpful data analyst. Answer the user's question.

User Question: {user_query}

Query:
- Collection: {query_plan.get('collection')}
- Operation: {query_plan.get('operation')}
- Reasoning: {query_plan.get('reasoning')}

Results:
{results_summary}

Provide a short, clear answer."""

            answer = self.groq_service.process_prompt(prompt)
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Found {len(results) if isinstance(results, list) else 0} results, but couldn't generate summary."
    
    def _format_schema_for_prompt(self, collections_info: Dict) -> str:
        """Format collections schema for AI prompt"""

        # FIX: truth check
        if collections_info is None:
            return "No schema available."

        schema_lines = []
        
        for collection_name, info in collections_info.items():
            schema_lines.append(f"\nCollection: {collection_name}")
            schema_lines.append(f"  Documents: {info.get('document_count', 0)}")
            schema_lines.append(f"  Fields: {', '.join(info.get('schema', {}).keys())}")
            
            if info.get('sample_documents'):
                sample = info['sample_documents'][0]
                schema_lines.append(f"  Sample fields: {list(sample.keys())}")
        
        return "\n".join(schema_lines)
    
    def _summarize_results(self, results: Any) -> str:
        """AI-friendly result summarization"""

        # FIX: if not results → use None check
        if results is None or results == []:
            return "No results found."
        
        if isinstance(results, list):
            summary = f"Found {len(results)} documents.\n"
            
            if len(results) <= 5:
                summary += f"Results:\n{json.dumps(results, indent=2)}"
            else:
                summary += f"First 5 results:\n{json.dumps(results[:5], indent=2)}"
                summary += f"\n... and {len(results) - 5} more"
            
            return summary
        
        return f"Result:\n{json.dumps(results, indent=2)}"
    
    def get_conversation_history(self) -> List[Dict]:
        return self.conversation_history
    
    def clear_history(self):
        self.conversation_history = []

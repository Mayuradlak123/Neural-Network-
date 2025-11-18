from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from services.mongodb_agent import get_session, cleanup_session
from services.mongodb_ai_agent import MongoDBChatAgent
from config.logger import logger
import uuid

mongodb_router = APIRouter(tags=["mongodb-agent"])


class ConnectRequest(BaseModel):
    connection_url: str
    database_name: str = None


class ChatRequest(BaseModel):
    session_id: str
    query: str


@mongodb_router.post("/connect")
async def connect_to_database(request: ConnectRequest):
    """
    Connect to user's MongoDB instance and extract schema
    """
    try:
        session_id = str(uuid.uuid4())
        logger.info(f"New connection request - Session: {session_id}")

        mongo_service = get_session(session_id)

        # Connect
        connection_result = mongo_service.connect(
            request.connection_url,
            request.database_name
        )

        # Extract schema
        schema_result = mongo_service.extract_schema()

        return {
            "success": True,
            "session_id": session_id,
            "database": schema_result["database"],
            "collections": list(schema_result["collections"].keys()),
            "total_collections": schema_result["total_collections"],
            "schema": schema_result["collections"]
        }

    except Exception as e:
        logger.error(f"Connection failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@mongodb_router.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """
    Chat with AI agent about your data
    """
    try:
        logger.info(f"Chat request - Session: {request.session_id}")

        mongo_service = get_session(request.session_id)

        # FIXED HERE ✔
        if mongo_service.db is None:
            raise HTTPException(
                status_code=400,
                detail="Session not found or database not connected"
            )

        chat_agent = MongoDBChatAgent(mongo_service)

        result = chat_agent.chat(request.query)

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mongodb_router.get("/schema/{session_id}")
async def get_schema(session_id: str):
    """
    Get current database schema
    """
    try:
        mongo_service = get_session(session_id)

        # FIXED HERE ✔
        if mongo_service.db is None:
            raise HTTPException(status_code=400, detail="Session not found")

        collections_info = mongo_service.get_collections_info()

        return {
            "success": True,
            "database": mongo_service.db.name,
            "collections": collections_info
        }

    except Exception as e:
        logger.error(f"Failed to get schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mongodb_router.delete("/disconnect/{session_id}")
async def disconnect_session(session_id: str):
    """
    Disconnect and cleanup session
    """
    try:
        cleanup_session(session_id)
        return {
            "success": True,
            "message": "Session disconnected successfully"
        }
    except Exception as e:
        logger.error(f"Failed to disconnect: {e}")
        raise HTTPException(status_code=500, detail=str(e))

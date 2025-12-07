from pathlib import Path
from fastapi import FastAPI, UploadFile, File, APIRouter, Request, HTTPException as FastAPIHTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import anyio
from config.database import connect_to_mongo, close_mongo_connection
from config.chroma import connect_to_chromadb, close_chromadb_connection
from config.groq import process_prompt, setup_groq
from routers.clustering_router import clustering_router
from routers.mongodb_agent_router import mongodb_router
from routers.mongodb_web_router import mongodb_web_router
from routers.etl_router import etl_router,ml_router
from routers.etl_web_router import etl_web_router
from routers.torch_router import torch_router
from routers.ann_regression_router import reg_router
from routers.model_interpretability_router import model_interpretability_router
# Import your routers
from routers.web_route import web_router
from routers.email_router import email_router
from routers.ann_router import ann_router
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run sync functions in safe thread
    await anyio.to_thread.run_sync(setup_groq)
    await anyio.to_thread.run_sync(connect_to_mongo)
    await anyio.to_thread.run_sync(connect_to_chromadb)

    yield

    await anyio.to_thread.run_sync(close_chromadb_connection)
    await anyio.to_thread.run_sync(close_mongo_connection)

app = FastAPI(
    title="Cold Email Generator API",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for CSS, JS, images if needed)
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(mongodb_web_router)
app.include_router(mongodb_router, prefix="/api/mongodb")
# Include routers
app.include_router(web_router)  # Web pages
app.include_router(email_router , prefix="/api")  # API endpoints

app.include_router(torch_router,prefix="/api/torch")
app.include_router(ann_router,prefix="/api/ann")
app.include_router(etl_web_router)
app.include_router(etl_router, prefix="/api/etl")
app.include_router(ml_router, prefix="/api/ml")
app.include_router(clustering_router,prefix="/api/clustering")
app.include_router(model_interpretability_router,prefix="/api/interpret")

# ANN Regression Router 
app.include_router(reg_router, prefix="/api/rag")
# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": 200,
        "success": True,
        "message": "Application is running"
    }



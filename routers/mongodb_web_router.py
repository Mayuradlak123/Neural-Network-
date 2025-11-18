from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Create router
mongodb_web_router = APIRouter(tags=["mongodb-web"])

# Setup Jinja2 templates
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@mongodb_web_router.get("/mongodb-agent", response_class=HTMLResponse)
async def mongodb_agent_page(request: Request):
    """
    Render the MongoDB AI Agent page
    """
    return templates.TemplateResponse("mongodb_agent.html", {"request": request})
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Create router
web_router = APIRouter(tags=["web"])

# Setup Jinja2 templates
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@web_router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the Cold Email Generator home page
    """
    return templates.TemplateResponse("index.html", {"request": request})
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Create router
etl_web_router = APIRouter(tags=["etl-web"])

# Setup Jinja2 templates
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@etl_web_router.get("/etl", response_class=HTMLResponse)
async def etl_page(request: Request):
    """
    Render the ETL & Feature Engineering page
    """
    return templates.TemplateResponse("etl.html", {"request": request})
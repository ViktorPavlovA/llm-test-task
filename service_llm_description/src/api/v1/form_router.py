from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

form_router = APIRouter()


@form_router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request) -> HTMLResponse:
    """
    Render the datasheet form used by catalog editors.

    Parameters
    ----------
    request : Request
        Incoming Starlette request required by Jinja2.

    Returns
    -------
    page : HTMLResponse
        ``index.html`` with the input form and result panel.

    """
    return templates.TemplateResponse(request, "index.html")

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.v1 import (description_router, 
                        form_router, 
                        health_router)
from src.repository.llm_client import llm_health
from src.utils import (_pydantic_errors2json, 
                       get_settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Log the LLM endpoint and probe health on startup.

    Parameters
    ----------
    app : FastAPI
        Application instance created by uvicorn.

    """
    settings = get_settings()
    logger.info("LLM endpoint: {}  model={}", settings.model_api, settings.llm_model)
    probe = await llm_health()
    if probe.get("ok"):
        logger.info("LLM health: ok")
    else:
        logger.warning("LLM is not ready yet: {}", probe)
    yield


app = FastAPI(
    title="AI product-card assistant",
    version="1.0.0",
    lifespan=lifespan,
    summary="SEO product-card draft for measuring instruments",
    description=(
        "Accepts instrument JSON. LangGraph calls the local LLM: when specs "
        "are missing, a research agent searches the web, then the writer "
        "builds the SEO card. Technical values are not invented. "
        "key_specs lists confirmed product-specific characteristics."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(form_router)
app.include_router(health_router, tags=["Health"])
app.include_router(description_router, prefix="/api", tags=["Description"])


@app.exception_handler(RequestValidationError)
async def validation_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Return a readable 422 body for malformed instrument JSON.

    Parameters
    ----------
    request : Request
        Failed request.
    exc : RequestValidationError
        Pydantic validation error.

    Returns
    -------
    response : JSONResponse
        Status 422 with ``detail`` and ``errors``.

    """
    errors = _pydantic_errors2json(exc.errors())
    logger.warning("Invalid input: {} {}", request.url.path, errors)
    return JSONResponse(
        status_code=422,
        content={
            "detail": (
                "Invalid JSON or empty required fields. Need name; "
                "manufacturer, specs, and description are optional."
            ),
            "errors": errors,
        },
    )

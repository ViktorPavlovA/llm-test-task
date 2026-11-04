from fastapi import APIRouter

from src.repository.llm_client import llm_health
from src.utils import get_settings

health_router = APIRouter(tags=["Health"])


@health_router.get("/health")
async def health() -> dict:
    """
    Report process liveness and LLM reachability.

    Returns
    -------
    payload : dict
        ``status``, LLM probe, search flag, and ``SEARCH_CONTEXT_CHARS``.

    """
    llm = await llm_health()
    settings = get_settings()
    return {
        "status": "ok",
        "llm": {
            "ok": bool(llm.get("ok")),
            "url": llm.get("url"),
            "detail": llm.get("body") or llm.get("error"),
        },
        "llm_search_internet": settings.llm_search_internet,
        "search_context_chars": settings.search_context_chars,
    }

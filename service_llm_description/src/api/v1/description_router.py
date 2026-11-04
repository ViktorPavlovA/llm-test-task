from typing import Literal

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import PlainTextResponse

from src.domain.exceptions import (LLMBadResponseError, 
                                       LLMUnavailableError)
from src.schemas import DeviceStatsModel
from src.service.get_description_service import get_description_service

router_description = APIRouter()


@router_description.post(
    "/description",
    status_code=status.HTTP_200_OK,
    summary="Build a product-card draft",
    response_model=None,
)
async def get_description(
    device_stats: DeviceStatsModel,
    response_format: Literal["json", "markdown"] = Query(
        default="json",
        alias="format",
        description="json for the card object, markdown for copy-paste text",
    ),
):
    """
    Assemble a B2B product card from instrument JSON.

    Parameters
    ----------
    device_stats : DeviceStatsModel
        Name, manufacturer, specs, and optional description.
    response_format : {"json", "markdown"}
        Response body shape. Query alias is ``format``.

    Returns
    -------
    card : CardResponse or PlainTextResponse
        JSON card or Markdown text.

    Raises
    ------
    HTTPException
        503 if the LLM is down, 502 if the model output is unusable.

    """
    try:
        card = await get_description_service(device_stats)
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except LLMBadResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    if response_format == "markdown":
        return PlainTextResponse(
            card.markdown,
            media_type="text/markdown; charset=utf-8",
        )
    return card

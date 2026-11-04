from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx
from loguru import logger

from src.domain.card_format import extract_json_object
from src.domain.exceptions import LLMBadResponseError, LLMUnavailableError
from src.utils import Settings, get_settings


def _trust_env_for(url: str) -> bool:
    """
    Decide whether httpx should honour HTTP(S)_PROXY for this URL.

    Loopback, Docker DNS names without a dot, and ``host.docker.internal``
    skip the host SOCKS proxy. Public hosts keep proxy support.

    Parameters
    ----------
    url : str
        Target base URL.

    Returns
    -------
    trust_env : bool
        True when proxy environment variables should be used.

    """
    host = (urlparse(url).hostname or "").strip().strip("[]").lower()
    if not host:
        return False
    if host in {"127.0.0.1", "localhost", "::1", "host.docker.internal"}:
        return False
    if "." not in host:
        return False
    return True


def _client(timeout: float, settings: Settings | None = None) -> httpx.AsyncClient:
    """
    Build an async HTTP client for the configured LLM base URL.

    Parameters
    ----------
    timeout : float
        Request timeout in seconds.
    settings : Settings or None
        App settings; loaded from cache when omitted.

    Returns
    -------
    client : httpx.AsyncClient
        Client with ``trust_env`` set for the LLM host.

    """
    settings = settings or get_settings()
    return httpx.AsyncClient(
        timeout=timeout,
        trust_env=_trust_env_for(settings.model_api),
    )


async def llm_health() -> dict[str, Any]:
    """
    Probe the llama.cpp ``/health`` endpoint.

    Returns
    -------
    probe : dict
        ``ok``, URL, and either the JSON body or an error string.

    """
    settings = get_settings()
    try:
        async with _client(5.0, settings) as client:
            response = await client.get(settings.health_url)
        body: Any
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text[:200]}
        return {
            "ok": response.status_code == 200,
            "status_code": response.status_code,
            "body": body,
            "url": settings.health_url,
        }
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "url": settings.health_url,
        }


def _message_text(message: Any) -> str:
    """
    Collect assistant text from ``content`` or ``reasoning_content``.

    Parameters
    ----------
    message : Any
        ``choices[0].message`` from the chat completions body.

    Returns
    -------
    text : str
        Concatenated text parts.

    """
    if not isinstance(message, dict):
        return ""
    chunks: list[str] = []
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    chunks.append(str(text))
            elif part:
                chunks.append(str(part))
    elif content:
        chunks.append(str(content))
    if not any(item.strip() for item in chunks):
        extra = message.get("reasoning_content")
        if extra:
            chunks.append(str(extra))
    return "".join(chunks)


async def complete_json(messages: list[dict[str, str]]) -> dict[str, Any]:
    """
    Ask the LLM for a JSON object and parse the content.

    Parameters
    ----------
    messages : list of dict
        Chat messages with ``role`` and ``content``.

    Returns
    -------
    payload : dict
        Parsed JSON object from the model.

    Raises
    ------
    LLMUnavailableError
        Network, timeout, or 5xx from the server.
    LLMBadResponseError
        Empty or non-JSON content after retries.

    """
    settings = get_settings()
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_error: Exception | None = None
    content = ""
    for attempt in range(settings.llm_retries + 1):
        try:
            content = await _post_chat(payload)
            return extract_json_object(content)
        except LLMUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "LLM JSON parse failed (attempt {}): {} | preview={}",
                attempt + 1,
                exc,
                repr(content)[:300],
            )
            if "response_format" in payload:
                payload = dict(payload)
                payload.pop("response_format", None)
    raise LLMBadResponseError(f"Model did not return valid JSON: {last_error}")


async def _post_chat(payload: dict[str, Any]) -> str:
    """
    POST ``/chat/completions`` and return the assistant content.

    Parameters
    ----------
    payload : dict
        OpenAI-compatible request body.

    Returns
    -------
    content : str
        ``choices[0].message.content``.

    Raises
    ------
    LLMUnavailableError
        Connection, timeout, or 5xx.
    LLMBadResponseError
        4xx, missing content, or a non-JSON envelope.

    """
    settings = get_settings()
    headers = {
        "Authorization": f"Bearer {settings.model_api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with _client(settings.llm_timeout_seconds, settings) as client:
            response = await client.post(
                settings.chat_url,
                headers=headers,
                json=payload,
            )
    except httpx.ConnectError as exc:
        raise LLMUnavailableError(
            "Local LLM is unavailable. Start the stack with "
            "`docker compose up` at the repo root or in service_llm_local."
        ) from exc
    except httpx.TimeoutException as exc:
        raise LLMUnavailableError(
            "Local LLM did not answer within the timeout."
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMUnavailableError(f"Network error talking to the LLM: {exc}") from exc

    if response.status_code >= 500:
        raise LLMUnavailableError(
            f"LLM returned {response.status_code}: {response.text[:400]}"
        )
    if response.status_code >= 400:
        raise LLMBadResponseError(
            f"LLM rejected the request ({response.status_code}): {response.text[:400]}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise LLMBadResponseError("LLM returned non-JSON") from exc

    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMBadResponseError(
            f"Unexpected LLM response shape: {repr(body)[:400]}"
        ) from exc

    content = _message_text(message)
    if not content.strip():
        raise LLMBadResponseError("LLM returned empty content")
    return content

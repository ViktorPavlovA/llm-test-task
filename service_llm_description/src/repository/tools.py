from __future__ import annotations

import ipaddress
import socket
import time
from html.parser import HTMLParser
from io import BytesIO
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from langchain_core.tools import tool
from loguru import logger
from pypdf import PdfReader

from src.utils import get_settings

_MAX_RAW_BYTES = 180_000
_MAX_PDF_BYTES = 2_500_000
_PDF_PAGES = 4
_SEARCH_ATTEMPTS = 3
_BLOCKED_HOSTS = {
    "localhost",
    "llm",
    "host.docker.internal",
    "metadata.google.internal",
    "metadata.internal",
}
_USER_AGENT = "Mozilla/5.0 (compatible; KIPCardResearch/1.0; +http://127.0.0.1:8000)"


class PublicURLError(ValueError):
    """
    Raised when a URL is not safe to fetch (scheme, private host, SSRF).

    """


class _HTMLText(HTMLParser):
    """
    Collect visible text from HTML, skipping script, style, and svg.

    """

    def __init__(self) -> None:
        """
        Initialize skip counters and the text buffer.

        """
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """
        Enter a skipped region for script, style, noscript, or svg.

        Parameters
        ----------
        tag : str
            HTML tag name.
        attrs : list of tuple
            Tag attributes, unused.

        """
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        """
        Leave a skipped region.

        Parameters
        ----------
        tag : str
            HTML tag name.

        """
        if tag in {"script", "style", "noscript", "svg"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        """
        Append collapsed whitespace text outside skipped regions.

        Parameters
        ----------
        data : str
            Raw character data.

        """
        if self._skip:
            return
        text = " ".join(data.split())
        if text:
            self.parts.append(text)


def _ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """
    Return True for private, loopback, link-local, or reserved addresses.

    Parameters
    ----------
    ip : ipaddress.IPv4Address or ipaddress.IPv6Address
        Resolved address. IPv4-mapped IPv6 is unwrapped.

    Returns
    -------
    blocked : bool
        True when the address must not be fetched.

    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or getattr(ip, "is_site_local", False)
    )


def assert_public_url(url: str) -> str:
    """
    Accept only public http(s) URLs; reject SSRF targets.

    Parameters
    ----------
    url : str
        Candidate page URL.

    Returns
    -------
    cleaned : str
        Stripped URL that passed the checks.

    Raises
    ------
    PublicURLError
        Wrong scheme, blocked host, private IP, or DNS failure.

    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise PublicURLError("http or https is required")
    host = (parsed.hostname or "").strip().strip("[]").lower()
    if not host:
        raise PublicURLError("URL has no host")
    if host in _BLOCKED_HOSTS or host.endswith(".internal") or host.endswith(".local"):
        raise PublicURLError(f"host {host} is blocked")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if _ip_blocked(ip):
            raise PublicURLError(f"address {host} must not be fetched")
        return url.strip()
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise PublicURLError(f"host {host} does not resolve") from exc
    if not infos:
        raise PublicURLError(f"host {host} does not resolve")
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if _ip_blocked(address):
            raise PublicURLError(f"host {host} resolves to an internal address")
    return url.strip()


def pdf_bytes_to_text(raw: bytes, *, max_pages: int = _PDF_PAGES) -> str:
    """
    Extract visible text from the first pages of a PDF.

    Parameters
    ----------
    raw : bytes
        PDF file body, already clipped.
    max_pages : int
        Number of leading pages to read.

    Returns
    -------
    text : str
        Collapsed page text, or empty when the file has no extractable text.

    """
    if not raw:
        return ""
    try:
        reader = PdfReader(BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        logger.info("pdf parse failed: {}", exc)
        return ""
    parts: list[str] = []
    for page in reader.pages[: max(1, max_pages)]:
        try:
            extracted = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            continue
        text = " ".join(extracted.split())
        if text:
            parts.append(text)
    return "\n".join(parts)


def looks_like_pdf(url: str, content_type: str, raw: bytes) -> bool:
    """
    Return True when the response is a PDF datasheet.

    Parameters
    ----------
    url : str
        Fetched URL.
    content_type : str
        HTTP Content-Type header.
    raw : bytes
        Response body prefix.

    Returns
    -------
    is_pdf : bool
        True for ``.pdf`` URLs, PDF content types, or ``%PDF-`` magic.

    """
    path = url.lower().split("?", 1)[0]
    if path.endswith(".pdf"):
        return True
    lowered = (content_type or "").lower()
    if "pdf" in lowered:
        return True
    return raw[:5] == b"%PDF-"


def html_to_text(raw: str) -> str:
    """
    Strip tags and return visible text.

    Parameters
    ----------
    raw : str
        HTML document.

    Returns
    -------
    text : str
        Collapsed visible text. Broken HTML is whitespace-normalized.

    """
    parser = _HTMLText()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:  # noqa: BLE001
        return " ".join(raw.split())
    return " ".join(parser.parts)


def _search_impl(query: str) -> list[dict[str, Any]]:
    """
    Run DuckDuckGo text search with the configured timeout and region.

    Parameters
    ----------
    query : str
        Search string.

    Returns
    -------
    rows : list of dict
        Hits with ``title``, ``href``, and ``body``.

    """
    from ddgs import DDGS

    settings = get_settings()
    cleaned = " ".join((query or "").split())
    if not cleaned:
        return []
    timeout = max(3, int(settings.search_timeout_seconds))
    last_error: Exception | None = None
    for attempt in range(_SEARCH_ATTEMPTS):
        try:
            with DDGS(timeout=timeout) as client:
                rows = client.text(
                    cleaned,
                    region="ru-ru",
                    max_results=settings.search_max_results,
                )
            return list(rows or [])
        except Exception as exc:  # noqa: BLE001
            message = str(exc).casefold()
            if "no results" in message:
                return []
            last_error = exc
            logger.warning(
                "search attempt {}/{} failed ({}): {}",
                attempt + 1,
                _SEARCH_ATTEMPTS,
                query,
                exc,
            )
            time.sleep(0.7 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return []


@tool
def search_web(query: str) -> str:
    """
    Search the web for a measuring-instrument datasheet and specs.

    Call this when input data is missing fields such as power, IP rating,
    interfaces, or manufacturer. Phrase ``query`` in Russian as
    ``"<name> <manufacturer> паспорт характеристики"``. Do not invent
    values; return search snippets only.

    Parameters
    ----------
    query : str
        Search string, for example ``манометр DM 02 Манотомь паспорт``.

    Returns
    -------
    results : str
        Numbered title / url / snippet list, or a recovery hint on error.

    """
    cleaned = " ".join((query or "").split())
    if not cleaned:
        return (
            "Ошибка: пустой query. Передай строку вида "
            "«манометр DM 02 Манотомь паспорт»."
        )
    try:
        rows = _search_impl(cleaned)
    except Exception as exc:  # noqa: BLE001
        logger.warning("search_web failed: {}", exc)
        return (
            f"Поиск не удался: {exc}. "
            "Сформулируй более короткий запрос на русском и вызови "
            "search_web снова."
        )
    if not rows:
        return "Ничего не найдено. Сократи запрос до названия модели и слова «паспорт»."
    snippet_chars = get_settings().search_snippet_chars
    lines = [f"Результаты по запросу «{cleaned}»:"]
    for index, row in enumerate(rows, start=1):
        title = str(row.get("title") or "").strip() or "без названия"
        url = str(row.get("href") or row.get("url") or "").strip()
        snippet = str(row.get("body") or row.get("snippet") or "").strip()
        lines.append(f"{index}. {title}")
        if url:
            lines.append(f"   url: {url}")
        if snippet:
            lines.append(f"   snippet: {snippet[:snippet_chars]}")
    return "\n".join(lines)


def _fetch_once(url: str) -> httpx.Response:
    """
    GET a URL without following redirects.

    Parameters
    ----------
    url : str
        Public http(s) URL already checked by ``assert_public_url``.

    Returns
    -------
    response : httpx.Response
        Raw HTTP response.

    """
    settings = get_settings()
    timeout = httpx.Timeout(settings.search_timeout_seconds)
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.5"
        ),
    }
    with httpx.Client(
        timeout=timeout,
        follow_redirects=False,
        trust_env=True,
        headers=headers,
    ) as client:
        return client.get(url)


def _fetch_page_impl(url: str) -> str:
    """
    Follow a short redirect chain and extract HTML or PDF text.

    Parameters
    ----------
    url : str
        Page URL from ``search_web``.

    Returns
    -------
    text : str
        Title plus visible text clipped to ``SEARCH_CONTEXT_CHARS``, or a hint.

    Raises
    ------
    PublicURLError
        Unsafe URL, zip/binary archive, or a redirect without Location.

    """
    current = assert_public_url(url)
    response: httpx.Response | None = None
    for _ in range(4):
        lower = current.split("?", 1)[0].lower()
        if lower.endswith((".zip", ".doc", ".docx", ".xls", ".xlsx")):
            raise PublicURLError("archives and office files are skipped")
        current = assert_public_url(current)
        response = _fetch_once(current)
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("location")
            if not location:
                raise PublicURLError("redirect without Location")
            current = urljoin(current, location)
            continue
        break
    if response is None:
        raise PublicURLError("failed to load the page")
    if response.status_code >= 400:
        return (
            f"Страница {current} ответила {response.status_code}. "
            "Возьми другой url из search_web."
        )
    content_type = (response.headers.get("content-type") or "").lower()
    if "zip" in content_type:
        return "Это архив, а не страница паспорта. Возьми HTML или PDF из search_web."
    path = current.lower().split("?", 1)[0]
    limit = _MAX_PDF_BYTES if ("pdf" in content_type or path.endswith(".pdf")) else _MAX_RAW_BYTES
    raw = response.content[:limit]
    if looks_like_pdf(current, content_type, raw):
        text = pdf_bytes_to_text(raw)
        if not text:
            return "В PDF нет извлекаемого текста. Возьми HTML-страницу из search_web."
        head = f"url: {current}\ntitle: PDF паспорт"
        body = text[: get_settings().search_context_chars]
        return f"{head}\n\n{body}"
    try:
        html = raw.decode(response.encoding or "utf-8", errors="replace")
    except LookupError:
        html = raw.decode("utf-8", errors="replace")
    text = html_to_text(html)
    if not text:
        return "На странице нет текста. Возьми другой url из search_web."
    title = ""
    start = html.lower().find("<title>")
    end = html.lower().find("</title>")
    if 0 <= start < end:
        title = " ".join(html[start + 7 : end].split())
    head = f"url: {current}"
    if title:
        head += f"\ntitle: {title[:200]}"
    body = text[: get_settings().search_context_chars]
    return f"{head}\n\n{body}"


@tool
def fetch_page(url: str) -> str:
    """
    Download visible text of an HTML page or PDF datasheet from ``search_web``.

    Call this when a snippet looks like a manufacturer site or datasheet.
    Skip marketplaces without a datasheet, archives, internal hosts, and
    file URLs.

    Parameters
    ----------
    url : str
        Public http(s) address from a ``search_web`` hit.

    Returns
    -------
    page : str
        Title and text clipped to ``SEARCH_CONTEXT_CHARS``, or a recovery hint.

    """
    cleaned = (url or "").strip()
    if not cleaned:
        return "Ошибка: пустой url. Передай адрес из поля url ответа search_web."
    try:
        return _fetch_page_impl(cleaned)
    except PublicURLError as exc:
        return (
            f"Ошибка: этот URL нельзя скачивать ({exc}). "
            "Возьми другой публичный http(s) адрес из search_web."
        )
    except httpx.HTTPError as exc:
        logger.warning("fetch_page network error: {}", exc)
        return f"Сеть не отдала страницу: {exc}. Попробуй другой url из search_web."
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_page failed: {}", exc)
        return f"Ошибка загрузки страницы: {exc}. Попробуй другой url из search_web."

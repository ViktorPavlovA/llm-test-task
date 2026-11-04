from __future__ import annotations

import re
from urllib.parse import urlparse

from loguru import logger

from src.repository.tools import PublicURLError, _fetch_page_impl, _search_impl
from src.schemas import DeviceStatsModel
from src.utils import get_settings

_VARIANT_RE = re.compile(r"\s*\(\s*вариант\s+\d+\s*\)\s*", re.IGNORECASE)
_ANALOG_RE = re.compile(
    r"манометр\s+показывающ|гигрометр\s+психрометрич|"
    r"термометр\s+стеклянн|дифманометр\s+показывающ",
    re.IGNORECASE,
)
_ELECTRONIC_RE = re.compile(
    r"преобразовател|сигнализатор|контроллер|датчик|"
    r"электроконтакт|цифров",
    re.IGNORECASE,
)
_ANALOG_SKIP = frozenset({"питание", "интерфейсы"})
_SITE_BY_MFR = {
    "овен": "owen.ru",
    "owen": "owen.ru",
    "wika": "wika.ru",
    "манотомь": "manotom.ru",
    "росма": "rosma.ru",
    "seitron": "seitron.com",
}

_DOMAIN_SCORE = {
    "owen.ru": 6,
    "www.owen.ru": 6,
    "docs.owen.ru": 6,
    "wika.ru": 6,
    "www.wika.ru": 6,
    "wika.com": 5,
    "www.wika.com": 5,
    "productpass.wika.com": 5,
    "manotom.ru": 6,
    "rosma.ru": 5,
    "seitron.com": 6,
    "www.seitron.com": 6,
    "teplopribor.ru": 4,
    "thermopribor.nt-rt.ru": 4,
    "souz-pribor.ru": 4,
    "etalongroup.ru": 3,
    "vit1.ru": 4,
}

_BAD_HOST_PARTS = (
    "startpage.com",
    "facebook.com",
    "pinterest.",
    "avito.ru",
    "wildberries.",
    "aliexpress.",
    "ozon.ru",
    "youtube.com",
    "vk.com",
)


def clean_search_name(name: str) -> str:
    """
    Strip synthetic ``(вариант N)`` suffixes from a product name.

    Parameters
    ----------
    name : str
        Raw instrument name.

    Returns
    -------
    cleaned : str
        Name safe to use as a search query.

    """
    text = _VARIANT_RE.sub(" ", name or "")
    return " ".join(text.split())


def _model_token(name: str) -> str:
    """
    Pick the most specific token from a cleaned product name.

    Parameters
    ----------
    name : str
        Cleaned instrument name.

    Returns
    -------
    token : str
        Catalog-like token (letters mixed with digits), or empty.

    """
    for part in reversed(name.split()):
        compact = part.replace("×", "x").replace("х", "x")
        if any(ch.isdigit() for ch in compact) and any(ch.isalpha() for ch in compact):
            return part
    return ""


def family_token(token: str) -> str:
    """
    Derive a product-line token from a catalog SKU.

    Parameters
    ----------
    token : str
        Specific model token, for example ``ПД100-ДИ0.25-29``.

    Returns
    -------
    family : str
        Shorter line name such as ``ПД100``, or empty.

    """
    text = (token or "").strip()
    if not text:
        return ""
    if "-" in text:
        head = text.split("-", 1)[0]
        if (
            head != text
            and any(ch.isdigit() for ch in head)
            and any(ch.isalpha() for ch in head)
            and len(head) >= 4
        ):
            return head
    match = re.search(r"^(.*[A-Za-zА-Яа-я])(\d{2,})$", text)
    if match and len(match.group(1)) >= 4:
        return match.group(1) + match.group(2)[0]
    return ""


def research_fields(device: DeviceStatsModel, missing: list[str]) -> list[str]:
    """
    Drop catalog gaps that do not apply to analog mechanical instruments.

    Parameters
    ----------
    device : DeviceStatsModel
        Instrument record.
    missing : list of str
        Fields still empty on the card.

    Returns
    -------
    fields : list of str
        Gaps worth sending to web search.

    """
    name = clean_search_name(device.name)
    if _ANALOG_RE.search(name) and not _ELECTRONIC_RE.search(name):
        return [item for item in missing if item not in _ANALOG_SKIP]
    return list(missing)


def build_search_queries(device: DeviceStatsModel, missing: list[str]) -> list[str]:
    """
    Build a short list of DuckDuckGo queries for missing catalog fields.

    Parameters
    ----------
    device : DeviceStatsModel
        Instrument record.
    missing : list of str
        Catalog fields still empty.

    Returns
    -------
    queries : list of str
        Deduplicated queries, at most four.

    """
    name = clean_search_name(device.name)
    mfr = " ".join((device.manufacturer or "").split())
    token = _model_token(name)
    queries: list[str] = []

    def add(text: str) -> None:
        cleaned = " ".join(text.split())
        if cleaned and cleaned not in queries:
            queries.append(cleaned)

    if mfr:
        add(f"{name} {mfr} паспорт")
    else:
        add(f"{name} паспорт характеристики")
    if token and mfr:
        add(f"{mfr} {token} паспорт")
    host = _SITE_BY_MFR.get(mfr.casefold())
    if host and (token or name):
        add(f"{token or name} site:{host}")
    family = family_token(token)
    if family and family != token and mfr:
        add(f"{mfr} {family} паспорт")
    if mfr and len(queries) < 4:
        add(f"{name} {mfr} характеристики")
    gap = next((item for item in missing if item not in {"текстовое описание"}), "")
    if gap and token and len(queries) < 4:
        add(f"{token} {mfr} {gap}".strip())
    return queries[:4]


def _host(url: str) -> str:
    """
    Lowercased hostname of a URL.

    Parameters
    ----------
    url : str
        Absolute http(s) URL.

    Returns
    -------
    host : str
        Hostname, or empty.

    """
    return (urlparse(url).hostname or "").strip().lower()


def score_hit(title: str, url: str, snippet: str) -> int:
    """
    Rank a search hit for datasheet usefulness.

    Parameters
    ----------
    title : str
        Result title.
    url : str
        Result URL.
    snippet : str
        Result snippet.

    Returns
    -------
    score : int
        Higher is better. Negative scores should be skipped.

    """
    host = _host(url)
    if not host:
        return -10
    if any(part in host for part in _BAD_HOST_PARTS):
        return -8
    score = _DOMAIN_SCORE.get(host, 0)
    blob = f"{title} {snippet} {url}".casefold()
    if "паспорт" in blob or "datasheet" in blob:
        score += 3
    if "характеристик" in blob or "specification" in blob:
        score += 2
    path = url.lower().split("?", 1)[0]
    if path.endswith(".pdf"):
        score += 2 if score >= 3 else -1
    return score


def gather_web_evidence(
    device: DeviceStatsModel,
    missing: list[str],
) -> tuple[str, list[str]]:
    """
    Search and fetch pages without asking the LLM to call tools.

    Parameters
    ----------
    device : DeviceStatsModel
        Instrument record.
    missing : list of str
        Fields the extractor should look for.

    Returns
    -------
    evidence : str
        Ranked snippets and page text for the extractor.
    sources : list of str
        Unique URLs that contributed text.

    """
    settings = get_settings()
    page_chars = min(settings.search_context_chars, 3200)
    snippet_chars = settings.search_snippet_chars
    queries = build_search_queries(device, missing)
    ranked: list[tuple[int, dict[str, str]]] = []
    seen: set[str] = set()
    for query in queries:
        try:
            rows = _search_impl(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("search query failed ({}): {}", query, exc)
            continue
        for row in rows:
            url = str(row.get("href") or row.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            title = str(row.get("title") or "").strip()
            snippet = str(row.get("body") or row.get("snippet") or "").strip()
            score = score_hit(title, url, snippet)
            if score < 0:
                continue
            ranked.append(
                (
                    score,
                    {"title": title, "url": url, "snippet": snippet, "query": query},
                )
            )
    ranked.sort(key=lambda item: item[0], reverse=True)
    blocks: list[str] = []
    sources: list[str] = []
    fetched = 0
    for score, hit in ranked[:8]:
        url = hit["url"]
        snippet = hit["snippet"][:snippet_chars]
        block = (
            f"query: {hit['query']}\n"
            f"title: {hit['title'] or 'без названия'}\n"
            f"url: {url}\n"
            f"snippet: {snippet}"
        )
        path = url.lower().split("?", 1)[0]
        is_office = path.endswith((".doc", ".docx", ".zip", ".xls", ".xlsx"))
        if fetched < 3 and score >= 3 and not is_office:
            try:
                page = _fetch_page_impl(url)
            except (PublicURLError, Exception) as exc:  # noqa: BLE001
                logger.info("skip fetch {}: {}", url, exc)
            else:
                page = page[:page_chars]
                if page:
                    block += f"\npage:\n{page}"
                    fetched += 1
        blocks.append(block)
        if url not in sources:
            sources.append(url)
        if len("\n\n".join(blocks)) > min(page_chars * 2, 9000):
            break
    return "\n\n".join(blocks), sources[:8]

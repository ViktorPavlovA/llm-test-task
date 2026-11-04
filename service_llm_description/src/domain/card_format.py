from __future__ import annotations

import json
import re
from typing import Any

from src.schemas import AnswerLLMModel

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _strip_think(text: str) -> str:
    """
    Drop closed and unclosed ``<think>`` blocks so JSON remains.

    Parameters
    ----------
    text : str
        Raw assistant content.

    Returns
    -------
    cleaned : str
        Text without thought traces.

    """
    cleaned = _THINK_RE.sub("", text or "")
    if "<think>" not in cleaned.casefold():
        return cleaned
    brace = cleaned.find("{")
    if brace != -1:
        return cleaned[brace:]
    parts = _THINK_OPEN_RE.split(cleaned, maxsplit=1)
    return parts[0] if parts else ""


def _as_object(payload: Any) -> dict[str, Any]:
    """
    Require a JSON object.

    Parameters
    ----------
    payload : Any
        Decoded JSON value.

    Returns
    -------
    obj : dict
        The object.

    Raises
    ------
    ValueError
        If the value is not an object.

    """
    if not isinstance(payload, dict):
        raise ValueError("Model JSON is not an object")
    return payload


def _loads_object(snippet: str) -> dict[str, Any]:
    """
    Decode the first JSON object, ignoring trailing junk.

    Parameters
    ----------
    snippet : str
        Text that starts at ``{``.

    Returns
    -------
    obj : dict
        First decoded object.

    """
    payload, _ = json.JSONDecoder().raw_decode(snippet)
    return _as_object(payload)


def _repair_truncated_object(snippet: str) -> str:
    """
    Close a JSON object cut off at ``max_tokens``.

    Parameters
    ----------
    snippet : str
        Text that starts at ``{``.

    Returns
    -------
    repaired : str
        Snippet with closed strings, arrays, and braces.

    """
    repaired = snippet.rstrip()
    curly = 0
    square = 0
    in_str = False
    escape = False
    for char in repaired:
        if in_str:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_str = False
            continue
        if char == '"':
            in_str = True
        elif char == "{":
            curly += 1
        elif char == "}":
            curly -= 1
        elif char == "[":
            square += 1
        elif char == "]":
            square -= 1
    if escape:
        repaired += "\\"
        escape = False
    if in_str:
        repaired += '"'
    repaired = re.sub(r",\s*$", "", repaired)
    repaired += "]" * max(0, square) + "}" * max(0, curly)
    return repaired


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Parse a JSON object from model output, ignoring think blocks and fences.

    Parameters
    ----------
    text : str
        Raw assistant content.

    Returns
    -------
    payload : dict
        First JSON object found in the text.

    Raises
    ------
    ValueError
        If the text has no JSON object.

    """
    cleaned = _strip_think(text or "").strip()
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    try:
        return _as_object(json.loads(cleaned))
    except (json.JSONDecodeError, ValueError):
        pass
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("Model reply has no JSON object")
    snippet = cleaned[start:]
    try:
        return _loads_object(snippet)
    except json.JSONDecodeError:
        pass
    try:
        return _loads_object(_repair_truncated_object(snippet))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model JSON is truncated or invalid: {exc}") from exc


def clip_title(title: str, max_len: int = 70) -> str:
    """
    Trim a title to ``max_len`` characters on a word boundary.

    Parameters
    ----------
    title : str
        Raw SEO title.
    max_len : int, optional
        Maximum length. Default is 70.

    Returns
    -------
    clipped : str
        Title that fits the limit.

    """
    cleaned = " ".join((title or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    chunk = cleaned[:max_len].rstrip()
    for sep in (", ", " — ", " – ", " - ", " "):
        cut = chunk.rfind(sep)
        if cut >= 32:
            chunk = chunk[:cut]
            break
    return chunk.rstrip(" ,.;:—-")


def clip_description(text: str, max_len: int = 1200) -> str:
    """
    Trim a description to ``max_len`` characters, preferring a sentence end.

    Parameters
    ----------
    text : str
        Raw SEO description.
    max_len : int, optional
        Maximum length. Default is 1200.

    Returns
    -------
    clipped : str
        Description that fits the limit.

    """
    cleaned = (text or "").strip()
    if len(cleaned) <= max_len:
        return cleaned
    chunk = cleaned[:max_len]
    for sep in (". ", "! ", "? ", "; "):
        pos = chunk.rfind(sep)
        if pos >= 799:
            return chunk[: pos + 1].strip()
    pos = chunk.rfind(" ")
    if pos >= 799:
        return chunk[:pos].rstrip(" ,;:") + "."
    return chunk[:max_len].rstrip()


def pad_description(
    text: str,
    extra_sentences: list[str],
    min_len: int = 800,
) -> str:
    """
    Append known-fact sentences until the description reaches ``min_len``.

    Parameters
    ----------
    text : str
        Current description.
    extra_sentences : list of str
        Sentences built from locked facts.
    min_len : int, optional
        Target minimum length. Default is 800.

    Returns
    -------
    padded : str
        Description with extra sentences that were not already present.

    """
    cleaned = (text or "").strip()
    for sentence in extra_sentences:
        if len(cleaned) >= min_len:
            break
        piece = sentence.strip()
        if not piece:
            continue
        if not piece.endswith((".", "!", "?")):
            piece += "."
        if piece in cleaned:
            continue
        cleaned = f"{cleaned} {piece}".strip()
    return cleaned


def normalize_queries(queries: Any, fallback: list[str]) -> list[str]:
    """
    Keep 3 to 5 unique search queries, filling from fallbacks if needed.

    Parameters
    ----------
    queries : Any
        Model output: a list, a string, or something else.
    fallback : list of str
        Queries derived from name and manufacturer.

    Returns
    -------
    items : list of str
        Three to five unique queries.

    """
    items: list[str] = []
    if isinstance(queries, str):
        queries = [queries]
    if isinstance(queries, list):
        for item in queries:
            text = " ".join(str(item).split())
            if text and text not in items:
                items.append(text)
    for item in fallback:
        if len(items) >= 3:
            break
        text = " ".join(item.split())
        if text and text not in items:
            items.append(text)
    return items[:5]


def fallback_queries(name: str, manufacturer: str | None) -> list[str]:
    """
    Build default search queries from the instrument name.

    Parameters
    ----------
    name : str
        Instrument name.
    manufacturer : str or None
        Maker, if known.

    Returns
    -------
    queries : list of str
        Up to five unique query strings.

    """
    queries = [name, f"{name} характеристики"]
    if manufacturer:
        queries.insert(1, f"{name} {manufacturer}")
        queries.append(f"{manufacturer} {name} купить")
    queries.append(f"{name} паспорт")
    seen: list[str] = []
    for item in queries:
        if item not in seen:
            seen.append(item)
    return seen[:5]


def number_tokens(text: str) -> set[str]:
    """
    Collect numeric tokens from text, normalizing commas to dots.

    Parameters
    ----------
    text : str
        Source or generated copy.

    Returns
    -------
    tokens : set of str
        Distinct number strings.

    """
    tokens: set[str] = set()
    for match in _NUMBER_RE.findall(text or ""):
        tokens.add(match.replace(",", "."))
    return tokens


def unexpected_numbers(output: str, source: str) -> list[str]:
    """
    List numbers that appear in the output but not in the source facts.

    Single digits 1-5 are ignored as likely list indices.

    Parameters
    ----------
    output : str
        Generated title, description, queries, and commercial text.
    source : str
        Concatenated input facts and web snippets.

    Returns
    -------
    extra : list of str
        Suspicious numeric tokens.

    """
    extra = number_tokens(output) - number_tokens(source)
    notable: list[str] = []
    for token in sorted(extra, key=lambda item: (len(item), item)):
        if re.fullmatch(r"[1-5]", token):
            continue
        notable.append(token)
    return notable


def to_markdown(
    card: AnswerLLMModel,
    sources: list[str] | None = None,
) -> str:
    """
    Render a card as Markdown for copy-paste.

    Parameters
    ----------
    card : AnswerLLMModel
        Structured card fields.
    sources : list of str or None
        Search URLs. Falls back to ``card.sources`` when omitted.

    Returns
    -------
    markdown : str
        Markdown document with heading sections.

    """
    specs = "\n".join(f"- {item}" for item in card.key_specs)
    queries = "\n".join(f"- {item}" for item in card.search_queries)
    missing = "\n".join(f"- {item}" for item in card.data_missing) or "- нет"
    notes = "\n".join(f"- {item}" for item in card.review_notes)
    extra_sources = (
        sources if sources is not None else list(getattr(card, "sources", None) or [])
    )
    parts = [
        f"# {card.title}",
        "",
        "## SEO-описание",
        "",
        card.description,
        "",
        "## Ключевые характеристики",
        "",
        specs,
        "",
        "## Поисковые запросы",
        "",
        queries,
        "",
        "## Для менеджера",
        "",
        card.commercial_block,
        "",
        "## Требуют уточнения",
        "",
        missing,
    ]
    if extra_sources:
        listed = "\n".join(f"- {item}" for item in extra_sources)
        parts.extend(["", "## Источники поиска", "", listed])
    if notes:
        parts.extend(["", "## Замечания перед публикацией", "", notes])
    return "\n".join(parts) + "\n"

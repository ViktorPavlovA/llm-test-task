from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.domain.card_format import (
    clip_description,
    clip_title,
    fallback_queries,
    normalize_queries,
    pad_description,
    to_markdown,
    unexpected_numbers,
)
from src.domain.exceptions import LLMBadResponseError
from src.domain.facts import (
    PLACEHOLDER_VALUES,
    apply_web_specs,
    build_key_specs,
    collect_missing,
    merge_data_missing,
    merge_key_specs,
    source_text,
    stringify_spec,
)
from src.domain.prompt import build_messages
from src.graph.research import (
    clean_search_name,
    gather_web_evidence,
    research_fields,
)
from src.repository.llm_client import complete_json
from src.schemas import AnswerLLMModel, CardResponse, DeviceStatsModel
from src.utils import get_settings

_SKIP_QUOTE_MARKERS = ("sorry, need more steps", "need more steps to process")

EXTRACT_SYSTEM = """Ты извлекаешь характеристики контрольно-измерительного прибора из текстов поиска.
Верни один JSON без markdown и без тегов <think>:
{"found_specs": {"питание": "..."}, "sources": ["https://..."], "quotes": ["цитата"]}
Правила:
- Бери только значения, которые буквально есть в тексте поиска.
- Уже известные specs не переписывай.
- Бери характеристики, полезные именно для этого типа прибора.
- Не выдумывай числа, IP, ГОСТ, питание, интерфейсы.
- Разрешено брать общие характеристики линейки с сайта производителя, если название прибора — модификация этой линейки (префикс совпадает: «ПД100-ДИ0.25-29» это линейка ПД100).
- Не подставляй характеристики другой линейки (САУ-М6 не подходит для САУ-М611, если модель 611 в тексте нет).
- Если не уверен — не включай поле. Пустой found_specs допустим.
- Сразу верни JSON. Не рассуждай вслух.

/no_think
"""


class CardState(TypedDict, total=False):
    """
    Shared state of the research-then-write graph.

    """

    device: dict[str, Any]
    key_specs: list[str]
    data_missing: list[str]
    web_specs: dict[str, str]
    sources: list[str]
    research_notes: list[str]
    web_used: bool
    card: dict[str, Any]


@dataclass
class _Draft:
    """
    Intermediate writer output before Pydantic validation.

    """

    title: str
    description: str
    key_specs: list[str]
    search_queries: list[str]
    commercial_block: str
    data_missing: list[str]
    review_notes: list[str] = field(default_factory=list)


def needs_web_research(
    data_missing: list[str],
    name: str,
    *,
    llm_search_internet: bool | None = None,
) -> bool:
    """
    Return True when the research node should run.

    Parameters
    ----------
    data_missing : list of str
        Catalog fields still empty.
    name : str
        Instrument name used as the search seed.
    llm_search_internet : bool or None
        Override for tests. Default reads ``Settings.llm_search_internet``.

    Returns
    -------
    needed : bool
        True when ``LLM_SEARCH_INTERNET`` is on, something is missing,
        and a name exists.

    """
    enabled = (
        get_settings().llm_search_internet
        if llm_search_internet is None
        else llm_search_internet
    )
    return bool(enabled) and bool(data_missing) and bool((name or "").strip())


def _route_start(state: CardState) -> Literal["research", "write"]:
    """
    Choose the first graph node from missing fields and settings.

    Parameters
    ----------
    state : CardState
        Initial graph state.

    Returns
    -------
    next_node : {"research", "write"}
        Node name to enter after START.

    """
    device = state.get("device") or {}
    if needs_web_research(
        state.get("data_missing") or [],
        str(device.get("name") or ""),
    ):
        return "research"
    return "write"


def _normalize_found(raw: Any) -> dict[str, str]:
    """
    Keep non-empty spec pairs and drop placeholder values.

    Parameters
    ----------
    raw : Any
        ``found_specs`` from the research agent.

    Returns
    -------
    found : dict of str
        Clean label-to-value mapping.

    """
    if not isinstance(raw, dict):
        return {}
    found: dict[str, str] = {}
    for key, value in raw.items():
        label = " ".join(str(key).split())
        text = stringify_spec(value)
        if not label or not text:
            continue
        if text.casefold() in PLACEHOLDER_VALUES:
            continue
        found[label] = text
    return found


def _normalize_sources(raw: Any) -> list[str]:
    """
    Flatten source strings or ``{url, title}`` objects.

    Parameters
    ----------
    raw : Any
        Agent ``sources`` field.

    Returns
    -------
    items : list of str
        Up to eight unique source lines.

    """
    items: list[str] = []
    if isinstance(raw, str) and raw.strip():
        raw = [raw]
    if not isinstance(raw, list):
        return items
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
            if text and text not in items:
                items.append(text)
            continue
        if isinstance(item, dict):
            url = str(item.get("url") or item.get("href") or "").strip()
            title = str(item.get("title") or "").strip()
            if url and title:
                text = f"{title} — {url}"
            else:
                text = url or title
            if text and text not in items:
                items.append(text)
    return items[:8]


def _is_noise_text(text: str) -> bool:
    """
    Detect LangGraph step-limit filler that is not research JSON.

    Parameters
    ----------
    text : str
        Assistant or quote text.

    Returns
    -------
    noisy : bool
        True when the text should be ignored.

    """
    low = text.casefold()
    return any(marker in low for marker in _SKIP_QUOTE_MARKERS)


async def research_node(state: CardState) -> dict[str, Any]:
    """
    Search the web for missing specs and merge them into empty fields.

    Parameters
    ----------
    state : CardState
        Graph state with ``device`` and ``data_missing``.

    Returns
    -------
    update : dict
        ``web_specs``, ``sources``, ``research_notes``, ``web_used``,
        and an optionally enriched ``device``.

    """
    device = DeviceStatsModel.model_validate(state["device"])
    missing = state.get("data_missing") or collect_missing(device)
    gaps = research_fields(device, missing)
    notes: list[str] = []
    found: dict[str, str] = {}
    sources: list[str] = []
    quotes: list[str] = []
    if not gaps:
        return {
            "web_specs": {},
            "sources": [],
            "research_notes": [
                "Поиск не запускался: оставшиеся поля не применимы "
                "к этому типу прибора."
            ],
            "web_used": False,
            "device": device.model_dump(),
        }
    try:
        evidence, sources = gather_web_evidence(device, gaps)
        if evidence:
            extracted = await complete_json(
                [
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Прибор: {clean_search_name(device.name)}\n"
                            f"Производитель: {device.manufacturer or 'не указан'}\n"
                            f"Уже известные specs: "
                            f"{json.dumps(device.specs, ensure_ascii=False)}\n"
                            f"Недостаёт: {', '.join(gaps)}\n\n"
                            "Тексты поиска:\n"
                            f"{evidence}"
                        ),
                    },
                ]
            )
            found = _normalize_found(extracted.get("found_specs"))
            extra_sources = _normalize_sources(extracted.get("sources"))
            for item in extra_sources:
                if item not in sources:
                    sources.append(item)
            raw_quotes = extracted.get("quotes")
            if isinstance(raw_quotes, list):
                quotes.extend(
                    str(item).strip()
                    for item in raw_quotes
                    if str(item).strip() and not _is_noise_text(str(item))
                )
        if found:
            notes.append(
                "Из интернета добавлены поля: "
                + ", ".join(found.keys())
                + ". Сверьте с паспортом перед публикацией."
            )
        elif sources:
            notes.append(
                "Поиск в интернете выполнен, недостающие характеристики "
                "не подтвердились."
            )
        else:
            notes.append("Поиск в интернете не дал пригодных характеристик.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("research node failed")
        notes.append(f"Поиск в интернете не удался: {str(exc).splitlines()[0][:180]}")
        return {
            "web_specs": {},
            "sources": [],
            "research_notes": notes,
            "web_used": False,
        }
    return {
        "web_specs": found,
        "sources": sources,
        "research_notes": notes + quotes[:3],
        "web_used": True,
    }


def _draft_text(draft: _Draft) -> str:
    """
    Join generated copy for the unexpected-number check.

    Parameters
    ----------
    draft : _Draft
        Writer output.

    Returns
    -------
    text : str
        Title, description, commercial block, and queries.

    """
    return " ".join(
        [draft.title, draft.description, draft.commercial_block, *draft.search_queries]
    )


def _fallback_commercial(device: DeviceStatsModel) -> str:
    """
    Build a generic commercial block from known facts only.

    Parameters
    ----------
    device : DeviceStatsModel
        Instrument record.

    Returns
    -------
    text : str
        Fallback commercial paragraph.

    """
    who = device.manufacturer or "заявленного производителя"
    purpose = device.description or "задач измерения и контроля по паспорту прибора"
    return (
        f"Прибор «{device.name}» производства {who} подходит снабженцам и инженерам, "
        f"которым нужен черновик карточки B2B-каталога. Он закрывает {purpose} "
        "в том объёме, который подтверждён входными данными. Перед публикацией "
        "сверьте характеристики с паспортом прибора."
    )


def _padding_sentences(
    device: DeviceStatsModel,
    key_specs: list[str],
    data_missing: list[str],
) -> list[str]:
    """
    Sentences used to pad a short description with known facts.

    Parameters
    ----------
    device : DeviceStatsModel
        Instrument record.
    key_specs : list of str
        Locked spec lines.
    data_missing : list of str
        Still-missing catalog fields.

    Returns
    -------
    sentences : list of str
        Fact sentences, without invented numbers.

    """
    sentences = [f"Прибор поставляется под наименованием «{device.name}»."]
    if device.manufacturer:
        sentences.append(
            f"Изготовитель, указанный во входных данных, — {device.manufacturer}."
        )
    if device.description:
        sentences.append(device.description.rstrip(".") + ".")
    provided = [item for item in key_specs if "данных нет" not in item.lower()]
    if provided:
        sentences.append(
            "В карточке можно опубликовать только подтверждённые параметры: "
            + "; ".join(provided)
            + "."
        )
    if data_missing:
        sentences.append(
            "Перед публикацией нужно запросить у поставщика недостающие сведения: "
            + ", ".join(data_missing)
            + "."
        )
    sentences.append(
        "Все технические значения в черновике взяты из входных данных или "
        "помеченных веб-источников."
    )
    return sentences


def _coerce_draft(
    raw: dict,
    device: DeviceStatsModel,
    source: str,
    web_specs: dict[str, str] | None = None,
) -> _Draft:
    """
    Clip copy and merge dynamic key specs from a raw model JSON object.

    Parameters
    ----------
    raw : dict
        Parsed writer JSON.
    device : DeviceStatsModel
        Instrument record.
    source : str
        Fact text used to flag invented numbers.
    web_specs : dict of str or None
        Search fills. They always appear in ``key_specs``.

    Returns
    -------
    draft : _Draft
        Normalized writer fields.

    """
    title = clip_title(str(raw.get("title") or device.name)) or clip_title(device.name)
    description = clip_description(str(raw.get("description") or "").strip())
    queries = normalize_queries(
        raw.get("search_queries"),
        fallback_queries(device.name, device.manufacturer),
    )
    commercial = str(raw.get("commercial_block") or "").strip() or _fallback_commercial(
        device
    )
    key_specs = merge_key_specs(
        device,
        raw.get("key_specs"),
        web_specs=web_specs,
    )
    data_missing = merge_data_missing(device, raw.get("data_missing"))
    notes: list[str] = []
    extras = unexpected_numbers(
        " ".join([title, description, commercial, *queries]),
        source,
    )
    if extras:
        notes.append(
            "Модель могла добавить числа вне входных данных: " + ", ".join(extras)
        )
    return _Draft(
        title=title,
        description=description,
        key_specs=key_specs,
        search_queries=queries,
        commercial_block=commercial,
        data_missing=data_missing,
        review_notes=notes,
    )


async def write_node(state: CardState) -> dict[str, Any]:
    """
    Generate SEO copy and assemble the final card payload.

    Parameters
    ----------
    state : CardState
        Graph state after optional research.

    Returns
    -------
    update : dict
        Locked specs, remaining missing fields, and ``card``.

    Raises
    ------
    LLMBadResponseError
        Description length is still outside 800-1200 after padding.

    """
    device_in = DeviceStatsModel.model_validate(state["device"])
    web_specs = dict(state.get("web_specs") or {})
    device = apply_web_specs(device_in, web_specs) if web_specs else device_in
    key_specs = build_key_specs(device)
    source = source_text(device)
    web_blob = "\n".join(
        [
            json.dumps(web_specs, ensure_ascii=False),
            *(state.get("sources") or []),
            *(state.get("research_notes") or []),
        ]
    )
    source_for_numbers = f"{source}\n{web_blob}"
    sources = list(state.get("sources") or [])
    web_used = bool(state.get("web_used"))

    raw = await complete_json(
        build_messages(
            device,
            key_specs,
            [],
            web_facts=web_specs or None,
            sources=sources,
        )
    )
    draft = _coerce_draft(raw, device, source_for_numbers, web_specs=web_specs)

    if len(draft.description) < 800:
        repair = (
            f"Сейчас в description {len(draft.description)} символов. "
            "Расширь текст до 900–1100 символов, не добавляя новых "
            "характеристик и чисел. Верни тот же JSON целиком."
        )
        raw = await complete_json(
            build_messages(
                device,
                key_specs,
                [],
                repair=repair,
                web_facts=web_specs or None,
                sources=sources,
            )
        )
        draft = _coerce_draft(raw, device, source_for_numbers, web_specs=web_specs)

    if len(draft.description) < 800:
        draft.description = pad_description(
            draft.description,
            _padding_sentences(device, draft.key_specs, draft.data_missing),
        )
        draft.description = clip_description(draft.description)

    if not (800 <= len(draft.description) <= 1200):
        raise LLMBadResponseError(
            f"Failed to get a 800-1200 character description "
            f"(got {len(draft.description)})"
        )

    extras = unexpected_numbers(_draft_text(draft), source_for_numbers)
    if extras and not any("числа" in note for note in draft.review_notes):
        draft.review_notes.append(
            "Проверьте числа, которых не было во входных данных: " + ", ".join(extras)
        )
    for note in state.get("research_notes") or []:
        if not note or note in draft.review_notes or note.startswith("url:"):
            continue
        if len(note) >= 400 or _is_noise_text(note) or note.startswith("Ошибка:"):
            continue
        if "нельзя скачивать" in note.casefold():
            continue
        draft.review_notes.append(note)

    card = AnswerLLMModel(
        title=draft.title,
        description=draft.description,
        key_specs=draft.key_specs,
        search_queries=draft.search_queries,
        commercial_block=draft.commercial_block,
        data_missing=draft.data_missing,
        review_notes=draft.review_notes,
    )
    response = CardResponse(
        **card.model_dump(),
        markdown=to_markdown(card, sources=sources),
        sources=sources,
        web_used=web_used,
    )
    return {
        "key_specs": draft.key_specs,
        "data_missing": draft.data_missing,
        "card": response.model_dump(),
    }


def _build_graph():
    """
    Compile START -> research|write -> write -> END.

    Returns
    -------
    graph : CompiledStateGraph
        Runnable card pipeline.

    """
    builder = StateGraph(CardState)
    builder.add_node("research", research_node)
    builder.add_node("write", write_node)
    builder.add_conditional_edges(
        START,
        _route_start,
        {"research": "research", "write": "write"},
    )
    builder.add_edge("research", "write")
    builder.add_edge("write", END)
    return builder.compile()


GRAPH = _build_graph()


async def build_card(device: DeviceStatsModel) -> CardResponse:
    """
    Run the LangGraph pipeline and return a product card.

    Parameters
    ----------
    device : DeviceStatsModel
        Parsed instrument JSON.

    Returns
    -------
    card : CardResponse
        SEO card, markdown, sources, and ``web_used``.

    Raises
    ------
    LLMBadResponseError
        Graph finished without a card payload.

    """
    initial: CardState = {
        "device": device.model_dump(),
        "key_specs": build_key_specs(device),
        "data_missing": collect_missing(device),
        "web_specs": {},
        "sources": [],
        "research_notes": [],
        "web_used": False,
    }
    final = await GRAPH.ainvoke(initial, config={"recursion_limit": 32})
    payload = final.get("card")
    if not payload:
        raise LLMBadResponseError("Graph did not return a card")
    return CardResponse.model_validate(payload)

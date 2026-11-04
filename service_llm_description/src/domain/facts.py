from __future__ import annotations

import re
from typing import Any

from src.schemas import DeviceStatsModel

MISSING_MARK = "данных нет"
PLACEHOLDER_VALUES = frozenset(
    {
        "данных нет",
        "нет данных",
        "не найдено",
        "не найдены",
        "не указан",
        "не указано",
        "не указаны",
        "не задано",
        "не заданы",
        "отсутствует",
        "отсутствуют",
        "unknown",
        "n/a",
        "-",
        "нет",
    }
)

# Typical measuring-instrument fields from the assignment plus JSON aliases.
EXPECTED_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("производитель", ("производитель", "manufacturer", "бренд", "изготовитель")),
    (
        "диапазон измерений",
        (
            "диапазон измерений",
            "диапазон измерения",
            "диапазон",
            "верхний предел измерений",
            "верхний предел",
            "предел измерений",
        ),
    ),
    (
        "класс точности",
        (
            "класс точности",
            "точность",
            "погрешность",
            "основная приведённая погрешность",
            "класс допуска",
        ),
    ),
    ("питание", ("питание", "напряжение питания", "напряжение")),
    ("тип корпуса", ("тип корпуса", "исполнение корпуса", "корпус")),
    (
        "интерфейсы",
        ("интерфейсы", "интерфейс", "выходной сигнал", "выходные сигналы"),
    ),
    ("степень защиты", ("степень защиты", "ip", "пыле-влагозащита")),
    (
        "присоединение",
        (
            "присоединение",
            "присоединительная резьба",
            "резьба",
            "штуцер",
            "расположение штуцера",
        ),
    ),
)


def _norm(value: str) -> str:
    """
    Fold a label for alias matching.

    Parameters
    ----------
    value : str
        Spec key or alias.

    Returns
    -------
    normalized : str
        Lowercase text with punctuation replaced by spaces.

    """
    text = value.casefold().replace("ё", "е")
    return re.sub(r"[^a-z0-9а-я]+", " ", text).strip()


def stringify_spec(value: Any) -> str:
    """
    Turn a spec value into a single-line string.

    Parameters
    ----------
    value : Any
        JSON value from ``specs``.

    Returns
    -------
    text : str
        Display string, or empty when the value is missing.

    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return format(value, "g")
    text = str(value).strip()
    return " ".join(text.split())


def _spec_map(device: DeviceStatsModel) -> dict[str, str]:
    """
    Map non-empty spec keys to display strings.

    Parameters
    ----------
    device : DeviceStatsModel
        Instrument record.

    Returns
    -------
    mapping : dict of str
        ``{label: value}`` for non-empty specs.

    """
    mapping: dict[str, str] = {}
    for key, raw in device.specs.items():
        text = stringify_spec(raw)
        if text:
            mapping[key] = text
    return mapping


def _has_alias(spec_keys: list[str], aliases: tuple[str, ...]) -> bool:
    """
    Return True when any spec key matches any alias.

    Parameters
    ----------
    spec_keys : list of str
        Keys already present on the device.
    aliases : tuple of str
        Canonical name and synonyms.

    Returns
    -------
    found : bool
        True on exact or substring match after normalization.

    """
    norms = [_norm(key) for key in spec_keys]
    alias_norms = [_norm(alias) for alias in aliases]
    for key in norms:
        for alias in alias_norms:
            if not alias:
                continue
            if key == alias or alias in key or key in alias:
                return True
    return False


def is_placeholder(value: str) -> bool:
    """
    Return True when a spec value is an empty-meaning placeholder.

    Parameters
    ----------
    value : str
        Display string.

    Returns
    -------
    empty : bool
        True for «данных нет» and similar marks.

    """
    return stringify_spec(value).casefold() in PLACEHOLDER_VALUES


def parse_spec_line(line: str) -> tuple[str, str] | None:
    """
    Split a ``label: value`` key-spec line.

    Parameters
    ----------
    line : str
        One key-spec entry.

    Returns
    -------
    pair : tuple of str or None
        Label and value, or None when the line is empty or a placeholder.

    """
    text = " ".join(str(line or "").split())
    if ":" not in text:
        return None
    label, raw = text.split(":", 1)
    label = label.strip()
    value = stringify_spec(raw)
    if not label or not value or is_placeholder(value):
        return None
    return label, value


def collect_missing(device: DeviceStatsModel) -> list[str]:
    """
    List expected catalog fields that the input does not cover.

    Parameters
    ----------
    device : DeviceStatsModel
        Instrument record.

    Returns
    -------
    missing : list of str
        Unique field labels still without a value.

    """
    missing: list[str] = []
    spec_keys = list(_spec_map(device).keys())

    if not device.manufacturer and not _has_alias(
        spec_keys, ("производитель", "manufacturer")
    ):
        missing.append("производитель")

    for label, aliases in EXPECTED_FIELDS:
        if label == "производитель":
            continue
        if not _has_alias(spec_keys, aliases):
            missing.append(label)

    if not device.description:
        missing.append("текстовое описание")

    seen: set[str] = set()
    ordered: list[str] = []
    for item in missing:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def known_spec_lines(device: DeviceStatsModel) -> list[str]:
    """
    Build confirmed ``label: value`` lines from the input and web merge.

    Parameters
    ----------
    device : DeviceStatsModel
        Instrument record.

    Returns
    -------
    lines : list of str
        Manufacturer and non-empty specs. Placeholders are omitted.

    """
    specs = _spec_map(device)
    lines = [
        f"{key}: {value}"
        for key, value in specs.items()
        if not is_placeholder(value)
    ]
    if device.manufacturer and not _has_alias(
        list(specs.keys()), ("производитель", "manufacturer")
    ):
        lines.insert(0, f"производитель: {device.manufacturer}")
    if not lines:
        lines.append(f"наименование: {device.name}")
    return lines


def build_key_specs(device: DeviceStatsModel) -> list[str]:
    """
    Build confirmed key-spec lines for the card.

    Catalog gaps are not filled with «данных нет»; the writer may reorder
    these lines and the merge step keeps every confirmed value.

    Parameters
    ----------
    device : DeviceStatsModel
        Instrument record.

    Returns
    -------
    lines : list of str
        ``"{label}: {value}"`` entries from known facts only.

    """
    return known_spec_lines(device)


def _llm_spec_items(raw: Any) -> list[str]:
    """
    Flatten model ``key_specs`` that arrived as a list or a mapping.

    Parameters
    ----------
    raw : Any
        Writer ``key_specs`` field.

    Returns
    -------
    items : list of str
        Spec lines.

    """
    if isinstance(raw, dict):
        return [f"{key}: {value}" for key, value in raw.items()]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str) and raw.strip():
        return [raw]
    return []


def merge_key_specs(
    device: DeviceStatsModel,
    llm_lines: Any,
    web_specs: dict[str, str] | None = None,
) -> list[str]:
    """
    Keep every input spec and every grounded web spec.

    The writer may reorder. Invented values and placeholders are dropped.
    Search fills never overwrite input JSON values.

    Parameters
    ----------
    device : DeviceStatsModel
        Instrument record. Web fills may already be merged in.
    llm_lines : Any
        ``key_specs`` from the writer JSON.
    web_specs : dict of str or None
        Specs extracted from search. They always appear on the card.

    Returns
    -------
    lines : list of str
        Product-specific specs.

    """
    working = apply_web_specs(device, web_specs) if web_specs else device
    known_lines = known_spec_lines(working)
    known: dict[str, tuple[str, str]] = {}
    for line in known_lines:
        parsed = parse_spec_line(line)
        if parsed is None:
            continue
        known[_norm(parsed[0])] = parsed
    source = source_text(working).casefold()
    ordered: list[str] = []
    seen: set[str] = set()

    def add(label: str, value: str, key: str) -> None:
        if key in seen or is_placeholder(value):
            return
        ordered.append(f"{label}: {value}")
        seen.add(key)

    def match_known(label: str) -> str | None:
        key = _norm(label)
        if key in known:
            return key
        field = _field_for_key(label)
        if not field:
            return None
        for known_key, pair in known.items():
            if _has_alias([pair[0]], _aliases_for(field)):
                return known_key
        return None

    for item in _llm_spec_items(llm_lines):
        parsed = parse_spec_line(item)
        if parsed is None:
            continue
        label, value = parsed
        matched = match_known(label)
        if matched is not None:
            orig, known_value = known[matched]
            add(orig, known_value, matched)
            continue
        compact = value.casefold()
        folded_source = source.replace(" ", "")
        if compact not in source and compact.replace(" ", "") not in folded_source:
            continue
        if len(compact) < 4 or compact.isdigit():
            continue
        add(label, value, _norm(label))

    for key, (label, value) in known.items():
        add(label, value, key)
    return ordered or known_lines


def merge_data_missing(device: DeviceStatsModel, llm_missing: Any) -> list[str]:
    """
    Keep LLM-chosen gaps that are not already covered by known specs.

    Parameters
    ----------
    device : DeviceStatsModel
        Instrument record.
    llm_missing : Any
        Writer ``data_missing`` field.

    Returns
    -------
    missing : list of str
        Relevant uncovered fields, without the rigid catalog dump.

    """
    if isinstance(llm_missing, str) and llm_missing.strip():
        items = [llm_missing]
    elif isinstance(llm_missing, list):
        items = [str(item) for item in llm_missing]
    else:
        items = []
    spec_keys = list(_spec_map(device).keys())
    if device.manufacturer:
        spec_keys.append("производитель")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        label = " ".join(item.split())
        if not label or is_placeholder(label):
            continue
        if label.casefold() in {"нет", "нет полей", "не требуется"}:
            continue
        field = _field_for_key(label) or label
        if _has_alias(spec_keys, _aliases_for(field) if field != label else (label,)):
            continue
        key = _norm(label)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(label)
        if len(ordered) >= 8:
            break
    return ordered


def _aliases_for(label: str) -> tuple[str, ...]:
    """
    Return synonym tuple for a canonical catalog field.

    Parameters
    ----------
    label : str
        Canonical field name.

    Returns
    -------
    aliases : tuple of str
        Synonyms, or ``(label,)`` when unknown.

    """
    for field, aliases in EXPECTED_FIELDS:
        if field == label:
            return aliases
    return (label,)


def _field_for_key(key: str) -> str | None:
    """
    Map a free-form spec key onto a canonical catalog field.

    Parameters
    ----------
    key : str
        Incoming spec name.

    Returns
    -------
    field : str or None
        Canonical label, or None when the key is unknown.

    """
    for label, aliases in EXPECTED_FIELDS:
        if _has_alias([key], aliases):
            return label
    return None


def apply_web_specs(
    device: DeviceStatsModel,
    found: dict[str, str],
) -> DeviceStatsModel:
    """
    Fill empty fields from web search. Input JSON values are never overwritten.

    Parameters
    ----------
    device : DeviceStatsModel
        Original instrument record.
    found : dict of str
        Specs extracted from search snippets or pages.

    Returns
    -------
    enriched : DeviceStatsModel
        Copy with gaps filled; existing keys stay as they were.

    """
    data = device.model_dump()
    specs: dict[str, Any] = dict(data.get("specs") or {})
    spec_keys = list(specs.keys())

    for raw_key, raw_value in (found or {}).items():
        label = " ".join(str(raw_key).split())
        text = stringify_spec(raw_value)
        if not label or not text:
            continue
        field = _field_for_key(label)
        norm_label = _norm(label)

        if field == "производитель" or norm_label in {
            _norm("производитель"),
            _norm("manufacturer"),
        }:
            if not data.get("manufacturer"):
                data["manufacturer"] = text
            continue

        if norm_label in {
            _norm("описание"),
            _norm("description"),
            _norm("текстовое описание"),
        }:
            if not data.get("description"):
                data["description"] = text[:2000]
            continue

        if field and _has_alias(spec_keys, _aliases_for(field)):
            continue
        if _has_alias(spec_keys, (label,)):
            continue

        stored_key = field or label
        specs[stored_key] = text
        spec_keys.append(stored_key)

    data["specs"] = specs
    return DeviceStatsModel.model_validate(data)


def facts_for_prompt(device: DeviceStatsModel) -> dict[str, Any]:
    """
    Compact fact dict injected into the writer prompt.

    Parameters
    ----------
    device : DeviceStatsModel
        Instrument record.

    Returns
    -------
    facts : dict
        Name, manufacturer, type, model, specs, description.

    """
    specs = _spec_map(device)
    facts: dict[str, Any] = {"name": device.name, "specs": specs}
    if device.manufacturer:
        facts["manufacturer"] = device.manufacturer
    if device.type:
        facts["type"] = device.type
    if device.model:
        facts["model"] = device.model
    if device.description:
        facts["description"] = device.description
    return facts


def source_text(device: DeviceStatsModel) -> str:
    """
    Concatenate known facts for the unexpected-number check.

    Parameters
    ----------
    device : DeviceStatsModel
        Instrument record.

    Returns
    -------
    text : str
        Newline-joined source strings.

    """
    parts = [device.name]
    if device.manufacturer:
        parts.append(device.manufacturer)
    if device.type:
        parts.append(device.type)
    if device.model:
        parts.append(device.model)
    if device.description:
        parts.append(device.description)
    for key, value in _spec_map(device).items():
        parts.append(f"{key} {value}")
    return "\n".join(parts)

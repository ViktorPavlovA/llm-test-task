from __future__ import annotations

import json
from typing import Any

from src.domain.facts import facts_for_prompt
from src.schemas import DeviceStatsModel

SYSTEM_PROMPT = """Ты помощник контент-менеджера B2B-магазина контрольно-измерительных приборов.

Собери черновик карточки только по переданным ФАКТАМ. Все значения из facts.specs обязательны: включи их в key_specs и своими словами в SEO-описание.

Правила:
1. Не выдумывай характеристики, числа, единицы измерения, классы точности, IP, питание, интерфейсы, материалы, ГОСТ, сертификаты, цену, гарантию и сроки поставки.
2. Если факта нет, не домысливай его. В коммерческом тексте пиши общими словами.
3. web_facts — данные из открытых источников. Их можно использовать. Без web_facts правило «не выдумывай» жёсткое.
4. key_specs — динамический список «подпись: значение» для ЭТОГО типа прибора.
   Сигнализатор загазованности: газ, диапазон, пороги, сенсор, питание, выходы, IP.
   Манометр: диапазон, класс точности, присоединение, диаметр.
   Термопреобразователь: НСХ, диапазон температур, класс допуска, арматура.
   Преобразователь давления: тип давления, предел, выход, погрешность, питание, резьба, IP.
   Включи все facts.specs и производителя, если он есть. Можно переставить по важности.
   Все web_facts тоже включи в key_specs: поиск нашёл их по этому прибору.
   Не добавляй поля вне facts и web_facts. Не пиши «данных нет» в key_specs.
5. data_missing — только поля, которые для этого прибора действительно нужны, но во фактах их нет. Не копируй общий каталог КИП.
6. Ответ — один JSON-объект без markdown и без комментариев.
7. Не пиши теги <think> и не рассуждай вслух. Сразу верни JSON.

/no_think

Формат:
{
  "title": "SEO-заголовок до 70 символов",
  "description": "SEO-описание 900-1100 символов, включая пробелы",
  "key_specs": ["производитель: Seitron", "контролируемый газ: CO+CH4"],
  "search_queries": ["запрос 1", "запрос 2", "запрос 3"],
  "commercial_block": "кому подходит прибор и какие задачи решает",
  "data_missing": ["межповерочный интервал"]
}

title: название прибора и одна-две характеристики из фактов.
description: связный текст 900-1100 символов. Перечисли все переданные характеристики.
search_queries: 3-5 формулировок, как ищут инженеры и снабженцы.
commercial_block: 2-4 предложения, без новых цифр.
"""


def build_messages(
    device: DeviceStatsModel,
    key_specs: list[str],
    data_missing: list[str],
    repair: str | None = None,
    web_facts: dict[str, str] | None = None,
    sources: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Build chat messages for the JSON card writer.

    Parameters
    ----------
    device : DeviceStatsModel
        Instrument record after optional web merge.
    key_specs : list of str
        Confirmed specs the model must keep in the card.
    data_missing : list of str
        Optional research gaps; not a catalog to dump into the card.
    repair : str or None
        Extra instruction used when the previous description was too short.
    web_facts : dict of str or None
        Specs filled from search snippets.
    sources : list of str or None
        URLs that produced ``web_facts``.

    Returns
    -------
    messages : list of dict
        System and user messages for ``/chat/completions``.

    """
    provided = [
        item for item in key_specs if item and "данных нет" not in item.casefold()
    ]
    payload: dict[str, Any] = {
        "facts": facts_for_prompt(device),
        "confirmed_specs": provided,
    }
    if web_facts:
        payload["web_facts"] = web_facts
        payload["web_sources"] = sources or []
    spec_block = (
        "\n".join(f"- {item}" for item in provided) or "- нет подтверждённых значений"
    )
    user = (
        "ФАКТЫ — основной источник истины:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Переданные характеристики (все обязаны быть в key_specs и в description):\n"
        f"{spec_block}\n\n"
        "Собери JSON карточки. В key_specs верни характеристики, важные именно "
        "для этого прибора: все confirmed_specs и все web_facts, без пустых "
        "каталожных полей и без «данных нет»."
    )
    if web_facts:
        user += (
            "\nweb_facts найдены в интернете по этому прибору — "
            "включи их в key_specs и в description. "
            "Не добавляй характеристики вне facts и web_facts."
        )
    if repair:
        user += f"\n\nИсправление предыдущего ответа: {repair}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

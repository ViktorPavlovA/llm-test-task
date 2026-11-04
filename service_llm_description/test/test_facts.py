from src.domain.facts import (
    apply_web_specs,
    build_key_specs,
    collect_missing,
    merge_data_missing,
    merge_key_specs,
)
from src.schemas import DeviceStatsModel


def test_missing_fields_for_sparse_product():
    """
    List expected catalog fields when the input has only a name.

    """
    device = DeviceStatsModel(name="Манометр X")
    missing = collect_missing(device)
    assert "производитель" in missing
    assert "диапазон измерений" in missing
    assert "класс точности" in missing
    assert "питание" in missing
    assert "текстовое описание" in missing


def test_key_specs_keep_given_values_without_placeholders():
    """
    Keep provided specs and skip empty catalog fields.

    """
    device = DeviceStatsModel(
        name="Манометр DM 02",
        manufacturer="Манотомь",
        specs={"диапазон измерений": "0–1,6 МПа", "класс точности": "1,5"},
        description="Для избыточного давления.",
    )
    specs = build_key_specs(device)
    assert "производитель: Манотомь" in specs
    assert "диапазон измерений: 0–1,6 МПа" in specs
    assert "класс точности: 1,5" in specs
    assert not any("данных нет" in item for item in specs)
    assert not any(item.startswith("питание:") for item in specs)
    assert not any(item.startswith("интерфейсы:") for item in specs)


def test_alias_covers_expected_field():
    """
    Treat known aliases as covering the canonical catalog field.

    """
    device = DeviceStatsModel(
        name="ПД100",
        manufacturer="ОВЕН",
        specs={"выходной сигнал": "4…20 мА", "напряжение питания": "12…36 В"},
        description="Преобразователь давления.",
    )
    missing = collect_missing(device)
    assert "интерфейсы" not in missing
    assert "питание" not in missing


def test_web_specs_fill_gaps_only():
    """
    Fill empty specs from the web without overwriting input values.

    """
    device = DeviceStatsModel(
        name="Манометр DM 02",
        manufacturer="Манотомь",
        specs={"диапазон измерений": "0–1,6 МПа"},
        description="Для избыточного давления.",
    )
    enriched = apply_web_specs(
        device,
        {
            "диапазон измерений": "0–25 МПа",
            "производитель": "Другой завод",
            "питание": "нет (механический)",
            "класс точности": "1,5",
        },
    )
    assert enriched.manufacturer == "Манотомь"
    assert enriched.specs["диапазон измерений"] == "0–1,6 МПа"
    assert enriched.specs["питание"] == "нет (механический)"
    assert enriched.specs["класс точности"] == "1,5"


def test_empty_spec_values_are_treated_as_missing():
    """
    Drop blank spec values so the catalog field stays in ``data_missing``.

    """
    device = DeviceStatsModel(
        name="Манометр X",
        specs={"диапазон измерений": "  ", "класс точности": "1,5"},
    )
    assert "диапазон измерений" not in device.specs
    assert device.specs["класс точности"] == "1,5"
    missing = collect_missing(device)
    assert "диапазон измерений" in missing
    lines = build_key_specs(device)
    assert "класс точности: 1,5" in lines
    assert not any("диапазон измерений" in item for item in lines)
    assert not any("данных нет" in item for item in lines)


def test_json_specs_all_appear_in_key_specs():
    """
    Copy every non-empty input spec into ``key_specs`` without catalog fillers.

    """
    device = DeviceStatsModel(
        name="Сигнализатор загазованности CO+CH4-109 (вариант 280)",
        manufacturer="Seitron",
        specs={
            "контролируемый газ": "CO+CH4",
            "диапазон измерений": "0–200 мг/м³",
            "пороги срабатывания": "20 / 100 мг/м³",
            "тип сенсора": "электрохимический",
            "степень защиты": "IP54",
            "питание": "220 В",
            "выходные сигналы": "реле, звуковая и световая сигнализация",
        },
        description=(
            "Стационарный сигнализатор для непрерывного контроля концентрации "
            "CO+CH4 в воздухе помещений."
        ),
    )
    lines = build_key_specs(device)
    joined = "\n".join(lines)
    assert "производитель: Seitron" in joined
    assert "контролируемый газ: CO+CH4" in joined
    assert "диапазон измерений: 0–200 мг/м³" in joined
    assert "пороги срабатывания: 20 / 100 мг/м³" in joined
    assert "тип сенсора: электрохимический" in joined
    assert "степень защиты: IP54" in joined
    assert "питание: 220 В" in joined
    assert "выходные сигналы: реле, звуковая и световая сигнализация" in joined
    missing = collect_missing(device)
    assert "питание" not in missing
    assert "степень защиты" not in missing
    assert "интерфейсы" not in missing
    assert "класс точности" in missing
    assert not any("данных нет" in item for item in lines)
    assert not any(item.startswith("класс точности:") for item in lines)
    assert not any(item.startswith("тип корпуса:") for item in lines)
    assert not any(item.startswith("присоединение:") for item in lines)


def test_merge_key_specs_keeps_input_and_drops_inventions():
    """
    Preserve input specs, honor LLM order, and drop invented values.

    """
    device = DeviceStatsModel(
        name="Сигнализатор загазованности CO+CH4-109",
        manufacturer="Seitron",
        specs={
            "контролируемый газ": "CO+CH4",
            "питание": "220 В",
            "степень защиты": "IP54",
        },
        description="Стационарный сигнализатор для контроля CO+CH4.",
    )
    merged = merge_key_specs(
        device,
        [
            "питание: 380 В",
            "контролируемый газ: CO+CH4",
            "давление: 25 МПа",
            "класс точности: данных нет",
            "исполнение: стационарный",
        ],
    )
    joined = "\n".join(merged)
    assert "контролируемый газ: CO+CH4" in joined
    assert "питание: 220 В" in joined
    assert "степень защиты: IP54" in joined
    assert "производитель: Seitron" in joined
    assert "380 В" not in joined
    assert "25 МПа" not in joined
    assert "данных нет" not in joined
    assert merged[0] == "питание: 220 В"
    assert "исполнение: стационарный" in joined
    missing = merge_data_missing(device, ["питание", "межповерочный интервал"])
    assert missing == ["межповерочный интервал"]


def test_merge_key_specs_keeps_web_finds():
    """
    Put search fills onto the card without overwriting input values.

    """
    device = DeviceStatsModel(
        name="Сигнализатор загазованности CO+CH4-109",
        manufacturer="Seitron",
        specs={
            "контролируемый газ": "CO+CH4",
            "питание": "220 В",
        },
        description="Стационарный сигнализатор для контроля CO+CH4.",
    )
    merged = merge_key_specs(
        device,
        ["контролируемый газ: CO+CH4"],
        web_specs={
            "присоединение": "M20x1.5",
            "класс точности": "1,5",
            "питание": "380 В",
        },
    )
    joined = "\n".join(merged)
    assert "контролируемый газ: CO+CH4" in joined
    assert "питание: 220 В" in joined
    assert "380 В" not in joined
    assert "присоединение: M20x1.5" in joined
    assert "класс точности: 1,5" in joined
    omitted = merge_key_specs(
        device,
        ["питание: 220 В"],
        web_specs={"тип корпуса": "пластиковый"},
    )
    assert "тип корпуса: пластиковый" in omitted
    assert "производитель: Seitron" in omitted


def test_web_specs_fill_empty_manufacturer():
    """
    Copy manufacturer and description from web facts when they are empty.

    """
    device = DeviceStatsModel(name="ПД100")
    enriched = apply_web_specs(
        device,
        {"производитель": "ОВЕН", "описание": "Преобразователь давления."},
    )
    assert enriched.manufacturer == "ОВЕН"
    assert enriched.description == "Преобразователь давления."
    assert "производитель" not in collect_missing(enriched)
    assert "текстовое описание" not in collect_missing(enriched)

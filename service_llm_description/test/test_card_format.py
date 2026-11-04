from src.domain.card_format import (
    clip_description,
    clip_title,
    extract_json_object,
    to_markdown,
    unexpected_numbers,
)
from src.repository.llm_client import _message_text, _trust_env_for
from src.schemas import AnswerLLMModel


def test_extract_json_from_fenced_and_think_block():
    """
    Parse a JSON object wrapped in a think block and a markdown fence.

    """
    raw = """<think>рассуждение</think>
```json
{"title": "Ок", "description": "текст"}
```
"""
    assert extract_json_object(raw)["title"] == "Ок"


def test_extract_json_from_unclosed_think_block():
    """
    Keep JSON that follows an unclosed ``<think>`` dump.

    """
    raw = """<think>модель не закрыла блок и потратила лимит
{"title": "Ок", "description": "текст"}
"""
    assert extract_json_object(raw)["title"] == "Ок"


def test_extract_json_ignores_trailing_junk():
    """
    Keep the first complete object when the model keeps generating.

    """
    raw = '{"title": "Ок", "description": "текст"} вот ещё текст {'
    assert extract_json_object(raw)["title"] == "Ок"


def test_extract_json_repairs_truncated_object():
    """
    Close a JSON object cut off inside a string.

    """
    raw = '{"title": "Ок", "description": "текст без конца'
    payload = extract_json_object(raw)
    assert payload["title"] == "Ок"
    assert "текст без конца" in payload["description"]


def test_extract_json_rejects_think_without_object():
    """
    Raise when the reply is only an unclosed think dump.

    """
    try:
        extract_json_object("<think>только рассуждение без JSON")
    except ValueError as exc:
        assert "no JSON object" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_message_text_joins_parts_and_reasoning():
    """
    Read string content, list parts, and reasoning_content fallback.

    """
    assert _message_text({"content": '{"ok": true}'}) == '{"ok": true}'
    assert (
        _message_text({"content": [{"type": "text", "text": '{"a": 1}'}]})
        == '{"a": 1}'
    )
    assert _message_text({"content": "", "reasoning_content": '{"ok": 1}'}) == '{"ok": 1}'
    assert _message_text({"content": None}) == ""


def test_clip_title_keeps_words():
    """
    Cut an overlong title on a word boundary within 70 characters.

    """
    title = "Манометр показывающий DM 02 Манотомь диапазон 0–1,6 МПа для теплоснабжения"
    clipped = clip_title(title)
    assert len(clipped) <= 70
    assert "Манометр" in clipped


def test_clip_description_stays_in_limit():
    """
    Keep a clipped description between 800 and 1200 characters.

    """
    text = ("Предложение одно. " * 80).strip()
    clipped = clip_description(text, 1200)
    assert 800 <= len(clipped) <= 1200


def test_unexpected_numbers_ignore_source_values():
    """
    Flag numbers that are not present in the source facts.

    """
    source = "диапазон 1,6 МПа класс 1,5"
    output = "Манометр 1,6 МПа, класс 1.5, гарантия 36 месяцев"
    extra = unexpected_numbers(output, source)
    assert "36" in extra
    assert "1.6" not in extra


def test_markdown_contains_sections():
    """
    Render required Markdown headings and optional search sources.

    """
    card = AnswerLLMModel(
        title="Манометр DM 02",
        description="А" * 800,
        key_specs=["класс точности: 1,5"],
        search_queries=["манометр dm 02", "манометр манотомь", "манометр 1,6"],
        commercial_block="Для теплосетей.",
        data_missing=["питание"],
    )
    md = to_markdown(card)
    assert md.startswith("# Манометр DM 02")
    assert "## SEO-описание" in md
    assert "питание" in md
    md_src = to_markdown(card, sources=["https://example.com/passport"])
    assert "## Источники поиска" in md_src
    assert "example.com/passport" in md_src


def test_httpx_skips_proxy_for_docker_and_loopback():
    """
    Disable proxy env for Docker DNS and loopback, keep it for public hosts.

    """
    assert _trust_env_for("http://llm:8686/v1") is False
    assert _trust_env_for("http://127.0.0.1:8686/v1") is False
    assert _trust_env_for("http://localhost:8686/v1") is False
    assert _trust_env_for("https://api.openai.com/v1") is True

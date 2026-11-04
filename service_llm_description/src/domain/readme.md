# Domain

Детерминированные правила карточки. Сюда не ходят HTTP и LangGraph.

- `facts.py` — входные `specs`, веб-находки, `merge_key_specs`, без строк «данных нет».
- `prompt.py` — системный и пользовательский промпт writer.
- `card_format.py` — разбор JSON модели, обрезка title/description, Markdown.
- `exceptions.py` — ошибки пайплайна (`LLMUnavailableError`, `LLMBadResponseError`).

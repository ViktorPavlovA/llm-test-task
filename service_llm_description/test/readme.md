# Тесты для данного сервиса

Юнит-тесты `service_llm_description`. Живую модель и интернет не вызывают: LLM подменяется, поиск и HTTP режутся заглушками.

Запуск из каталога сервиса:

```bash
cd service_llm_description
pytest
```

Из корня репозитория, если установлен venv:

```bash
.venv/bin/pytest service_llm_description/test
```

Конфиг — `pytest.ini` в корне сервиса: `pythonpath = .`, `testpaths = test`, `asyncio_mode = auto`. Сейчас 42 теста.

## Файлы

| Файл | Что проверяет |
| --- | --- |
| `test_facts.py` | Сборка характеристик. Все входные `specs` попадают в `key_specs`, пустые поля и «данных нет» не выводятся. Веб-находки дополняют карточку и не перезаписывают вход. Выдуманные значения модели отбрасываются. |
| `test_card_format.py` | Разбор JSON ответа модели: `<think>`, markdown-fence, незакрытый think, мусор после объекта, обрезанный JSON. Обрезка title ≤ 70 и description 800–1200. Markdown-секции карточки. Числа, которых не было во входных фактах. Прокси для Docker DNS и loopback выключен. |
| `test_tools.py` | SSRF: `fetch_page` не ходит на private/loopback. HTML без script/style. Пустой поисковый запрос. Из имени снимается `(вариант N)`. Запросы и скоринг предпочитают паспорт/сайт производителя. У аналоговых манометров не ищут питание и интерфейсы, у электронных преобразователей — ищут. Текст из PDF. Маршрут research vs write по флагу `LLM_SEARCH_INTERNET`. |
| `test_service.py` | Сквозной `get_description_service` с подменённым LLM: выдуманные 25 МПа / 220 В не попадают в карточку. Writer-промпт содержит все входные specs и требует web_facts в `key_specs`. `write_node` дописывает веб-поля, даже если модель их не вернула. API 422 на пустое имя. Во входных полях режутся jailbreak/system-prompt, обычный текст про «систему измерения» проходит. |
| `test_settings.py` | `LLM_SEARCH_INTERNET=true/false` и `SEARCH_CONTEXT_CHARS` читаются из окружения. |

## Фикстуры

`json_requests/synthetic_products_500.json` — 500 синтетических карточек КИП для ручной и живой проверки поиска. Pytest этот файл сам не гоняет.

Живой прогон против llama (`POST /api/description`) в этот набор не входит.
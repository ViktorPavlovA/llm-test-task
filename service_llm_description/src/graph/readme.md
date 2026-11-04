# Graph

LangGraph-пайплайн карточки: `START → research? → write → END`.

- `graph.py` — состояние, узлы research/write, `build_card`.
- `research.py` — запросы, скоринг выдачи, HTML/PDF, извлечение `found_specs`.

Вызов модели — через `src.repository.llm_client`. Веб — через `src.repository.tools`. Склейка характеристик — `src.domain.facts`.

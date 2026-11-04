# API

HTTP-слой FastAPI.

- `form_router.py` — страница формы `/`.
- `health_router.py` — `GET /health`.
- `description_router.py` — `POST /api/description`.

Обработчик 422 живёт на приложении в `main.py`, не на роутере.

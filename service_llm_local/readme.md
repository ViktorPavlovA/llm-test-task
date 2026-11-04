# Локальный LLM-сервер 

Загружает и запускает LLM-модель на ваш компьютер:
Модель: [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF).



## API

```bash
curl http://127.0.0.1:8686/health

curl http://127.0.0.1:8686/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-llm",
    "messages": [{"role": "user", "content": "Привет"}],
    "temperature": 0.6,
    "max_tokens": 128
  }'
```

Инструмент:

```bash
curl http://127.0.0.1:8686/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-llm",
    "messages": [{"role": "user", "content": "Сколько сейчас времени по МСК?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "current_time",
        "description": "Текущие дата и время в IANA-зоне, для Москвы Europe/Moscow.",
        "parameters": {
          "type": "object",
          "properties": {
            "tz": {"type": "string", "description": "IANA timezone"}
          }
        }
      }
    }],
    "tool_choice": "auto"
  }'
```
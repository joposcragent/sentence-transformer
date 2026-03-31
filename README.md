# sentence-transformer (REST)

Сервис векторизации текста на базе [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) через [sentence-transformers](https://www.sbert.net/), [FastAPI](https://fastapi.tiangolo.com/) и [Uvicorn](https://www.uvicorn.org/).

Контракт API описан в репозитории спецификаций: `specifications/services/sentence-transformer/openapi.yaml` (базовый путь `/sentence-transformer`).

## Требования

- Python 3.11+
- Переменные окружения (или файл `.env` в корне проекта) — см. `.env.example`

## Разработка (venv)

```bash
cd app/sentence-transformer
python -m venv .venv
# Windows:
source .venv/Scripts/activate
# Linux/macOS:
# source .venv/bin/activate

pip install -U pip
pip install -e ".[dev]"
```

Скопируйте `.env.example` в `.env` и при необходимости измените `ST_MODEL_NAME`, `ST_MODEL_CACHE_DIR`, `ST_DEVICE`, `ST_HOST`, `ST_PORT`.

Запуск API:

```bash
python -m uvicorn st_service.main:app --host 0.0.0.0 --port 8000
```

Либо через entrypoint:

```bash
st-service
```

При первом запуске модель загрузится с Hugging Face (или из локального кэша). Убедитесь, что задан достаточный `ST_MODEL_CACHE_DIR` и доступ в интернет, если модели ещё нет в кэше.

Проверка здоровья:

```bash
curl http://localhost:8000/sentence-transformer/health
```

Тесты и покрытие:

```bash
pytest tests/ --cov=st_service --cov-report=term-missing --cov-fail-under=70
```

(При необходимости задайте `PYTHONPATH=src`, если пакет установлен без editable-режима.)

## Продакшен (Docker)

Сборка образа из каталога `app/sentence-transformer`:

```bash
docker build -t sentence-transformer:latest .
```

Запуск (порт 8000, модель и кэш через переменные окружения):

```bash
docker run --rm -p 8000:8000 \
  -e ST_MODEL_NAME=BAAI/bge-m3 \
  -e ST_MODEL_CACHE_DIR=/app/.cache \
  -e ST_DEVICE=cpu \
  sentence-transformer:latest
```

Рекомендуется смонтировать том для кэша модели, чтобы не скачивать веса при каждом старте:

```bash
docker run --rm -p 8000:8000 \
  -v st_hf_cache:/app/.cache \
  sentence-transformer:latest
```

Образ запускает процесс от непривилегированного пользователя `appuser`; каталог кэша по умолчанию — `/app/.cache`.

## Переменные окружения


| Переменная           | Описание                                                                                                                    |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `ST_MODEL_NAME`      | Идентификатор модели в Hugging Face или путь к локальным весам (по умолчанию `BAAI/bge-m3`)                                 |
| `ST_MODEL_CACHE_DIR` | Каталог кэша загрузки модели (`cache_folder` в `SentenceTransformer`)                                                       |
| `ST_DEVICE`          | `cpu` или `cuda` / `cuda:0`                                                                                                 |
| `ST_HOST`            | Адрес привязки Uvicorn (в Docker используется `0.0.0.0`)                                                                    |
| `ST_PORT`            | Порт (в контейнере CMD зафиксирован на 8000; смена порта — через `docker run -p` и переопределение `CMD` при необходимости) |



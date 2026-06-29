FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    DATABASE_URL=sqlite+pysqlite:////app/data/rag_project.db

WORKDIR /app

RUN addgroup --system rag \
    && adduser --system --ingroup rag rag \
    && mkdir -p /app/data \
    && chown rag:rag /app /app/data

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY .env.example ./.env.example

USER rag

EXPOSE 8000

CMD ["uvicorn", "rag_project.main:app", "--host", "0.0.0.0", "--port", "8000"]

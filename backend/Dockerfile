FROM python:3.12.8-slim-bookworm as builder

RUN apt-get update \
    && apt-get install -y \
        curl \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.5 \
    POETRY_HOME=/opt/poetry \
    PATH="/opt/poetry/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry --version

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry export --without-hashes -f requirements.txt -o requirements.txt

FROM python:3.12.8-slim-bookworm AS runner

WORKDIR /app

COPY --from=builder /app/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src

WORKDIR /app/src

ENV PYTHONUNBUFFERED=1
ENV IS_PROD=1

CMD ["uvicorn", "entrypoint:app", "--host", "0.0.0.0", "--port", "80"]

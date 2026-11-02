FROM python:3.13-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
ENV LLM_BASE_URL=http://host.docker.internal:11434/v1
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

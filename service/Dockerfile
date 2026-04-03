FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY service ./service
COPY artifacts ./artifacts
COPY .env.example ./.env.example

ENV PYTHONPATH=/app/src:/app

CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10-slim

WORKDIR /app

COPY service/requirements-service.txt ./service/requirements-service.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
RUN pip install --no-cache-dir -r ./service/requirements-service.txt

COPY src ./src
COPY service ./service
COPY artifacts ./artifacts
COPY .env.example ./.env.example

ENV PYTHONPATH=/app/src:/app

CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

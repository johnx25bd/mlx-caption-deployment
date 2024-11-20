
# Use an official Python runtime as a parent image
FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt --verbose

COPY api /app/

EXPOSE 8000

ENV PYTHONDEVMODE=1
ENV UVICORN_LOG_LEVEL=debug
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY scripts/mocks/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY scripts/mocks/ /app/scripts/mocks/
ENV PYTHONPATH=/app

CMD ["python", "-m", "scripts.mocks"]

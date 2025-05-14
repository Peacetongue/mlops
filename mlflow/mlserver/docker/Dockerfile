FROM python:3.11.4-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir && \
    rm -rf /root/.cache/pip

COPY . .
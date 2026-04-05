FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y \
    build-essential \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

COPY cluster_analytics/requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir "numpy>=1.24.0,<2.0.0" scipy matplotlib && \
    pip install --no-cache-dir -r requirements.txt

CMD ["tail", "-f", "/dev/null"]

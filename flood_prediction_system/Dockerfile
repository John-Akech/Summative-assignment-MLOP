FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY src/requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy application code
COPY src/app.py .
COPY src/static /app/static
COPY src/templates /app/templates

# Create data directory and copy data file
RUN mkdir -p /app/data
COPY data/flood_processed.csv /app/data/

EXPOSE 5000
CMD ["python", "app.py"]
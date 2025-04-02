# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any are needed)
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p data/train models static templates

# Copy application files
COPY app.py .
COPY static/ static/
COPY templates/ templates/

# Copy training data (ensure this path matches your local structure)
COPY data/train/flood_processed.csv data/train/

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
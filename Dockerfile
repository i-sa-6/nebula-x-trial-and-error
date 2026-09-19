# Google Cloud Run Container Specification for NebulaX
FROM python:3.11-slim

WORKDIR /app

# Set environment variables for Cloud Run
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV CONTAINER=1

# Install required build tools if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install minimal python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application, champion model, and test predictions
COPY src/ ./src/
COPY models/ ./models/
COPY app/ ./app/
COPY submission/ ./submission/
COPY predict.py .

# Prepare data directories
RUN mkdir -p data/Test data/Train

# Pre-populate sample test files for instant cloud demo
COPY data/Test/Test1.csv data/Test/
COPY data/Test/Test2.csv data/Test/
COPY data/Test/Test11.csv data/Test/

EXPOSE 8080

CMD ["python", "app/run_app.py", "8080"]

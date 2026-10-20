# Base python image
FROM python:3.11-slim

WORKDIR /app

# Install basic system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU (lightweight ~150MB, portable on any host CPU/Mac/Windows/Linux)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY Plant_Disease/requirements.txt /app/requirements.txt
RUN grep -v "torch" /app/requirements.txt | grep -v "extra-index-url" > /app/req_clean.txt && \
    pip install --no-cache-dir -r /app/req_clean.txt

# Copy source code and pre-trained artifacts
COPY preprocessing /app/preprocessing
COPY Plant_Disease /app/Plant_Disease

WORKDIR /app/Plant_Disease

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]


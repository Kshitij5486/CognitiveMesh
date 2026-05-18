FROM python:3.13-slim

LABEL maintainer="Kshitij Srivastava"
LABEL description="CognitiveMesh v1.0.0 — Distributed Byzantine-Resilient Causal Computing Fabric"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pillar1-causal/ ./pillar1-causal/
COPY .env.docker .env

# Expose all service ports
EXPOSE 8080 8081 8082 8083 8084 8085 8086 8087 8088 8089

# Health check via Byzantine Recovery API
HEALTHCHECK --interval=30s --timeout=10s \
            --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8089/health || exit 1

# Entrypoint
CMD ["python", "pillar1-causal/causal/byzantine_recovery_api.py"]
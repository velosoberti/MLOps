# Multi-stage Dockerfile for ML Pipeline Flask API
# Uses uv for fast, reproducible dependency installation

# =============================================================================
# Stage 1: Build stage - Install dependencies
# =============================================================================
FROM python:3.10-slim AS builder

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (production only, no dev dependencies)
RUN uv sync --frozen --no-dev --no-install-project

# =============================================================================
# Stage 2: Runtime stage - Minimal production image
# =============================================================================
FROM python:3.10-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # Default configuration (can be overridden at runtime)
    ML_API_HOST=0.0.0.0 \
    ML_API_PORT=5005 \
    ML_API_DEBUG=false

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY config/ ./config/
COPY src/ ./src/
COPY api/ ./api/
COPY etl_functions/ ./etl_functions/

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 5005

# Health check - verify API is responding
# Checks /health endpoint every 30 seconds, with 10s timeout
# Allows 3 retries before marking unhealthy, starts checking after 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5005/health', timeout=5)" || exit 1

# Run the Flask API
# Using python directly instead of flask run for better control
CMD ["python", "-m", "api.api"]

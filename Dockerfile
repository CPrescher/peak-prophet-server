# Stage 1: Build dependencies
FROM python:3.11 AS builder

# Install Poetry
ENV POETRY_VERSION=1.8.2 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN curl -sSL https://install.python-poetry.org | python3 -

# Set the path to Poetry
ENV PATH="$POETRY_HOME/bin:$PATH"

# Create and set work directory
WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --only main --no-root

# Stage 2: Build runtime image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

# Copy Poetry from builder
COPY --from=builder /opt/poetry /opt/poetry
COPY --from=builder /app/.venv /app/.venv

# Set the path to Poetry and Virtual Environment
ENV PATH="/app/.venv/bin:$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install project
RUN poetry install --only main --no-dev --no-interaction --no-root

# Set default port
ENV PORT=8009

# Expose the port (uses default if PORT not set during runtime)
EXPOSE ${PORT}

# Command to run the server with uvicorn using the PORT environment variable
CMD uvicorn run:app --host 0.0.0.0 --port ${PORT}
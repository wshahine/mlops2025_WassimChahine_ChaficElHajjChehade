# Requirement: Use Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional, but good practice)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Requirement: Install dependencies using uv
# We install uv first
RUN pip install uv

# Copy necessary project files
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY README.md .

# Create a virtual environment and sync dependencies using uv
# Requirement: Install the ml-project package
RUN uv venv .venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies and the project itself in editable mode
RUN uv pip install -e .

# Copy the entrypoint script (we will create this next)
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set the entrypoint to handle commands
ENTRYPOINT ["./entrypoint.sh"]

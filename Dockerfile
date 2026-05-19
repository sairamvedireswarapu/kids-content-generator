# Start with Python 3.11
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install ffmpeg (needed by moviepy)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Copy dependency files first (faster rebuilds)
COPY pyproject.toml .
COPY uv.lock .

# Install all Python dependencies
RUN uv sync --frozen


# Copy your source code
COPY main.py .
COPY nodes.py .
COPY graph.py .
COPY video_generator.py .
COPY mcp_server.py .
COPY api.py .
COPY index.html .

# Expose port
EXPOSE 8080

# Run the server

CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
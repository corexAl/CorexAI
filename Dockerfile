# Use a lightweight Python image
FROM python:3.12-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Disable output buffering
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install build tools (some Python packages require them)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for Docker cache
COPY requirements.txt .
COPY requirements-dev.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project
COPY . .

# Default command
CMD ["python", "main.py"]

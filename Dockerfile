FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files
# and to ensure stdout/stderr are flushed immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Install system dependencies (required for some Python packages)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements and setup files first for caching
COPY pyproject.toml requirements.txt ./
COPY backend/requirements.txt ./backend/

# Install dependencies from all requirements files
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy all the application code into the container
COPY . .

# Install the app itself in editable mode so part1, bonus, shared are accessible
RUN pip install -e .

# Expose the API port
EXPOSE $PORT

# Run the FastAPI server via Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

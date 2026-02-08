# Use an official lightweight Python image.
FROM python:3.13-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Debug: Show what files are actually in /app
RUN echo "=== CONTENTS OF /app ===" && \
    ls -la /app && \
    echo "=== SEARCHING FOR main.py ===" && \
    find /app -name "main.py" -o -name "*.py" | head -20

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the application
ENTRYPOINT ["streamlit", "run", "main_auth.py", "--server.port=8501", "--server.address=0.0.0.0"]

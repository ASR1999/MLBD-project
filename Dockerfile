# Choose appropriate Python base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
# --no-cache-dir reduces image size
# Consider platform-specific build args if needed
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on (choose one, e.g., 8000)
# Some platforms override this (like HF Spaces uses 7860 often)
EXPOSE 8000

# Command to run the application using Gunicorn
# Use environment variables for host/port if needed by platform
# Example: CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
# Check platform docs for recommended CMD/ENTRYPOINT
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:${PORT:-8000}"]
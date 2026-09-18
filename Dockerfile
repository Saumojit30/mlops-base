# Use lightweight official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Prevent Python from writing .pyc files & buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies first (leverage Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files into container
COPY . .

# Expose ports for Gradio UI (7860) and High-Concurrency FastAPI Microservice (8000)
EXPOSE 7860
EXPOSE 8000

# Default command launches Gradio UI.
# To launch the high-concurrency FastAPI microservice instead:
#   docker run -p 8000:8000 <image> uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
CMD ["python", "app.py"]

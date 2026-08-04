FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIDEO_SOURCE=0

WORKDIR /app

# Install system dependencies required for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY edge_app/requirements.txt ./edge_app/requirements.txt
RUN pip install --no-cache-dir -r ./edge_app/requirements.txt

# Copy application files
COPY edge_app/ ./edge_app/
COPY dashboard/ ./dashboard/

# Create uploads directory
RUN mkdir -p /app/edge_app/uploads

WORKDIR /app/edge_app

EXPOSE 5000

CMD ["python", "server.py"]

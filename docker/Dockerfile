# Dockerfile for Prithvi-EO-Segmentation
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY ../requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# (Optional) Install Jupyter for notebook support
RUN pip install jupyter

# Copy project files
COPY .. .

# Set environment variables (if needed)
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["bash"]

# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install PyMuPDF dependencies
# For PyMuPDF, you might need some system libraries like freetype, libjpeg, etc.
# Check PyMuPDF's documentation for specific system dependencies if you encounter issues.
# Often, `python:slim-buster` has most common ones, but sometimes not.
# If issues, you might need: 
RUN apt-get update && apt-get install -y --no-install-recommends \
libfreetype6-dev libjpeg-dev zlib1g-dev \
&& rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Command to run your main script
# This script will then handle reading from /app/input and writing to /app/output
CMD ["python", "process_pdfs.py"]
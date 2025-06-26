# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt ./
COPY faceapp/requirements.txt ./faceapp/requirements.txt
COPY Silent-Face-Anti-Spoofing-master/requirements.txt ./Silent-Face-Anti-Spoofing-master/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -r faceapp/requirements.txt \
    && pip install -r Silent-Face-Anti-Spoofing-master/requirements.txt

# Copy backend code
COPY . .

# Expose port for Flask
EXPOSE 5000

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Run the backend
CMD ["python", "faceapp/backend.py"] 
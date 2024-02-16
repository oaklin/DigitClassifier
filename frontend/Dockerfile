# Use Python 3.11 slim image as the base
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app/

# Copy the source code from the host machine to the container's /app directory
COPY ./src /app

# Create a directory for configuration files
RUN mkdir -p /app/config

# Install system dependencies required for the application
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Install Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a new user 'appuser' and assign ownership of the /app directory to it
# This enhances security by running the application as a non-root user
RUN useradd appuser && chown -R appuser /app
USER appuser

# Expose port 8501 to allow access from outside the container
EXPOSE 8501

# Set environment variable for backend URL
ENV URL_BACKEND="http://localhost:8000/"

# Command to run Streamlit when the container starts
CMD ["streamlit", "run", "main.py"]
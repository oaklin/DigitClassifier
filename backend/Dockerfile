# Set the base image for the new image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Create a directory for the cache folder
RUN mkdir -p /app/.cache/huggingface/hub

# Set environment variable for the cache directory
ENV TRANSFORMERS_CACHE /app/.cache/huggingface/hub

# Copy the current directory contents into the container at /app
COPY ./src /app

# Install the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a new user 'appuser' and switch to it
# This user will run the application in the container
RUN useradd appuser && chown -R appuser /app
USER appuser

# Set environment variables for the server
ENV MPLCONFIGDIR /app/config
ENV MNIST_MODEL_PATH ./model/vit-base-mnist

# Cache the predict model by running it once
RUN python -m model.predict

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
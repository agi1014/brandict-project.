# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies needed by OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python packages (using the correct --no-cache-dir flag)
RUN pip install --no-cache-dir -r requirements.txt

# Add the application directory to the PYTHONPATH
ENV PYTHONPATH=/app

# Copy your application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run your app
CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8000"]
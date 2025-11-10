# Use the stable and well-supported Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables to prevent Python from creating .pyc files and buffering logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code and the data/model artifacts
# NOTE: You must run train.py first to generate the joblib files locally

COPY crop_yield.csv .
COPY train.py .
COPY predict.py .
# COPY models /app/models

# Run the training script to generate the model artifacts inside the container
# This ensures the model is generated fresh during the build process

RUN python train.py

# Expose the port that Flask will run on (8080 is common for Docker web services)

EXPOSE 8080

# Command to run the application

CMD ["python", "predict.py"]
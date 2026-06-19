# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces require the app to run on port 7860
EXPOSE 7860

# Run FastAPI using uvicorn on port 7860 (without modifying api.py!)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]

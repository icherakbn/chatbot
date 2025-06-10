# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to cache dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

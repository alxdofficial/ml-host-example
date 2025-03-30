FROM python:3.10-slim


# Install system dependencies needed by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application files
COPY . .

# Expose the Flask port
EXPOSE 5000

# Command to start the Flask application
CMD ["python", "app.py"]

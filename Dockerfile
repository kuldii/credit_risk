FROM python:3.10-slim

WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (including app.py, models/, etc.)
COPY . .

# Run the app
CMD ["python", "app.py"]

# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Run gunicorn
CMD ["gunicorn", "tc5_fiap.wsgi:application", "--bind", "0.0.0.0:8000"]
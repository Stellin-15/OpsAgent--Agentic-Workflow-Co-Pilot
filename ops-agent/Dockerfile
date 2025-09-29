# Start with an official Python image
FROM python:3.11-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Copy and install the Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of our application code
COPY . .

# Tell Docker our app runs on port 8000
EXPOSE 8000

# The command to start our web server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
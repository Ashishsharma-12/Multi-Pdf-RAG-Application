# Use a minimal Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file to the container
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the entire application to the container
COPY . .

# Use JSON array syntax for CMD
CMD ["streamlit", "run", "rag.py", "0.0.0.0:8501"]
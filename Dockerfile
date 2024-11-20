# Use a base image with Python 3.9
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the local project files into the container
COPY . /app/

# Copy the Kaggle credentials to the appropriate location
COPY kaggle.json /root/.kaggle/kaggle.json

# Set permissions (recommended for security)
RUN chmod 600 /root/.kaggle/kaggle.json

# Install Docker CLI tools (optional)
RUN apt-get update && apt-get install -y \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Install numpy first to avoid compatibility issues with Metaflow
RUN pip install --no-cache-dir numpy==1.23.5

# Install required Python packages from requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install Metaflow with optional services you might need (adjust services if necessary)
RUN pip install --no-cache-dir "metaflow[all]"

# Install spaCy model separately after numpy installation
RUN python -m spacy download en_core_web_sm

# Expose the Metaflow UI port (adjust this if necessary)
EXPOSE 8080

# Set the entrypoint or default command
# Ensure this points to your main script, adjust the filename accordingly
CMD ["python", "/app/external/scripts/main_pipeline.py"]

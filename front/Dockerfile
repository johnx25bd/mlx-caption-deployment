# Use the official Debian image
FROM python:3.12

# Install Python and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    gcc \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py"]
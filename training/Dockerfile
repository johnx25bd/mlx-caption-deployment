FROM python:3.11-slim

# Install readline and other useful terminal tools
RUN apt-get update && apt-get install -y \
    readline-common \
    libreadline-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

# Keep container running
CMD ["tail", "-f", "/dev/null"]
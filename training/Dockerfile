# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

CMD ["python", "-m", "train"]
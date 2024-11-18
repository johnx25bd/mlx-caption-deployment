# FastAPI API Container

This is a containerized FastAPI project designed to quickly start building APIs with Python. The project is structured with modularity and scalability in mind, and it uses Poetry for dependency management.

## Project Structure

```plaintext
api/
├── __init__.py
├── main.py
├── routes.py
└── models/
    └── __init__.py
```

## Prerequisites

- Python 3.7+
- [Poetry](https://python-poetry.org/) (for dependency management)
- Docker (for containerization)

## Setup Instructions

### Clone the Repository

### Create and Activate a Virtual Environment

Poetry automatically manages virtual environments. To create and activate one, simply run:

```bash
poetry shell
```

### Install Dependencies

Install all necessary packages using Poetry:

```bash
poetry install
```

### Environment Variables (Optional)

If your application requires environment variables, set them up in a `.env` file in the root directory.

Example `.env`:

```plaintext
DATABASE_URL=postgresql://user:password@localhost/dbname
SECRET_KEY=your_secret_key
```

## Running the Server Locally

Start the FastAPI server using Uvicorn:

```bash
poetry run uvicorn main:app --reload
```

- `main:app` specifies the `app` instance in `main.py`.
- `--reload` enables hot-reloading, useful for development.

After starting the server, you should see the output indicating that it’s running at `http://127.0.0.1:8000`.

## API Documentation

FastAPI automatically generates interactive documentation for your API:

- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## Testing the API

Once the server is running, you can test the API endpoints by visiting `http://127.0.0.1:8000/docs`, where you’ll find interactive documentation to execute requests.

Alternatively, use `curl` or a tool like **Postman** to test your API endpoints.

Example request with `curl`:

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/' \
  -H 'accept: application/json'
```

## Deployment

For production, it’s recommended to run Uvicorn with a process manager like **Gunicorn** and additional workers:

```bash
gunicorn -k uvicorn.workers.UvicornWorker main:app --workers 4
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
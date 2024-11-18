# Search App, Containerized

## Quickstart

- Copy the four data files from this [Google Drive folder](https://drive.google.com/drive/folders/16aTFOtnjFngyJLgHIHA8S8n23hF9tItP?usp=drive_link) into the `/api/data` directory.
- Run `docker-compose up --build` to start the services.
- Access the application at `http://localhost:80`.

This repository contains a containerized web application using FastAPI for the backend API, Streamlit for the frontend, and Nginx as a reverse proxy. The application is orchestrated using Docker Compose.

## Architecture

Client Request → Nginx (Port 80) → FastAPI (Port 8000)
→ Streamlit (Port 8501)

- **Nginx**: Reverse proxy that routes requests to the appropriate service
- **FastAPI**: Backend API service
- **Streamlit**: Frontend web application

## Prerequisites

- Docker and Docker Compose
- Git
- Python 3.12 (for local development)

## Project Structure

```
.
├── README.md
├── docker-compose.yml
├── api/
│ ├── Dockerfile
│ ├── data/ # 👇 these need to be downloaded from the Google Drive folder
│ │ ├── word-vector-embeddings.model
│ │ ├── two_tower_state_dict.pth
│ │ ├── doc-index-64.faiss
│ │ └── training-with-tokens.parquet
│ ├── app/
│ │ ├── main.py
│ │ └── ...
│ └── requirements.txt
├── streamlit/
│ ├── Dockerfile
│ ├── app.py
│ └── requirements.txt
└── nginx/
|    ├── Dockerfile
|    └── conf/
|    └── nginx.conf
```

## Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Build and start the containers:
   ```bash
   docker-compose up --build
   ```

3. Access the applications:
   - Frontend (Streamlit): `http://localhost/`
   - API (FastAPI): `http://localhost/api/`

## Development

### Local Development Setup

1. Set up Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   cd api && pip install -r requirements.txt
   cd ../streamlit && pip install -r requirements.txt
   ```

3. Run services locally:
   - FastAPI:
     ```bash
     cd api
     uvicorn app.main:app --reload
     ```
   - Streamlit:
     ```bash
     cd streamlit
     streamlit run app.py
     ```

### Docker Commands

- Build containers:
  ```bash
  docker-compose build
  ```

- Start services:
  ```bash
  docker-compose up
  ```

- Stop services:
  ```bash
  docker-compose down
  ```

## Deployment

1. Push changes to repository
2. SSH into your server
3. Pull latest changes:
   ```bash
   git pull
   ```
4. Build and start containers:
   ```bash
   docker-compose up --build -d
   ```

## Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Docker Documentation](https://docs.docker.com/)

## Known Issues

- Streamlit may have connectivity issues in Brave browser due to privacy settings
  - Solution: Disable Brave Shields for the site or use a different browser

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## MIT License

Copyright (c) 2024 johnx25bd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

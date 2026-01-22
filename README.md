# Phishing Detection Backend

FastAPI backend for AI-powered phishing detection using DistilBERT.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
uvicorn app.main:app --reload
```

API will be available at `http://localhost:8000`

## Docker

Build and run with Docker:
```bash
docker build -t phishing-backend .
docker run -p 8000:8000 phishing-backend
```

## API Endpoints

- `POST /scan` - Scan text or URL for phishing
- `GET /history` - Get scan history
- `POST /clear-history` - Clear scan history
- `POST /feedback` - Submit feedback on scan results

## Features

- Uses DistilBERT for lightweight, fast phishing detection
- Pattern-based analysis for URLs and text
- SQLite database for scan history
- CORS enabled for frontend integration

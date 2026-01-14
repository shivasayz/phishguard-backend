# Phishing Detection Backend

FastAPI backend for AI-powered phishing detection using BERT.

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

3. Train the model (optional):
```bash
python train_model.py
```

4. Run the server:
```bash
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`

## API Endpoints

- `POST /scan` - Scan text or URL for phishing
- `GET /history` - Get scan history
- `POST /clear-history` - Clear scan history
- `POST /feedback` - Submit feedback on scan results

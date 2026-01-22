from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional
import sqlite3
import json
from datetime import datetime
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from urllib.parse import urlparse
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading DistilBERT model...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
print("Model loaded successfully")

class ScanRequest(BaseModel):
    content_type: str
    text_content: Optional[str] = None
    url: Optional[str] = None
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v, info):
        if info.data.get('content_type') == 'url':
            if v is None:
                raise ValueError("URL must be provided for URL scan")
            if not v.startswith(('http://', 'https://')):
                v = 'http://' + v
            if '.' not in v:
                raise ValueError("Invalid URL format. URL must contain a domain name with a dot (e.g., example.com)")
        return v
    
    @field_validator('text_content')
    @classmethod
    def validate_text(cls, v, info):
        if info.data.get('content_type') == 'text':
            if v is None or v.strip() == '':
                raise ValueError("Text content must be provided for text scan")
        return v

class FeedbackRequest(BaseModel):
    scan_id: int
    is_correct: bool
    feedback_text: Optional[str] = None

class ScanResult(BaseModel):
    risk_level: str
    confidence_score: float
    flagged_keywords: List[str]
    domains_found: List[dict]
    screenshot: Optional[str] = None

def extract_urls(text: str) -> List[dict]:
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    domains = []
    
    for url in urls:
        try:
            parsed = urlparse(url)
            domains.append({"domain": parsed.netloc, "url": url})
        except:
            continue
    
    return domains

def analyze_text_with_bert(text: str) -> tuple:
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence_score = predictions[0][1].item()

        suspicious_patterns = {
            'urgency': {
                'patterns': [r'urgent', r'immediate action', r'account suspended', r'security alert',
                    r'unauthorized access', r'suspicious activity', r'limited time', r'expires soon'],
                'weight': 0.4
            },
            'credentials': {
                'patterns': [r'verify.{0,20}account', r'confirm.{0,20}identity', r'login.{0,20}details',
                    r'password.{0,20}expired', r'security.{0,20}update', r'update.{0,20}information'],
                'weight': 0.5
            },
            'financial': {
                'patterns': [r'bank.{0,20}account', r'credit.{0,20}card', r'payment.{0,20}info',
                    r'billing.{0,20}address', r'financial.{0,20}activity', r'transaction'],
                'weight': 0.4
            },
            'threats': {
                'patterns': [r'account.{0,20}terminated', r'legal.{0,20}action', r'police',
                    r'lawsuit', r'criminal.{0,20}charges', r'investigation'],
                'weight': 0.6
            },
            'rewards': {
                'patterns': [r'won.{0,20}prize', r'lottery.{0,20}winner', r'inheritance',
                    r'unclaimed.{0,20}funds', r'compensation', r'reward'],
                'weight': 0.5
            }
        }

        found_keywords = []
        additional_score = 0.0
        
        for category, data in suspicious_patterns.items():
            category_matches = []
            for pattern in data['patterns']:
                matches = re.findall(pattern, text.lower())
                if matches:
                    category_matches.extend(matches)
                    additional_score += data['weight'] * len(matches)
            
            if category_matches:
                found_keywords.append(f"{category.title()}: {', '.join(set(category_matches))}")

        final_score = (confidence_score + min(additional_score, 1.0)) / 2

        if final_score > 0.8:
            risk_level = "High"
        elif final_score > 0.5:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return risk_level, final_score, found_keywords

    except Exception as e:
        return "Error", 0.0, [f"Error in text analysis: {str(e)}"]

def analyze_url(url: str) -> tuple:
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
            
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        if not domain or '.' not in domain:
            return "Error", 0.0, ["Invalid URL: Unable to parse domain"]
            
        flagged_patterns = []
        
        legitimate_domains = {'google.com', 'amazon.com', 'microsoft.com', 'apple.com', 
            'facebook.com', 'twitter.com', 'linkedin.com', 'github.com', 'youtube.com',
            'instagram.com', 'netflix.com', 'paypal.com', 'ebay.com', 'walmart.com'}
        
        if domain in legitimate_domains or domain.endswith(tuple(f'.{d}' for d in legitimate_domains)):
            return "Low", 0.1, ["Legitimate domain"]
        
        risk_score = 0.0
        
        for legit_domain in legitimate_domains:
            if domain != legit_domain and legit_domain[:-4] in domain:
                flagged_patterns.append(f"Possible lookalike domain for {legit_domain}")
                risk_score += 0.6

        suspicious_patterns = [
            (r'bit\.ly|tinyurl\.com|goo\.gl', "URL shortener service", 0.3),
            (r'[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+', "IP address in URL", 0.5),
            (r'(password|login|signin|verify|account|update|security)', "Sensitive terms in URL", 0.4),
            (r'@', "@ symbol in URL", 0.6),
            (r'data:|javascript:|file:', "Suspicious URL scheme", 0.8),
            (r'(bank|paypal|ebay|amazon|apple|microsoft).*\.(tk|ga|gq|ml|cf)', "Suspicious TLD combination", 0.8),
            (r'secure[0-9]*\.', "Numbered secure subdomain", 0.5),
            (r'-?update-?account', "Account update keywords", 0.5)
        ]

        for pattern, description, score in suspicious_patterns:
            if re.search(pattern, url, re.I):
                flagged_patterns.append(description)
                risk_score += score

        if len(domain) > 30:
            flagged_patterns.append("Unusually long domain name")
            risk_score += 0.3

        subdomain_count = len(domain.split('.')) - 1
        if subdomain_count > 3:
            flagged_patterns.append(f"Excessive subdomains ({subdomain_count})")
            risk_score += 0.3

        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.club', '.work', '.date', '.bid']
        for tld in suspicious_tlds:
            if domain.endswith(tld):
                flagged_patterns.append(f"Suspicious TLD ({tld})")
                risk_score += 0.5
                break

        if risk_score > 0.7:
            risk_level = "High"
        elif risk_score > 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        if not flagged_patterns:
            flagged_patterns.append("No suspicious patterns detected")

        return risk_level, min(risk_score, 1.0), flagged_patterns

    except Exception as e:
        return "Error", 0.0, [f"Error in URL analysis: {str(e)}"]

@app.post("/scan")
async def scan_content(scan_request: ScanRequest) -> ScanResult:
    try:
        if scan_request.content_type == "text":
            if not scan_request.text_content or not scan_request.text_content.strip():
                raise HTTPException(status_code=400, detail="Text content is required for text analysis")
            risk_level, confidence_score, flagged_keywords = analyze_text_with_bert(scan_request.text_content)
            domains_found = extract_urls(scan_request.text_content)
            content_preview = scan_request.text_content[:100] + "..." if len(scan_request.text_content) > 100 else scan_request.text_content

        elif scan_request.content_type == "url":
            if not scan_request.url or not scan_request.url.strip():
                raise HTTPException(status_code=400, detail="URL is required for URL analysis")
            
            url_to_analyze = scan_request.url
            if not url_to_analyze.startswith(('http://', 'https://')):
                url_to_analyze = 'http://' + url_to_analyze
            
            try:
                parsed = urlparse(url_to_analyze)
                if not parsed.netloc or '.' not in parsed.netloc:
                    raise HTTPException(status_code=400, detail="Invalid URL format. URL must contain a valid domain name.")
            except Exception:
                raise HTTPException(status_code=400, detail="Could not parse URL. Please check the format.")
            
            risk_level, confidence_score, flagged_keywords = analyze_url(url_to_analyze)
            domains_found = [{"domain": urlparse(url_to_analyze).netloc, "url": url_to_analyze}]
            content_preview = url_to_analyze

        else:
            raise HTTPException(status_code=400, detail="Invalid content type. Must be 'text', 'url', or 'image'")

        conn = sqlite3.connect('phishing_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scan_history 
            (content_type, email_preview, url, risk_level, confidence_score, flagged_keywords, domains_found, scan_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            scan_request.content_type,
            content_preview,
            scan_request.url if scan_request.content_type == "url" else None,
            risk_level,
            confidence_score,
            json.dumps(flagged_keywords),
            json.dumps(domains_found),
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()

        return ScanResult(
            risk_level=risk_level,
            confidence_score=confidence_score,
            flagged_keywords=flagged_keywords,
            domains_found=domains_found
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in scan_content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")

def init_db():
    conn = sqlite3.connect('phishing_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS scan_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content_type TEXT,
        email_preview TEXT,
        url TEXT,
        risk_level TEXT,
        confidence_score REAL,
        flagged_keywords TEXT,
        domains_found TEXT,
        scan_date TEXT,
        is_feedback_correct INTEGER DEFAULT NULL,
        feedback_text TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

init_db()

@app.get("/history")
async def get_history():
    try:
        conn = sqlite3.connect('phishing_history.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, content_type, email_preview, url, risk_level, confidence_score, 
               flagged_keywords, domains_found, scan_date, is_feedback_correct, feedback_text
        FROM scan_history
        ORDER BY id DESC LIMIT 100
        ''')
        
        rows = cursor.fetchall()
        result = []
        
        for row in rows:
            item = dict(row)
            item['flagged_keywords'] = json.loads(item['flagged_keywords'])
            item['domains_found'] = json.loads(item['domains_found'])
            result.append(item)
        
        conn.close()
        return result
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
        init_db()
        return []
    except Exception as e:
        print(f"Error in get_history: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/clear-history")
async def clear_history():
    conn = sqlite3.connect('phishing_history.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM scan_history')
    conn.commit()
    conn.close()
    return {"message": "History cleared successfully"}

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    conn = sqlite3.connect('phishing_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    UPDATE scan_history
    SET is_feedback_correct = ?, feedback_text = ?
    WHERE id = ?
    ''', (1 if feedback.is_correct else 0, feedback.feedback_text, feedback.scan_id))
    
    conn.commit()
    conn.close()
    
    return {"message": "Feedback submitted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

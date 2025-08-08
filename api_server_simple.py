from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Content Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str
    platform: str = "unknown"

class AnalysisResponse(BaseModel):
    risk_score: float
    threat_level: str
    content_length: int
    indicators: dict = {}

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    # Simple risk calculation based on suspicious keywords
    text_lower = request.text.lower()
    risk_score = 0.0
    
    # Check for misinformation indicators
    if "urgent" in text_lower: risk_score += 0.2
    if "shocking" in text_lower: risk_score += 0.2
    if "secret" in text_lower: risk_score += 0.15
    if "they don't want" in text_lower or "they dont want" in text_lower: risk_score += 0.25
    if "share before" in text_lower: risk_score += 0.2
    if "deleted" in text_lower: risk_score += 0.15
    if "!!!" in request.text: risk_score += 0.1
    
    # Determine threat level
    if risk_score >= 0.7: threat_level = "red"
    elif risk_score >= 0.5: threat_level = "orange"
    elif risk_score >= 0.3: threat_level = "yellow"
    else: threat_level = "green"
    
    return AnalysisResponse(
        risk_score=min(risk_score, 1.0),
        threat_level=threat_level,
        content_length=len(request.text),
        indicators={
            "urgency_detected": "urgent" in text_lower,
            "emotional_manipulation": "shocking" in text_lower,
            "conspiracy_language": "they don" in text_lower,
            "false_scarcity": "deleted" in text_lower
        }
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "content-analysis"}

if __name__ == "__main__":
    print("Starting Simple Content Analysis API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

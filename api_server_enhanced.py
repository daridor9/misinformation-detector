from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import re

app = FastAPI(title="Content Analysis API", version="2.0.0")

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
    detailed_analysis: dict = {}

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    text = request.text
    text_lower = text.lower()
    risk_components = {}
    
    # 1. Urgency patterns (0-0.25)
    urgency_score = 0
    urgency_words = ['urgent', 'breaking', 'alert', 'warning', 'immediate', 'act now', 'hurry']
    for word in urgency_words:
        if word in text_lower:
            urgency_score += 0.05
    if 'urgent' in text_lower and '!' in text:
        urgency_score += 0.1
    risk_components['urgency'] = min(urgency_score, 0.25)
    
    # 2. Emotional manipulation (0-0.25)
    emotion_score = 0
    emotion_words = ['shocking', 'horrific', 'outrageous', 'disgusting', 'unbelievable', 'mind-blowing']
    for word in emotion_words:
        if word in text_lower:
            emotion_score += 0.05
    risk_components['emotional_manipulation'] = min(emotion_score, 0.25)
    
    # 3. Conspiracy language (0-0.25)
    conspiracy_score = 0
    conspiracy_phrases = [
        "they don't want", "they dont want", "hidden truth", "wake up",
        "mainstream media", "what they're hiding", "secret agenda",
        "truth revealed", "exposed", "cover up", "conspiracy"
    ]
    for phrase in conspiracy_phrases:
        if phrase in text_lower:
            conspiracy_score += 0.08
    risk_components['conspiracy_language'] = min(conspiracy_score, 0.25)
    
    # 4. False scarcity/urgency (0-0.15)
    scarcity_score = 0
    scarcity_phrases = [
        "before it's deleted", "before its deleted", "share before",
        "removed soon", "won't last", "limited time", "act fast"
    ]
    for phrase in scarcity_phrases:
        if phrase in text_lower:
            scarcity_score += 0.08
    risk_components['false_scarcity'] = min(scarcity_score, 0.15)
    
    # 5. Excessive punctuation (0-0.1)
    punct_score = 0
    if '!!!' in text: punct_score += 0.05
    if '???' in text: punct_score += 0.05
    exclamation_count = text.count('!')
    if exclamation_count > 5: punct_score += 0.05
    risk_components['excessive_punctuation'] = min(punct_score, 0.1)
    
    # 6. ALL CAPS abuse (0-0.1)
    caps_words = re.findall(r'\b[A-Z]{4,}\b', text)
    caps_score = min(len(caps_words) * 0.03, 0.1)
    risk_components['caps_abuse'] = caps_score
    
    # 7. Clickbait patterns (0-0.15)
    clickbait_score = 0
    clickbait_phrases = [
        "you won't believe", "doctors hate", "this one trick",
        "what happened next", "number 5 will shock"
    ]
    for phrase in clickbait_phrases:
        if phrase in text_lower:
            clickbait_score += 0.05
    risk_components['clickbait'] = min(clickbait_score, 0.15)
    
    # Calculate total risk score
    risk_score = sum(risk_components.values())
    
    # Determine threat level
    if risk_score >= 0.7: threat_level = "red"
    elif risk_score >= 0.5: threat_level = "orange"
    elif risk_score >= 0.3: threat_level = "yellow"
    else: threat_level = "green"
    
    # Detailed analysis
    detailed_analysis = {
        "total_exclamations": text.count('!'),
        "total_questions": text.count('?'),
        "caps_words_found": caps_words,
        "risk_breakdown": {k: f"{v*100:.1f}%" for k, v in risk_components.items()},
        "top_risks": sorted(risk_components.items(), key=lambda x: x[1], reverse=True)[:3]
    }
    
    return AnalysisResponse(
        risk_score=round(min(risk_score, 1.0), 3),
        threat_level=threat_level,
        content_length=len(request.text),
        indicators={k: v > 0 for k, v in risk_components.items()},
        detailed_analysis=detailed_analysis
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "content-analysis", "version": "2.0"}

if __name__ == "__main__":
    print("Starting Enhanced Content Analysis API v2.0...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

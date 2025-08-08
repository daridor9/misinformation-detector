from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import sys
import os
from datetime import datetime
import uuid

# Import your detection system
try:
    import defensive_misinformation_detection as dmd
except ImportError:
    print("Error: Could not import defensive_misinformation_detection.py")
    print("Make sure the file is in the same directory")
    sys.exit(1)

app = FastAPI(title="Content Analysis API", version="1.0.0")

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the detection system
print("Initializing Misinformation Defense System...")
defense_system = dmd.MisinformationDefenseSystem()

class TextRequest(BaseModel):
    text: str
    platform: str = "unknown"

class AnalysisResponse(BaseModel):
    risk_score: float
    threat_level: str
    content_length: int
    indicators: Dict = {}

@app.get("/")
async def root():
    return {"message": "Content Analysis API is running", "version": "1.0.0"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: TextRequest):
    try:
        # Create content item for the detection system
        content = dmd.ContentItem(
            id=str(uuid.uuid4()),
            text=request.text,
            source="api-user",
            timestamp=datetime.now(),
            platform=request.platform,
            engagement_metrics={
                "shares": 0,
                "time_elapsed_minutes": 0,
                "bot_like_shares": 0,
                "new_account_shares": 0,
                "temporal_clustering_score": 0,
                "similar_content_score": 0
            }
        )
        
        # Process through the full detection system
        threat_assessment = defense_system.process_content(content)
        
        # Extract key indicators
        indicators = {}
        if "content_analysis" in threat_assessment.indicators:
            analysis = threat_assessment.indicators["content_analysis"]
            if "risk_components" in analysis:
                indicators = {
                    "linguistic_risk": analysis["risk_components"].get("linguistic_risk", 0),
                    "network_risk": analysis["risk_components"].get("network_risk", 0),
                    "media_risk": analysis["risk_components"].get("media_risk", 0),
                    "fact_check_score": analysis["risk_components"].get("fact_check_score", 0.5)
                }
        
        return AnalysisResponse(
            risk_score=round(threat_assessment.risk_score, 3),
            threat_level=threat_assessment.threat_level.value,
            content_length=len(request.text),
            indicators=indicators
        )
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "content-analysis", "detection_system": "active"}

if __name__ == "__main__":
    print("Starting Content Analysis API with Misinformation Detection System...")
    print("API documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

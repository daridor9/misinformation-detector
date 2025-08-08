from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os

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

@app.get("/")
async def read_index():
    return FileResponse('index_simple_enhanced.html')

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "content-analysis", "version": "2.0"}

# Your existing analyze endpoint here...
@app.post("/analyze")
async def analyze_text(request: TextRequest):
    # Your analysis logic
    text_lower = request.text.lower()
    risk_score = 0.0
    
    if "urgent" in text_lower: risk_score += 0.2
    if "shocking" in text_lower: risk_score += 0.2
    # ... rest of your logic
    
    return {"risk_score": risk_score, "threat_level": "low", "content_length": len(request.text)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

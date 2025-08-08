from api_server_enhanced import app
import os

# Add a root endpoint
@app.get("/")
async def root():
    return {
        "message": "Misinformation Detection API",
        "docs": "Visit /docs for API documentation",
        "health": "Visit /health to check status"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

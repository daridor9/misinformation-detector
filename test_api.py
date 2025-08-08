import requests

# Test the API
response = requests.post(
    "http://localhost:8000/analyze",
    json={"text": "This is a test message to analyze", "platform": "test"}
)

print("Response:", response.json())

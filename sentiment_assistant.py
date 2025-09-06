import os
import requests

# Get endpoint and key from environment variables (recommended for security)
ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "https://senti-service.cognitiveservices.azure.com/")
API_KEY = os.getenv("AZURE_LANGUAGE_KEY", "7WxSIvGdIRzOSf6mrLXlFxhOoZM5UPLKr8I0Dv1BwFUgw6fDlOxnJQQJ99BIACqBBLyXJ3w3AAAaACOGbl9o")

# API path for sentiment analysis
PATH = "/language/:analyze-text?api-version=2023-04-01"

def analyze_sentiment(text: str):
    """
    Send text to Azure Cognitive Service for sentiment analysis.
    Returns sentiment result and confidence scores.
    """
    url = ENDPOINT.rstrip("/") + PATH
    headers = {
        "Ocp-Apim-Subscription-Key": API_KEY,
        "Content-Type": "application/json"
    }
    body = {
        "kind": "SentimentAnalysis",
        "parameters": {"opinionMining": False},
        "analysisInput": {
            "documents": [
                {"id": "1", "language": "en", "text": text}
            ]
        }
    }

    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code}, {response.text}")

    result = response.json()
    doc = result["results"]["documents"][0]

    sentiment = doc["sentiment"]
    confidence = doc["confidenceScores"]

    return {
        "sentiment": sentiment,
        "confidence": confidence
    }

def get_spiel(sentiment: str) -> str:
    """
    Returns empathy spiel (if angry/negative) or upsell spiel (otherwise).
    """
    if sentiment == "negative":
        return "I understand your concern. Let me do my best to resolve this for you as quickly as possible."
    else:
        return "By the way, I’d like to share some of our new services that might be useful for you."

if __name__ == "__main__":
    # Demo texts
    samples = [
        "I am very upset because my internet is not working at all!",
        "Thank you for fixing the issue so quickly!"
    ]

    for text in samples:
        print(f"\nCustomer: {text}")
        result = analyze_sentiment(text)
        sentiment = result["sentiment"]
        spiel = get_spiel(sentiment)
        print(f" Sentiment: {sentiment}")
        print(f" Confidence: {result['confidence']}")
        print(f" Suggested spiel: {spiel}")

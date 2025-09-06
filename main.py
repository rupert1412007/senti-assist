import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

# Load Azure endpoint and key
ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "https://senti-service.cognitiveservices.azure.com/")
API_KEY = os.getenv("AZURE_LANGUAGE_KEY", "7WxSIvGdIRzOSf6mrLXlFxhOoZM5UPLKr8I0Dv1BwFUgw6fDlOxnJQQJ99BIACqBBLyXJ3w3AAAaACOGbl9o")
PATH = "/language/:analyze-text?api-version=2023-04-01"

app = FastAPI(title="Sentiment & Tone API")

class TextInput(BaseModel):
    text: str

# --- Azure Sentiment Analysis ---
def analyze_sentiment(text: str):
    url = ENDPOINT.rstrip("/") + PATH
    headers = {
        "Ocp-Apim-Subscription-Key": API_KEY,
        "Content-Type": "application/json"
    }
    body = {
        "kind": "SentimentAnalysis",
        "parameters": {"opinionMining": False},
        "analysisInput": {
            "documents": [{"id": "1", "language": "en", "text": text}]
        }
    }

    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"Azure request failed: {response.status_code}, {response.text}")

    result = response.json()
    doc = result["results"]["documents"][0]
    return doc["sentiment"]

# --- Tone Classifier ---
def classify_tone(text: str, sentiment: str):
    t = text.lower()

    if sentiment == "negative":
        if any(word in t for word in ["angry", "furious", "upset", "mad"]):
            return "angry"
        if any(word in t for word in ["tired", "waiting", "slow", "delay"]):
            return "frustrated"
        if any(word in t for word in ["don’t understand", "don't understand", "confused", "unclear", "why"]):
            return "confused"
        if any(word in t for word in ["not happy", "bad", "poor", "unhappy"]):
            return "upset"
        return "negative"
    
    if sentiment == "positive":
        if any(word in t for word in ["thank", "glad", "great", "happy"]):
            return "happy"
        if any(word in t for word in ["fine", "good", "satisfied", "okay now"]):
            return "satisfied"
        return "positive"
    
    if sentiment == "neutral":
        if "not sure" in t or "maybe" in t:
            return "uncertain"
        return "neutral"

    return sentiment

# --- Spiel Generator ---
def get_spiel(sentiment: str, tone: str):
    spiels = {
        "angry": "I understand your frustration. Let’s work together to fix this immediately.",
        "frustrated": "I get that this has been frustrating. I’ll make sure to simplify things for you.",
        "confused": "I see this is confusing. Let me clarify and guide you step by step.",
        "upset": "I’m sorry you’re experiencing this. I’ll do my best to resolve it quickly.",
        "happy": "I’m glad to hear that! By the way, would you like to hear about some new services we offer?",
        "satisfied": "I’m happy things are working fine. Is there anything else I can do for you?",
        "neutral": "Thank you for sharing. Let’s continue and see how I can assist further.",
        "uncertain": "I’ll take a closer look at this and make sure we get clarity."
    }
    return spiels.get(tone, "Thank you for reaching out. I’ll do my best to assist you.")

# --- API Endpoint ---
@app.post("/analyze")
def analyze(input: TextInput):
    sentiment = analyze_sentiment(input.text)
    tone = classify_tone(input.text, sentiment)
    spiel = get_spiel(sentiment, tone)

    return {
        "text": input.text,
        "sentiment": sentiment,
        "tone": tone,
        "spiel": spiel
    }

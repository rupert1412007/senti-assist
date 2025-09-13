# main.py
import os
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict

from sentiment_assistant import (
    analyze_sentiment,
    classify_tone_and_intent,
    build_spiel,
    mask_pii,
    decide_mode,
    SentimentResult,
)

app = FastAPI(title="Senti Assist — Empathy & Tone API", version="1.3.0")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    customer_name: Optional[str] = None
    agent_name: Optional[str] = None
    context: Optional[str] = None


class AnalyzeResponse(BaseModel):
    text: str
    sentiment: Literal["positive", "neutral", "negative"]
    confidence: Dict[str, float]
    tone: str
    intent: Optional[str]
    mode: Literal["deescalate", "neutral", "upsell"]
    spiel: str


@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "senti-assist"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text.")

    # 1) sanitize
    clean_text = mask_pii(req.text)

    # 2) Azure Sentiment
    senti: SentimentResult = analyze_sentiment(clean_text)

    # 3) Tone + intent
    tone, intent = classify_tone_and_intent(clean_text, senti.label)

    # 4) Decide mode (deescalate / neutral / upsell)
    mode = decide_mode(senti.label, senti.confidence)

    # 5) Humanized spiel
    spiel = build_spiel(
        sentiment=senti.label,
        tone=tone,
        intent=intent,
        customer_name=req.customer_name,
        agent_name=req.agent_name,
        context=req.context,
        confidence=senti.confidence,
        mode=mode,
        variety_seed=random.random(),
    )

    return AnalyzeResponse(
        text=req.text,
        sentiment=senti.label,
        confidence=senti.confidence,
        tone=tone,
        intent=intent,
        mode=mode,
        spiel=spiel,
    )


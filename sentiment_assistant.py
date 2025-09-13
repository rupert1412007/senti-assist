# sentiment_assistant.py
import re
import requests
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal

# --- Azure endpoint & key (kept as in your original files) ---
ENDPOINT = "https://senti-service.cognitiveservices.azure.com/"
API_KEY = "7WxSIvGdIRzOSf6mrLXlFxhOoZM5UPLKr8I0Dv1BwFUgw6fDlOxnJQQJ99BIACqBBLyXJ3w3AAAaACOGbl9o"
PATH = "/language/:analyze-text?api-version=2023-04-01"

# --- Basic PII masking ---
PII_REGEX = re.compile(r"\b(\+?\d[\d\-\s]{7,}|\d{13,19})\b")
def mask_pii(text: str) -> str:
    return PII_REGEX.sub("[REDACTED]", text)


@dataclass
class SentimentResult:
    label: Literal["positive", "neutral", "negative"]
    confidence: Dict[str, float]


def analyze_sentiment(text: str) -> SentimentResult:
    url = ENDPOINT.rstrip("/") + PATH
    headers = {
        "Ocp-Apim-Subscription-Key": API_KEY,
        "Content-Type": "application/json",
    }
    body = {
        "kind": "SentimentAnalysis",
        "parameters": {"opinionMining": False},
        "analysisInput": {"documents": [{"id": "1", "language": "en", "text": text}]},
    }

    r = requests.post(url, headers=headers, json=body, timeout=8)
    if r.status_code != 200:
        raise RuntimeError(f"Azure request failed: {r.status_code}, {r.text}")

    data = r.json()
    doc = data["results"]["documents"][0]
    conf = doc["confidenceScores"]
    return SentimentResult(
        label=doc["sentiment"],
        confidence={
            "positive": float(conf["positive"]),
            "neutral": float(conf["neutral"]),
            "negative": float(conf["negative"]),
        },
    )


# --- Tone & intent classification ---
ANGER = {"angry", "furious", "mad", "fuming"}
FRUSTRATION = {"frustrated", "annoyed", "delay", "late", "slow", "waiting", "tired"}
CONFUSION = {"confused", "unclear", "don’t understand", "don't understand", "why"}
DISAPPOINT = {"disappointed", "not happy", "unhappy", "poor", "bad", "terrible"}
GRATITUDE = {"thank", "thanks", "appreciate", "grateful"}
RELIEF = {"finally", "good now", "okay now", "fixed", "resolved"}
UNCERTAINTY = {"maybe", "not sure", "possibly", "might"}

INTENT_CANCEL = {"cancel", "terminate", "close my account"}
INTENT_REFUND = {"refund", "money back", "chargeback"}
INTENT_ESCALATE = {"supervisor", "manager", "escalate", "complaint"}
INTENT_PURCHASE = {"buy", "upgrade", "add", "order", "sign me up"}


def classify_tone_and_intent(text: str, sentiment: str) -> Tuple[str, Optional[str]]:
    t = text.lower()

    # intent
    if any(k in t for k in INTENT_REFUND):
        intent = "refund"
    elif any(k in t for k in INTENT_CANCEL):
        intent = "cancel"
    elif any(k in t for k in INTENT_ESCALATE):
        intent = "escalate"
    elif any(k in t for k in INTENT_PURCHASE):
        intent = "purchase"
    else:
        intent = None

    # tone
    tone = "neutral"
    if sentiment == "negative":
        if any(k in t for k in ANGER):
            tone = "angry"
        elif any(k in t for k in FRUSTRATION):
            tone = "frustrated"
        elif any(k in t for k in CONFUSION):
            tone = "confused"
        elif any(k in t for k in DISAPPOINT):
            tone = "disappointed"
        else:
            tone = "negative"
    elif sentiment == "positive":
        if any(k in t for k in GRATITUDE):
            tone = "grateful"
        elif any(k in t for k in RELIEF):
            tone = "relieved"
        else:
            tone = "positive"
    else:
        if any(k in t for k in UNCERTAINTY):
            tone = "uncertain"
        else:
            tone = "neutral"

    return tone, intent


def decide_mode(
    sentiment: Literal["positive", "neutral", "negative"], confidence: Dict[str, float]
) -> Literal["deescalate", "neutral", "upsell"]:
    neg = confidence.get("negative", 0.0)
    pos = confidence.get("positive", 0.0)
    if sentiment == "negative" and neg >= 0.35:
        return "deescalate"
    if sentiment == "positive" and pos >= 0.65:
        return "upsell"
    return "neutral"


def build_spiel(
    sentiment: str,
    tone: str,
    intent: Optional[str],
    customer_name: Optional[str],
    agent_name: Optional[str],
    context: Optional[str],
    confidence: Dict[str, float],
    mode: Literal["deescalate", "neutral", "upsell"],
    variety_seed: float = 0.0,
) -> str:
    # Acknowledge – Align – Assure – Act (+Upsell if mode=upsell)
    ack = align = assure = act = upsell = ""

    if mode == "deescalate":
        if tone in {"angry", "frustrated", "disappointed"}:
            ack = "I hear this has been really tough—and that’s not the experience we want for you."
        elif tone == "confused":
            ack = "I can see this got confusing, and that’s on us to make clearer."
        else:
            ack = "I get that this hasn’t been smooth, and I appreciate you sticking with me."
        align = "You shouldn’t have to chase this—I’m on it now."
        assure = "I’ll take ownership and keep you updated as we fix it."
        if intent == "refund":
            act = "I can start a refund request right away or offer a replacement—what works better for you?"
        elif intent == "cancel":
            act = "I can process the cancellation now or pause the account while we resolve the issue—your call."
        elif intent == "escalate":
            act = "I can bring in a supervisor, and I’ll brief them so you don’t have to repeat yourself."
        else:
            act = "First step: let me check your case and confirm the next steps."

    elif mode == "upsell":
        ack = "I’m glad that worked out—thanks for confirming."
        align = "Since you’re here, there’s an option that often helps folks in your situation."
        assure = "No pressure—just sharing in case it adds value."
        hint = f" for {context}" if context else ""
        upsell = f"Many customers add our Care Plan{hint}; it covers accidental issues and speeds support. Want me to add it?"
        act = "If not today, totally fine—we can always revisit later."

    else:  # neutral
        if tone == "uncertain":
            ack = "Totally fair to be unsure here."
            align = "Let’s make this simple so you can decide confidently."
        else:
            ack = "Got it—thanks for explaining."
            align = "Let me make sure I’m solving the right thing."
        assure = "I’ll be clear and quick."
        act = "What outcome would feel like a win for you today?"

    return " ".join(p for p in [ack, align, assure, act, upsell] if p)

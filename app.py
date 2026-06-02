from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
import numpy as np

from roman_urdu_map import normalize_roman_urdu
from symptom_anchor import SYMPTOM_ANCHORS
from ai_helper import verify_model_prediction, ai_fallback_prediction

app = FastAPI(title="Disease Prediction API")

# Allow Flutter app to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For FYP/testing. In production, restrict this.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model & vectorizer
model = pickle.load(open("final_disease_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
VOCAB = set(vectorizer.get_feature_names_out())


class InputData(BaseModel):
    symptoms: str


def clean_text(text: str) -> str:
    text = normalize_roman_urdu(text.lower())

    for w in ["mein", "ka", "ki", "hai", "ha"]:
        text = text.replace(w, "")

    text = text.replace("dard", "pain")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def vocab_overlap(text: str):
    words = set(text.split())
    overlap = words & VOCAB
    return len(overlap), overlap


def anchor_boost(cleaned_text: str, probs):
    boosted_probs = probs.copy()

    for anchor, diseases in SYMPTOM_ANCHORS.items():
        if all(w in cleaned_text for w in anchor.split()):
            for disease in diseases:
                if disease in model.classes_:
                    idx = list(model.classes_).index(disease)
                    boosted_probs[idx] *= 1.8

    return boosted_probs


@app.get("/")
def root():
    return {
        "status": "Disease Prediction API running",
        "message": "Use POST /predict with symptoms"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "vectorizer_loaded": True
    }


@app.post("/predict")
def predict(data: InputData):
    raw_symptoms = data.symptoms.strip()

    if not raw_symptoms or len(raw_symptoms) < 3:
        return {
            "status": "invalid",
            "message": "Please enter valid symptoms."
        }

    cleaned = clean_text(raw_symptoms)
    overlap_count, overlap_words = vocab_overlap(cleaned)

    # Invalid input check
    # Example: random words with no medical meaning
    if overlap_count == 0:
        return {
            "status": "invalid",
            "message": "No medical intent detected. Please enter valid symptoms.",
            "cleaned_text": cleaned,
            "recognized_terms": []
        }

    try:
        # Vectorize symptoms
        vec = vectorizer.transform([cleaned])

        # Model probabilities
        probs = model.predict_proba(vec)[0]

        # Apply symptom anchor boost
        probs = anchor_boost(cleaned, probs)

        # Normalize probabilities safely
        total_prob = probs.sum()
        if total_prob == 0:
            return {
                "status": "error",
                "message": "Model probability error. Please try again."
            }

        probs = probs / total_prob

        # Get top predicted disease
        idx = int(np.argmax(probs))
        predicted_disease = str(model.classes_[idx])

        confidence = float(probs[idx])
        confidence_percent = round(confidence * 100, 2)

        # Default values
        final_disease = predicted_disease
        model_disease = predicted_disease
        ai_used = False
        source = "model"
        note = "High confidence model prediction."
        assurity_level = "High"

        # ===============================
        # Confidence-based hybrid logic
        # ===============================

        if confidence_percent >= 70:
            # High confidence: use your trained model directly
            final_disease = predicted_disease
            assurity_level = "High"
            ai_used = False
            source = "model"
            note = "High confidence model prediction."

        elif confidence_percent >= 40:
            # Medium confidence: ask ChatGPT to verify model result
            assurity_level = "Medium"

            ai_result = verify_model_prediction(
                symptoms=raw_symptoms,
                model_disease=predicted_disease,
                confidence_percent=confidence_percent
            )

            final_disease = ai_result.get("disease", predicted_disease)
            ai_used = True
            source = "model_ai_verified"
            note = ai_result.get(
                "note",
                "AI reviewed the model prediction."
            )

        else:
            # Low confidence: use ChatGPT fallback
            assurity_level = "Low"

            ai_result = ai_fallback_prediction(
                symptoms=raw_symptoms
            )

            final_disease = ai_result.get("disease", "Uncertain")
            ai_used = True
            source = "chatgpt_fallback"
            note = ai_result.get(
                "note",
                "Low model confidence, AI fallback used."
            )

        return {
            "status": "success",

            # Final disease app should use
            "disease": final_disease,

            # Original model prediction for debugging/report
            "model_disease": model_disease,

            # Model confidence
            "confidence_percent": confidence_percent,
            "assurity_level": assurity_level,

            # Tells Flutter whether ChatGPT was used
            "ai_used": ai_used,
            "source": source,

            # Extra information
            "recognized_terms": list(overlap_words),
            "cleaned_text": cleaned,
            "note": note,

            # Safety message for app/report
            "disclaimer": "This is an AI-assisted prediction, not a final medical diagnosis. Please consult a doctor for confirmation."
        }

    except Exception as e:
        return {
            "status": "error",
            "message": "Prediction failed.",
            "error": str(e)
        }
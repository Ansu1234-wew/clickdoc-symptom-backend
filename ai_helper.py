import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()


def get_openai_api_key():
    """
    Reads OpenAI API key at request time.
    This works better on Railway after variables are added/redeployed.
    """
    return os.getenv("OPENAI_API_KEY")


def _extract_output_text(result: dict) -> str:
    """
    Extract text safely from OpenAI Responses API result.
    """

    if not isinstance(result, dict):
        return ""

    if result.get("output_text"):
        return str(result["output_text"])

    output_items = result.get("output", [])
    extracted_parts = []

    for item in output_items:
        content_items = item.get("content", [])

        for content in content_items:
            if content.get("type") in ["output_text", "text"]:
                text_value = content.get("text", "")
                if text_value:
                    extracted_parts.append(str(text_value))

    return "\n".join(extracted_parts).strip()


def _extract_json_from_text(text: str):
    """
    Convert AI text into JSON safely.
    """

    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end > start:
            json_part = text[start:end]
            return json.loads(json_part)
    except Exception:
        pass

    return None


def _call_openai_for_medical_json(prompt: str, schema_name: str, schema: dict):
    """
    Calls OpenAI Responses API and forces JSON schema output.
    """

    openai_api_key = get_openai_api_key()

    if not openai_api_key:
        return None, "OpenAI API key missing."

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "instructions": (
                    "You are a careful medical triage assistant. "
                    "You do not give a final diagnosis. "
                    "You only suggest a possible condition. "
                    "Return only JSON matching the given schema."
                ),
                "input": prompt,
                "temperature": 0.1,
                "max_output_tokens": 250,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema
                    }
                }
            },
            timeout=25,
        )

        if response.status_code != 200:
            return None, (
                f"OpenAI API failed: {response.status_code} - {response.text}"
            )

        result = response.json()
        output_text = _extract_output_text(result)

        parsed = _extract_json_from_text(output_text)

        if not parsed:
            return None, f"AI response parsing failed. Raw response: {output_text}"

        return parsed, None

    except Exception as e:
        return None, f"OpenAI request error: {str(e)}"


def verify_model_prediction(
    symptoms: str,
    model_disease: str,
    confidence_percent: float,
):
    """
    Used when model confidence is medium: 40% to 69%.
    ChatGPT checks whether model disease is medically plausible.
    """

    schema = {
        "type": "object",
        "properties": {
            "is_model_plausible": {
                "type": "boolean"
            },
            "disease": {
                "type": "string"
            },
            "note": {
                "type": "string"
            }
        },
        "required": [
            "is_model_plausible",
            "disease",
            "note"
        ],
        "additionalProperties": False
    }

    prompt = f"""
Patient symptoms:
{symptoms}

Trained ML model predicted:
{model_disease}

Model confidence:
{confidence_percent}%

Task:
Check whether the model disease is medically plausible for these symptoms.

Rules:
- If the model disease is plausible, keep the same disease.
- If it is not plausible, suggest one better possible condition.
- Do not say this is a final diagnosis.
- Keep the note short.
"""

    parsed, error = _call_openai_for_medical_json(
        prompt=prompt,
        schema_name="verify_model_prediction",
        schema=schema,
    )

    if error or not parsed:
        return {
            "is_model_plausible": True,
            "disease": model_disease,
            "note": f"AI verification failed. Using model result. {error or ''}".strip()
        }

    disease = parsed.get("disease", model_disease)

    if not disease or disease.strip() == "":
        disease = model_disease

    return {
        "is_model_plausible": bool(parsed.get("is_model_plausible", True)),
        "disease": disease.strip(),
        "note": parsed.get("note", "AI reviewed the model prediction.")
    }


def ai_fallback_prediction(symptoms: str):
    """
    Used when model confidence is low: below 40%.
    ChatGPT suggests a possible condition.
    """

    schema = {
        "type": "object",
        "properties": {
            "disease": {
                "type": "string"
            },
            "note": {
                "type": "string"
            }
        },
        "required": [
            "disease",
            "note"
        ],
        "additionalProperties": False
    }

    prompt = f"""
Patient symptoms:
{symptoms}

The trained ML model confidence is low.

Task:
Suggest one most likely possible condition or general disease category.

Rules:
- Do not say this is a final diagnosis.
- Keep the disease name short.
- Keep the note short.
"""

    parsed, error = _call_openai_for_medical_json(
        prompt=prompt,
        schema_name="ai_fallback_prediction",
        schema=schema,
    )

    if error or not parsed:
        return {
            "disease": "Uncertain",
            "note": f"AI fallback failed. {error or ''}".strip()
        }

    disease = parsed.get("disease", "Uncertain")

    if not disease or disease.strip() == "":
        disease = "Uncertain"

    return {
        "disease": disease.strip(),
        "note": parsed.get("note", "Low confidence. AI fallback used.")
    }

ROMAN_URDU_MAP = {
    "sar dard": "headache",
    "sir dard": "headache",
    "bukhar": "fever",
    "khansi": "cough",
    "seene mein dard": "chest pain",
    "saans ka masla": "shortness of breath",
    "ulti": "vomiting",
    "kamzori": "weakness",
    "thakan": "fatigue",
    "gurda": "kidney",
    "gurday": "kidney",
    "per": "leg",
    "zayada": "severe",
    "bahat": "severe"
}

def normalize_roman_urdu(text):
    for ru, en in ROMAN_URDU_MAP.items():
        text = text.replace(ru, en)
    return text

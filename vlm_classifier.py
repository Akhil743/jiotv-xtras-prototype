"""
VLM-based video segment classifier using Ollama vision models.
Sends frames to a local VLM and parses structured classification output.

Strategy: Use simple description prompts (works with small models like moondream)
then classify the description via keyword scoring.
"""

import requests
import base64
import json
import re
from pathlib import Path


OLLAMA_URL = "http://localhost:11434"

DESCRIPTION_PROMPT = "Describe this image in detail. Mention people, actions, colors, text, and setting."

CLASSIFICATION_PROMPT = "Classify this movie frame as: dialogue, musical, action, transition, or credits. Reply with only the category."

CATEGORY_KEYWORDS = {
    "musical": {
        "strong": ["dance", "dancing", "singing", "song", "choreograph", "performer",
                    "stage", "musical", "lip-sync", "chorus", "melody", "instrument",
                    "guitar", "drum", "celebration", "festival", "colorful costumes",
                    "group formation", "bollywood"],
        "weak": ["colorful", "bright", "crowd", "group", "moving together",
                 "rhythmic", "joyful", "party", "wedding"],
    },
    "action": {
        "strong": ["fight", "fighting", "chase", "chasing", "punch", "kick",
                    "explosion", "combat", "battle", "sword", "gun", "shooting",
                    "stunt", "car chase", "running", "attack", "warrior", "blood"],
        "weak": ["fast", "intense", "destruction", "fire", "crash", "danger",
                 "aggressive", "violent"],
    },
    "dialogue": {
        "strong": ["talking", "conversation", "speaking", "discuss", "sitting",
                    "two people", "face to face", "emotional", "crying", "arguing",
                    "interview", "meeting", "dinner", "restaurant"],
        "weak": ["person", "room", "indoor", "looking at", "standing", "close-up",
                 "expression", "serious", "calm", "quiet"],
    },
    "transition": {
        "strong": ["landscape", "aerial", "establishing", "scenery", "city",
                    "building", "sunset", "sunrise", "mountain", "ocean",
                    "empty room", "no people", "exterior"],
        "weak": ["sky", "clouds", "wide shot", "panoramic", "nature"],
    },
    "credits": {
        "strong": ["text", "credits", "title", "opening", "closing", "names",
                    "written by", "directed by", "black background", "white text",
                    "logo"],
        "weak": ["letters", "words", "dark", "simple"],
    },
}


def check_ollama_available() -> tuple[bool, list[str]]:
    """Check if Ollama is running and list available vision models."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            vision_models = []
            for m in models:
                name = m["name"].lower()
                if any(v in name for v in ["moondream", "llava", "minicpm", "bakllava"]):
                    vision_models.append(m["name"])
            return True, vision_models
        return False, []
    except Exception:
        return False, []


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def classify_frame(image_path: str, model: str = "moondream-local") -> dict:
    """Classify a single frame using the VLM."""
    image_b64 = encode_image(image_path)

    is_small_model = any(s in model.lower() for s in ["moondream", "local"])

    prompt = DESCRIPTION_PROMPT if is_small_model else CLASSIFICATION_PROMPT

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 150}
    }

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=120
        )
        if resp.status_code == 200:
            text = resp.json().get("response", "")
            if is_small_model:
                return classify_description(text)
            return parse_vlm_response(text)
        return {"category": "unknown", "confidence": 0.0, "description": f"API error: {resp.status_code}"}
    except requests.exceptions.Timeout:
        return {"category": "unknown", "confidence": 0.0, "description": "VLM timeout"}
    except Exception as e:
        return {"category": "unknown", "confidence": 0.0, "description": str(e)}


def classify_description(text: str) -> dict:
    """Score a VLM description against category keywords."""
    text_lower = text.lower().strip()

    if not text_lower:
        return {"category": "dialogue", "confidence": 0.3, "description": "Empty VLM response"}

    scores = {}
    for category, kw_groups in CATEGORY_KEYWORDS.items():
        score = 0.0
        for word in kw_groups["strong"]:
            if word in text_lower:
                score += 0.25
        for word in kw_groups["weak"]:
            if word in text_lower:
                score += 0.1
        scores[category] = min(score, 1.0)

    total = sum(scores.values())
    if total > 0:
        scores = {k: round(v / total, 3) for k, v in scores.items()}
    else:
        scores = {"dialogue": 0.4, "musical": 0.2, "action": 0.2,
                  "transition": 0.1, "credits": 0.1}

    category = max(scores, key=scores.get)
    confidence = scores[category]

    return {
        "category": category,
        "confidence": round(confidence, 3),
        "scores": scores,
        "description": f"VLM: {text[:150].strip()}",
    }


def classify_frames_batch(image_paths: list[str], model: str = "moondream-local") -> dict:
    """Classify multiple frames (from one clip) and return majority-vote result."""
    results = []
    for path in image_paths:
        r = classify_frame(path, model)
        results.append(r)

    if not results:
        return {"category": "unknown", "confidence": 0.0, "description": "No frames to classify"}

    category_votes = {}
    for r in results:
        cat = r.get("category", "unknown")
        category_votes[cat] = category_votes.get(cat, 0) + 1

    winner = max(category_votes, key=category_votes.get)
    confidence = category_votes[winner] / len(results)
    best_desc = next(
        (r["description"] for r in results if r.get("category") == winner),
        "Classified by majority vote across frames"
    )

    return {
        "category": winner,
        "confidence": round(confidence, 3),
        "description": best_desc,
        "frame_results": results,
        "vote_distribution": category_votes,
    }


def parse_vlm_response(text: str) -> dict:
    """Parse VLM text output into structured classification (for larger models)."""
    text = text.strip()

    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            valid_cats = {"dialogue", "musical", "action", "transition", "credits"}
            if parsed.get("category") in valid_cats:
                return {
                    "category": parsed["category"],
                    "confidence": min(1.0, max(0.0, float(parsed.get("confidence", 0.7)))),
                    "description": parsed.get("description", "Classified by VLM"),
                }
        except (json.JSONDecodeError, ValueError):
            pass

    return classify_description(text)

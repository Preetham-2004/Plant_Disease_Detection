import asyncio

import requests

from ..config import GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL


def _build_prompt(disease_label: str, plant_part: str, language: str) -> str:
    return (
        f"You are an expert plant pathologist and agronomist. "
        f"The detected disease label is '{disease_label}' and the affected plant part is '{plant_part}'.\n\n"
        f"Your job is to give a treatment plan that is as accurate as possible for that exact disease and that exact plant part. "
        f"Do not give generic advice unless the disease label is too broad. "
        f"Only suggest treatments that are commonly used for this type of disease. "
        f"When giving medicines, prefer active ingredient names such as copper oxychloride, mancozeb, chlorothalonil, "
        f"azoxystrobin, propiconazole, neem oil, or potassium bicarbonate when they genuinely fit the disease. "
        f"Do not invent unsafe dosages, brand names, or unrealistic cures. "
        f"If the disease could have multiple causes, say that clearly and recommend careful monitoring or local expert confirmation.\n\n"
        f"Write the content in {language}. "
        f"Keep medicine names and active ingredients in English when needed for safety and easy purchase, "
        f"but explain the recommendation in {language}. "
        f"Use simple farmer-friendly wording and keep it easy to scan in the UI.\n\n"
        f"Return exactly this format and nothing else:\n"
        f"What it means: one short sentence specific to the disease.\n"
        f"Do now:\n"
        f"- 2 short bullets that match the disease and affected part.\n"
        f"Medicines:\n"
        f"- 2 or 3 short bullets with suitable medicine or treatment options.\n"
        f"- Mention the active ingredient first.\n"
        f"Watch for:\n"
        f"- 1 short bullet about what symptom to monitor next or when to seek expert help.\n\n"
        f"Keep the section headings exactly as written in English so the UI can format them. "
        f"Only the sentence and bullet content should be in {language}. "
        f"Keep the total output under 140 words. Avoid markdown bold text, long paragraphs, or extra notes."
    )


def _fallback_plan(disease_label: str, plant_part: str) -> str:
    return (
        f"What it means: {disease_label} was detected on the {plant_part}.\n"
        f"Do now:\n"
        f"- Remove badly affected areas if practical and keep the plant isolated from healthy plants.\n"
        f"- Reduce leaf wetness, improve airflow, and monitor nearby plants for the same symptoms.\n"
        f"Medicines:\n"
        f"- Use a treatment suited to the disease type and confirm the active ingredient with a local agriculture expert.\n"
        f"- Follow the label dosage and safety instructions exactly before spraying any product.\n"
        f"Watch for:\n"
        f"- If spots spread quickly, new leaves curl, or fruit damage increases, get local expert advice soon."
    )


def _call_groq(prompt: str) -> str:
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You create concise, accurate plant disease treatment plans.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.2,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip()


def _call_gemini(prompt: str) -> str:
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [
                {
                    "parts": [
                        {"text": "You create concise, accurate plant disease treatment plans."},
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
            },
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    candidates = payload.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise ValueError("Gemini returned empty content.")
    return text


def generate_treatment_plan_sync(disease_label: str, plant_part: str, language: str) -> str:
    prompt = _build_prompt(disease_label, plant_part, language)

    if GROQ_API_KEY:
        try:
            message = _call_groq(prompt)
            if message:
                return message
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            # Common auth / quota / provider-side failures should gracefully fall through to Gemini.
            if status_code not in {401, 403, 429, 500, 502, 503, 504}:
                return "Treatment plan service is temporarily unavailable. Please try again."
        except Exception:
            pass

    if GEMINI_API_KEY:
        try:
            message = _call_gemini(prompt)
            if message:
                return message
        except Exception:
            pass

    return _fallback_plan(disease_label, plant_part)


async def generate_treatment_plan(disease_label: str, plant_part: str, language: str) -> str:
    return await asyncio.to_thread(generate_treatment_plan_sync, disease_label, plant_part, language)

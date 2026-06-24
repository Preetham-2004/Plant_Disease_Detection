import asyncio
import requests

from ..config import GROQ_API_KEY, GROQ_MODEL


def generate_treatment_plan_sync(disease_label: str, plant_part: str, language: str) -> str:
    if not GROQ_API_KEY:
        return (
            "No Groq API key configured. "
            f"Please ensure you configure the GROQ_API_KEY to receive a treatment plan for {disease_label} on the {plant_part}."
        )

    try:
        prompt = (
            f"You are an expert plant pathologist and agronomist. "
            f"The detected disease label is '{disease_label}' and the affected plant part is '{plant_part}'.\n\n"
            f"Your job is to give a treatment plan that is as accurate as possible for that exact disease and that exact plant part. "
            f"Do not give generic advice unless the disease label is too broad. "
            f"Only suggest treatments that are commonly used for this type of disease. "
            f"When giving medicines, prefer active ingredient names such as copper oxychloride, mancozeb, chlorothalonil, azoxystrobin, propiconazole, neem oil, or potassium bicarbonate when they genuinely fit the disease. "
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
        message = payload["choices"][0]["message"]["content"].strip()
        return message or "Groq returned an empty treatment plan."
    except Exception as exc:
        return f"Failed to generate treatment plan using Groq API: {exc}"


async def generate_treatment_plan(disease_label: str, plant_part: str, language: str) -> str:
    return await asyncio.to_thread(generate_treatment_plan_sync, disease_label, plant_part, language)

from __future__ import annotations

import json
import re
import requests


def call_mistral(prompt: str, api_key: str, model: str, temperature: float, max_tokens: int) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def build_prompt(question: dict, fields: dict, chunks: list[dict], description: str) -> str:
    options = question.get("options") or []
    context = question.get("context") or ""
    question_text = question.get("question") or ""
    question_id = question.get("question_id") or ""

    fields_text = "\n".join(f"- {key}: {value}" for key, value in fields.items() if value)
    chunks_text = "\n\n".join(
        f"[chunk score={chunk['score']:.3f} start={chunk['start_char']}]\n{chunk['text']}"
        for chunk in chunks
    )
    options_text = "\n".join(f"- {opt}" for opt in options) if options else "Aucune option fournie."

    return (
        "Tu es un assistant d'analyse de dossiers de projets.\n"
        "Ta tache: repondre a la question a partir des reponses du formulaire et des extraits.\n"
        "Retourne uniquement un JSON strict avec les cles: text, qcm.\n"
        "Si aucune option ne convient, mets qcm a \"UNKNOWN\".\n\n"
        f"QuestionID: {question_id}\n"
        f"Question: {question_text}\n"
        f"Contexte: {context}\n\n"
        "Description du projet:\n"
        f"{description}\n\n"
        "Reponses formulaire:\n"
        f"{fields_text}\n\n"
        "Extraits pertinents:\n"
        f"{chunks_text}\n\n"
        "Options QCM:\n"
        f"{options_text}\n"
    )


def parse_llm_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}

from __future__ import annotations

import json
import re
import requests


MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


def call_llm(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    base_url: str | None = None,
) -> str:
    """
    Call any OpenAI-compatible LLM API.

    Args:
        prompt: The prompt to send
        api_key: API key for authentication
        model: Model name (e.g., "mistral-large-latest", "gpt-4", "llama3")
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        base_url: API base URL. Defaults to Mistral API.
                  Examples:
                  - Mistral: https://api.mistral.ai/v1
                  - OpenAI: https://api.openai.com/v1
                  - Local Ollama: http://localhost:11434/v1
                  - Local vLLM: http://localhost:8000/v1
    """
    if base_url is None:
        base_url = MISTRAL_BASE_URL

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    print(data["choices"][0]["message"]["content"])
    return data["choices"][0]["message"]["content"]


# Backward compatibility alias
def call_mistral(prompt: str, api_key: str, model: str, temperature: float, max_tokens: int) -> str:
    return call_llm(prompt, api_key, model, temperature, max_tokens, base_url=MISTRAL_BASE_URL)


def build_prompt(question: dict, fields: dict, chunks: list[dict]) -> str:
    options = question.get("options") or []
    context = question.get("context") or ""
    question_text = question.get("question") or ""
    question_id = question.get("question_id") or ""

    fields_text = "\n".join(f"- {key}: {value}" for key, value in fields.items() if value)
    chunks_text = "\n\n".join(
        f"[chunk score={chunk['score']:.3f} start={chunk['start_char']}]\n{chunk['text']}"
        for chunk in chunks
    )
    if options:
        options_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
        valid_letters = ", ".join(chr(65 + i) for i in range(len(options)))
    else:
        options_text = "Aucune option fournie."
        valid_letters = ""

    return (
        "Tu es un assistant d'analyse de dossiers de projets.\n"
        "Ta tache: repondre a la question a partir des reponses du formulaire et des extraits.\n"
        "Ne cherche pas de liens indirects entre la question et les extraits. "
        "Si des elements de reponse ne sont pas explicitement mentionnes, ne les infere pas.\n\n"
        f"QuestionID: {question_id}\n"
        f"Question: {question_text}\n"
        f"Contexte: {context}\n\n"
        "Reponses formulaire:\n"
        f"{fields_text}\n\n"
        "Extraits pertinents:\n"
        f"{chunks_text}\n\n"
        "Options QCM:\n"
        f"{options_text}\n\n"
        "INSTRUCTIONS DE FORMAT:\n"
        f"- Reponds UNIQUEMENT avec un JSON: {{\"text\": \"...\", \"qcm\": \"...\"}}\n"
        f"- Pour qcm, utilise UNIQUEMENT une lettre parmi: {valid_letters}\n"
        "- Si l'information est absente des documents, choisis l'option qui indique cette absence.\n"
        "- Utilise UNKNOWN seulement si aucune option ne convient.\n"
        f"- Exemple de reponse valide: {{\"text\": \"Le projet mentionne X.\", \"qcm\": \"A\"}}\n"
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


def validate_qcm(answer: str, options: list[str]) -> bool:
    """Check if answer is a valid letter for the given options."""
    if not options:
        return True
    if answer == "UNKNOWN":
        return True
    valid_letters = [chr(65 + i) for i in range(len(options))]
    return answer.upper() in valid_letters


def call_llm_with_validation(
    prompt: str,
    options: list[str],
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    base_url: str | None = None,
    max_retries: int = 2,
) -> tuple[str, dict]:
    """Call LLM and validate QCM response, retry if invalid."""
    for attempt in range(max_retries + 1):
        response_text = call_llm(prompt, api_key, model, temperature, max_tokens, base_url)
        parsed = parse_llm_json(response_text)
        qcm = parsed.get("qcm", "")

        if validate_qcm(qcm, options):
            return response_text, parsed

        if attempt < max_retries:
            print(f"Invalid QCM '{qcm}', retrying ({attempt + 1}/{max_retries})...")

    # After all retries, return last response with qcm set to UNKNOWN
    parsed["qcm"] = "UNKNOWN"
    parsed["validation_failed"] = True
    return response_text, parsed


# Backward compatibility alias
def call_mistral_with_validation(
    prompt: str,
    options: list[str],
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 2,
) -> tuple[str, dict]:
    return call_llm_with_validation(
        prompt, options, api_key, model, temperature, max_tokens,
        base_url=MISTRAL_BASE_URL, max_retries=max_retries
    )

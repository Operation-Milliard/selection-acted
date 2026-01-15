from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import yaml
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]
DRIVE_FILE_ID_RE = re.compile(r"(?:file/d/|open\?id=|uc\?id=)([-\w]{10,})")
CHUNK_SIZE_CHARS = 4000
CHUNK_OVERLAP_CHARS = 400


def load_config(path: str | None) -> dict:
    data: dict = {}
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

    google_cfg = data.get("google", {})
    sheets_cfg = google_cfg.get("sheets", {})
    drive_cfg = google_cfg.get("drive", {})
    project_cfg = data.get("project", {})
    output_cfg = data.get("output", {})
    rag_cfg = data.get("rag", {})
    mistral_cfg = data.get("mistral", {})

    def env_or_default(key: str, default: str | None) -> str | None:
        value = os.getenv(key)
        return value if value is not None else default

    config = {
        "oauth_client_path": env_or_default(
            "GOOGLE_OAUTH_CLIENT_PATH",
            google_cfg.get("oauth_client_path"),
        ),
        "token_path": env_or_default("GOOGLE_TOKEN_PATH", google_cfg.get("token_path")),
        "grid_spreadsheet_id": env_or_default(
            "GRID_SPREADSHEET_ID",
            sheets_cfg.get("grid_spreadsheet_id"),
        ),
        "grid_tab": env_or_default("GRID_TAB", sheets_cfg.get("grid_tab", "Grille")),
        "responses_spreadsheet_id": env_or_default(
            "RESPONSES_SPREADSHEET_ID",
            sheets_cfg.get("responses_spreadsheet_id"),
        ),
        "responses_tab": env_or_default(
            "RESPONSES_TAB",
            sheets_cfg.get("responses_tab", "Form responses 1"),
        ),
        "drive_folder_id": env_or_default("DRIVE_FOLDER_ID", drive_cfg.get("folder_id")),
        "project_name_column": env_or_default(
            "PROJECT_NAME_COLUMN",
            project_cfg.get("name_column"),
        ),
        "description_column": env_or_default(
            "PROJECT_DESCRIPTION_COLUMN",
            project_cfg.get("description_column"),
        ),
        "file_column": env_or_default(
            "FILE_COLUMN",
            project_cfg.get("file_column"),
        ),
        "status_column": env_or_default(
            "STATUS_COLUMN",
            project_cfg.get("status_column", "Status"),
        ),
        "dry_run_report_path": env_or_default(
            "DRY_RUN_REPORT_PATH",
            output_cfg.get("dry_run_report_path", "out/dry_run_report.json"),
        ),
        "projects_output_dir": env_or_default(
            "PROJECTS_OUTPUT_DIR",
            output_cfg.get("projects_output_dir", "out/projects"),
        ),
        "llm_output_dir": env_or_default(
            "LLM_OUTPUT_DIR",
            output_cfg.get("llm_output_dir", "out/llm"),
        ),
        "results_spreadsheet_id": env_or_default(
            "RESULTS_SPREADSHEET_ID",
            output_cfg.get("results_spreadsheet_id"),
        ),
        "results_write_mode": env_or_default(
            "RESULTS_WRITE_MODE",
            output_cfg.get("results_write_mode", "skip"),
        ),
        "chunk_size_chars": int(
            env_or_default(
                "CHUNK_SIZE_CHARS",
                output_cfg.get("chunk_size_chars", CHUNK_SIZE_CHARS),
            )
        ),
        "chunk_overlap_chars": int(
            env_or_default(
                "CHUNK_OVERLAP_CHARS",
                output_cfg.get("chunk_overlap_chars", CHUNK_OVERLAP_CHARS),
            )
        ),
        "embedding_model_name": env_or_default(
            "EMBEDDING_MODEL_NAME",
            rag_cfg.get("model_name", "intfloat/multilingual-e5-small"),
        ),
        "rag_top_k": int(env_or_default("RAG_TOP_K", rag_cfg.get("top_k", 6))),
        "mistral_api_key": env_or_default("MISTRAL_API_KEY", mistral_cfg.get("api_key")),
        "mistral_model": env_or_default("MISTRAL_MODEL", mistral_cfg.get("model", "mistral-large-latest")),
        "mistral_temperature": float(
            env_or_default("MISTRAL_TEMPERATURE", mistral_cfg.get("temperature", 0.2))
        ),
        "mistral_max_tokens": int(
            env_or_default("MISTRAL_MAX_TOKENS", mistral_cfg.get("max_tokens", 512))
        ),
    }

    missing = [
        key
        for key in (
            "oauth_client_path",
            "token_path",
            "grid_spreadsheet_id",
            "responses_spreadsheet_id",
            "drive_folder_id",
            "file_column",
        )
        if not config.get(key)
    ]
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")

    if not config.get("results_spreadsheet_id"):
        config["results_spreadsheet_id"] = config["grid_spreadsheet_id"]

    return config


def get_credentials(oauth_client_path: str, token_path: str):
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(oauth_client_path, SCOPES)
            creds = flow.run_local_server(port=0)
        os.makedirs(os.path.dirname(token_path) or ".", exist_ok=True)
        with open(token_path, "w", encoding="utf-8") as handle:
            handle.write(creds.to_json())

    return creds


def get_sheet_values(credentials, spreadsheet_id: str, range_a1: str) -> list[list[str]]:
    service = build("sheets", "v4", credentials=credentials)
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=range_a1)
        .execute()
    )
    return result.get("values", [])


def update_sheet_values(credentials, spreadsheet_id: str, range_a1: str, values: list[list[str]]):
    service = build("sheets", "v4", credentials=credentials)
    body = {"values": values}
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=range_a1,
        valueInputOption="RAW",
        body=body,
    ).execute()


def list_files_in_folder(credentials, folder_id: str) -> list[dict]:
    service = build("drive", "v3", credentials=credentials)
    query = f"'{folder_id}' in parents and trashed = false"
    files = []
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageToken=page_token,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return files


def list_sheet_titles(credentials, spreadsheet_id: str) -> set[str]:
    service = build("sheets", "v4", credentials=credentials)
    response = service.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        fields="sheets(properties(title))",
    ).execute()
    return {sheet["properties"]["title"] for sheet in response.get("sheets", [])}


def clear_sheet(credentials, spreadsheet_id: str, title: str) -> None:
    service = build("sheets", "v4", credentials=credentials)
    service.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=title,
        body={},
    ).execute()


def sanitize_sheet_title(value: str) -> str:
    if not value:
        return "Project"
    cleaned = re.sub(r"[:\\\\/?*\\[\\]]", "_", value)
    cleaned = cleaned.strip()
    return cleaned or "Project"


def ensure_unique_title(base_title: str, existing: set[str]) -> str:
    title = base_title[:100]
    if title not in existing:
        return title
    for idx in range(2, 1000):
        suffix = f" ({idx})"
        trimmed = title[: 100 - len(suffix)]
        candidate = f"{trimmed}{suffix}"
        if candidate not in existing:
            return candidate
    raise ValueError("Unable to generate unique sheet title")


def ensure_sheet(credentials, spreadsheet_id: str, title: str, write_mode: str) -> str | None:
    existing = list_sheet_titles(credentials, spreadsheet_id)
    base_title = sanitize_sheet_title(title)
    mode = write_mode.lower()
    if base_title in existing:
        if mode == "skip":
            return None
        if mode == "overwrite":
            clear_sheet(credentials, spreadsheet_id, base_title)
            return base_title
        raise ValueError("results_write_mode must be 'skip' or 'overwrite'")
    service = build("sheets", "v4", credentials=credentials)
    body = {"requests": [{"addSheet": {"properties": {"title": base_title}}}]}
    service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()
    return base_title


def download_drive_file(credentials, file_id: str) -> bytes:
    service = build("drive", "v3", credentials=credentials)
    request = service.files().get_media(fileId=file_id)
    buffer = BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages).strip()


def chunk_text(text: str, size: int, overlap: int) -> list[dict]:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        chunks.append(
            {
                "start_char": start,
                "end_char": end,
                "text": text[start:end],
            }
        )
        if end == length:
            break
        start = max(0, end - overlap)
    return chunks


def load_embedding_model(model_name: str) -> SentenceTransformer:
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    print(f"Embedding {len(texts)} text(s)")
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def select_top_k_chunks(
    model: SentenceTransformer,
    question: str,
    chunks: list[dict],
    top_k: int,
) -> list[dict]:
    if not chunks:
        return []
    texts = [f"passage: {chunk['text']}" for chunk in chunks]
    chunk_embeddings = embed_texts(model, texts)
    question_embedding = embed_texts(model, [f"query: {question}"])[0]
    scores = chunk_embeddings @ question_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "score": float(scores[idx]),
            **chunks[idx],
        }
        for idx in top_indices
    ]


def select_top_k_chunks_with_embeddings(
    model: SentenceTransformer,
    question: str,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    top_k: int,
) -> list[dict]:
    if not chunks:
        return []
    question_embedding = embed_texts(model, [f"query: {question}"])[0]
    scores = chunk_embeddings @ question_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "score": float(scores[idx]),
            **chunks[idx],
        }
        for idx in top_indices
    ]


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


def parse_grid(rows: list[list[str]]) -> list[dict]:
    questions = []
    for idx, row in enumerate(rows):
        if idx == 0:
            continue
        question_id = (row[0] if len(row) > 0 else "").strip()
        question = (row[1] if len(row) > 1 else "").strip()
        q_type = (row[2] if len(row) > 2 else "").strip()
        raw_options = row[3] if len(row) > 3 else ""
        options = [opt.strip() for opt in raw_options.splitlines() if opt.strip()]
        context = (row[4] if len(row) > 4 else "").strip()
        if not question:
            continue
        questions.append(
            {
                "question_id": question_id,
                "question": question,
                "type": q_type,
                "options": options,
                "context": context,
            }
        )
    return questions


def extract_drive_file_ids(cell_value: str) -> list[str]:
    if not cell_value:
        return []
    return DRIVE_FILE_ID_RE.findall(cell_value)


def load_sources(config: dict):
    credentials = get_credentials(config["oauth_client_path"], config["token_path"])

    grid_rows = get_sheet_values(
        credentials,
        config["grid_spreadsheet_id"],
        config["grid_tab"],
    )
    responses_rows = get_sheet_values(
        credentials,
        config["responses_spreadsheet_id"],
        config["responses_tab"],
    )
    drive_files = list_files_in_folder(credentials, config["drive_folder_id"])

    return credentials, grid_rows, responses_rows, drive_files


def build_report(
    grid_rows: list[list[str]],
    responses_rows: list[list[str]],
    drive_files: list[dict],
    project_name_column: str | None,
    file_column: str,
) -> dict:
    grid_questions = parse_grid(grid_rows)

    if not responses_rows:
        headers = []
        response_rows = []
    else:
        headers = responses_rows[0]
        response_rows = responses_rows[1:]

    drive_files_index = {item["id"]: item for item in drive_files}

    project_name_idx = None
    if project_name_column in headers:
        project_name_idx = headers.index(project_name_column)

    if file_column not in headers:
        raise ValueError(f"Missing file column '{file_column}' in responses sheet")
    file_column_idx = headers.index(file_column)

    responses = []
    for row_idx, row in enumerate(response_rows, start=2):
        project_name = row[project_name_idx] if project_name_idx is not None and len(row) > project_name_idx else ""
        row_map = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
        file_ids = extract_drive_file_ids(row[file_column_idx] if len(row) > file_column_idx else "")
        file_info = []
        for file_id in file_ids:
            info = drive_files_index.get(file_id, {"id": file_id, "name": "(not in folder list)"})
            file_info.append(info)

        responses.append(
            {
                "row_number": row_idx,
                "project_name": project_name,
                "fields": row_map,
                "file_ids": file_ids,
                "files": file_info,
            }
        )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "grid_questions_count": len(grid_questions),
        "responses_count": len(responses),
        "grid_questions": grid_questions,
        "responses": responses,
    }


def sanitize_filename(value: str) -> str:
    if not value:
        return "project"
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    return safe.strip("_") or "project"


def write_project_json(output_dir: Path, project_name: str, row_number: int, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sanitize_filename(project_name)}_{row_number}.json"
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def export_projects(
    credentials,
    grid_questions: list[dict],
    responses_rows: list[list[str]],
    drive_files: list[dict],
    project_name_column: str | None,
    file_column: str,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    output_dir: Path,
) -> list[Path]:
    if not responses_rows:
        return []

    headers = responses_rows[0]
    rows = responses_rows[1:]
    drive_files_index = {item["id"]: item for item in drive_files}

    project_name_idx = None
    if project_name_column in headers:
        project_name_idx = headers.index(project_name_column)

    if file_column not in headers:
        raise ValueError(f"Missing file column '{file_column}' in responses sheet")
    file_column_idx = headers.index(file_column)

    written = []
    for row_idx, row in enumerate(rows, start=2):
        project_name = row[project_name_idx] if project_name_idx is not None and len(row) > project_name_idx else ""
        row_map = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
        file_ids = extract_drive_file_ids(row[file_column_idx] if len(row) > file_column_idx else "")

        files_payload = []
        for file_id in file_ids:
            info = drive_files_index.get(file_id, {"id": file_id, "name": "(not in folder list)"})
            text = ""
            try:
                content = download_drive_file(credentials, file_id)
            except Exception as exc:
                print(f"[{row_idx}] download failed for {file_id}: {exc}")
                content = None
            if content is not None and info.get("mimeType") == "application/pdf":
                try:
                    text = extract_pdf_text(content)
                except Exception as exc:
                    print(f"[{row_idx}] extract failed for {info.get('name', file_id)}: {exc}")
            elif content is not None:
                print(f"[{row_idx}] skipped non-pdf {info.get('name', file_id)} ({info.get('mimeType')})")
            if text:
                char_count = len(text)
                line_count = text.count("\n") + 1
                print(f"[{row_idx}] extracted {char_count} chars / {line_count} lines from {info.get('name', file_id)}")
            else:
                print(f"[{row_idx}] no text extracted from {info.get('name', file_id)}")
            chunks = chunk_text(text, size=chunk_size_chars, overlap=chunk_overlap_chars)
            if chunks:
                print(f"[{row_idx}] {len(chunks)} chunks for {info.get('name', file_id)}")
            files_payload.append(
                {
                    "id": file_id,
                    "name": info.get("name"),
                    "mimeType": info.get("mimeType"),
                    "text": text,
                    "chunks": chunks,
                }
            )

        payload = {
            "row_number": row_idx,
            "project_name": project_name,
            "fields": row_map,
            "grid_questions": grid_questions,
            "files": files_payload,
        }
        written.append(write_project_json(output_dir, project_name, row_idx, payload))

    return written


def write_project_sheets(
    credentials,
    llm_dir: Path,
    spreadsheet_id: str,
    write_mode: str,
) -> None:
    for path in sorted(llm_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        project_name = data.get("project_name", "")
        row_number = data.get("row_number", "")
        answers = data.get("answers", [])

        if not answers:
            print(f"Skipping {path.name}: no answers")
            continue

        title = f"{project_name}_{row_number}" if project_name else f"Project_{row_number}"
        sheet_title = ensure_sheet(credentials, spreadsheet_id, title, write_mode)
        if sheet_title is None:
            print(f"Skipping existing sheet for {path.name}")
            continue

        rows = [["Question", "TEXT", "QCM"]]
        for entry in answers:
            rows.append(
                [
                    entry.get("question", "") or entry.get("question_id", ""),
                    entry.get("text", ""),
                    entry.get("qcm", ""),
                ]
            )

        range_a1 = f"{sheet_title}!A1:C{len(rows)}"
        update_sheet_values(credentials, spreadsheet_id, range_a1, rows)
        print(f"Wrote sheet {sheet_title} for {path.name}")


def column_index_to_letter(index: int) -> str:
    index += 1
    letters = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def update_status_column(
    credentials,
    spreadsheet_id: str,
    tab_name: str,
    responses_rows: list[list[str]],
    status_column: str,
    status_value: str,
) -> int:
    if not responses_rows:
        return 0

    headers = responses_rows[0]
    rows = responses_rows[1:]

    if status_column not in headers:
        raise ValueError(f"Missing status column '{status_column}' in responses sheet")

    status_index = headers.index(status_column)
    status_letter = column_index_to_letter(status_index)

    values = []
    updated_count = 0
    for row in rows:
        existing = row[status_index] if len(row) > status_index else ""
        if existing.strip():
            values.append([existing])
        else:
            values.append([status_value])
            updated_count += 1

    if values:
        range_a1 = f"{tab_name}!{status_letter}2:{status_letter}{len(rows) + 1}"
        update_sheet_values(credentials, spreadsheet_id, range_a1, values)

    return updated_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run: read grid + responses + drive folder.")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument(
        "--mark-status",
        action="store_true",
        help="Write a human-readable date in the Status column for unprocessed rows",
    )
    parser.add_argument(
        "--export-projects",
        action="store_true",
        help="Download PDFs and write one JSON per project to out/projects",
    )
    parser.add_argument(
        "--test-embeddings",
        action="store_true",
        help="Load embedding model and run a small test encoding",
    )
    parser.add_argument(
        "--llm-generate",
        action="store_true",
        help="Generate LLM answers from project JSON files",
    )
    parser.add_argument(
        "--write-sheets",
        action="store_true",
        help="Write LLM answers to one sheet per project",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    credentials, grid_rows, responses_rows, drive_files = load_sources(config)

    grid_questions = parse_grid(grid_rows)

    report = build_report(
        grid_rows,
        responses_rows,
        drive_files,
        config.get("project_name_column"),
        config["file_column"],
    )

    output_path = Path(config["dry_run_report_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Dry-run report written to {output_path}")

    if args.test_embeddings:
        model = load_embedding_model(config["embedding_model_name"])
        vector = embed_texts(model, ["test"])
        print(f"Embedding test vector shape: {vector.shape}")

    if args.export_projects:
        output_dir = Path(config["projects_output_dir"])
        written = export_projects(
            credentials,
            grid_questions,
            responses_rows,
            drive_files,
            config.get("project_name_column"),
            config["file_column"],
            config["chunk_size_chars"],
            config["chunk_overlap_chars"],
            output_dir,
        )
        print(f"Wrote {len(written)} project JSON files to {output_dir}")

    if args.llm_generate:
        if not config.get("mistral_api_key"):
            raise ValueError("Missing Mistral API key (set MISTRAL_API_KEY or config.mistral.api_key)")
        model = load_embedding_model(config["embedding_model_name"])
        input_dir = Path(config["projects_output_dir"])
        output_dir = Path(config["llm_output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(input_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            fields = data.get("fields", {})
            description = fields.get(config.get("description_column") or "", "")
            grid_questions = data.get("grid_questions", [])
            files = data.get("files", [])

            chunks = []
            for file_entry in files:
                for idx, chunk in enumerate(file_entry.get("chunks", [])):
                    chunks.append(
                        {
                            **chunk,
                            "file_id": file_entry.get("id"),
                            "file_name": file_entry.get("name"),
                            "chunk_index": idx,
                        }
                    )

            chunk_embeddings = None
            if chunks:
                chunk_texts = [f"passage: {chunk['text']}" for chunk in chunks]
                chunk_embeddings = embed_texts(model, chunk_texts)

            answers = []
            for question in grid_questions:
                if chunks and chunk_embeddings is not None:
                    selected = select_top_k_chunks_with_embeddings(
                        model,
                        question.get("question", ""),
                        chunks,
                        chunk_embeddings,
                        config["rag_top_k"],
                    )
                else:
                    selected = []
                prompt = build_prompt(question, fields, selected, description)
                response_text = call_mistral(
                    prompt,
                    api_key=config["mistral_api_key"],
                    model=config["mistral_model"],
                    temperature=config["mistral_temperature"],
                    max_tokens=config["mistral_max_tokens"],
                )
                parsed = parse_llm_json(response_text)
                answers.append(
                    {
                        "question_id": question.get("question_id"),
                        "question": question.get("question"),
                        "text": parsed.get("text", ""),
                        "qcm": parsed.get("qcm", ""),
                        "chunks_used": selected,
                        "prompt": prompt,
                        "raw_response": response_text,
                    }
                )

            output = {
                **data,
                "llm": {
                    "model": config["mistral_model"],
                    "temperature": config["mistral_temperature"],
                    "max_tokens": config["mistral_max_tokens"],
                    "rag_top_k": config["rag_top_k"],
                    "note": "Rerank could improve chunk selection if quality is insufficient.",
                },
                "answers": answers,
            }
            out_path = output_dir / path.name
            out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote LLM answers to {out_path}")

    if args.write_sheets:
        llm_dir = Path(config["llm_output_dir"])
        write_project_sheets(
            credentials,
            llm_dir,
            config["results_spreadsheet_id"],
            config["results_write_mode"],
        )

    if args.mark_status:
        status_value = datetime.now().strftime("%Y-%m-%d %H:%M")
        updated = update_status_column(
            credentials,
            config["responses_spreadsheet_id"],
            config["responses_tab"],
            responses_rows,
            config.get("status_column", "Status"),
            status_value,
        )
        print(f"Status updated for {updated} rows")


if __name__ == "__main__":
    main()

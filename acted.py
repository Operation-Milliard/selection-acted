from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import yaml
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]
DRIVE_FILE_ID_RE = re.compile(r"(?:file/d/|open\?id=|uc\?id=)([-\w]{10,})")


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
            files_payload.append(
                {
                    "id": file_id,
                    "name": info.get("name"),
                    "mimeType": info.get("mimeType"),
                    "text": text,
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

    if args.export_projects:
        output_dir = Path("out/projects")
        written = export_projects(
            credentials,
            grid_questions,
            responses_rows,
            drive_files,
            config.get("project_name_column"),
            config["file_column"],
            output_dir,
        )
        print(f"Wrote {len(written)} project JSON files to {output_dir}")

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

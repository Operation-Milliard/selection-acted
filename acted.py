from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from lib.config import AppConfig, load_config
from lib.google import (
    add_sheet,
    clear_sheet,
    download_drive_file,
    get_credentials,
    get_sheet_values,
    list_files_in_folder,
    list_sheet_titles,
    update_sheet_values,
)
from lib.llm import build_prompt, call_mistral_with_validation
from lib.pdf import chunk_text, extract_drive_file_ids, extract_pdf_text
from lib.rag import embed_texts, load_embedding_model, select_top_k_chunks_with_embeddings









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
    add_sheet(credentials, spreadsheet_id, base_title)
    return base_title




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



def load_sources(config: AppConfig):
    credentials = get_credentials(config.oauth_client_path, config.token_path)

    grid_rows = get_sheet_values(
        credentials,
        config.grid_spreadsheet_id,
        config.grid_tab,
    )
    responses_rows = get_sheet_values(
        credentials,
        config.responses_spreadsheet_id,
        config.responses_tab,
    )
    drive_files = list_files_in_folder(credentials, config.drive_folder_id)

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
        config.project_name_column,
        config.file_column,
    )

    output_path = Path(config.dry_run_report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Dry-run report written to {output_path}")

    if args.test_embeddings:
        model = load_embedding_model(config.embedding_model_name)
        vector = embed_texts(model, ["test"])
        print(f"Embedding test vector shape: {vector.shape}")

    if args.export_projects:
        output_dir = Path(config.projects_output_dir)
        written = export_projects(
            credentials,
            grid_questions,
            responses_rows,
            drive_files,
            config.project_name_column,
            config.file_column,
            config.chunk_size_chars,
            config.chunk_overlap_chars,
            output_dir,
        )
        print(f"Wrote {len(written)} project JSON files to {output_dir}")

    if args.llm_generate:
        if not config.mistral_api_key:
            raise ValueError("Missing Mistral API key (set MISTRAL_API_KEY or config.mistral.api_key)")
        model = load_embedding_model(config.embedding_model_name)
        input_dir = Path(config.projects_output_dir)
        output_dir = Path(config.llm_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(input_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            fields = data.get("fields", {})
            description = fields.get(config.description_column or "", "")
            print(description)
            if config.project_name_column and config.project_name_column in fields:
                prompt_fields = {config.project_name_column: fields.get(config.project_name_column, "")}
            else:
                prompt_fields = {}
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
                        config.rag_top_k,
                    )
                else:
                    selected = []
                prompt = build_prompt(question, prompt_fields, selected, description)
                response_text, parsed = call_mistral_with_validation(
                    prompt,
                    options=question.get("options", []),
                    api_key=config.mistral_api_key,
                    model=config.mistral_model,
                    temperature=config.mistral_temperature,
                    max_tokens=config.mistral_max_tokens,
                )
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
                    "model": config.mistral_model,
                    "temperature": config.mistral_temperature,
                    "max_tokens": config.mistral_max_tokens,
                    "rag_top_k": config.rag_top_k,
                    "note": "Rerank could improve chunk selection if quality is insufficient.",
                },
                "answers": answers,
            }
            out_path = output_dir / path.name
            out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote LLM answers to {out_path}")

    if args.write_sheets:
        llm_dir = Path(config.llm_output_dir)
        write_project_sheets(
            credentials,
            llm_dir,
            config.results_spreadsheet_id,
            config.results_write_mode,
        )

    if args.mark_status:
        status_value = datetime.now().strftime("%Y-%m-%d %H:%M")
        updated = update_status_column(
            credentials,
            config.responses_spreadsheet_id,
            config.responses_tab,
            responses_rows,
            config.status_column,
            status_value,
        )
        print(f"Status updated for {updated} rows")


if __name__ == "__main__":
    main()

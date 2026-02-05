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
    send_email_with_attachment,
    update_sheet_values,
)
from lib.reviewer import (
    assign_reviewers_round_robin,
    build_email_body,
    build_form_url,
    find_html_reports,
    load_reviewers,
)
from lib.html_report import convert_all_json_to_html
from lib.llm import build_prompt, call_llm_with_validation
from lib.pdf import chunk_text, chunk_text_smart, extract_drive_file_ids, extract_pdf_markdown, extract_pdf_text
from lib.rag import embed_texts, load_embedding_model, select_top_k_chunks_with_embeddings









def validate_columns(
    headers: list[str],
    config: AppConfig,
) -> list[str]:
    """Validate configured columns exist in headers.

    Returns list of missing column descriptions.
    """
    missing = []
    print("HEADERS:")
    print(headers)
    for col in config.file_columns:
        if col not in headers:
            missing.append(f"file_columns: {col}")

    if config.project_name_column and config.project_name_column not in headers:
        missing.append(f"name_column: {config.project_name_column}")

    for field in config.prompt_fields:
        if field not in headers:
            missing.append(f"prompt_fields: {field}")

    return missing


def sanitize_sheet_title(value: str) -> str:
    if not value:
        return "Project"
    cleaned = re.sub(r"[:\\/?\*\[\]]", "_", value)
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
    file_columns: list[str],
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

    # Build indices for file columns that exist in headers
    file_column_indices = []
    for file_column in file_columns:
        if file_column in headers:
            file_column_indices.append(headers.index(file_column))

    responses = []
    for row_idx, row in enumerate(response_rows, start=2):
        project_name = row[project_name_idx] if project_name_idx is not None and len(row) > project_name_idx else ""
        row_map = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}

        # Extract file IDs from all file columns
        file_ids = []
        for file_column_idx in file_column_indices:
            cell_value = row[file_column_idx] if len(row) > file_column_idx else ""
            file_ids.extend(extract_drive_file_ids(cell_value))

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
    file_columns: list[str],
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    chunk_min_chars: int,
    chunk_max_chars: int,
    chunk_overlap_sentences: int,
    use_smart_chunking: bool,
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

    # Build indices for file columns that exist in headers
    file_column_indices = []
    for file_column in file_columns:
        if file_column in headers:
            file_column_indices.append(headers.index(file_column))

    written = []
    for row_idx, row in enumerate(rows, start=2):
        project_name = row[project_name_idx] if project_name_idx is not None and len(row) > project_name_idx else ""
        row_map = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}

        # Extract file IDs from all file columns
        file_ids = []
        for file_column_idx in file_column_indices:
            cell_value = row[file_column_idx] if len(row) > file_column_idx else ""
            file_ids.extend(extract_drive_file_ids(cell_value))

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
                    if use_smart_chunking:
                        text = extract_pdf_markdown(content)
                    else:
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
            if use_smart_chunking:
                chunks = chunk_text_smart(
                    text,
                    min_chars=chunk_min_chars,
                    max_chars=chunk_max_chars,
                    overlap_sentences=chunk_overlap_sentences,
                )
            else:
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


def send_review_emails(config: AppConfig, credentials, dry_run: bool = False) -> None:
    """Assign reviewers to projects and send review emails with HTML reports."""
    # Validate config
    if not config.reviewers_file_path:
        raise ValueError("Missing reviewers.file_path in config")
    if not config.google_form_url:
        raise ValueError("Missing reviewers.google_form_url in config")
    if not config.form_project_field:
        raise ValueError("Missing reviewers.form_project_field in config")

    # Load reviewers
    print(f"Loading reviewers from {config.reviewers_file_path}...")
    reviewers = load_reviewers(config.reviewers_file_path)
    print(f"  {len(reviewers)} reviewers found.")

    # Find HTML reports
    llm_dir = Path(config.llm_output_dir)
    reports = find_html_reports(llm_dir)
    print(f"Found {len(reports)} projects with HTML reports.")

    if not reports:
        print("No HTML reports found. Run --export-html first.")
        return

    # Assign reviewers
    project_names = list(reports.keys())
    print(f"\nAssigning reviewers (round-robin, {config.reviewers_per_project} per project)...")
    assignments = assign_reviewers_round_robin(
        project_names,
        reviewers,
        config.reviewers_per_project,
    )

    # Send emails
    print("\nSending review emails...")
    sent_count = 0
    failed_count = 0

    for project_name, assigned_reviewers in assignments.items():
        html_path = reports[project_name]
        form_url = build_form_url(
            config.google_form_url,
            config.form_project_field,
            project_name,
        )

        for reviewer_email in assigned_reviewers:
            subject = config.email_subject_template.format(project_name=project_name)
            body = build_email_body(project_name, form_url, config.email_from_name)

            if dry_run:
                print(f"  [DRY-RUN] {project_name} -> {reviewer_email}")
                sent_count += 1
            else:
                try:
                    send_email_with_attachment(
                        credentials,
                        to=reviewer_email,
                        subject=subject,
                        body=body,
                        attachment_path=html_path,
                        from_name=config.email_from_name,
                    )
                    print(f"  OK {project_name} -> {reviewer_email}")
                    sent_count += 1
                except Exception as e:
                    print(f"  FAILED {project_name} -> {reviewer_email}: {e}")
                    failed_count += 1

    # Summary
    if dry_run:
        print(f"\nDry-run complete: {sent_count} emails would be sent.")
    else:
        print(f"\nSummary: {sent_count} emails sent successfully.")
        if failed_count:
            print(f"         {failed_count} emails failed.")


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
    parser.add_argument(
        "--export-html",
        action="store_true",
        help="Export LLM answers as HTML reports",
    )
    parser.add_argument(
        "--send-reviews",
        action="store_true",
        help="Assign reviewers and send review emails with HTML reports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually sending emails",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    credentials, grid_rows, responses_rows, drive_files = load_sources(config)

    # Validate columns interactively
    if responses_rows:
        headers = responses_rows[0]
        missing = validate_columns(headers, config)

        print("\nChecking column configuration...")
        for col in config.file_columns:
            status = "✓" if col in headers else "✗"
            print(f"  {status} file_columns: {col}")
        if config.project_name_column:
            status = "✓" if config.project_name_column in headers else "✗"
            print(f"  {status} name_column: {config.project_name_column}")
        for field in config.prompt_fields:
            status = "✓" if field in headers else "✗"
            print(f"  {status} prompt_fields: {field}")

        if missing:
            print(f"\nWarning: {len(missing)} column(s) not found.")
            response = input("Continue anyway? [y/N]: ").strip().lower()
            if response != "y":
                raise SystemExit(0)
        else:
            print("All columns found.\n")

    grid_questions = parse_grid(grid_rows)

    report = build_report(
        grid_rows,
        responses_rows,
        drive_files,
        config.project_name_column,
        config.file_columns,
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
            config.file_columns,
            config.chunk_size_chars,
            config.chunk_overlap_chars,
            config.chunk_min_chars,
            config.chunk_max_chars,
            config.chunk_overlap_sentences,
            config.use_smart_chunking,
            output_dir,
        )
        print(f"Wrote {len(written)} project JSON files to {output_dir}")

    if args.llm_generate:
        if not config.llm_api_key and not config.llm_base_url:
            raise ValueError("Missing LLM API key (set LLM_API_KEY or config.llm.api_key)")
        model = load_embedding_model(config.embedding_model_name)
        input_dir = Path(config.projects_output_dir)
        output_dir = Path(config.llm_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(input_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            fields = data.get("fields", {})
            # Build prompt_fields dict from config.prompt_fields list
            prompt_fields = {key: fields.get(key, "") for key in config.prompt_fields if key in fields}
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
                prompt = build_prompt(question, prompt_fields, selected)
                print("PROMPT:")
                print(prompt)
                response_text, parsed = call_llm_with_validation(
                    prompt,
                    options=question.get("options", []),
                    api_key=config.llm_api_key,
                    model=config.llm_model,
                    temperature=config.llm_temperature,
                    max_tokens=config.llm_max_tokens,
                    base_url=config.llm_base_url,
                )
                qcm_letter = parsed.get("qcm", "")
                qcm_text = ""
                if qcm_letter and qcm_letter != "UNKNOWN":
                    options = question.get("options", [])
                    idx = ord(qcm_letter.upper()) - 65
                    if 0 <= idx < len(options):
                        qcm_text = options[idx]
                answers.append(
                    {
                        "question_id": question.get("question_id"),
                        "question": question.get("question"),
                        "text": parsed.get("text", ""),
                        "qcm": qcm_letter,
                        "qcm_text": qcm_text,
                        "chunks_used": selected,
                        "prompt": prompt,
                        "raw_response": response_text,
                    }
                )

            output = {
                **data,
                "llm": {
                    "base_url": config.llm_base_url,
                    "model": config.llm_model,
                    "temperature": config.llm_temperature,
                    "max_tokens": config.llm_max_tokens,
                    "rag_top_k": config.rag_top_k,
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

    if args.export_html:
        llm_dir = Path(config.llm_output_dir)
        written = convert_all_json_to_html(llm_dir)
        print(f"Exported {len(written)} HTML reports to {llm_dir}")

    if args.send_reviews:
        send_review_emails(config, credentials, dry_run=args.dry_run)

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

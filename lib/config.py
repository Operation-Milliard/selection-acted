from __future__ import annotations

from dataclasses import dataclass
import os
import yaml

CHUNK_SIZE_CHARS = 4000
CHUNK_OVERLAP_CHARS = 400
CHUNK_MIN_CHARS = 500
CHUNK_MAX_CHARS = 2000
CHUNK_OVERLAP_SENTENCES = 1


@dataclass
class AppConfig:
    oauth_client_path: str
    token_path: str
    grid_spreadsheet_id: str
    grid_tab: str
    responses_spreadsheet_id: str
    responses_tab: str
    drive_folder_id: str
    project_name_column: str | None
    description_column: str | None
    file_column: str
    status_column: str
    dry_run_report_path: str
    projects_output_dir: str
    llm_output_dir: str
    results_spreadsheet_id: str
    results_write_mode: str
    chunk_size_chars: int
    chunk_overlap_chars: int
    chunk_min_chars: int
    chunk_max_chars: int
    chunk_overlap_sentences: int
    use_smart_chunking: bool
    embedding_model_name: str
    rag_top_k: int
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int


def load_config(path: str | None) -> AppConfig:
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
    # Support both "llm" and legacy "mistral" config sections
    llm_cfg = data.get("llm", {}) or data.get("mistral", {})

    def env_or_default(key: str, default: str | None) -> str | None:
        value = os.getenv(key)
        return value if value is not None else default

    oauth_client_path = env_or_default("GOOGLE_OAUTH_CLIENT_PATH", google_cfg.get("oauth_client_path"))
    token_path = env_or_default("GOOGLE_TOKEN_PATH", google_cfg.get("token_path"))
    grid_spreadsheet_id = env_or_default("GRID_SPREADSHEET_ID", sheets_cfg.get("grid_spreadsheet_id"))
    responses_spreadsheet_id = env_or_default("RESPONSES_SPREADSHEET_ID", sheets_cfg.get("responses_spreadsheet_id"))
    drive_folder_id = env_or_default("DRIVE_FOLDER_ID", drive_cfg.get("folder_id"))

    missing = [
        key
        for key, value in (
            ("oauth_client_path", oauth_client_path),
            ("token_path", token_path),
            ("grid_spreadsheet_id", grid_spreadsheet_id),
            ("responses_spreadsheet_id", responses_spreadsheet_id),
            ("drive_folder_id", drive_folder_id),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")

    file_column = env_or_default("FILE_COLUMN", project_cfg.get("file_column"))
    if not file_column:
        raise ValueError("Missing project.file_column in config")

    results_spreadsheet_id = env_or_default("RESULTS_SPREADSHEET_ID", output_cfg.get("results_spreadsheet_id"))
    if not results_spreadsheet_id:
        results_spreadsheet_id = grid_spreadsheet_id

    return AppConfig(
        oauth_client_path=oauth_client_path,
        token_path=token_path,
        grid_spreadsheet_id=grid_spreadsheet_id,
        grid_tab=env_or_default("GRID_TAB", sheets_cfg.get("grid_tab", "Grille")) or "Grille",
        responses_spreadsheet_id=responses_spreadsheet_id,
        responses_tab=env_or_default("RESPONSES_TAB", sheets_cfg.get("responses_tab", "Form responses 1"))
        or "Form responses 1",
        drive_folder_id=drive_folder_id,
        project_name_column=env_or_default("PROJECT_NAME_COLUMN", project_cfg.get("name_column")),
        description_column=env_or_default("PROJECT_DESCRIPTION_COLUMN", project_cfg.get("description_column")),
        file_column=file_column,
        status_column=env_or_default("STATUS_COLUMN", project_cfg.get("status_column", "Status")) or "Status",
        dry_run_report_path=env_or_default(
            "DRY_RUN_REPORT_PATH",
            output_cfg.get("dry_run_report_path", "out/dry_run_report.json"),
        )
        or "out/dry_run_report.json",
        projects_output_dir=env_or_default(
            "PROJECTS_OUTPUT_DIR",
            output_cfg.get("projects_output_dir", "out/projects"),
        )
        or "out/projects",
        llm_output_dir=env_or_default(
            "LLM_OUTPUT_DIR",
            output_cfg.get("llm_output_dir", "out/llm"),
        )
        or "out/llm",
        results_spreadsheet_id=results_spreadsheet_id,
        results_write_mode=env_or_default(
            "RESULTS_WRITE_MODE",
            output_cfg.get("results_write_mode", "skip"),
        )
        or "skip",
        chunk_size_chars=int(
            env_or_default("CHUNK_SIZE_CHARS", output_cfg.get("chunk_size_chars", CHUNK_SIZE_CHARS))
            or CHUNK_SIZE_CHARS
        ),
        chunk_overlap_chars=int(
            env_or_default("CHUNK_OVERLAP_CHARS", output_cfg.get("chunk_overlap_chars", CHUNK_OVERLAP_CHARS))
            or CHUNK_OVERLAP_CHARS
        ),
        chunk_min_chars=int(
            env_or_default("CHUNK_MIN_CHARS", rag_cfg.get("chunk_min_chars", CHUNK_MIN_CHARS))
            or CHUNK_MIN_CHARS
        ),
        chunk_max_chars=int(
            env_or_default("CHUNK_MAX_CHARS", rag_cfg.get("chunk_max_chars", CHUNK_MAX_CHARS))
            or CHUNK_MAX_CHARS
        ),
        chunk_overlap_sentences=int(
            env_or_default("CHUNK_OVERLAP_SENTENCES", rag_cfg.get("chunk_overlap_sentences", CHUNK_OVERLAP_SENTENCES))
            or CHUNK_OVERLAP_SENTENCES
        ),
        use_smart_chunking=str(
            env_or_default("USE_SMART_CHUNKING", rag_cfg.get("use_smart_chunking", "true"))
        ).lower() in ("true", "1", "yes"),
        embedding_model_name=env_or_default(
            "EMBEDDING_MODEL_NAME",
            rag_cfg.get("model_name", "intfloat/multilingual-e5-small"),
        )
        or "intfloat/multilingual-e5-small",
        rag_top_k=int(env_or_default("RAG_TOP_K", rag_cfg.get("top_k", 6)) or 6),
        llm_base_url=env_or_default("LLM_BASE_URL", llm_cfg.get("base_url")),
        llm_api_key=env_or_default("LLM_API_KEY", llm_cfg.get("api_key"))
        or env_or_default("MISTRAL_API_KEY", None),  # Backward compat
        llm_model=env_or_default("LLM_MODEL", llm_cfg.get("model", "mistral-large-latest"))
        or "mistral-large-latest",
        llm_temperature=float(
            env_or_default("LLM_TEMPERATURE", llm_cfg.get("temperature", 0.2)) or 0.2
        ),
        llm_max_tokens=int(env_or_default("LLM_MAX_TOKENS", llm_cfg.get("max_tokens", 512)) or 512),
    )

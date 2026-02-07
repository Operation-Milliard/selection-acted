from __future__ import annotations

import itertools
import json
from pathlib import Path
from urllib.parse import quote_plus


def load_reviewers(path: str) -> list[str]:
    """Load reviewer emails from text file (one per line)."""
    reviewers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            email = line.strip()
            if email and not email.startswith("#"):
                reviewers.append(email)
    return reviewers


def assign_reviewers_round_robin(
    projects: list[str],
    reviewers: list[str],
    per_project: int = 2,
) -> dict[str, list[str]]:
    """Assign reviewers to projects using round-robin distribution."""
    if not reviewers:
        raise ValueError("No reviewers provided")
    if per_project < 1:
        raise ValueError("per_project must be at least 1")

    assignments = {}
    reviewer_cycle = itertools.cycle(reviewers)
    for project in projects:
        assignments[project] = [next(reviewer_cycle) for _ in range(per_project)]
    return assignments


def build_form_url(base_url: str, field_id: str, project_name: str) -> str:
    """Build Google Form URL with pre-filled project name."""
    encoded_name = quote_plus(project_name)
    return f"{base_url}?{field_id}={encoded_name}"


def build_email_body(
    project_name: str,
    form_url: str,
    from_name: str,
) -> str:
    """Build email body text in French."""
    return f"""Bonjour,

Vous avez été assigné(e) à la revue du projet "{project_name}".

Veuillez trouver en pièce jointe le rapport d'évaluation.

Pour soumettre votre avis, utilisez ce formulaire:
{form_url}

Cordialement,
{from_name}
"""


def find_html_reports(llm_output_dir: Path) -> dict[str, Path]:
    """Find all HTML reports and map project names to file paths.

    Returns dict mapping project_name to HTML file path.
    Reads the original project name from the corresponding JSON file.
    """
    reports = {}
    for html_path in llm_output_dir.glob("*.html"):
        # Get project name from corresponding JSON file
        json_path = html_path.with_suffix(".json")
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            project_name = data.get("project_name", html_path.stem)
        else:
            # Fallback to filename if JSON doesn't exist
            stem = html_path.stem
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                project_name = parts[0]
            else:
                project_name = stem
        reports[project_name] = html_path
    return reports

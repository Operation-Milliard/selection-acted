from __future__ import annotations

import html
import json
from pathlib import Path


def generate_html_report(data: dict) -> str:
    """Generate an HTML report from LLM answer JSON data."""
    project_name = html.escape(data.get("project_name", "Unknown"))
    row_number = data.get("row_number", "")
    fields = data.get("fields", {})
    answers = data.get("answers", [])
    llm_info = data.get("llm", {})

    fields_html = ""
    for key, value in fields.items():
        if value:
            fields_html += f"<tr><td><strong>{html.escape(key)}</strong></td><td>{html.escape(str(value))}</td></tr>\n"

    answers_html = ""
    for idx, answer in enumerate(answers, 1):
        question_id = html.escape(answer.get("question_id", ""))
        question = html.escape(answer.get("question", ""))
        text = html.escape(answer.get("text", ""))
        qcm = html.escape(answer.get("qcm", ""))
        qcm_text = html.escape(answer.get("qcm_text", ""))
        chunks_used = answer.get("chunks_used", [])

        qcm_display = f"<span class='qcm-letter'>{qcm}</span>"
        if qcm_text:
            qcm_display += f" - {qcm_text}"

        chunks_html = ""
        if chunks_used:
            chunks_html = "<details><summary>Chunks utilisés ({} extraits)</summary><div class='chunks'>".format(len(chunks_used))
            for chunk in chunks_used:
                score = chunk.get("score", 0)
                chunk_text = html.escape(chunk.get("text", ""))
                file_name = html.escape(chunk.get("file_name", ""))
                chunks_html += f"""
                <div class='chunk'>
                    <div class='chunk-meta'>Score: {score:.3f} | Fichier: {file_name}</div>
                    <div class='chunk-text'>{chunk_text}</div>
                </div>
                """
            chunks_html += "</div></details>"

        answers_html += f"""
        <div class='answer'>
            <div class='question-header'>
                <span class='question-id'>{question_id}</span>
                <span class='question-text'>{question}</span>
            </div>
            <div class='response'>
                <div class='response-text'><strong>Réponse:</strong> {text}</div>
                <div class='response-qcm'><strong>QCM:</strong> {qcm_display}</div>
            </div>
            {chunks_html}
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport - {project_name}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .project-info {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .project-info table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .project-info td {{
            padding: 8px;
            border-bottom: 1px solid #eee;
            vertical-align: top;
        }}
        .project-info td:first-child {{
            width: 30%;
            color: #666;
        }}
        .answer {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }}
        .question-header {{
            margin-bottom: 15px;
        }}
        .question-id {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            margin-right: 10px;
        }}
        .question-text {{
            font-weight: 600;
            color: #2c3e50;
        }}
        .response {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        .response-text {{
            margin-bottom: 10px;
        }}
        .response-qcm {{
            color: #27ae60;
        }}
        .qcm-letter {{
            display: inline-block;
            background: #27ae60;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        details {{
            margin-top: 15px;
        }}
        summary {{
            cursor: pointer;
            color: #666;
            font-size: 0.9em;
        }}
        summary:hover {{
            color: #3498db;
        }}
        .chunks {{
            margin-top: 10px;
        }}
        .chunk {{
            background: #fff9e6;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
            border: 1px solid #f0e6cc;
        }}
        .chunk-meta {{
            font-size: 0.8em;
            color: #888;
            margin-bottom: 5px;
        }}
        .chunk-text {{
            font-size: 0.9em;
            white-space: pre-wrap;
            max-height: 150px;
            overflow-y: auto;
        }}
        .llm-info {{
            background: #ecf0f1;
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 0.85em;
            color: #666;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>{project_name}</h1>

    <div class="llm-info">
        Modèle: {html.escape(llm_info.get('model', 'N/A'))} |
        Température: {llm_info.get('temperature', 'N/A')} |
        Top-K RAG: {llm_info.get('rag_top_k', 'N/A')}
    </div>

    <h2>Informations du projet</h2>
    <div class="project-info">
        <table>
            {fields_html}
        </table>
    </div>

    <h2>Réponses ({len(answers)} questions)</h2>
    {answers_html}
</body>
</html>
"""


def convert_json_to_html(json_path: Path, output_path: Path | None = None) -> Path:
    """Convert a JSON file to HTML report."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    html_content = generate_html_report(data)

    if output_path is None:
        output_path = json_path.with_suffix(".html")

    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def convert_all_json_to_html(input_dir: Path, output_dir: Path | None = None) -> list[Path]:
    """Convert all JSON files in a directory to HTML reports."""
    if output_dir is None:
        output_dir = input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for json_path in sorted(input_dir.glob("*.json")):
        output_path = output_dir / json_path.with_suffix(".html").name
        convert_json_to_html(json_path, output_path)
        written.append(output_path)
        print(f"Generated {output_path}")

    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert LLM JSON to HTML report")
    parser.add_argument("input", help="JSON file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if input_path.is_file():
        result = convert_json_to_html(input_path, output_path)
        print(f"Generated {result}")
    elif input_path.is_dir():
        results = convert_all_json_to_html(input_path, output_path)
        print(f"Generated {len(results)} HTML files")
    else:
        print(f"Error: {input_path} not found")

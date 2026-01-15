# Acted - Project Selection Assistant

Day 1 deliverable: read Google Forms responses + grid, list Drive file links, and generate a dry-run report.

## Setup (OAuth user)
1) Create an OAuth client (Desktop app) and download the JSON to `secrets/client_secret.json`.
2) On first run, the script opens a browser to authorize access to Sheets + Drive.
3) Add a `Status` column in the responses sheet to track processed rows.
4) Set `project.file_column` in `config.yaml` to the upload column name.

## Install
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure
Copy the example config and edit it:
```
cp config.example.yaml config.yaml
```

## Run dry-run
```
python acted.py --config config.yaml
```

## Export projects JSON
```
python acted.py --config config.yaml --export-projects
```

Chunking is configured in `config.yaml` under `output.chunk_size_chars` and `output.chunk_overlap_chars`.

Local embeddings (for RAG) are configured under `rag.model_name` and `rag.top_k`.

## Generate LLM answers (Mistral)
Set `mistral.api_key` in `config.yaml` or export `MISTRAL_API_KEY`.
```
python acted.py --config config.yaml --llm-generate
```

## Write LLM answers to Google Sheets
By default, results are written to the same spreadsheet as the grid. To target another spreadsheet, set `output.results_spreadsheet_id`. Choose `output.results_write_mode` as `skip` or `overwrite`.
```
python acted.py --config config.yaml --write-sheets
```

## Mark processed rows
```
python acted.py --config config.yaml --mark-status
```

Writes a human-readable date into the `Status` column for rows that are still empty.

Outputs a JSON report at `out/dry_run_report.json`.

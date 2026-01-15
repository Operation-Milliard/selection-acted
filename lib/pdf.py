from __future__ import annotations

from io import BytesIO
import re
from pypdf import PdfReader

DRIVE_FILE_ID_RE = re.compile(r"(?:file/d/|open\?id=|uc\?id=)([-\w]{10,})")


def extract_drive_file_ids(cell_value: str) -> list[str]:
    if not cell_value:
        return []
    return DRIVE_FILE_ID_RE.findall(cell_value)


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

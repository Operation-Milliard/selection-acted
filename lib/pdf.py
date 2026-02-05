from __future__ import annotations

import re
import tempfile
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

DRIVE_FILE_ID_RE = re.compile(r"(?:file/d/|open\?id=|uc\?id=)([-\w]{10,})")

# Regex patterns for structure detection
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


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
    """Legacy fixed-size character chunking."""
    if not text:
        return []
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be less than size ({size})")
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
        start = end - overlap
    return chunks


def extract_pdf_markdown(pdf_bytes: bytes) -> str:
    """Extract PDF as markdown with structure using pymupdf4llm."""
    try:
        import pymupdf4llm
    except ImportError:
        print("pymupdf4llm not installed, falling back to basic extraction")
        return extract_pdf_text(pdf_bytes)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        md_text = pymupdf4llm.to_markdown(str(tmp_path))
        return md_text
    finally:
        tmp_path.unlink()


def split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitting."""
    # Split on sentence-ending punctuation followed by space or newline
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_pattern.split(text)
    return [s.strip() for s in sentences if s.strip()]


def split_by_structure(md_text: str) -> list[dict]:
    """Split markdown text by headers and paragraphs, preserving structure info."""
    sections = []
    current_header = ""
    current_pos = 0

    # Split by double newlines (paragraphs)
    parts = PARAGRAPH_SPLIT.split(md_text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check if this part is a header
        header_match = HEADER_PATTERN.match(part)
        if header_match:
            current_header = header_match.group(2)

        # Find position in original text
        pos = md_text.find(part, current_pos)
        if pos == -1:
            pos = current_pos

        sections.append({
            "text": part,
            "header": current_header,
            "start_char": pos,
            "end_char": pos + len(part),
        })
        current_pos = pos + len(part)

    return sections


def chunk_by_structure(
    text: str,
    min_chars: int = 500,
    max_chars: int = 2000,
    overlap_sentences: int = 1,
) -> list[dict]:
    """
    Structure-aware chunking with size constraints.

    1. Split by paragraphs/headers
    2. Merge small sections (< min_chars)
    3. Split large sections at sentence boundaries (> max_chars)
    4. Add sentence overlap between chunks
    """
    if not text:
        return []

    sections = split_by_structure(text)
    if not sections:
        return []

    chunks = []
    buffer_text = ""
    buffer_start = 0
    buffer_header = ""

    def flush_buffer():
        nonlocal buffer_text, buffer_start, buffer_header
        if buffer_text.strip():
            chunks.append({
                "start_char": buffer_start,
                "end_char": buffer_start + len(buffer_text),
                "text": buffer_text.strip(),
                "header": buffer_header,
            })
        buffer_text = ""
        buffer_start = 0
        buffer_header = ""

    def split_large_section(section: dict) -> list[dict]:
        """Split a large section at sentence boundaries."""
        result = []
        sentences = split_into_sentences(section["text"])
        if not sentences:
            return [section]

        current_chunk = ""
        chunk_start = section["start_char"]

        for i, sentence in enumerate(sentences):
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                # Flush current chunk
                result.append({
                    "start_char": chunk_start,
                    "end_char": chunk_start + len(current_chunk),
                    "text": current_chunk.strip(),
                    "header": section.get("header", ""),
                })

                # Start new chunk with overlap
                overlap_start = max(0, i - overlap_sentences)
                overlap_text = " ".join(sentences[overlap_start:i])
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                chunk_start = section["start_char"]  # Approximate
            else:
                current_chunk += (" " if current_chunk else "") + sentence

        if current_chunk.strip():
            result.append({
                "start_char": chunk_start,
                "end_char": chunk_start + len(current_chunk),
                "text": current_chunk.strip(),
                "header": section.get("header", ""),
            })

        return result

    for section in sections:
        section_text = section["text"]
        section_len = len(section_text)

        if section_len > max_chars:
            # Section too large: flush buffer and split section
            flush_buffer()
            chunks.extend(split_large_section(section))

        elif len(buffer_text) + section_len > max_chars:
            # Adding this section would exceed max: flush buffer first
            flush_buffer()
            buffer_text = section_text
            buffer_start = section["start_char"]
            buffer_header = section.get("header", "")

        elif len(buffer_text) + section_len < min_chars:
            # Combined still too small: accumulate
            if not buffer_text:
                buffer_start = section["start_char"]
                buffer_header = section.get("header", "")
            buffer_text += "\n\n" + section_text if buffer_text else section_text

        else:
            # Good size: flush buffer and start new one
            flush_buffer()
            buffer_text = section_text
            buffer_start = section["start_char"]
            buffer_header = section.get("header", "")

    # Don't forget remaining buffer
    flush_buffer()

    return chunks


def chunk_text_smart(
    text: str,
    min_chars: int = 500,
    max_chars: int = 2000,
    overlap_sentences: int = 1,
    use_structure: bool = True,
) -> list[dict]:
    """
    Smart chunking with fallback.

    Args:
        text: Text to chunk (plain text or markdown)
        min_chars: Minimum chunk size
        max_chars: Maximum chunk size
        overlap_sentences: Number of sentences to overlap
        use_structure: Whether to use structure-aware chunking

    Returns:
        List of chunk dicts with start_char, end_char, text, and optional header
    """
    if not text:
        return []

    if use_structure:
        chunks = chunk_by_structure(text, min_chars, max_chars, overlap_sentences)
        if chunks:
            return chunks

    # Fallback to simple chunking
    return chunk_text(text, size=max_chars, overlap=min_chars // 2)

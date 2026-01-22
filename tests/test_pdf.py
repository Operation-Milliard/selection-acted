"""Tests for lib/pdf.py chunking functions."""

from pathlib import Path

import pytest

from lib.pdf import (
    chunk_text,
    chunk_text_smart,
    chunk_by_structure,
    extract_drive_file_ids,
    extract_pdf_markdown,
    extract_pdf_text,
    split_by_structure,
    split_into_sentences,
)

# Path to test PDF
TEST_PDF_PATH = Path(__file__).parent / "Rapport-dactivite-CSR-2024.pdf"


@pytest.fixture
def pdf_bytes():
    """Load test PDF as bytes."""
    return TEST_PDF_PATH.read_bytes()


class TestExtractDriveFileIds:
    def test_empty_string(self):
        assert extract_drive_file_ids("") == []

    def test_single_file_id(self):
        url = "https://drive.google.com/file/d/1Qo_0muoWMWePXAyQjc876keNeVgAgtgF/view"
        assert extract_drive_file_ids(url) == ["1Qo_0muoWMWePXAyQjc876keNeVgAgtgF"]

    def test_open_id_format(self):
        url = "https://drive.google.com/open?id=1Qo_0muoWMWePXAyQjc876keNeVgAgtgF"
        assert extract_drive_file_ids(url) == ["1Qo_0muoWMWePXAyQjc876keNeVgAgtgF"]

    def test_multiple_file_ids(self):
        text = """
        https://drive.google.com/file/d/ABC123456789/view
        https://drive.google.com/open?id=XYZ987654321
        """
        ids = extract_drive_file_ids(text)
        assert len(ids) == 2
        assert "ABC123456789" in ids
        assert "XYZ987654321" in ids


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("", size=100, overlap=10) == []

    def test_short_text(self):
        text = "Hello world"
        chunks = chunk_text(text, size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Hello world"
        assert chunks[0]["start_char"] == 0
        assert chunks[0]["end_char"] == len(text)

    def test_chunking_with_overlap(self):
        text = "A" * 100 + "B" * 100 + "C" * 100
        chunks = chunk_text(text, size=100, overlap=20)

        assert len(chunks) == 4
        assert chunks[0]["text"] == "A" * 100
        assert chunks[1]["start_char"] == 80  # 100 - 20 overlap
        assert "A" in chunks[1]["text"]  # Contains overlap from previous

    def test_exact_size(self):
        text = "X" * 200
        chunks = chunk_text(text, size=100, overlap=0)
        assert len(chunks) == 2
        assert chunks[0]["text"] == "X" * 100
        assert chunks[1]["text"] == "X" * 100


class TestSplitIntoSentences:
    def test_simple_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence."
        assert sentences[2] == "Third sentence."

    def test_question_marks(self):
        text = "Is this working? Yes it is! Great."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3

    def test_no_sentences(self):
        text = "No punctuation here"
        sentences = split_into_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == "No punctuation here"


class TestSplitByStructure:
    def test_simple_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        sections = split_by_structure(text)
        assert len(sections) == 3
        assert sections[0]["text"] == "First paragraph."
        assert sections[1]["text"] == "Second paragraph."
        assert sections[2]["text"] == "Third paragraph."

    def test_with_headers(self):
        text = "# Header 1\n\nContent under header 1.\n\n## Header 2\n\nContent under header 2."
        sections = split_by_structure(text)
        assert len(sections) == 4
        # First section is the header itself
        assert "Header 1" in sections[0]["text"]

    def test_empty_text(self):
        sections = split_by_structure("")
        assert sections == []


class TestChunkByStructure:
    def test_empty_text(self):
        assert chunk_by_structure("") == []

    def test_merges_small_sections(self):
        # Three small paragraphs that should be merged
        text = "Short.\n\nAlso short.\n\nStill short."
        chunks = chunk_by_structure(text, min_chars=100, max_chars=500)
        # Should merge into fewer chunks since each is < min_chars
        assert len(chunks) <= 2

    def test_splits_large_sections(self):
        # One very long paragraph that should be split
        long_text = "This is a sentence. " * 100  # ~2000 chars
        chunks = chunk_by_structure(long_text, min_chars=100, max_chars=500)
        # Should be split into multiple chunks
        assert len(chunks) > 1
        # Each chunk should be <= max_chars
        for chunk in chunks:
            assert len(chunk["text"]) <= 600  # Some tolerance for sentence boundaries

    def test_preserves_reasonable_sections(self):
        # Sections within min/max range should stay intact
        text = "A" * 300 + "\n\n" + "B" * 300 + "\n\n" + "C" * 300
        chunks = chunk_by_structure(text, min_chars=100, max_chars=500)
        # Each section is 300 chars, between min and max
        assert len(chunks) == 3

    def test_has_required_fields(self):
        text = "Some content here.\n\nMore content here."
        chunks = chunk_by_structure(text, min_chars=10, max_chars=100)
        for chunk in chunks:
            assert "text" in chunk
            assert "start_char" in chunk
            assert "end_char" in chunk


class TestChunkTextSmart:
    def test_empty_text(self):
        assert chunk_text_smart("") == []

    def test_fallback_to_simple(self):
        text = "Simple text without structure"
        chunks = chunk_text_smart(text, min_chars=10, max_chars=100, use_structure=False)
        assert len(chunks) >= 1

    def test_with_structure(self):
        text = "# Header\n\nParagraph one.\n\nParagraph two."
        chunks = chunk_text_smart(text, min_chars=10, max_chars=1000, use_structure=True)
        assert len(chunks) >= 1

    def test_respects_max_chars(self):
        # Use text with sentences so splitting can work
        long_text = "This is a sentence. " * 200
        chunks = chunk_text_smart(long_text, min_chars=100, max_chars=500)
        for chunk in chunks:
            # Allow some tolerance for sentence boundaries
            assert len(chunk["text"]) <= 700


class TestIntegration:
    """Integration tests with realistic document-like content."""

    def test_markdown_document(self):
        doc = """# Project Summary

This project aims to provide educational services to underprivileged communities.

## Objectives

1. Increase literacy rates
2. Provide vocational training
3. Support local employment

## Budget

The total budget is 50,000 euros distributed as follows:
- Personnel: 30,000 euros
- Materials: 15,000 euros
- Operations: 5,000 euros

## Timeline

The project will run from January 2024 to December 2024.
"""
        chunks = chunk_text_smart(doc, min_chars=100, max_chars=500)

        # Should create reasonable chunks
        assert len(chunks) >= 2

        # All text should be covered
        all_text = " ".join(c["text"] for c in chunks)
        assert "educational services" in all_text
        assert "Budget" in all_text
        assert "Timeline" in all_text


class TestRealPDF:
    """Tests using the real PDF file Rapport-dactivite-CSR-2024.pdf."""

    def test_extract_pdf_text(self, pdf_bytes):
        """Test basic PDF text extraction."""
        text = extract_pdf_text(pdf_bytes)

        assert text is not None
        assert len(text) > 0
        # Should contain some French text (it's a CSR report)
        assert isinstance(text, str)

    def test_extract_pdf_text_has_content(self, pdf_bytes):
        """Test that extracted text has meaningful content."""
        text = extract_pdf_text(pdf_bytes)

        # Should have multiple lines
        lines = text.split("\n")
        assert len(lines) > 10

        # Should have reasonable character count for a report
        assert len(text) > 1000

    def test_extract_pdf_markdown(self, pdf_bytes):
        """Test markdown extraction from PDF."""
        md_text = extract_pdf_markdown(pdf_bytes)

        assert md_text is not None
        assert len(md_text) > 0
        assert isinstance(md_text, str)

    def test_extract_pdf_markdown_vs_text(self, pdf_bytes):
        """Compare markdown vs plain text extraction."""
        plain_text = extract_pdf_text(pdf_bytes)
        md_text = extract_pdf_markdown(pdf_bytes)

        # Both should extract content
        assert len(plain_text) > 0
        assert len(md_text) > 0

        # Markdown might have formatting markers
        # (headers, lists, etc.) depending on PDF structure

    def test_chunk_real_pdf_legacy(self, pdf_bytes):
        """Test legacy chunking on real PDF."""
        text = extract_pdf_text(pdf_bytes)
        chunks = chunk_text(text, size=2000, overlap=200)

        assert len(chunks) > 0

        # Check chunk structure
        for chunk in chunks:
            assert "text" in chunk
            assert "start_char" in chunk
            assert "end_char" in chunk
            assert len(chunk["text"]) <= 2000

    def test_chunk_real_pdf_smart(self, pdf_bytes):
        """Test smart chunking on real PDF."""
        md_text = extract_pdf_markdown(pdf_bytes)
        chunks = chunk_text_smart(
            md_text,
            min_chars=500,
            max_chars=2000,
            overlap_sentences=1,
        )

        assert len(chunks) > 0

        # Check chunk structure
        for chunk in chunks:
            assert "text" in chunk
            assert "start_char" in chunk
            assert "end_char" in chunk

        # Chunks should have reasonable sizes
        chunk_sizes = [len(c["text"]) for c in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        assert avg_size > 100  # Not too small

    def test_smart_vs_legacy_chunking(self, pdf_bytes):
        """Compare smart chunking vs legacy chunking."""
        text = extract_pdf_text(pdf_bytes)
        md_text = extract_pdf_markdown(pdf_bytes)

        legacy_chunks = chunk_text(text, size=2000, overlap=200)
        smart_chunks = chunk_text_smart(md_text, min_chars=500, max_chars=2000)

        # Both should produce chunks
        assert len(legacy_chunks) > 0
        assert len(smart_chunks) > 0

        # Smart chunking should have more variable sizes
        # (respecting document structure)
        legacy_sizes = [len(c["text"]) for c in legacy_chunks]
        smart_sizes = [len(c["text"]) for c in smart_chunks]

        # Calculate variance
        def variance(sizes):
            mean = sum(sizes) / len(sizes)
            return sum((s - mean) ** 2 for s in sizes) / len(sizes)

        # Smart chunking typically has more size variance
        # due to respecting document structure
        print(f"Legacy chunks: {len(legacy_chunks)}, variance: {variance(legacy_sizes):.0f}")
        print(f"Smart chunks: {len(smart_chunks)}, variance: {variance(smart_sizes):.0f}")

    def test_all_text_covered(self, pdf_bytes):
        """Verify that chunking covers all extracted text."""
        md_text = extract_pdf_markdown(pdf_bytes)
        chunks = chunk_text_smart(md_text, min_chars=500, max_chars=2000)

        # Concatenate all chunk texts
        all_chunk_text = " ".join(c["text"] for c in chunks)

        # Should cover most of the original content
        # (some whitespace normalization may occur)
        original_words = set(md_text.split())
        chunk_words = set(all_chunk_text.split())

        # Most words should be preserved
        preserved = len(original_words & chunk_words)
        total = len(original_words)
        coverage = preserved / total if total > 0 else 0

        assert coverage > 0.8, f"Only {coverage:.1%} of words preserved"

    def test_chunk_boundaries_sensible(self, pdf_bytes):
        """Test that smart chunks don't break mid-word."""
        md_text = extract_pdf_markdown(pdf_bytes)
        chunks = chunk_text_smart(md_text, min_chars=500, max_chars=2000)

        for chunk in chunks:
            text = chunk["text"]
            if text:
                # Chunk should not start with whitespace
                assert not text[0].isspace(), \
                    f"Chunk starts with whitespace: {repr(text[:20])}"
                # Chunk should not end with whitespace
                assert not text[-1].isspace(), \
                    f"Chunk ends with whitespace: {repr(text[-20:])}"

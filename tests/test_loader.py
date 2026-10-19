import shutil

import pymupdf
import pytest

from src.loader import load_pdf

LONG_TEXT = "ChatGPT reached 100 million users in about two months after launch in November 2022."


def make_pdf(tmp_path, scanned_page=False):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), LONG_TEXT, fontsize=12)
    if scanned_page:
        png = page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes("png")
        doc.new_page().insert_image(doc[0].rect, stream=png)
    path = tmp_path / "sample.pdf"
    doc.save(path)
    return path


def test_text_pdf_is_read_without_ocr(tmp_path):
    pages = load_pdf(make_pdf(tmp_path))
    assert len(pages) == 1
    assert pages[0].metadata["loader"] == "pymupdf"
    assert "100 million" in pages[0].page_content


def test_missing_file_and_wrong_type_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "nope.pdf")
    other = tmp_path / "notes.txt"
    other.write_text("hi")
    with pytest.raises(ValueError):
        load_pdf(other)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_scanned_page_falls_back_to_ocr(tmp_path):
    pages = load_pdf(make_pdf(tmp_path, scanned_page=True))
    assert [p.metadata["loader"] for p in pages] == ["pymupdf", "ocr"]
    assert "100 million" in pages[1].page_content
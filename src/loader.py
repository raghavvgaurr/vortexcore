"""Read a PDF into one Document per page, using OCR for pages without a text layer."""
import io
import shutil
from pathlib import Path

import pymupdf
import pytesseract
from langchain_core.documents import Document
from PIL import Image

MIN_TEXT_CHARS = 50  # below this, a page is treated as scanned
OCR_ZOOM = 2         # same zoom as the original notebook


def _ocr_page(page: pymupdf.Page) -> str:
    if shutil.which("tesseract") is None:
        raise RuntimeError(
            "Tesseract is not installed. Install it locally, or add "
            "'tesseract-ocr' to packages.txt on Streamlit Cloud."
        )
    pix = page.get_pixmap(matrix=pymupdf.Matrix(OCR_ZOOM, OCR_ZOOM))
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(image)


def load_pdf(path: str | Path, force_ocr: bool = False) -> list[Document]:
    """Return one Document per page. force_ocr=True OCRs every page."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF files are supported, got '{path.suffix}'")

    pages = []
    with pymupdf.open(path) as pdf:
        for number, page in enumerate(pdf):
            text = "" if force_ocr else page.get_text()
            method = "pymupdf"
            if force_ocr or len(text.strip()) < MIN_TEXT_CHARS:
                text = _ocr_page(page)
                method = "ocr"
            pages.append(Document(
                page_content=text,
                metadata={"source": str(path), "page": number, "loader": method},
            ))
    return pages
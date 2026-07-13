import fitz


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text and text.strip():
            pages.append({
                "page": page_num + 1,
                "text": text,
                "source": pdf_path,
            })
    return pages

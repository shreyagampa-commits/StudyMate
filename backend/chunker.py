def chunk_text(pages, chunk_size=180, overlap=40):
    """Split extracted PDF pages into overlapping word chunks."""
    chunks = []
    for page in pages:
        words = page["text"].split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = start + chunk_size
            text = " ".join(words[start:end]).strip()
            if len(text) > 40:
                chunks.append({
                    "page": page["page"],
                    "text": text,
                    "source": page.get("source", "uploaded_pdf"),
                })
            if end >= len(words):
                break
            start += max(1, chunk_size - overlap)
    return chunks

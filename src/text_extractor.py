import re
from docx import Document

def extract_text_from_docs(path: str):
    doc = Document(path)

    paragraphs = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    print("Количество абзацев:", len(paragraphs))
    #print("Первые 5 абзацев", paragraphs[:5])

    return "\n".join(paragraphs)

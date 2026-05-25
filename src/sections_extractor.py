import re
from docx import Document

SECTION_PATTERN = re.compile(
    r"(Узел|Система).*схем.*лист",
    re.IGNORECASE
)

def extract_sections(docx_path: str):
    doc = Document(docx_path)

    sections = []
    current_section = None

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        text = re.sub(r"\поз\.", "позиция", text, flags=re.IGNORECASE)

        if not text:
            continue

        if SECTION_PATTERN.search(text):
            if current_section:
                sections.append(current_section)

            current_section = {
                "title": text,
                "paragraphs": []
            }

        else:
            if current_section:
                current_section["paragraphs"].append(text)

    if current_section:
        sections.append(current_section)

    return sections
import re

def is_bullet(text: str):
    return bool(re.match(r"^\s*[-–—•]\s+", text))

def starts_with_capital(text: str):
    return bool(re.match(r"^[А-ЯA-ZЁ]", text.strip()))

def ends_with_dot(text: str):
    return bool(text.endswith("."))

def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    
    sentences = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        if is_bullet(part):
            if sentences:
                sentences[-1] += "\n" + part
            continue

        if starts_with_capital(part) and ends_with_dot(part):
            sentences.append(part)
    
    return sentences


def build_sentence_chunks(sections: list[dict], window_size = 3, overlap = 1):
    chunks = []

    for section in sections:
        section_title = section["title"]
        paragraphs = section["paragraphs"]

        for paragraph_id, paragraph in enumerate(paragraphs):
            sentences = split_sentences(paragraph)

            step = max(1, window_size - overlap)

            for start in range(0, len(sentences), step):
                window = sentences[start:start + window_size]

                if not window:
                    continue

                text = " ".join(window)
                chunk_text = f"Раздел: {section_title}\nТекст: {text}"

                chunks.append({
                    "section": section_title,
                    "paragraph_id": paragraph_id,
                    "sentence_start": start,
                    "sentence_end": start + len(window) - 1,
                    "text": text,
                    "chunk_text": chunk_text,
                    "source_paragraph": paragraph,
                })

    return chunks
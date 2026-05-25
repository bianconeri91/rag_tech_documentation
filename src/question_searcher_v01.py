import numpy as np
import faiss

def search_question(question: str, model, index, chunks: list[dict], section: str, k: int = 15):
    if section:
        search_query = f"Раздел: {section}\nВопрос: {question}"
    else:
        search_query = question
    
    query = model.encode([search_query])
    query = np.array(query).astype("float32")

    faiss.normalize_L2(query)

    search_k = 100 if section else k

    D, I = index.search(query, k=search_k)

    results = []

    for score, idx in zip(D[0], I[0]):
        chunk = chunks[idx]

        if section and chunk.get("section") != section:
            continue

        results.append({
            "rank": len(results) + 1,
            "score": float(score),
            "idx": int(idx),
            "section": chunk.get("section"),
            "text": chunk.get("text"),
            "chunk_text": chunk.get("chunk_text"),
            "source_paragraph": chunk.get("source_paragraph")
        })

        if len(results) >= k:
            break

    return results

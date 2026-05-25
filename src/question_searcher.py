import numpy as np
import faiss

from .results_reranker import rerank_results


def build_search_query(question: str, section: str | None = None):
    if section:
        return f"Раздел: {section}\nВопрос: {question}"
    return question


def encode_query(search_query: str, model):
    query = model.encode([search_query])
    query = np.array(query).astype("float32")
    faiss.normalize_L2(query)
    return query


def search_faiss(query: np.ndarray, index, k = int):
    scores, indices = index.search(query, k=k)
    return zip(scores[0], indices[0])


def build_search_result(
        score: float,
        idx: int,
        chunk: dict,
        rank: int,
        match_type: str,
):
    return {
        "rank": rank,
        "score": score,
        "idx": idx,
        "section": chunk.get("section"),
        "text": chunk.get("text"),
        "chunk_text": chunk.get("chunk_text"),
        "source_paragraph": chunk.get("source_paragraph"),
        "match_type": match_type,
    }


def is_matching_section(chunk: dict, section: str | None):
    if not section:
        return True
    
    return chunk.get("section") == section


def exact_search_in_section(
        chunks: list[dict],
        question: str,
        section: str | None = None,
        k: int = 15,
):
    query_text = question.lower()

    results = []

    for idx, chunk in enumerate(chunks):
        if not is_matching_section(chunk, section):
            continue

        chunk_text = chunk.get("chunk_text")

        if not chunk_text:
            continue

        if query_text not in chunk_text.lower():
            continue

        results.append(
            build_search_result(
                score=999.0,
                idx=idx,
                chunk=chunk,
                rank=len(results) + 1,
                match_type="exact",
            )
        )

        if len(results) >= k:
            break
    
    return results


def bm25_search_in_section(
        question: str,
        chunks: list[dict],
        bm25,
        section: str | None = None,
        k: int = 15,
):
    query_tokens = question.lower().split()
    scores = bm25.get_scores(query_tokens)

    results = []

    for idx, score in enumerate(scores):
        chunk = chunks[idx]

        if not is_matching_section(chunk, section):
            continue

        result = build_search_result(
            score=float(score),
            idx=idx,
            chunk=chunk,
            rank=0,
            match_type="bm25",
        )

        results.append(result)
    
    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    results = results[:k]

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    
    return results


def faiss_search_in_section(
        question: str,
        model,
        index,
        chunks: list[dict],
        section: str | None = None,
        k: int = 15,
        search_k: int = 100,
):
    search_query = build_search_query(question, section)
    query = encode_query(search_query, model)

    faiss_results = search_faiss(query, index, k=search_k)

    results = []

    for score, idx in faiss_results:
        chunk = chunks[idx]

        if not is_matching_section(chunk, section):
            continue

        result = build_search_result(
            score=score,
            idx=idx,
            chunk=chunk,
            rank=len(results) + 1,
            match_type="faiss",
        )

        results.append(result)

        if len(results) >= k:
            break
    
    return results


def merge_results(exact_results, bm25_results, faiss_results, k: int = 10):
    merged = {}

    def add_results(results, source_name):
        for result in results:
            idx = result["idx"]

            if idx not in merged:
                merged[idx] = result.copy()
                merged[idx]["rrf_score"] = 0.0
                merged[idx]["sources"] = set()

            rank = result["rank"]

            merged[idx]["rrf_score"] += 1 / (20 + rank)
            merged[idx]["sources"].add(source_name)

    add_results(bm25_results, "bm25")
    add_results(faiss_results, "faiss")

    for result in exact_results:
        idx = result["idx"]

        if idx not in merged:
            merged[idx] = result.copy()
            merged[idx]["rrf_score"] = 1000.0
            merged[idx]["sources"] = {"exact"}
        else:
            merged[idx]["rrf_score"] += 1000.0
            merged[idx]["sources"].add("exact")
    

    results = list(merged.values())

    for result in results:
        result["match_type"] = "+".join(
            sorted(result["sources"])
        )

    results.sort(
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    final_results = results[:k]

    for rank, result in enumerate(final_results, start=1):
        result["rank"] = rank
        result["score"] = result["rrf_score"]
        result.pop("sources", None)

    return final_results


def search_question(
        question: str, 
        model, 
        index, 
        chunks: list[dict], 
        section: str, 
        bm25,
        k: int = 10
):
    
    exact_results = exact_search_in_section(
        chunks=chunks,
        question=question,
        section=section,
        k=k,
    )

    bm25_results = bm25_search_in_section(
        chunks=chunks,
        question=question,
        section=section,
        bm25=bm25,
        k=k,
    )
    
    faiss_results = faiss_search_in_section(
        question=question,
        model=model,
        index=index,
        chunks=chunks,
        section=section,
        k=k,
        search_k=100 if section else k,
    )

    rerank_faiss_results = rerank_results(
        question=question,
        results=faiss_results,
        k=k,
    )

    results = merge_results(
        exact_results=exact_results,
        bm25_results=bm25_results,
        faiss_results=rerank_faiss_results,
        k=k,
    )

    return results

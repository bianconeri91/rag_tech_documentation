from sentence_transformers import CrossEncoder

_reranker = None


def get_reranker():
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return _reranker

def rerank_results(question: str, results: list[dict], k: int = 5):
    reranker = get_reranker()

    pairs = [
        [question, result["chunk_text"] or result["text"] or ""]
        for result in results
    ]

    rerank_scores = reranker.predict(pairs)

    for result, rerank_score in zip(results, rerank_scores):
        result["rerank_score"] = float(rerank_score)

    reranked = sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    for rank, result in enumerate(reranked[:k], start=1):
        result["rank"] = rank

    return reranked[:k]

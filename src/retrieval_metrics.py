def find_expected_rank(results: list[dict], expected_substring: str):
    """
    Возвращает rank первого результата, в котором найден expected_substring.
    Если ожидаемый фрагмент не найден, возвращает None.
    """
    expected = expected_substring.lower()

    for result in results:
        text = result.get("text") or result.get("chunk_text") or ""

        if expected in text.lower():
            return result["rank"]

    return None


def reciprocal_rank(rank: int | None):
    """
    Считает Reciprocal Rank: 1 / rank.
    Если правильный результат не найден, возвращает 0.0.
    """
    if rank is None:
        return 0.0

    return 1 / rank


def hit_at_k(rank: int | None, k: int):
    """
    Проверяет, найден ли правильный результат в top-k.
    Возвращает 1, если найден, иначе 0.
    """
    if rank is None:
        return 0

    return int(rank <= k)


def calculate_retrieval_metrics(
    evaluation_results: list[dict],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
):
    """
    Считает агрегированные retrieval-метрики:
    Hit Rate@k и MRR.
    """
    total = len(evaluation_results)

    if total == 0:
        return {}

    metrics = {}

    for k in k_values:
        hits = [
            hit_at_k(item["found_rank"], k)
            for item in evaluation_results
        ]
        metrics[f"hit_rate@{k}"] = sum(hits) / total

    reciprocal_ranks = [
        reciprocal_rank(item["found_rank"])
        for item in evaluation_results
    ]

    metrics["mrr"] = sum(reciprocal_ranks) / total
    metrics["total_questions"] = total

    return metrics
from .question_searcher import search_question
from .retrieval_metrics import calculate_retrieval_metrics, find_expected_rank

def evaluate_questions(
        test_questions: list[dict], 
        model, 
        bm25, 
        index, 
        chunks, 
        k: int = 10,
):
    """
    Прогоняет тестовые вопросы через retrieval pipeline
    и считает итоговые метрики качества поиска.
    """
    evaluation_results = []

    for item in test_questions:
        question = item["question"]
        expected_substring = item["expected_substring"]
        q_section = item["section"]

        results = search_question(
            question=question,
            section=q_section,
            model=model,
            index=index,
            chunks=chunks,
            bm25=bm25,
            k=k,
        )

        found_rank = find_expected_rank(
            results=results,
            expected_substring=expected_substring,
        )

        evaluation_results.append({
            "id": item["id"],
            "question": question,
            "section": q_section,
            "expected_substring": expected_substring,
            "found_rank": found_rank,
            "is_found_in_top_k": found_rank is not None,
            "top_results": results,
        })

        metrics = calculate_retrieval_metrics(
            evaluation_results=evaluation_results,
            k_values=(1, 3, 5, 10),
        )

    return evaluation_results, metrics

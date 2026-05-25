import argparse

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .chunks_builder import build_sentence_chunks
from .question_evaluator import evaluate_questions
from .sections_extractor import extract_sections


TEST_QUESTIONS = [
    {
        "id": "q001",
        "section": "Узел активации катализатора – 4110 (схема ВТР-АГХК-4000-08-25, лист 17)",
        "question": "подается технологический воздух с давлением",
        "expected_substring": "Для активации катализатора от границы",
    },
    {
        "id": "q002",
        "section": "Узел подготовки сырья – 4120 (схемы ВТР-АГХК-4000-08-25, листы 18 – 21)",
        "question": "кориолисовым расходомером",
        "expected_substring": "Контроль расхода гексена",
    },
    {
        "id": "q003",
        "section": "Узел подготовки катализаторов и сокатализаторов – 4130 (схемы ВТР-АГХК-4000-08-25, листы 22 - 25)",
        "question": "выгружается в емкость загрузки катализатора",
        "expected_substring": "Транспортная емкость с катализатором",
    },
    {
        "id": "q004",
        "section": "Узел приема химреагентов для реакторов – 4140 (схемы ВТР-АГХК-4000-08-25, листы 26, 27)",
        "question": "хромового катализатора",
        "expected_substring": "Антистатик для производства полимера",
    },
    {
        "id": "q005",
        "section": "Система реактора А – 4141 (схемы ВТР-АГХК-4000-08-25, листы 28 - 32)",
        "question": "устанавливается регулятором расхода",
        "expected_substring": "Скорость подачи безолефинового изобутана",
    },
    {
        "id": "q006",
        "section": "Система рециркуляции и регенерации – 4150 (схемы ВТР-АГХК-4000-08-25, листы 38 – 42)",
        "question": "ротационного клапана продувочной колонны",
        "expected_substring": "Регулирование уровня в колонне продувки",
    },
    {
        "id": "q007",
        "section": "Система экструзии – 4160 (схемы ВТР-АГХК-4000-08-25, листы 59-66)",
        "question": "прием компрессора подачи",
        "expected_substring": "Охлажденный транспортирующий газ",
    },
    {
        "id": "q008",
        "section": "Система воздуха КИП, технического воздуха и азота – 4650 (схемы ВТР-АГХК-4000-08-25, листы 50 - 55)",
        "question": "установлены предохранительные клапаны",
        "expected_substring": "Для защиты оборудования",
    },
    {
        "id": "q009",
        "section": "Узел распределения топливного газа – 4660. Система факельных сбросов ВД и НД – 4670 (схема ВТР-АГХК-4000-08-25, лист 56)",
        "question": "входного сепаратора УРИА",
        "expected_substring": "Факельный коллектор НД",
    },
    {
        "id": "q010",
        "section": "Система сточных вод – 4680 (схемы ВТР-АГХК-4000-08-25, лист 57)",
        "question": "собирается в подземной дренажной емкости",
        "expected_substring": "Бытовая канализация",
    },
    {
        "id": "q011",
        "section": "Система экструзии – 4160 (схемы ВТР-АГХК-4000-08-25, листы 59-66)",
        "question": "куда поступает охлажденный транспортирующий газ после охладителя",
        "expected_substring": "Охлажденный транспортирующий газ",
    },
    {
        "id": "q012",
        "section": "Узел активации катализатора – 4110 (схема ВТР-АГХК-4000-08-25, лист 17)",
        "question": "какое давление у воздуха при подаче для активации катализатора",
        "expected_substring": "Для активации катализатора от границы",
    },
    {
        "id": "q013",
        "section": "Узел подготовки сырья – 4120 (схемы ВТР-АГХК-4000-08-25, листы 18 – 21)",
        "question": "какой расходомер контролирует расход гексена",
        "expected_substring": "Контроль расхода гексена",
    },
    {
        "id": "q014",
        "section": "Узел подготовки катализаторов и сокатализаторов – 4130 (схемы ВТР-АГХК-4000-08-25, листы 22 - 25)",
        "question": "в какую емкость выгружается катализатор Циглера-Натта",
        "expected_substring": "Транспортная емкость с катализатором",
    },
    {
        "id": "q015",
        "section": "Узел приема химреагентов для реакторов – 4140 (схемы ВТР-АГХК-4000-08-25, листы 26, 27)",
        "question": "куда подается антистатик для производства полимера",
        "expected_substring": "Антистатик для производства полимера",
    },
    {
        "id": "q016",
        "section": "Система реактора А – 4141 (схемы ВТР-АГХК-4000-08-25, листы 28 - 32)",
        "question": "каким датчиком устанавливается скорость подачи безолефинового изобутана",
        "expected_substring": "Скорость подачи безолефинового изобутана",
    },
    {
        "id": "q017",
        "section": "Система рециркуляции и регенерации – 4150 (схемы ВТР-АГХК-4000-08-25, листы 38 – 42)",
        "question": "какой последовательностью регулируется уровень в колонне продувки",
        "expected_substring": "Регулирование уровня в колонне продувки",
    },
    {
        "id": "q018",
        "section": "Система воздуха КИП, технического воздуха и азота – 4650 (схемы ВТР-АГХК-4000-08-25, листы 50 - 55)",
        "question": "какой прибор защищает оборудование от превышения давления",
        "expected_substring": "Для защиты оборудования",
    },
    {
        "id": "q019",
        "section": "Узел распределения топливного газа – 4660. Система факельных сбросов ВД и НД – 4670 (схема ВТР-АГХК-4000-08-25, лист 56)",
        "question": "для чего предназначен факельный коллектор НД",
        "expected_substring": "Факельный коллектор НД",
    },
    {
        "id": "q020",
        "section": "Система сточных вод – 4680 (схемы ВТР-АГХК-4000-08-25, лист 57)",
        "question": "где собирается бытовая канализация К1",
        "expected_substring": "Бытовая канализация",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG retrieval evaluation for a DOCX file.")
    parser.add_argument("--docx", required=True, help="Path to the source DOCX document.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of search results to evaluate.")
    return parser.parse_args()


def build_bm25(chunks):
    tokenized_corpus = [(chunk["text"] or "").lower().split() for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def build_faiss_index(embeddings):
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def main():
    args = parse_args()

    sections = extract_sections(args.docx)
    chunks = build_sentence_chunks(sections)

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    sentences = [chunk["chunk_text"] for chunk in chunks]
    embeddings = model.encode(sentences)

    bm25 = build_bm25(chunks)
    index, embeddings = build_faiss_index(embeddings)

    print(f"embeddings dtype: {embeddings.dtype}")
    print(f"faiss index trained: {index.is_trained}")
    print(f"faiss index size: {index.ntotal}")

    _, metrics = evaluate_questions(
        test_questions=TEST_QUESTIONS,
        model=model,
        index=index,
        bm25=bm25,
        chunks=chunks,
        k=args.top_k,
    )

    print(metrics)


if __name__ == "__main__":
    main()

# Industrial RAG Pipeline for Technical Documentation

Local Retrieval-Augmented Generation (RAG) pipeline for industrial technical documentation with hybrid retrieval, section-aware search, and retrieval quality evaluation.

---

# Project Overview

This project implements a local RAG retrieval pipeline for large industrial technical documents written in Russian.

The system is designed for:
- technological regulations;
- process descriptions;
- instrumentation documentation;
- engineering specifications;
- industrial operating procedures.

The primary goal is accurate retrieval of relevant technical fragments from large documents containing:
- engineering terminology;
- instrument tags;
- long structured sections;
- semantically similar process descriptions.

The project focuses on retrieval quality engineering rather than chatbot orchestration.

---

# Business Problem

Industrial companies operate with large volumes of technical documentation:
- technological regulations;
- operating manuals;
- instrumentation descriptions;
- process diagrams;
- engineering standards.

These documents may contain:
- hundreds of pages;
- complex technical terminology;
- duplicated process descriptions;
- highly specific equipment references.

Manual search inside such documents is slow and inefficient.

Traditional keyword search often fails because:
- the same concept may be described differently;
- terminology may vary across sections;
- relevant information may span multiple sentences;
- lexical search does not capture semantic similarity.

At the same time, cloud-based AI solutions are often unacceptable due to:
- confidentiality requirements;
- internal corporate restrictions;
- industrial data sensitivity.

This creates the need for:
- fully local retrieval;
- explainable retrieval logic;
- measurable retrieval quality;
- hybrid search strategies.

---

# Why RAG

Pure LLM-based solutions are insufficient for technical documentation because:
- LLMs may hallucinate;
- responses must remain grounded in source documents;
- engineering queries require factual precision.

RAG (Retrieval-Augmented Generation) addresses this problem by:
1. retrieving relevant document fragments;
2. providing grounded context;
3. enabling explainable answer generation.

However, vector search alone is not enough for industrial documentation.

This project explores hybrid retrieval strategies combining:
- exact matching;
- semantic search;
- BM25 lexical retrieval;
- section-aware filtering.

---

# Retrieval Evolution

The retrieval pipeline was improved iteratively.

## Stage 1 — Pure Semantic Search

Initial implementation:
- sentence embeddings;
- FAISS vector search.

Problems:
- unstable rankings;
- semantically similar but incorrect chunks;
- section leakage.

---

## Stage 2 — Section-Aware Retrieval

Added:
- document section metadata;
- retrieval constrained by section.

Improvements:
- reduced false positives;
- improved contextual relevance.

---

## Stage 3 — Exact Match Retrieval

Added:
- exact substring matching inside target section.

Purpose:
- improve deterministic retrieval for highly specific engineering terms;
- improve retrieval for exact process descriptions.

---

## Stage 4 — Hybrid Retrieval (Semantic + BM25)

Added:
- BM25 lexical search;
- result merging across retrieval strategies.

Purpose:
- improve robustness;
- capture both semantic similarity and lexical overlap;
- reduce failures caused by synonym mismatch.

Current retrieval combines:
- exact search;
- BM25;
- semantic search.

---

# Architecture

```text
DOCX
 ↓
Section extraction
 ↓
Sentence splitting
 ↓
Window chunking
 ↓
Embedding generation
 ↓
FAISS indexing
 ↓
Hybrid retrieval:
    - Exact match
    - BM25
    - Semantic search
 ↓
CrossEncoder reranking for Semantic Search
 ↓
Result merging
 ↓
Retrieved context
```
---

# Chunking Strategy

The pipeline uses:
- sentence-level splitting;
- rolling window chunking;
- overlap between neighboring chunks.

Each chunk contains:
- section metadata;
- source paragraph reference;
- sentence boundaries;
- original text context.

This improves retrieval stability for technical descriptions spanning multiple sentences.

---

# Evaluation Strategy

The project includes an automated retrieval evaluation pipeline.

Evaluation dataset:

- manually prepared engineering questions;
- expected target fragments;
- section constraints.

---

# Reranking

The project includes an additional reranking stage using CrossEncoder models.

Purpose:
* improve ranking precision;
* reorder semantically similar chunks;
* increase relevance of top-ranked results.

This stage is applied after hybrid retrieval and before final context selection.

---

# Retrieval Metrics

The following metrics are used:

## Hit Rate@K

Measures whether the correct chunk appears in top-K results.

Examples:
- Hit Rate@1 → correct result is first;
- Hit Rate@3 → correct result exists in top-3.

This metric is important because RAG systems usually pass only top-ranked chunks to the LLM.

---

## Mean Reciprocal Rank (MRR)

MRR evaluates ranking quality.

Higher MRR means:
- relevant chunks appear closer to the top;
- retrieval quality is more stable.

---

# Current Results

Evaluation on 20 technical questions:

| Metric      | Value |
| ----------- | ----- |
| Hit Rate@1  | 0.75  |
| Hit Rate@3  | 1.00  |
| Hit Rate@5  | 1.00  |
| Hit Rate@10 | 1.00  |
| MRR         | 0.867 |

Interpretation:
- the correct chunk appears in top-3 results for all evaluation queries;
- most correct chunks are ranked first;
- hybrid retrieval significantly improved retrieval stability.

---

# Tech Stack

- Python
- PyTorch
- sentence-transformers
- FAISS
- rank-bm25
- pandas
- python-docx

---

# Future Improvements

Planned improvements:
- query expansion;
- answer generation with local LLMs;
- web interface.

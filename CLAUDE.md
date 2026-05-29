# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-based product classification system that predicts 4-level categories from Chinese product names. Uses BGE embedding for semantic retrieval and LLM APIs for final classification.

**Core Pipeline**: Knowledge Base → BGE Vector Retrieval → Brand Boosting → LLM Classification

## Common Commands

```bash
# Build knowledge base and vector index (first time setup)
python main.py build

# Interactive classification mode
python main.py interactive

# Batch classify products from CSV
python main.py classify data/商品.csv output/结果.csv

# Evaluate system performance
python main.py evaluate data/test.csv [level]  # level: 1-4, default 4
```

## Configuration

All settings in `.env` file:

| Key | Purpose | Default |
|-----|---------|---------|
| `LLM_PROVIDER` | openai, yi, custom | openai |
| `LLM_MODEL` | Model name | gpt-3.5-turbo |
| `OPENAI_API_KEY` | API key | - |
| `API_BASE_URL` | For custom/compatible APIs | - |
| `BGE_MODEL_NAME` | BAAI/bge-base-zh-v1.5 | - |
| `RETRIEVAL_TOP_K` | Final results count | 3 |
| `RETRIEVAL_CANDIDATE_K` | Candidate pool size | 8 |
| `BRAND_BOOST_SCORE` | Brand match bonus | 0.15 |
| `ENABLE_CACHE` | Cache LLM results | true |

## Architecture

### Core Components

- **`src/knowledge_base.py`** - Loads `data/品类定义详情.csv`, creates 437 level-4 category documents with brand/flavor/packaging indexes
- **`src/vector_store.py`** - BGE model encoding, vector indexing (numpy-based, optional Faiss)
- **`src/retriever.py`** - Hybrid retrieval: BGE semantic + brand boosting
- **`src/llm_classifier.py`** - OpenAI-compatible API client with prompt engineering for product profiling
- **`src/evaluator.py`** - Accuracy metrics, error analysis, per-category performance

### Data Flow

```
Product Name → BGE Encode → Cosine Similarity → Top-K Categories
                     ↓
              Brand Extraction (from name) → Brand Index Lookup → Boost Scores
                     ↓
              Re-rank → Top-3 Categories → LLM Prompt → Structured Output
```

### Key Design Decisions

1. **Chunk Strategy**: Each level-4 category = one document (~500 chars). Preserves complete definition + brands + flavors.
2. **Brand Boosting**: If product name contains known brand, add `BRAND_BOOST_SCORE` to matching categories.
3. **LLM Prompting**: Strict "no hallucination" policy - outputs "未知" (unknown) when KB lacks info, not fabricated guesses.

## Directory Structure

```
data/               # Source data (品类定义详情.csv)
knowledge_base/     # JSON knowledge documents, brand/flavor indexes
bge_vector_index/   # Vector embeddings (npy), documents.json
output/             # Batch classification results
logs/               # Structured logs (app.log with rotation)
```

## LLM Prompt Engineering

Critical constraint: **Prevent hallucination**. The prompt (`src/llm_classifier.py:_build_product_profile_prompt`) explicitly requires:
- Only output information explicitly provided in knowledge base
- Use "未知" for unknown fields
- No vague terms like "全国", "常见", "通常"
- Explain reasoning for each extracted field

## Adding New LLM Providers

Set in `.env`:
```
LLM_PROVIDER=custom
API_BASE_URL=https://your-api.com/v1
OPENAI_API_KEY=your-key
```

Any OpenAI-compatible API works (Yi, LMStudio, local vLLM, etc.).

## Performance

- **Accuracy**: ~92% on level-4 classification
- **Latency**: ~1.2s per product (retrieval + LLM)
- **Cache**: Enabled by default, LRU with `CACHE_SIZE` limit

## Troubleshooting

- **Model loading fails**: Check network, first run downloads 400MB BGE model
- **API call fails**: Verify `OPENAI_API_KEY` and `API_BASE_URL`
- **Low accuracy**: Check `logs/app.log` for retrieval details, adjust `BRAND_BOOST_SCORE`
- **"瞎编" outputs**: Prompt should prevent this; if persists, check `llm_classifier.py` prompt template

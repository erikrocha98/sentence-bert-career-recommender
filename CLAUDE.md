# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

This repo currently contains **only the planning PDFs** — no code yet:
- `_FundIA__TP3_TransferLearning_2026_1.pdf` — the official assignment (UFMG, Fundamentos de IA, TP3).
- `Proposta_Refinada_NLP_UFMG.pdf` — the team's refined project proposal (the design being built).

When code is added it will live in a Jupyter/Colab notebook (the primary deliverable). There is no build/lint/test tooling set up yet; this file documents what the project *is* and the constraints/contracts it must satisfy, so the first implementation doesn't drift from the agreed design.

## The project

**Sistema Inteligente de Recomendação de Disciplinas e Trilhas de Carreira com Busca Semântica.**

A student describes career goals in free-text Portuguese; the system returns matching careers, recommended UFMG courses for each, and a prerequisite-respecting study roadmap. The NLP task is **semantic search via sentence embeddings + cosine similarity** — this is the single primary task; do not expand scope into open-ended chat/QA generation.

### Two-hop pipeline (the core architecture)
1. **Hop 1 — student → career:** embed the student's free-text query, rank career profiles by cosine similarity.
2. **Hop 2 — career → courses:** take the selected career's competencies/technologies as a new semantic query against course-syllabus embeddings; rank courses.
3. **Roadmap:** topological sort over the course prerequisite DAG, distributing recommended courses across semesters.

The two-hop design exists to give **explainability** ("we recommend *Machine Learning* because career *ML Engineer* requires *Deep Learning*"). Preserve that traceability when changing the pipeline.

### Data sources
- **Fonte A — Courses:** ~100–150 UFMG courses. Fields: name, ementa (syllabus), objectives, prerequisites (structured as a list of course codes → DAG edges), workload. Prerequisites-as-graph is the trickiest data work.
- **Fonte C — Career profiles:** 20–30 hand-curated profiles. Fields: role name, function description, required competencies, technologies, area. Curated from sources like roadmap.sh and domain knowledge — **deliberately not scraped** (scraping job postings was removed to avoid ToS/scraping risk).

## Integration contracts (agreed; do not change unilaterally)

These three contracts let the two authors work in parallel — embeddings generated under different functions are incompatible, so honor them exactly:

1. **Embedding function:** `gerar_embedding(texto: str) -> np.array` — a single shared function used everywhere.
2. **Persisted record schema:** each collection stored as records of `{id, texto, embedding}`. Persisted as `.npy`/`.pkl` (e.g. `disciplinas_emb.npy`, `perfis_emb.npy`). Embeddings are computed **offline once** and reloaded; search is online.
3. **Search function:** `buscar(query_emb, base_emb, top_k) -> [(id, score)]` — generic cosine-similarity ranking, reused by both hops.

For each course, embed `nome + ementa + objetivos` concatenated. For each profile, embed `descrição + competências + tecnologias`.

## Technical constraints (from the assignment — non-negotiable)

- **Must fine-tune a pretrained model OR use its embeddings** (BERT / Sentence-BERT / RoBERTa). Pure commercial-API calls with no local training/adaptation are rejected. This project satisfies it via Sentence-BERT embeddings.
- **Model:** `paraphrase-multilingual-MiniLM-L12-v2` (Sentence-BERT). Portuguese alternatives: STS-BERTimbau, `neuralmind/bert-base-portuguese-cased`. Portuguese-pretrained models are strongly recommended since the data is in Portuguese.
- **Stack:** PyTorch + Hugging Face Transformers/`sentence-transformers`, on free Google Colab (T4 GPU). At this scale, `numpy`/`scikit-learn` cosine similarity is enough — **no FAISS or vector DB**.
- **Dataset size:** hundreds to low thousands of examples; prioritize curation over volume.
- **No data leakage** between train/test — this is weighted heavily in grading.

## Evaluation (build these, they are graded)

- ~15–20 test queries with expected answers; measure **top-3 precision**.
- **TF-IDF baseline** to compare against Sentence-BERT (demonstrate the semantic gain over lexical search).
- **Similarity-matrix heatmap** as visual demonstration.
- The error-analysis section ("where and why the model fails") matters more than raw accuracy — grading explicitly rewards critical analysis (F1/accuracy) over a high score alone.

## Plans / documentation

When the user asks for a plan, save it as a Markdown file inside the `documentacao/` folder. This folder is gitignored (local working notes, not committed).

## Testing

Write unit tests that cover the **main behavior** of each feature — focus on behavior, only the essentials. Keep the suite small: this project needs to be implemented quickly, so do **not** create many tests. Prefer a handful of tests that exercise the core path (e.g. `buscar` ranks by cosine correctly, the prerequisite graph topological sort respects ordering) over broad edge-case coverage.

## Deliverables

1. Self-contained Jupyter/Colab notebook with Markdown cells explaining reasoning, data handling, training loop, and evaluation.
2. A 4–6 page LaTeX technical report (intro/motivation, EDA, methodology, results + error discussion).
3. Optional extra credit: a simple Gradio/Streamlit demo UI.

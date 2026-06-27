"""Avaliacao quantitativa do TP3: Sentence-BERT vs. TF-IDF.

O script mede dois pontos do pipeline:

1. Hop 1 (texto do aluno -> carreira): top-1, hit@3 e MRR para 20 consultas
   curadas, cada uma com uma carreira esperada.
2. Hop 2 (competencias -> disciplinas): Precision@3, Recall@3 e MRR para 13
   consultas de competencias, cada uma com tres disciplinas esperadas.

Tambem gera figuras usadas no relatorio LaTeX:
- distribuicao de perfis por area;
- comparacao de metricas entre TF-IDF e Sentence-BERT;
- mapa de similaridade das consultas de carreira com os perfis.

Uso:
    python scripts/avaliar_modelos.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.busca import buscar, carregar_colecao
from src.embeddings import gerar_embedding
from src.schema import texto_para_embedding, texto_para_embedding_perfil

CAMINHO_AVALIACAO = RAIZ / "data" / "consultas_avaliacao.json"
CAMINHO_PERFIS = RAIZ / "data" / "perfis.json"
CAMINHO_DISCIPLINAS = RAIZ / "data" / "disciplinas.json"
CAMINHO_PERFIS_EMB = RAIZ / "data" / "perfis_emb.pkl"
CAMINHO_DISCIPLINAS_EMB = RAIZ / "data" / "disciplinas_emb.pkl"
DIR_RESULTADOS = RAIZ / "results"
DIR_FIGURAS = DIR_RESULTADOS / "figures"


def _rank_tfidf(query: str, ids: list[str], textos: list[str], top_k: int) -> list[tuple[str, float]]:
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2))
    matriz = vectorizer.fit_transform(textos)
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, matriz).ravel()
    ordem = np.argsort(scores)[::-1][:top_k]
    return [(ids[i], float(scores[i])) for i in ordem]


def _mrr(ranking: list[str], esperados: set[str]) -> float:
    for i, item in enumerate(ranking, start=1):
        if item in esperados:
            return 1.0 / i
    return 0.0


def _avaliar_carreiras(casos, perfis, perfis_emb):
    ids = [p["id"] for p in perfis]
    textos = [texto_para_embedding_perfil(p) for p in perfis]
    linhas = []

    for caso in casos:
        query = caso["query"]
        esperado = caso["esperado"]
        ranking_sbert = buscar(gerar_embedding(query), perfis_emb, top_k=5)
        ranking_tfidf = _rank_tfidf(query, ids, textos, top_k=5)

        for metodo, ranking in [("Sentence-BERT", ranking_sbert), ("TF-IDF", ranking_tfidf)]:
            top_ids = [r[0] for r in ranking]
            linhas.append(
                {
                    "metodo": metodo,
                    "query": query,
                    "esperado": esperado,
                    "top1": top_ids[0] if top_ids else "",
                    "top3": "|".join(top_ids[:3]),
                    "top5_scores": "|".join(f"{cid}:{score:.4f}" for cid, score in ranking),
                    "acerto_top1": int(top_ids[:1] == [esperado]),
                    "hit_at_3": int(esperado in top_ids[:3]),
                    "mrr": _mrr(top_ids, {esperado}),
                }
            )

    metricas = {}
    for metodo in ["TF-IDF", "Sentence-BERT"]:
        subset = [l for l in linhas if l["metodo"] == metodo]
        metricas[metodo] = {
            "top1_accuracy": float(np.mean([l["acerto_top1"] for l in subset])),
            "hit_at_3": float(np.mean([l["hit_at_3"] for l in subset])),
            "mrr": float(np.mean([l["mrr"] for l in subset])),
        }

    return metricas, linhas


def _avaliar_disciplinas(casos, disciplinas, disciplinas_emb):
    ids = [d["codigo"] for d in disciplinas]
    textos = [texto_para_embedding(d) for d in disciplinas]
    linhas = []

    for caso in casos:
        query = caso["query"]
        esperados = set(caso["esperados"])
        ranking_sbert = buscar(gerar_embedding(query), disciplinas_emb, top_k=5)
        ranking_tfidf = _rank_tfidf(query, ids, textos, top_k=5)

        for metodo, ranking in [("Sentence-BERT", ranking_sbert), ("TF-IDF", ranking_tfidf)]:
            top_ids = [r[0] for r in ranking]
            acertos_top3 = len(esperados.intersection(top_ids[:3]))
            linhas.append(
                {
                    "metodo": metodo,
                    "query": query,
                    "esperados": "|".join(sorted(esperados)),
                    "top3": "|".join(top_ids[:3]),
                    "top5_scores": "|".join(f"{cid}:{score:.4f}" for cid, score in ranking),
                    "precision_at_3": acertos_top3 / 3.0,
                    "recall_at_3": acertos_top3 / len(esperados),
                    "mrr": _mrr(top_ids, esperados),
                }
            )

    metricas = {}
    for metodo in ["TF-IDF", "Sentence-BERT"]:
        subset = [l for l in linhas if l["metodo"] == metodo]
        metricas[metodo] = {
            "precision_at_3": float(np.mean([l["precision_at_3"] for l in subset])),
            "recall_at_3": float(np.mean([l["recall_at_3"] for l in subset])),
            "mrr": float(np.mean([l["mrr"] for l in subset])),
        }

    return metricas, linhas


def _salvar_csv(caminho: Path, linhas: list[dict]) -> None:
    if not linhas:
        return
    with caminho.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(linhas[0]))
        writer.writeheader()
        writer.writerows(linhas)


def _fig_distribuicao_areas(perfis: list[dict]) -> None:
    contagem = Counter(p["area"] for p in perfis)
    areas = list(contagem)
    valores = [contagem[a] for a in areas]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(areas, valores, color="#3b82f6")
    ax.set_xlabel("Numero de perfis")
    ax.set_title("Distribuicao dos perfis de carreira por area")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "distribuicao_perfis_area.png", dpi=180)
    plt.close(fig)


def _fig_comparacao_metricas(metricas: dict) -> None:
    labels = ["Hop 1 Top-1", "Hop 1 Hit@3", "Hop 2 P@3"]
    tfidf = [
        metricas["carreiras"]["TF-IDF"]["top1_accuracy"],
        metricas["carreiras"]["TF-IDF"]["hit_at_3"],
        metricas["disciplinas"]["TF-IDF"]["precision_at_3"],
    ]
    sbert = [
        metricas["carreiras"]["Sentence-BERT"]["top1_accuracy"],
        metricas["carreiras"]["Sentence-BERT"]["hit_at_3"],
        metricas["disciplinas"]["Sentence-BERT"]["precision_at_3"],
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - width / 2, tfidf, width, label="TF-IDF", color="#94a3b8")
    ax.bar(x + width / 2, sbert, width, label="Sentence-BERT", color="#2563eb")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metrica media")
    ax.set_xticks(x, labels)
    ax.set_title("Comparacao quantitativa dos metodos")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "comparacao_metricas.png", dpi=180)
    plt.close(fig)


def _fig_heatmap_carreiras(casos, perfis, perfis_emb) -> None:
    ids = [p["id"] for p in perfis]
    nomes = {p["id"]: p["nome"] for p in perfis}
    matriz_emb = np.vstack([np.asarray(r["embedding"], dtype=np.float32) for r in perfis_emb])
    matriz_norm = matriz_emb / (np.linalg.norm(matriz_emb, axis=1, keepdims=True) + 1e-12)

    scores = []
    labels_y = []
    for i, caso in enumerate(casos, start=1):
        q = gerar_embedding(caso["query"])
        q = q / (np.linalg.norm(q) + 1e-12)
        scores.append(matriz_norm @ q)
        labels_y.append(f"Q{i}")

    matriz = np.vstack(scores)
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(matriz, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.85, float(matriz.max())))
    ax.set_yticks(np.arange(len(labels_y)), labels_y)
    ax.set_xticks(np.arange(len(ids)), [nomes[i].replace("Engenheiro de ", "Eng. ") for i in ids], rotation=65, ha="right", fontsize=7)
    ax.set_title("Matriz de similaridade Sentence-BERT: consultas x perfis")
    ax.set_xlabel("Perfil de carreira")
    ax.set_ylabel("Consulta de avaliacao")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Similaridade de cosseno")
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "similaridade_queries_carreiras.png", dpi=180)
    plt.close(fig)


def main() -> None:
    DIR_RESULTADOS.mkdir(exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    avaliacao = json.loads(CAMINHO_AVALIACAO.read_text(encoding="utf-8"))
    perfis = json.loads(CAMINHO_PERFIS.read_text(encoding="utf-8"))
    disciplinas = json.loads(CAMINHO_DISCIPLINAS.read_text(encoding="utf-8"))
    perfis_emb = carregar_colecao(CAMINHO_PERFIS_EMB)
    disciplinas_emb = carregar_colecao(CAMINHO_DISCIPLINAS_EMB)

    metricas_car, linhas_car = _avaliar_carreiras(avaliacao["carreiras"], perfis, perfis_emb)
    metricas_disc, linhas_disc = _avaliar_disciplinas(
        avaliacao["disciplinas"], disciplinas, disciplinas_emb
    )

    resultados = {
        "n_casos": {
            "carreiras": len(avaliacao["carreiras"]),
            "disciplinas": len(avaliacao["disciplinas"]),
        },
        "carreiras": metricas_car,
        "disciplinas": metricas_disc,
    }

    (DIR_RESULTADOS / "avaliacao_metricas.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _salvar_csv(DIR_RESULTADOS / "avaliacao_carreiras.csv", linhas_car)
    _salvar_csv(DIR_RESULTADOS / "avaliacao_disciplinas.csv", linhas_disc)

    _fig_distribuicao_areas(perfis)
    _fig_comparacao_metricas(resultados)
    _fig_heatmap_carreiras(avaliacao["carreiras"], perfis, perfis_emb)

    print(json.dumps(resultados, ensure_ascii=False, indent=2))
    print(f"Resultados salvos em: {DIR_RESULTADOS}")


if __name__ == "__main__":
    main()

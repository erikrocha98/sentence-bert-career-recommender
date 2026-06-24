"""Validação MANUAL do Hop 1 e do pipeline ponta a ponta.

Imprime, para algumas consultas em texto livre do aluno:
- o top-k de carreiras (Hop 1, score | id | nome) — conferência humana de coerência;
- a trilha completa de uma consulta (carreira + disciplinas + roadmap + justificativa).

Não há asserts sobre o conteúdo: a validação é humana, por design.

Uso: ``python scripts/validar_carreiras.py``
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.busca import carregar_colecao
from src.carreiras import recomendar_carreiras
from src.pipeline import recomendar_trilha

CAMINHO_PERFIS = RAIZ / "data" / "perfis.json"
CAMINHO_DISCIPLINAS = RAIZ / "data" / "disciplinas.json"
CAMINHO_PERFIS_EMB = RAIZ / "data" / "perfis_emb.pkl"
CAMINHO_DISCIPLINAS_EMB = RAIZ / "data" / "disciplinas_emb.pkl"
TOP_K = 5

QUERIES = [
    "quero trabalhar com IA generativa e processamento de linguagem natural",
    "tenho interesse em construir pipelines de dados e data warehouses",
    "quero atuar com estatística, regressão e análise de dados",
    "gosto de robôs, sensores e controle de sistemas autônomos",
]


def main() -> None:
    perfis = json.loads(CAMINHO_PERFIS.read_text(encoding="utf-8"))
    disciplinas = json.loads(CAMINHO_DISCIPLINAS.read_text(encoding="utf-8"))
    nomes_perfil = {p["id"]: p["nome"] for p in perfis}
    nomes_disc = {d["codigo"]: d["nome"] for d in disciplinas}
    perfis_emb = carregar_colecao(CAMINHO_PERFIS_EMB)
    disciplinas_emb = carregar_colecao(CAMINHO_DISCIPLINAS_EMB)
    print(f"{len(perfis_emb)} perfis e {len(disciplinas_emb)} disciplinas na base.\n")

    print("=== Hop 1: texto livre do aluno -> carreiras ===")
    for q in QUERIES:
        print(f"Consulta: {q!r}")
        for cid, score in recomendar_carreiras(q, perfis_emb, top_k=TOP_K):
            print(f"    {score:.3f}  {cid}  {nomes_perfil.get(cid, '?')}")
        print()

    print("=== Pipeline ponta a ponta (1 consulta) ===")
    res = recomendar_trilha(
        QUERIES[0], perfis, perfis_emb, disciplinas, disciplinas_emb, top_k=3
    )
    print(f"Carreira: {res['carreira']}  (score {res['score']:.3f})")
    print("Disciplinas recomendadas:")
    for cod, score in res["disciplinas"]:
        print(f"    {score:.3f}  {cod}  {nomes_disc.get(cod, '?')}")
    print("Roadmap por semestre:")
    for sem, itens in res["roadmap"].items():
        rotulos = ", ".join(f"{it['codigo']}({it['origem']})" for it in itens)
        print(f"    {sem}º semestre: {rotulos}")
    print(f"\nJustificativa: {res['justificativa']}")


if __name__ == "__main__":
    main()

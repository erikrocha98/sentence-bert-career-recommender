"""Validação MANUAL da recomendação de disciplinas por competências (Hop 2).

Imprime o top-k (score | código | nome) para alguns conjuntos de competências, para
que a conferência "na unha" — se as disciplinas retornadas fazem sentido — seja feita
pelo usuário. Não há asserts sobre o conteúdo: a validação é humana, por design.

Uso: ``python scripts/validar_recomendacao.py``
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.busca import carregar_colecao
from src.recomendacao import recomendar_disciplinas

CAMINHO_DISCIPLINAS = RAIZ / "data" / "disciplinas.json"
CAMINHO_EMB = RAIZ / "data" / "disciplinas_emb.pkl"
TOP_K = 5

# Conjuntos representativos de áreas distintas — ajuste à vontade para validar.
CONJUNTOS_COMPETENCIAS = [
    "Python, Deep Learning, MLOps, redes neurais, visão computacional",
    "SQL, modelagem de dados, ETL, data warehouse, BI",
    "estatística, regressão, probabilidade, análise de dados",
]


def main() -> None:
    disciplinas = json.loads(CAMINHO_DISCIPLINAS.read_text(encoding="utf-8"))
    nomes = {d["codigo"]: d["nome"] for d in disciplinas}
    base = carregar_colecao(CAMINHO_EMB)
    print(f"{len(base)} disciplinas na base de embeddings.\n")

    for competencias in CONJUNTOS_COMPETENCIAS:
        print(f"Competências: {competencias!r}")
        print(f"  top-{TOP_K} disciplinas:")
        for codigo, score in recomendar_disciplinas(competencias, base, top_k=TOP_K):
            print(f"    {score:.3f}  {codigo}  {nomes.get(codigo, '?')}")
        print()


if __name__ == "__main__":
    main()

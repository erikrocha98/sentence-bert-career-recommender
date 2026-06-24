"""Validação MANUAL do roadmap de estudos por semestre (Hop 3).

Monta a trilha a partir de um conjunto de disciplinas recomendadas e imprime a
distribuição por semestre, para conferência "na unha" de que a ordem respeita os
pré-requisitos. O conjunto de exemplo simula a saída do Hop 2.

Uso: ``python scripts/validar_roadmap.py``
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.roadmap import montar_roadmap

CAMINHO_DISCIPLINAS = RAIZ / "data" / "disciplinas.json"

# Conjunto de exemplo (perfil ML/IA): disciplinas "recomendadas" cujos pré-requisitos
# vão puxar a cadeia de programação e matemática.
RECOMENDADAS = ["EEEXXA", "DCC043", "DCC057", "DCC212", "DCC011"]


def imprimir(roadmap, nomes) -> None:
    for sem, itens in roadmap.items():
        print(f"  Semestre {sem}:")
        for it in itens:
            marca = "" if it["origem"] == "recomendada" else "  [pré-requisito]"
            print(f"    {it['codigo']}  {nomes.get(it['codigo'], '?')}{marca}")


def main() -> None:
    disciplinas = json.loads(CAMINHO_DISCIPLINAS.read_text(encoding="utf-8"))
    nomes = {d["codigo"]: d["nome"] for d in disciplinas}

    print(f"Recomendadas: {RECOMENDADAS}\n")

    print("incluir_prerequisitos=True (puxa dependências faltantes):")
    imprimir(montar_roadmap(RECOMENDADAS, disciplinas, incluir_prerequisitos=True), nomes)

    print("\nincluir_prerequisitos=False (só as recomendadas):")
    imprimir(montar_roadmap(RECOMENDADAS, disciplinas, incluir_prerequisitos=False), nomes)


if __name__ == "__main__":
    main()

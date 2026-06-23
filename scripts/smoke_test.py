"""Smoke test local (substitui o passo "abra o Colab e teste" da proposta original).

Confirma que o ambiente em CPU está funcional:
1. carrega o modelo e gera o embedding de uma frase (imprime shape/dtype);
2. carrega e valida as disciplinas-semente (``data/disciplinas.json``);
3. roda uma busca real sobre as disciplinas, demonstrando o salto 2 do pipeline.

Uso: ``python scripts/smoke_test.py``
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.busca import buscar
from src.embeddings import MODELO_NOME, gerar_embedding
from src.schema import texto_para_embedding, validar_disciplina


def main() -> None:
    print(f"Carregando modelo: {MODELO_NOME} (pode baixar ~120MB na 1ª vez)...")
    emb = gerar_embedding("Quero trabalhar com aprendizado de máquina e inteligência artificial.")
    print(f"OK. shape do embedding: {emb.shape}, dtype: {emb.dtype}")
    print(f"primeiros valores: {emb[:5]}\n")

    disciplinas = json.loads((RAIZ / "data" / "disciplinas.json").read_text(encoding="utf-8"))
    for d in disciplinas:
        validar_disciplina(d)
    print(f"{len(disciplinas)} disciplinas carregadas e validadas.\n")

    base = [
        {
            "id": d["codigo"],
            "texto": texto_para_embedding(d),
            "embedding": gerar_embedding(texto_para_embedding(d)),
        }
        for d in disciplinas
    ]

    consulta = "inteligência artificial, aprendizado de máquina e ciência de dados"
    print(f"Busca de sanidade — consulta: {consulta!r}")
    print("top-5 disciplinas:")
    nomes = {d["codigo"]: d["nome"] for d in disciplinas}
    for codigo, score in buscar(gerar_embedding(consulta), base, top_k=5):
        print(f"  {score:.3f}  {codigo}  {nomes[codigo]}")


if __name__ == "__main__":
    main()

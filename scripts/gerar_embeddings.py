"""Geração offline dos embeddings das disciplinas (etapa pré-computada do pipeline).

Para cada disciplina concatena nome + ementa + objetivos (via ``texto_para_embedding``),
gera o embedding (via ``gerar_embedding``) e salva a coleção como registros
``{id, texto, embedding}`` em ``data/disciplinas_emb.pkl`` (via ``salvar_colecao``).

Ao final, recarrega o arquivo e verifica que a matriz de embeddings é
``nº de disciplinas × 384`` (dimensão do paraphrase-multilingual-MiniLM-L12-v2).

Uso: ``python scripts/gerar_embeddings.py``
"""
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.busca import carregar_colecao, salvar_colecao
from src.embeddings import MODELO_NOME, gerar_embedding
from src.schema import texto_para_embedding, validar_disciplina

CAMINHO_DISCIPLINAS = RAIZ / "data" / "disciplinas.json"
CAMINHO_SAIDA = RAIZ / "data" / "disciplinas_emb.pkl"
DIM_ESPERADA = 384


def gerar() -> list[dict]:
    """Carrega e valida as disciplinas, gera os embeddings e persiste a coleção."""
    disciplinas = json.loads(CAMINHO_DISCIPLINAS.read_text(encoding="utf-8"))
    for d in disciplinas:
        validar_disciplina(d)
    print(f"{len(disciplinas)} disciplinas carregadas e validadas.")

    print(f"Gerando embeddings com {MODELO_NOME} (pode baixar ~120MB na 1ª vez)...")
    base = []
    for d in disciplinas:
        texto = texto_para_embedding(d)
        base.append(
            {
                "id": d["codigo"],
                "texto": texto,
                "embedding": gerar_embedding(texto),
            }
        )

    salvar_colecao(base, CAMINHO_SAIDA)
    print(f"Coleção salva em {CAMINHO_SAIDA.relative_to(RAIZ)}")
    return base


def verificar() -> None:
    """Recarrega o arquivo salvo e confere len, shape, dtype e a matriz nº × 384."""
    disciplinas = json.loads(CAMINHO_DISCIPLINAS.read_text(encoding="utf-8"))
    colecao = carregar_colecao(CAMINHO_SAIDA)

    assert len(colecao) == len(disciplinas), (
        f"esperado {len(disciplinas)} registros, obtido {len(colecao)}"
    )
    for r in colecao:
        emb = np.asarray(r["embedding"])
        assert emb.shape == (DIM_ESPERADA,), f"{r['id']}: shape {emb.shape} != ({DIM_ESPERADA},)"
        assert emb.dtype == np.float32, f"{r['id']}: dtype {emb.dtype} != float32"

    ids_json = [d["codigo"] for d in disciplinas]
    assert [r["id"] for r in colecao] == ids_json, "ids da coleção não batem com o JSON"

    matriz = np.vstack([r["embedding"] for r in colecao])
    print(f"Matriz de embeddings: {matriz.shape}  (dtype {matriz.dtype})")
    assert matriz.shape == (len(disciplinas), DIM_ESPERADA)
    print("OK — verificação passou.")


def main() -> None:
    gerar()
    verificar()


if __name__ == "__main__":
    main()

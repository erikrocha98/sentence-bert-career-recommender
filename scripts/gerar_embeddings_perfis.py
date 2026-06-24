"""Geração offline dos embeddings dos perfis de carreira (Fonte C do pipeline).

Para cada perfil concatena descrição + competências + tecnologias (via
``texto_para_embedding_perfil``), gera o embedding (via ``gerar_embedding``) e salva a
coleção como registros ``{id, texto, embedding}`` em ``data/perfis_emb.pkl`` (via
``salvar_colecao``).

Ao final, recarrega o arquivo e verifica que a matriz de embeddings é
``nº de perfis × 384`` (dimensão do paraphrase-multilingual-MiniLM-L12-v2).

Uso: ``python scripts/gerar_embeddings_perfis.py``
"""
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.busca import carregar_colecao, salvar_colecao
from src.embeddings import MODELO_NOME, gerar_embedding
from src.schema import texto_para_embedding_perfil, validar_perfil

CAMINHO_PERFIS = RAIZ / "data" / "perfis.json"
CAMINHO_SAIDA = RAIZ / "data" / "perfis_emb.pkl"
DIM_ESPERADA = 384


def gerar() -> list[dict]:
    """Carrega e valida os perfis, gera os embeddings e persiste a coleção."""
    perfis = json.loads(CAMINHO_PERFIS.read_text(encoding="utf-8"))
    for p in perfis:
        validar_perfil(p)
    print(f"{len(perfis)} perfis carregados e validados.")

    print(f"Gerando embeddings com {MODELO_NOME} (pode baixar ~120MB na 1ª vez)...")
    base = []
    for p in perfis:
        texto = texto_para_embedding_perfil(p)
        base.append(
            {
                "id": p["id"],
                "texto": texto,
                "embedding": gerar_embedding(texto),
            }
        )

    salvar_colecao(base, CAMINHO_SAIDA)
    print(f"Coleção salva em {CAMINHO_SAIDA.relative_to(RAIZ)}")
    return base


def verificar() -> None:
    """Recarrega o arquivo salvo e confere len, shape, dtype e a matriz nº × 384."""
    perfis = json.loads(CAMINHO_PERFIS.read_text(encoding="utf-8"))
    colecao = carregar_colecao(CAMINHO_SAIDA)

    assert len(colecao) == len(perfis), (
        f"esperado {len(perfis)} registros, obtido {len(colecao)}"
    )
    for r in colecao:
        emb = np.asarray(r["embedding"])
        assert emb.shape == (DIM_ESPERADA,), f"{r['id']}: shape {emb.shape} != ({DIM_ESPERADA},)"
        assert emb.dtype == np.float32, f"{r['id']}: dtype {emb.dtype} != float32"

    ids_json = [p["id"] for p in perfis]
    assert [r["id"] for r in colecao] == ids_json, "ids da coleção não batem com o JSON"

    matriz = np.vstack([r["embedding"] for r in colecao])
    print(f"Matriz de embeddings: {matriz.shape}  (dtype {matriz.dtype})")
    assert matriz.shape == (len(perfis), DIM_ESPERADA)
    print("OK — verificação passou.")


def main() -> None:
    gerar()
    verificar()


if __name__ == "__main__":
    main()

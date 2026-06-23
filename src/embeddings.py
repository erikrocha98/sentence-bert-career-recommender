"""Carregamento (lazy) do modelo de embeddings e a função compartilhada ``gerar_embedding``.

Contrato (CLAUDE.md): ``gerar_embedding(texto: str) -> np.ndarray``.
Modelo: ``paraphrase-multilingual-MiniLM-L12-v2`` (Sentence-BERT, multilíngue, 384 dims).
Roda em CPU. O modelo é carregado uma única vez (singleton de módulo).
"""
from __future__ import annotations

import numpy as np

MODELO_NOME = "paraphrase-multilingual-MiniLM-L12-v2"

_modelo = None


def carregar_modelo():
    """Carrega o ``SentenceTransformer`` na primeira chamada e reusa depois."""
    global _modelo
    if _modelo is None:
        from sentence_transformers import SentenceTransformer

        _modelo = SentenceTransformer(MODELO_NOME)
    return _modelo


def gerar_embedding(texto: str) -> np.ndarray:
    """Retorna o embedding (float32, vetor 1-D) do texto.

    Embedding "cru", sem normalizar — a similaridade de cosseno é calculada em
    ``busca.buscar``, mantendo as responsabilidades separadas.
    """
    modelo = carregar_modelo()
    emb = modelo.encode(texto, convert_to_numpy=True, normalize_embeddings=False)
    return emb.astype(np.float32)

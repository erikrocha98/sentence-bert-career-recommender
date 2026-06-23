"""Busca por similaridade de cosseno e persistência das coleções de embeddings.

Contratos (CLAUDE.md):
- ``buscar(query_emb, base_emb, top_k) -> list[(id, score)]``
- registro persistido por coleção: ``{id, texto, embedding}``
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def _cosseno(query: np.ndarray, matriz: np.ndarray) -> np.ndarray:
    """Similaridade de cosseno entre um vetor e cada linha de uma matriz."""
    q = query / (np.linalg.norm(query) + 1e-12)
    m = matriz / (np.linalg.norm(matriz, axis=1, keepdims=True) + 1e-12)
    return m @ q


def buscar(query_emb, base_emb, top_k):
    """Ranqueia os registros de ``base_emb`` por similaridade de cosseno com ``query_emb``.

    ``base_emb``: lista de registros ``{id, texto, embedding}``.
    Retorna lista de ``(id, score)`` ordenada por score desc, com no máximo ``top_k`` itens.
    """
    if not base_emb:
        return []
    ids = [r["id"] for r in base_emb]
    matriz = np.vstack([np.asarray(r["embedding"], dtype=np.float32) for r in base_emb])
    scores = _cosseno(np.asarray(query_emb, dtype=np.float32), matriz)
    ordem = np.argsort(scores)[::-1][:top_k]
    return [(ids[i], float(scores[i])) for i in ordem]


def salvar_colecao(registros, caminho) -> None:
    """Persiste uma coleção de registros ``{id, texto, embedding}`` em ``.pkl``."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("wb") as f:
        pickle.dump(registros, f)


def carregar_colecao(caminho):
    """Carrega uma coleção previamente salva por ``salvar_colecao``."""
    with Path(caminho).open("rb") as f:
        return pickle.load(f)

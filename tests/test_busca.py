"""Testes mínimos do comportamento principal (ver política de testes no CLAUDE.md).

Foco no essencial: o ranqueamento por cosseno de ``buscar`` e o shape do embedding.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.busca import buscar

_TEM_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None


def test_buscar_ranqueia_por_cosseno_e_respeita_top_k():
    # Query aponta na direção de 'a'; 'b' é ortogonal, 'c' é oposto.
    query = np.array([1.0, 0.0], dtype=np.float32)
    base = [
        {"id": "a", "texto": "", "embedding": np.array([1.0, 0.0], dtype=np.float32)},
        {"id": "b", "texto": "", "embedding": np.array([0.0, 1.0], dtype=np.float32)},
        {"id": "c", "texto": "", "embedding": np.array([-1.0, 0.0], dtype=np.float32)},
    ]
    resultado = buscar(query, base, top_k=2)

    assert [id_ for id_, _ in resultado] == ["a", "b"]  # ordem por similaridade + top_k
    assert resultado[0][1] > resultado[1][1]            # scores em ordem decrescente


@pytest.mark.skipif(not _TEM_SENTENCE_TRANSFORMERS, reason="sentence-transformers não instalado")
def test_gerar_embedding_tem_shape_esperado():
    from src.embeddings import gerar_embedding

    emb = gerar_embedding("frase de teste")
    assert emb.ndim == 1
    assert emb.shape[0] == 384

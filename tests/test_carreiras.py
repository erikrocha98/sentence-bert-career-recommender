"""Teste mínimo do Hop 1 (ver política de testes no CLAUDE.md).

Verifica que ``recomendar_carreiras`` gera o embedding da consulta e delega a ``buscar``,
devolvendo o ranking esperado — sem carregar o modelo (embedding fake).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.carreiras as carreiras
from src.carreiras import recomendar_carreiras


def test_recomendar_carreiras_ranqueia_por_cosseno(monkeypatch):
    # Embedding fake: a consulta aponta na direção de 'a'; 'b' ortogonal, 'c' oposto.
    monkeypatch.setattr(
        carreiras, "gerar_embedding", lambda _texto: np.array([1.0, 0.0], dtype=np.float32)
    )
    base = [
        {"id": "a", "texto": "", "embedding": np.array([1.0, 0.0], dtype=np.float32)},
        {"id": "b", "texto": "", "embedding": np.array([0.0, 1.0], dtype=np.float32)},
        {"id": "c", "texto": "", "embedding": np.array([-1.0, 0.0], dtype=np.float32)},
    ]

    resultado = recomendar_carreiras("quero trabalhar com IA", base, top_k=2)

    assert [id_ for id_, _ in resultado] == ["a", "b"]  # ordena por similaridade + top_k
    assert resultado[0][1] > resultado[1][1]            # scores decrescentes

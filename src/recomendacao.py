"""Recomendação de disciplinas por competências (mecanismo do Hop 2 do pipeline).

Dado um texto de competências (ex.: "Python, Deep Learning, MLOps"), gera o embedding
da consulta e ranqueia as disciplinas mais similares por cosseno, reutilizando os
primitivos compartilhados ``gerar_embedding`` e ``buscar``.
"""
from __future__ import annotations

from src.busca import buscar
from src.embeddings import gerar_embedding


def recomendar_disciplinas(texto_competencias: str, base_emb, top_k: int = 5):
    """Ranqueia ``base_emb`` pela similaridade com o texto de competências.

    ``base_emb``: coleção de registros ``{id, texto, embedding}`` (ex.: ``disciplinas_emb.pkl``).
    Retorna ``[(id, score)]`` ordenado por score desc — mesmo contrato de ``buscar``.
    """
    emb = gerar_embedding(texto_competencias)
    return buscar(emb, base_emb, top_k)

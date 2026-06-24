"""Recomendação de carreiras a partir do texto livre do aluno (Hop 1 do pipeline).

Dado o texto em que o aluno descreve seus objetivos de carreira (ex.: "quero trabalhar
com IA generativa e NLP"), gera o embedding da consulta e ranqueia os perfis de carreira
mais similares por cosseno, reutilizando os primitivos compartilhados ``gerar_embedding``
e ``buscar``.
"""
from __future__ import annotations

from src.busca import buscar
from src.embeddings import gerar_embedding


def recomendar_carreiras(query_aluno: str, perfis_emb, top_k: int = 5):
    """Ranqueia ``perfis_emb`` pela similaridade com o texto livre do aluno.

    ``perfis_emb``: coleção de registros ``{id, texto, embedding}`` (ex.: ``perfis_emb.pkl``).
    Retorna ``[(id, score)]`` ordenado por score desc — mesmo contrato de ``buscar``.
    """
    emb = gerar_embedding(query_aluno)
    return buscar(emb, perfis_emb, top_k)

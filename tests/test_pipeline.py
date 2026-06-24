"""Teste de integração mínimo do pipeline ponta a ponta (ver CLAUDE.md).

Com dados sintéticos minúsculos e embeddings fake (sem carregar o modelo), verifica que
``recomendar_trilha`` amarra os três Hops e monta a estrutura de saída esperada:
carreira escolhida + disciplinas recomendadas + roadmap por semestre, com a justificativa.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.carreiras as carreiras
import src.recomendacao as recomendacao
from src.pipeline import recomendar_trilha

PERFIS = [
    {"id": "p1", "nome": "Carreira X", "descricao": "", "competencias": ["alg"],
     "tecnologias": ["py"], "area": "A"},
    {"id": "p2", "nome": "Carreira Y", "descricao": "", "competencias": ["b"],
     "tecnologias": ["q"], "area": "B"},
]
PERFIS_EMB = [
    {"id": "p1", "texto": "", "embedding": np.array([1.0, 0.0], dtype=np.float32)},
    {"id": "p2", "texto": "", "embedding": np.array([0.0, 1.0], dtype=np.float32)},
]
DISCIPLINAS = [
    {"codigo": "A", "nome": "Disc A", "ementa": "", "objetivos": "", "carga_horaria": 60, "prerequisitos": []},
    {"codigo": "B", "nome": "Disc B", "ementa": "", "objetivos": "", "carga_horaria": 60, "prerequisitos": ["A"]},
]
DISCIPLINAS_EMB = [
    {"id": "A", "texto": "", "embedding": np.array([1.0, 0.0], dtype=np.float32)},
    {"id": "B", "texto": "", "embedding": np.array([0.9, 0.1], dtype=np.float32)},
]


def test_recomendar_trilha_monta_estrutura_ponta_a_ponta(monkeypatch):
    # Hop 1 e Hop 2 apontam ambos para a direção [1,0] (fake, sem modelo).
    monkeypatch.setattr(carreiras, "gerar_embedding",
                        lambda _t: np.array([1.0, 0.0], dtype=np.float32))
    monkeypatch.setattr(recomendacao, "gerar_embedding",
                        lambda _t: np.array([1.0, 0.0], dtype=np.float32))

    res = recomendar_trilha(
        "quero a carreira X", PERFIS, PERFIS_EMB, DISCIPLINAS, DISCIPLINAS_EMB, top_k=2
    )

    # Hop 1 escolheu p1.
    assert res["carreira"] == "Carreira X"
    assert res["carreira_id"] == "p1"
    # Hop 2 recomendou as duas disciplinas.
    assert {cod for cod, _ in res["disciplinas"]} == {"A", "B"}
    # Hop 3 respeitou o pré-requisito (A antes de B).
    codigos_sem = {sem: [it["codigo"] for it in itens] for sem, itens in res["roadmap"].items()}
    assert codigos_sem == {1: ["A"], 2: ["B"]}
    # Explicabilidade preservada.
    assert "Carreira X" in res["justificativa"]

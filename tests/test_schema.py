"""Testes mínimos do schema de perfil (ver política de testes no CLAUDE.md).

Cobre o essencial da Fonte C: ``texto_para_embedding_perfil`` concatena descrição +
competências + tecnologias (contrato do CLAUDE.md) e ``validar_perfil`` rejeita campo ausente.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schema import texto_para_embedding_perfil, validar_perfil

PERFIL = {
    "id": "ml",
    "nome": "Engenheiro de ML",
    "descricao": "Treina modelos.",
    "competencias": ["deep learning", "redes neurais"],
    "tecnologias": ["Python", "PyTorch"],
    "area": "IA",
}


def test_texto_para_embedding_perfil_concatena_descricao_competencias_tecnologias():
    texto = texto_para_embedding_perfil(PERFIL)
    assert texto == "Treina modelos.. deep learning, redes neurais. Python, PyTorch"


def test_validar_perfil_rejeita_campo_ausente():
    incompleto = {k: v for k, v in PERFIL.items() if k != "competencias"}
    with pytest.raises(ValueError):
        validar_perfil(incompleto)

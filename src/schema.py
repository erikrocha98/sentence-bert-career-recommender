"""Esquema de dados de uma disciplina e utilitários de validação.

Contrato de dados (ver CLAUDE.md): cada disciplina é um dicionário com os campos
abaixo. A função ``texto_para_embedding`` define a granularidade usada na vetorização
(nome + ementa + objetivos), conforme a proposta do projeto.
"""
from __future__ import annotations

from typing import TypedDict


class Disciplina(TypedDict):
    codigo: str
    nome: str
    ementa: str
    objetivos: str
    carga_horaria: int
    prerequisitos: list[str]  # lista de códigos de outras disciplinas


CAMPOS_OBRIGATORIOS = (
    "codigo",
    "nome",
    "ementa",
    "objetivos",
    "carga_horaria",
    "prerequisitos",
)


def validar_disciplina(d: dict) -> None:
    """Valida o mínimo: presença dos campos e tipos básicos. Levanta ``ValueError``."""
    faltando = [c for c in CAMPOS_OBRIGATORIOS if c not in d]
    if faltando:
        raise ValueError(f"Disciplina {d.get('codigo', '?')}: campos ausentes: {faltando}")
    if not isinstance(d["prerequisitos"], list):
        raise ValueError(
            f"Disciplina {d['codigo']}: 'prerequisitos' deve ser uma lista de códigos"
        )
    if not isinstance(d["carga_horaria"], int):
        raise ValueError(f"Disciplina {d['codigo']}: 'carga_horaria' deve ser int")


def texto_para_embedding(d: Disciplina) -> str:
    """Concatena nome + ementa + objetivos em um único texto para vetorização.

    Campos vazios são ignorados, então disciplinas ainda sem ementa curada caem de
    volta para o nome (sinal semântico mínimo) — útil enquanto as ementas não foram
    coletadas.
    """
    partes = [d.get("nome", ""), d.get("ementa", ""), d.get("objetivos", "")]
    return ". ".join(p.strip() for p in partes if p and p.strip())

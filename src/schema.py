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


class Perfil(TypedDict):
    id: str
    nome: str  # nome do cargo/carreira
    descricao: str
    competencias: list[str]
    tecnologias: list[str]
    area: str


CAMPOS_OBRIGATORIOS_PERFIL = (
    "id",
    "nome",
    "descricao",
    "competencias",
    "tecnologias",
    "area",
)


def validar_perfil(p: dict) -> None:
    """Valida o mínimo: presença dos campos e tipos básicos. Levanta ``ValueError``."""
    faltando = [c for c in CAMPOS_OBRIGATORIOS_PERFIL if c not in p]
    if faltando:
        raise ValueError(f"Perfil {p.get('id', '?')}: campos ausentes: {faltando}")
    if not isinstance(p["competencias"], list):
        raise ValueError(f"Perfil {p['id']}: 'competencias' deve ser uma lista")
    if not isinstance(p["tecnologias"], list):
        raise ValueError(f"Perfil {p['id']}: 'tecnologias' deve ser uma lista")


def texto_para_embedding_perfil(p: Perfil) -> str:
    """Concatena descrição + competências + tecnologias em um texto para vetorização.

    Contrato do CLAUDE.md (Fonte C). As listas de competências e tecnologias são unidas
    com ``", "``; campos/itens vazios são ignorados, espelhando ``texto_para_embedding``.
    """
    competencias = ", ".join(c.strip() for c in p.get("competencias", []) if c and c.strip())
    tecnologias = ", ".join(t.strip() for t in p.get("tecnologias", []) if t and t.strip())
    partes = [p.get("descricao", ""), competencias, tecnologias]
    return ". ".join(p_.strip() for p_ in partes if p_ and p_.strip())

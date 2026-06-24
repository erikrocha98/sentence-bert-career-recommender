"""Integração ponta a ponta dos três Hops, preservando a explicabilidade.

Amarra o pipeline completo do sistema:
- **Hop 1** (``recomendar_carreiras``): texto livre do aluno → carreira mais aderente.
- **Hop 2** (``recomendar_disciplinas``): competências/tecnologias da carreira escolhida →
  disciplinas mais aderentes.
- **Hop 3** (``montar_roadmap``): ordenação topológica das disciplinas recomendadas em
  semestres, respeitando os pré-requisitos.

A rastreabilidade exigida pelo CLAUDE.md ("recomendamos *Aprendizado de Máquina* porque a
carreira *ML Engineer* exige *Deep Learning*") é mantida: o Hop 2 usa explicitamente as
competências/tecnologias do perfil selecionado como consulta, e o resultado inclui uma
``justificativa`` textual ligando carreira → competências → disciplinas.
"""
from __future__ import annotations

from src.carreiras import recomendar_carreiras
from src.recomendacao import recomendar_disciplinas
from src.roadmap import montar_roadmap


def _texto_competencias(perfil: dict) -> str:
    """Consulta do Hop 2: competências + tecnologias do perfil, unidas com ``", "``."""
    itens = list(perfil.get("competencias", [])) + list(perfil.get("tecnologias", []))
    return ", ".join(i.strip() for i in itens if i and i.strip())


def trilha_para_perfil(
    perfil: dict,
    score: float,
    disciplinas: list[dict],
    disciplinas_emb,
    top_k: int = 3,
):
    """Executa os Hops 2 e 3 para um perfil de carreira já escolhido.

    Separado de ``recomendar_trilha`` para que a interface possa deixar o aluno escolher
    entre as carreiras do top-k do Hop 1 e re-disparar os Hops 2–3 com a selecionada.
    ``score`` é a similaridade dessa carreira no Hop 1 (só para a justificativa).

    Retorna o mesmo dicionário de ``recomendar_trilha``.
    """
    nomes_disc = {d["codigo"]: d["nome"] for d in disciplinas}

    # Hop 2: competências/tecnologias da carreira escolhida → disciplinas.
    texto_competencias = _texto_competencias(perfil)
    disciplinas_rec = recomendar_disciplinas(texto_competencias, disciplinas_emb, top_k=top_k)

    # Hop 3: roadmap por semestre das disciplinas recomendadas.
    roadmap = montar_roadmap([cod for cod, _ in disciplinas_rec], disciplinas)

    # Justificativa textual (explicabilidade do pipeline).
    nomes_rec = ", ".join(nomes_disc.get(cod, cod) for cod, _ in disciplinas_rec)
    competencias_str = ", ".join(perfil.get("competencias", []))
    justificativa = (
        f"Recomendamos a carreira '{perfil['nome']}' por melhor casar com seus objetivos "
        f"(similaridade {score:.2f}). As disciplinas {nomes_rec} foram sugeridas porque essa "
        f"carreira exige competências como {competencias_str}."
    )

    return {
        "carreira": perfil["nome"],
        "carreira_id": perfil["id"],
        "score": score,
        "competencias": texto_competencias,
        "disciplinas": disciplinas_rec,
        "roadmap": roadmap,
        "justificativa": justificativa,
    }


def recomendar_trilha(
    query_aluno: str,
    perfis: list[dict],
    perfis_emb,
    disciplinas: list[dict],
    disciplinas_emb,
    top_k: int = 3,
):
    """Executa os três Hops e devolve a trilha recomendada com a justificativa.

    ``perfis`` / ``disciplinas``: catálogos completos (para resolver ids → nomes/campos).
    ``perfis_emb`` / ``disciplinas_emb``: coleções ``{id, texto, embedding}`` pré-computadas.

    Retorna ``{carreira, score, competencias, disciplinas, roadmap, justificativa}``, onde
    ``disciplinas`` é ``[(codigo, score)]`` e ``roadmap`` é a saída de ``montar_roadmap``.
    Levanta ``ValueError`` se nenhuma carreira for encontrada.
    """
    perfis_por_id = {p["id"]: p for p in perfis}

    # Hop 1: texto livre do aluno → carreira.
    carreiras = recomendar_carreiras(query_aluno, perfis_emb, top_k=1)
    if not carreiras:
        raise ValueError("Nenhuma carreira encontrada (base de perfis vazia?).")
    carreira_id, score = carreiras[0]
    perfil = perfis_por_id[carreira_id]

    # Hops 2–3 sobre a carreira escolhida.
    return trilha_para_perfil(perfil, score, disciplinas, disciplinas_emb, top_k=top_k)

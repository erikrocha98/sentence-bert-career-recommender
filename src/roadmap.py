"""Montagem do roadmap de estudos por semestre (Hop 3 do pipeline).

A partir das disciplinas recomendadas (Hop 2), ordena topologicamente respeitando os
pré-requisitos e distribui por semestre: uma disciplina só entra num semestre depois que
TODOS os seus pré-requisitos já entraram em semestres anteriores.

Usa o algoritmo de Kahn em camadas (sem dependências externas). O semestre de cada
disciplina é ``1 + max(semestre dos seus pré-requisitos)``.
"""
from __future__ import annotations


def montar_roadmap(
    recomendadas, disciplinas, incluir_prerequisitos: bool = True
) -> dict[int, list[dict]]:
    """Distribui as disciplinas recomendadas por semestre respeitando pré-requisitos.

    ``recomendadas``: iterável de códigos (ex.: ids de ``recomendar_disciplinas``).
    ``disciplinas``: catálogo completo, para consultar ``prerequisitos``.
    ``incluir_prerequisitos``: se ``True`` (padrão), puxa os pré-requisitos faltantes para
    a trilha (fecho transitivo), marcados com ``origem="prerequisito"``; se ``False``,
    mantém só as recomendadas e ignora arestas para pré-requisitos ausentes.

    Retorna ``{semestre: [{"codigo", "origem"}, ...]}`` ordenado por semestre, com os
    códigos ordenados dentro de cada semestre.
    """
    prereqs = {d["codigo"]: list(d["prerequisitos"]) for d in disciplinas}
    recomendadas = list(dict.fromkeys(recomendadas))  # remove duplicatas, preserva ordem

    desconhecidas = [c for c in recomendadas if c not in prereqs]
    if desconhecidas:
        print(f"Aviso: códigos recomendados ausentes do catálogo, ignorados: {desconhecidas}")
    recomendadas = [c for c in recomendadas if c in prereqs]

    # Conjunto de nós a agendar e a origem de cada um.
    origem: dict[str, str] = {c: "recomendada" for c in recomendadas}
    if incluir_prerequisitos:
        pilha = list(recomendadas)
        while pilha:
            atual = pilha.pop()
            for p in prereqs.get(atual, []):
                if p not in prereqs:
                    continue  # pré-requisito externo ao catálogo: ignora
                if p not in origem:
                    origem[p] = "prerequisito"
                    pilha.append(p)

    nos = set(origem)
    # Arestas restritas ao conjunto: pré-requisitos fora de ``nos`` são desconsiderados.
    deps = {c: [p for p in prereqs[c] if p in nos] for c in nos}

    # Kahn em camadas: semestre = 1 + max(semestre dos pré-requisitos).
    semestre: dict[str, int] = {}
    while len(semestre) < len(nos):
        prontos = [
            c for c in nos
            if c not in semestre and all(p in semestre for p in deps[c])
        ]
        if not prontos:
            restantes = sorted(nos - set(semestre))
            raise ValueError(f"Ciclo de pré-requisitos detectado entre: {restantes}")
        for c in prontos:
            semestre[c] = 1 + max((semestre[p] for p in deps[c]), default=0)

    roadmap: dict[int, list[dict]] = {}
    for codigo in sorted(nos):
        roadmap.setdefault(semestre[codigo], []).append(
            {"codigo": codigo, "origem": origem[codigo]}
        )
    return dict(sorted(roadmap.items()))

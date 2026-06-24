"""Teste mínimo do roadmap (ver política de testes no CLAUDE.md).

Cobre o caminho principal: a ordenação topológica respeita os pré-requisitos (cada um
cai num semestre anterior ao da disciplina) e os dois modos de tratar pré-requisito
ausente da lista de recomendadas.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.roadmap import montar_roadmap

# Catálogo sintético: cadeia linear C <- B <- A.
DISCIPLINAS = [
    {"codigo": "A", "nome": "A", "ementa": "", "objetivos": "", "carga_horaria": 60, "prerequisitos": []},
    {"codigo": "B", "nome": "B", "ementa": "", "objetivos": "", "carga_horaria": 60, "prerequisitos": ["A"]},
    {"codigo": "C", "nome": "C", "ementa": "", "objetivos": "", "carga_horaria": 60, "prerequisitos": ["B"]},
]


def _codigos_por_semestre(roadmap):
    return {sem: [it["codigo"] for it in itens] for sem, itens in roadmap.items()}


def test_cadeia_linear_distribui_em_semestres_crescentes():
    roadmap = montar_roadmap(["A", "B", "C"], DISCIPLINAS)
    assert _codigos_por_semestre(roadmap) == {1: ["A"], 2: ["B"], 3: ["C"]}


def test_inclui_prerequisitos_ausentes_como_dependencia():
    roadmap = montar_roadmap(["C"], DISCIPLINAS, incluir_prerequisitos=True)
    origem = {it["codigo"]: it["origem"] for itens in roadmap.values() for it in itens}
    assert origem == {"A": "prerequisito", "B": "prerequisito", "C": "recomendada"}
    assert _codigos_por_semestre(roadmap) == {1: ["A"], 2: ["B"], 3: ["C"]}


def test_ignora_prerequisitos_fora_da_lista():
    roadmap = montar_roadmap(["C"], DISCIPLINAS, incluir_prerequisitos=False)
    assert _codigos_por_semestre(roadmap) == {1: ["C"]}

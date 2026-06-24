"""Interface de demonstração (Gradio) do pipeline de recomendação two-hop.

O aluno descreve seus objetivos de carreira em texto livre; a interface mostra:
- **Hop 1:** carreiras sugeridas (com a possibilidade de escolher entre as top-k);
- **Hop 2:** disciplinas recomendadas para a carreira selecionada;
- **Hop 3:** roadmap por semestre respeitando os pré-requisitos;
- a **justificativa** de explicabilidade ("recomendamos X porque a carreira Y exige Z").

Os artefatos (modelo, perfis e disciplinas + seus embeddings) são carregados UMA vez no
startup; a cada submit só o embedding da query do aluno é calculado (barato, roda em CPU).

Uso local: ``python app.py`` → http://127.0.0.1:7860
No Colab: ``demo.launch(share=True)`` gera link público temporário.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.busca import carregar_colecao
from src.carreiras import recomendar_carreiras
from src.embeddings import carregar_modelo
from src.pipeline import trilha_para_perfil

RAIZ = Path(__file__).resolve().parent

# ---- Carga única dos artefatos no startup ------------------------------------------------
PERFIS = json.loads((RAIZ / "data" / "perfis.json").read_text(encoding="utf-8"))
DISCIPLINAS = json.loads((RAIZ / "data" / "disciplinas.json").read_text(encoding="utf-8"))
PERFIS_EMB = carregar_colecao(RAIZ / "data" / "perfis_emb.pkl")
DISCIPLINAS_EMB = carregar_colecao(RAIZ / "data" / "disciplinas_emb.pkl")

PERFIS_POR_ID = {p["id"]: p for p in PERFIS}
NOMES_DISC = {d["codigo"]: d["nome"] for d in DISCIPLINAS}

carregar_modelo()  # aquece o modelo para o 1º submit não pagar o carregamento

EXEMPLOS = [
    "Quero trabalhar com Machine Learning e IA generativa",
    "Tenho interesse em engenharia de dados e pipelines de ETL",
    "Quero atuar com visão computacional e robótica",
    "Gosto de estatística, regressão e análise de dados",
]


def buscar_carreiras(query: str, top_k: int):
    """Hop 1: ranqueia carreiras e prepara o seletor das top-k."""
    if not query or not query.strip():
        return (
            gr.update(choices=[], value=None),
            "_Digite seus objetivos de carreira e clique em **Recomendar**._",
            {},
        )
    resultados = recomendar_carreiras(query, PERFIS_EMB, top_k=int(top_k))
    if not resultados:
        return gr.update(choices=[], value=None), "_Nenhuma carreira encontrada._", {}

    choices = [(f"{PERFIS_POR_ID[cid]['nome']} (similaridade {s:.2f})", cid) for cid, s in resultados]
    scores = {cid: s for cid, s in resultados}

    linhas = ["| # | Carreira | Área | Similaridade |", "|---|---|---|---|"]
    for i, (cid, s) in enumerate(resultados, 1):
        p = PERFIS_POR_ID[cid]
        linhas.append(f"| {i} | {p['nome']} | {p['area']} | {s:.3f} |")
    md = "### Hop 1 — Carreiras sugeridas\n" + "\n".join(linhas)

    return gr.update(choices=choices, value=choices[0][1]), md, scores


def montar_trilha(carreira_id: str, scores: dict, top_k: int):
    """Hops 2–3: disciplinas + roadmap + justificativa para a carreira escolhida."""
    if not carreira_id:
        return "", "", ""
    perfil = PERFIS_POR_ID[carreira_id]
    score = float(scores.get(carreira_id, 0.0)) if scores else 0.0
    res = trilha_para_perfil(perfil, score, DISCIPLINAS, DISCIPLINAS_EMB, top_k=int(top_k))

    # Hop 2 — tabela de disciplinas.
    linhas = ["| Similaridade | Código | Disciplina |", "|---|---|---|"]
    for cod, s in res["disciplinas"]:
        linhas.append(f"| {s:.3f} | {cod} | {NOMES_DISC.get(cod, '?')} |")
    disc_md = "### Hop 2 — Disciplinas recomendadas\n" + "\n".join(linhas)

    # Hop 3 — roadmap por semestre.
    if res["roadmap"]:
        partes = ["### Hop 3 — Roadmap por semestre"]
        for sem, itens in res["roadmap"].items():
            partes.append(f"**{sem}º semestre**")
            for it in itens:
                marca = " _[pré-requisito]_" if it["origem"] == "prerequisito" else ""
                partes.append(f"- {it['codigo']} — {NOMES_DISC.get(it['codigo'], '?')}{marca}")
        roadmap_md = "\n".join(partes)
    else:
        roadmap_md = "### Hop 3 — Roadmap por semestre\n_Sem disciplinas para montar o roadmap._"

    justificativa_md = f"### Justificativa\n{res['justificativa']}"
    return disc_md, roadmap_md, justificativa_md


with gr.Blocks(title="Recomendação de Disciplinas e Trilhas de Carreira") as demo:
    gr.Markdown(
        "# 🎓 Recomendação de Disciplinas e Trilhas de Carreira\n"
        "Descreva seus objetivos de carreira em texto livre. O sistema sugere carreiras "
        "(Hop 1), as disciplinas da UFMG mais aderentes (Hop 2) e um roadmap por semestre "
        "respeitando os pré-requisitos (Hop 3)."
    )

    scores_state = gr.State({})

    with gr.Row():
        query = gr.Textbox(
            label="Seus objetivos de carreira",
            placeholder="Ex.: quero trabalhar com Machine Learning e IA generativa",
            lines=2,
            scale=4,
        )
        top_k = gr.Slider(1, 10, value=5, step=1, label="Top-k", scale=1)
    botao = gr.Button("Recomendar", variant="primary")
    gr.Examples(examples=EXEMPLOS, inputs=query)

    carreiras_md = gr.Markdown()
    carreira_sel = gr.Radio(label="Escolha uma carreira para detalhar a trilha", choices=[])

    disciplinas_md = gr.Markdown()
    roadmap_md = gr.Markdown()
    justificativa_md = gr.Markdown()

    saidas_trilha = [disciplinas_md, roadmap_md, justificativa_md]

    # Botão: Hop 1 e, em seguida, Hops 2–3 para a melhor carreira (radio já selecionado).
    botao.click(
        buscar_carreiras, inputs=[query, top_k], outputs=[carreira_sel, carreiras_md, scores_state]
    ).then(
        montar_trilha, inputs=[carreira_sel, scores_state, top_k], outputs=saidas_trilha
    )
    query.submit(
        buscar_carreiras, inputs=[query, top_k], outputs=[carreira_sel, carreiras_md, scores_state]
    ).then(
        montar_trilha, inputs=[carreira_sel, scores_state, top_k], outputs=saidas_trilha
    )
    # Trocar a carreira no seletor re-dispara apenas os Hops 2–3.
    carreira_sel.change(
        montar_trilha, inputs=[carreira_sel, scores_state, top_k], outputs=saidas_trilha
    )


if __name__ == "__main__":
    demo.launch()

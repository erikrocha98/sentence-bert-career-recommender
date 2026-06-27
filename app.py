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

from html import escape
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
    "Quero trabalhar com processamento de linguagem natural, chatbots e busca semântica",
    "Tenho interesse em engenharia de dados, ETL e data warehouses",
    "Quero desenvolver APIs, backend escalável e sistemas distribuídos",
    "Quero otimizar rotas, logística e alocação de recursos",
]

CSS = """
:root {
  --paper: #f7f3ea;
  --ink: #182026;
  --muted: #61707a;
  --line: rgba(24, 32, 38, 0.16);
  --blue: #1e5fd7;
  --blue-dark: #123d8a;
  --green: #17765b;
  --amber: #b86a12;
  --panel: rgba(255, 255, 255, 0.84);
  --shadow: 0 18px 55px rgba(24, 32, 38, 0.13);
  --page-bg: linear-gradient(135deg, #faf7ef 0%, #eef4f8 48%, #edf5ed 100%);
}

html,
body,
#root {
  min-height: 100%;
  margin: 0;
  background: #eef4f8 !important;
}

body::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
  background:
    linear-gradient(90deg, rgba(30, 95, 215, 0.055) 1px, transparent 1px),
    linear-gradient(0deg, rgba(23, 118, 91, 0.045) 1px, transparent 1px),
    var(--page-bg);
  background-size: 56px 56px, 56px 56px, auto;
}

.gradio-container,
.gradio-container .main,
.app,
.contain {
  max-width: none !important;
}

.gradio-container {
  width: 100% !important;
  min-height: 100vh !important;
  box-sizing: border-box;
  padding: clamp(18px, 2.8vw, 44px) clamp(18px, 3.6vw, 72px) 64px !important;
  color: var(--ink);
  background: transparent !important;
}

#hero-shell {
  width: 100%;
  box-sizing: border-box;
  margin: 0 0 22px;
}

.gradio-html:has(#hero-shell),
.block:has(#hero-shell) {
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

#hero {
  width: 100%;
  box-sizing: border-box;
  margin: 0;
  padding: clamp(24px, 3vw, 44px);
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.70);
  box-shadow: var(--shadow);
  position: relative;
  overflow: hidden;
}

#hero::after {
  content: "";
  position: absolute;
  right: -80px;
  top: -80px;
  width: 240px;
  height: 240px;
  border: 1px solid rgba(30, 95, 215, 0.24);
  transform: rotate(18deg);
}

#hero .kicker {
  color: var(--blue-dark);
  font: 700 12px/1.2 "Avenir Next", "Helvetica Neue", sans-serif;
  letter-spacing: 0;
  text-transform: uppercase;
}

#hero h1 {
  margin: 8px 0;
  font-family: Georgia, "Times New Roman", serif;
  max-width: 1180px;
  color: var(--ink);
  font-size: clamp(34px, 4.8vw, 78px);
  line-height: 0.98;
  letter-spacing: 0;
}

#hero p {
  max-width: 980px;
  margin: 0;
  color: var(--muted);
  font-size: 16px;
}

.stat-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 18px;
}

.stat-chip {
  color: var(--ink);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.68);
  font-size: 13px;
}

.panel {
  color: var(--ink);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
  background: var(--panel);
  box-shadow: 0 10px 32px rgba(24, 32, 38, 0.08);
  animation: rise-in 420ms ease both;
}

.panel h3 {
  margin: 0 0 12px;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 22px;
  letter-spacing: 0;
}

.result-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.result-table th {
  color: var(--muted);
  font-weight: 700;
  text-align: left;
  border-bottom: 1px solid var(--line);
  padding: 8px 6px;
}

.result-table td {
  border-bottom: 1px solid rgba(24, 32, 38, 0.08);
  padding: 9px 6px;
}

.score {
  display: inline-block;
  min-width: 48px;
  border-radius: 999px;
  padding: 4px 8px;
  color: white;
  background: var(--blue);
  font-weight: 700;
  text-align: center;
}

.roadmap-grid {
  display: grid;
  gap: 10px;
}

.semester {
  border-left: 4px solid var(--green);
  padding: 10px 12px;
  background: rgba(23, 118, 91, 0.08);
}

.semester strong {
  display: block;
  margin-bottom: 6px;
}

.course-tag {
  display: inline-block;
  margin: 3px 5px 3px 0;
  padding: 5px 8px;
  border-radius: 999px;
  background: rgba(30, 95, 215, 0.10);
}

.course-tag small {
  color: var(--amber);
  font-weight: 700;
}

.empty-state {
  color: var(--muted);
  padding: 10px 0;
}

.primary button {
  transition: transform 160ms ease, box-shadow 160ms ease !important;
}

.primary button:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 24px rgba(30, 95, 215, 0.22);
}

html.dark,
body.dark,
.dark,
[data-theme="dark"] {
  --ink: #f4f7fb;
  --muted: #c8d2dc;
  --line: rgba(226, 232, 240, 0.22);
  --blue: #64a3ff;
  --blue-dark: #9ec5ff;
  --green: #65d6b0;
  --amber: #ffb86b;
  --panel: rgba(25, 29, 36, 0.94);
  --shadow: 0 18px 55px rgba(0, 0, 0, 0.34);
}

html.dark,
html.dark body,
body.dark,
.dark .gradio-container,
[data-theme="dark"] .gradio-container {
  background: #0f1115 !important;
}

html.dark body::before,
body.dark::before,
.dark body::before,
[data-theme="dark"] body::before {
  background:
    linear-gradient(90deg, rgba(100, 163, 255, 0.05) 1px, transparent 1px),
    linear-gradient(0deg, rgba(101, 214, 176, 0.04) 1px, transparent 1px),
    linear-gradient(135deg, #101318 0%, #151a20 48%, #101816 100%) !important;
  background-size: 56px 56px, 56px 56px, auto !important;
}

html.dark #hero,
body.dark #hero,
.dark #hero,
[data-theme="dark"] #hero {
  background: rgba(22, 26, 33, 0.96) !important;
  border-color: rgba(226, 232, 240, 0.22) !important;
  color: #f8fafc !important;
}

html.dark #hero h1,
body.dark #hero h1,
.dark #hero h1,
[data-theme="dark"] #hero h1 {
  color: #f8fafc !important;
}

html.dark #hero p,
body.dark #hero p,
.dark #hero p,
[data-theme="dark"] #hero p {
  color: #d7e0ea !important;
}

html.dark #hero .kicker,
body.dark #hero .kicker,
.dark #hero .kicker,
[data-theme="dark"] #hero .kicker {
  color: #9ec5ff !important;
}

html.dark .stat-chip,
body.dark .stat-chip,
.dark .stat-chip,
[data-theme="dark"] .stat-chip {
  color: #eef4fb !important;
  background: rgba(255, 255, 255, 0.08) !important;
  border-color: rgba(226, 232, 240, 0.24) !important;
}

html.dark .panel,
body.dark .panel,
.dark .panel,
[data-theme="dark"] .panel {
  background: rgba(25, 29, 36, 0.94) !important;
  border-color: rgba(226, 232, 240, 0.18) !important;
  color: #f4f7fb !important;
}

html.dark .result-table th,
body.dark .result-table th,
.dark .result-table th,
[data-theme="dark"] .result-table th {
  color: #c8d2dc !important;
  border-bottom-color: rgba(226, 232, 240, 0.22) !important;
}

html.dark .result-table td,
body.dark .result-table td,
.dark .result-table td,
[data-theme="dark"] .result-table td {
  color: #eef4fb !important;
  border-bottom-color: rgba(226, 232, 240, 0.10) !important;
}

html.dark .gradio-container label,
body.dark .gradio-container label,
.dark .gradio-container label,
[data-theme="dark"] .gradio-container label,
html.dark .gradio-container button,
body.dark .gradio-container button,
.dark .gradio-container button,
[data-theme="dark"] .gradio-container button {
  color: #eef4fb !important;
}

html.dark .gradio-container textarea,
body.dark .gradio-container textarea,
.dark .gradio-container textarea,
[data-theme="dark"] .gradio-container textarea,
html.dark .gradio-container input,
body.dark .gradio-container input,
.dark .gradio-container input,
[data-theme="dark"] .gradio-container input {
  color: #f8fafc !important;
  background: #171b22 !important;
  border-color: rgba(226, 232, 240, 0.18) !important;
}

html.dark .gradio-container textarea::placeholder,
body.dark .gradio-container textarea::placeholder,
.dark .gradio-container textarea::placeholder,
[data-theme="dark"] .gradio-container textarea::placeholder {
  color: #8f9baa !important;
}

html.dark .primary button,
body.dark .primary button,
.dark .primary button,
[data-theme="dark"] .primary button {
  color: #ffffff !important;
  background: #f45a0b !important;
  border-color: #f45a0b !important;
}

@media (max-width: 780px) {
  .gradio-container {
    padding: 14px 12px 36px !important;
  }

  #hero {
    padding: 22px 18px;
  }

  #hero h1 {
    font-size: clamp(30px, 12vw, 46px);
  }
}

@keyframes rise-in {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
"""


def _panel(titulo: str, corpo: str) -> str:
    return f'<section class="panel"><h3>{escape(titulo)}</h3>{corpo}</section>'


def buscar_carreiras(query: str, top_k: int):
    """Hop 1: ranqueia carreiras e prepara o seletor das top-k."""
    if not query or not query.strip():
        return (
            gr.update(choices=[], value=None),
            _panel(
                "Carreiras sugeridas",
                '<div class="empty-state">Digite seus objetivos e execute a recomendação.</div>',
            ),
            {},
        )
    resultados = recomendar_carreiras(query, PERFIS_EMB, top_k=int(top_k))
    if not resultados:
        return (
            gr.update(choices=[], value=None),
            _panel("Carreiras sugeridas", '<div class="empty-state">Nenhuma carreira encontrada.</div>'),
            {},
        )

    choices = [(f"{PERFIS_POR_ID[cid]['nome']} (similaridade {s:.2f})", cid) for cid, s in resultados]
    scores = {cid: s for cid, s in resultados}

    linhas = [
        '<table class="result-table">',
        "<thead><tr><th>#</th><th>Carreira</th><th>Área</th><th>Similaridade</th></tr></thead>",
        "<tbody>",
    ]
    for i, (cid, s) in enumerate(resultados, 1):
        p = PERFIS_POR_ID[cid]
        linhas.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{escape(p['nome'])}</td>"
            f"<td>{escape(p['area'])}</td>"
            f'<td><span class="score">{s:.2f}</span></td>'
            "</tr>"
        )
    linhas.extend(["</tbody>", "</table>"])
    html = _panel("Hop 1 - carreiras sugeridas", "\n".join(linhas))

    return gr.update(choices=choices, value=choices[0][1]), html, scores


def montar_trilha(carreira_id: str, scores: dict, top_k: int):
    """Hops 2–3: disciplinas + roadmap + justificativa para a carreira escolhida."""
    if not carreira_id:
        return "", "", ""
    perfil = PERFIS_POR_ID[carreira_id]
    score = float(scores.get(carreira_id, 0.0)) if scores else 0.0
    res = trilha_para_perfil(perfil, score, DISCIPLINAS, DISCIPLINAS_EMB, top_k=int(top_k))

    # Hop 2 — tabela de disciplinas.
    linhas = [
        '<table class="result-table">',
        "<thead><tr><th>Score</th><th>Código</th><th>Disciplina</th></tr></thead>",
        "<tbody>",
    ]
    for cod, s in res["disciplinas"]:
        linhas.append(
            "<tr>"
            f'<td><span class="score">{s:.2f}</span></td>'
            f"<td>{escape(cod)}</td>"
            f"<td>{escape(NOMES_DISC.get(cod, '?'))}</td>"
            "</tr>"
        )
    linhas.extend(["</tbody>", "</table>"])
    disc_html = _panel("Hop 2 - disciplinas recomendadas", "\n".join(linhas))

    # Hop 3 — roadmap por semestre.
    if res["roadmap"]:
        partes = ['<div class="roadmap-grid">']
        for sem, itens in res["roadmap"].items():
            partes.append(f'<div class="semester"><strong>{sem}º semestre</strong>')
            for it in itens:
                marca = " <small>pré-requisito</small>" if it["origem"] == "prerequisito" else ""
                partes.append(
                    f'<span class="course-tag">{escape(it["codigo"])} - '
                    f'{escape(NOMES_DISC.get(it["codigo"], "?"))}{marca}</span>'
                )
            partes.append("</div>")
        partes.append("</div>")
        roadmap_html = _panel("Hop 3 - roadmap por semestre", "\n".join(partes))
    else:
        roadmap_html = _panel(
            "Hop 3 - roadmap por semestre",
            '<div class="empty-state">Sem disciplinas para montar o roadmap.</div>',
        )

    justificativa_html = _panel("Justificativa", f"<p>{escape(res['justificativa'])}</p>")
    return disc_html, roadmap_html, justificativa_html


with gr.Blocks(
    title="Recomendação de Disciplinas e Trilhas de Carreira",
    fill_width=True,
) as demo:
    gr.HTML(
        f"""
        <div id="hero-shell">
          <header id="hero">
            <div class="kicker">UFMG | Fundamentos de IA | TP3</div>
            <h1>Recomendação de trilhas acadêmicas por busca semântica</h1>
            <p>Digite objetivos profissionais em português. O sistema ranqueia carreiras,
            traduz competências em disciplinas e monta um percurso que respeita pré-requisitos.</p>
            <div class="stat-row">
              <span class="stat-chip">Sentence-BERT multilíngue</span>
              <span class="stat-chip">{len(DISCIPLINAS)} disciplinas curadas</span>
              <span class="stat-chip">{len(PERFIS)} perfis de carreira</span>
              <span class="stat-chip">Roadmap por pré-requisitos</span>
            </div>
          </header>
        </div>
        """
    )

    scores_state = gr.State({})

    with gr.Row():
        query = gr.Textbox(
            label="Seus objetivos de carreira",
            placeholder="Ex.: quero trabalhar com processamento de linguagem natural, chatbots e busca semântica",
            lines=2,
            scale=4,
        )
        top_k = gr.Slider(1, 10, value=5, step=1, label="Top-k", scale=1)
    botao = gr.Button("Recomendar trilha", variant="primary", elem_classes=["primary"])
    gr.Examples(examples=EXEMPLOS, inputs=query)

    carreiras_md = gr.HTML()
    carreira_sel = gr.Radio(label="Escolha uma carreira para detalhar a trilha", choices=[])

    disciplinas_md = gr.HTML()
    roadmap_md = gr.HTML()
    justificativa_md = gr.HTML()

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
    demo.launch(css=CSS)

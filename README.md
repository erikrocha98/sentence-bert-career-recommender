# Sistema de Recomendação de Disciplinas e Trilhas de Carreira

Projeto da disciplina **Fundamentos de IA (TP3 — Transfer Learning)**, UFMG.

O aluno descreve seus objetivos de carreira em **texto livre em português** e o sistema
retorna as carreiras mais aderentes, as disciplinas da UFMG recomendadas para cada uma e um
**roadmap de estudos por semestre** que respeita os pré-requisitos. A tarefa de NLP é
**busca semântica** com embeddings de sentenças (Sentence-BERT) + similaridade de cosseno.

## Arquitetura: pipeline two-hop

O desenho em dois saltos existe para garantir **explicabilidade** ("recomendamos *Aprendizado
de Máquina* porque a carreira *Engenheiro de ML* exige *Deep Learning*"):

1. **Hop 1 — aluno → carreira:** embute a consulta do aluno e ranqueia os perfis de carreira
   por cosseno.
2. **Hop 2 — carreira → disciplinas:** usa as competências/tecnologias da carreira escolhida
   como nova consulta semântica contra os embeddings das ementas das disciplinas.
3. **Roadmap (Hop 3):** ordenação topológica sobre o DAG de pré-requisitos, distribuindo as
   disciplinas recomendadas pelos semestres.

Os embeddings são **pré-computados offline uma vez** (`data/*_emb.pkl`) e recarregados; só a
consulta do aluno é vetorizada a cada busca.

- **Modelo:** `paraphrase-multilingual-MiniLM-L12-v2` (Sentence-BERT multilíngue, 384 dims), em CPU.

## Organização

```
.
├── src/                 # núcleo do sistema
│   ├── embeddings.py    # gerar_embedding(texto) -> np.array  (função compartilhada)
│   ├── busca.py         # buscar(query, base, top_k) + salvar/carregar coleções (.pkl)
│   ├── schema.py        # schemas e validação de Disciplina e Perfil + textos p/ embedding
│   ├── carreiras.py     # Hop 1 — recomendar_carreiras(query_aluno, ...)
│   ├── recomendacao.py  # Hop 2 — recomendar_disciplinas(competencias, ...)
│   ├── roadmap.py       # Hop 3 — montar_roadmap(recomendadas, disciplinas)
│   └── pipeline.py      # integração ponta a ponta — recomendar_trilha(...)
├── data/
│   ├── disciplinas.json # catálogo de disciplinas da UFMG (nome, ementa, pré-requisitos…)
│   ├── perfis.json      # perfis de carreira curados à mão (Fonte C)
│   ├── README.md        # proveniência e limitações dos dados
│   └── *_emb.pkl        # embeddings pré-computados (registros {id, texto, embedding})
├── scripts/
│   ├── gerar_embeddings.py         # gera data/disciplinas_emb.pkl
│   ├── gerar_embeddings_perfis.py  # gera data/perfis_emb.pkl
│   ├── avaliar_modelos.py          # compara Sentence-BERT vs TF-IDF e gera figuras
│   ├── smoke_test.py               # checagem rápida do ambiente (modelo + busca)
│   └── validar_*.py                # validação MANUAL (impressão p/ conferência humana)
├── tests/               # testes mínimos (cosseno, roadmap, schema, hops, integração)
├── results/             # métricas, CSVs e figuras da avaliação quantitativa
├── TP3_Recomendacao_Carreiras_UFMG.ipynb # notebook/Colab do trabalho
├── app.py               # interface de demonstração em Gradio
└── requirements.txt
```

### Contratos de integração

- **Embedding:** `gerar_embedding(texto: str) -> np.array` — função única usada em todo lugar.
- **Registro persistido:** cada coleção é uma lista de `{id, texto, embedding}` em `.pkl`.
- **Busca:** `buscar(query_emb, base_emb, top_k) -> [(id, score)]` — cosseno genérico, reusado
  pelos dois hops.

## Como executar

### 1. Ambiente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Gerar os embeddings (offline, uma vez)

> Os `.pkl` já estão versionados em `data/`. Rode apenas se editar `disciplinas.json`/`perfis.json`.
> Na primeira execução o modelo (~120 MB) é baixado.

```bash
python scripts/gerar_embeddings.py          # data/disciplinas_emb.pkl
python scripts/gerar_embeddings_perfis.py   # data/perfis_emb.pkl
```

### 3. Interface de demonstração (Gradio)

```bash
python app.py
```

Abre em `http://127.0.0.1:7860`. Digite seus objetivos de carreira, escolha uma das carreiras
sugeridas e veja as disciplinas + roadmap + justificativa. No Google Colab, use
`demo.launch(share=True)` para gerar um link público temporário.

### 4. Uso programático

```python
import json
from src.busca import carregar_colecao
from src.pipeline import recomendar_trilha

perfis = json.load(open("data/perfis.json", encoding="utf-8"))
disciplinas = json.load(open("data/disciplinas.json", encoding="utf-8"))
res = recomendar_trilha(
    "quero trabalhar com IA generativa e NLP",
    perfis, carregar_colecao("data/perfis_emb.pkl"),
    disciplinas, carregar_colecao("data/disciplinas_emb.pkl"),
    top_k=3,
)
print(res["carreira"], res["disciplinas"], res["justificativa"])
```

### Testes e validação

```bash
pytest tests/ -q                      # testes automatizados
python scripts/smoke_test.py          # sanidade do ambiente (modelo + busca)
python scripts/validar_carreiras.py   # conferência humana do Hop 1 + pipeline
python scripts/avaliar_modelos.py      # avaliação quantitativa + figuras do relatório
```

### Resultados medidos

A avaliação quantitativa usa `data/consultas_avaliacao.json` com 20 consultas de carreira
e 13 consultas de competências para disciplinas. O baseline TF-IDF é treinado apenas sobre os
textos dos catálogos, sem usar os rótulos esperados.

Última execução local:

| Tarefa | Método | Métrica principal |
|---|---:|---:|
| Hop 1 — carreira | TF-IDF | top-1 = 0.90, hit@3 = 1.00 |
| Hop 1 — carreira | Sentence-BERT | top-1 = 0.95, hit@3 = 1.00 |
| Hop 2 — disciplinas | TF-IDF | precision@3 = 0.77 |
| Hop 2 — disciplinas | Sentence-BERT | precision@3 = 0.64 |

Esse resultado é analisado no relatório: Sentence-BERT ficou melhor no mapeamento
texto-livre → carreira, enquanto TF-IDF ainda vence em disciplinas porque as ementas curadas
são curtas e muito lexicalizadas. A avaliação expõe casos em que a curadoria dos textos e um
reranking híbrido ainda melhorariam o sistema.

### Proveniência dos dados

A Fonte A é um catálogo curado para protótipo acadêmico, baseado na estrutura curricular
presente em `V2-1_4_Apendice_2_Estrutura_Curricular_FEV2024.pdf`. A versão atual mantém 30
disciplinas curadas. Os detalhes estão em `data/README.md`.

Os arquivos `data/*_emb.pkl` são artefatos derivados dos JSONs, não fontes independentes.

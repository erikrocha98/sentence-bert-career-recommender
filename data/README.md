# Proveniência dos dados

Este projeto usa bases curadas para um protótipo acadêmico de busca semântica.

## `disciplinas.json`

Catálogo curado de 30 disciplinas/tópicos curriculares relacionados à computação, IA,
dados, otimização, robótica e software.

- A base foi construída a partir da estrutura curricular disponível no repositório
  (`V2-1_4_Apendice_2_Estrutura_Curricular_FEV2024.pdf`) e curadoria manual das ementas
  e objetivos. Para os registros extraídos do PDF, nomes, códigos, cargas horárias e
  pré-requisitos vieram da estrutura curricular quando identificáveis.
- Os textos de ementa e objetivos são curados para fins experimentais; a base não deve
  ser apresentada como ementário oficial completo da UFMG.

## `perfis.json`

Base de 23 perfis de carreira curados manualmente a partir de conhecimento de domínio
e referências públicas de trilhas profissionais. Não houve scraping de vagas.

## `consultas_avaliacao.json`

Conjunto curado de consultas de teste para medir a qualidade do ranqueamento:
20 consultas de carreira e 13 consultas de disciplinas.

## Arquivos `*_emb.pkl`

`disciplinas_emb.pkl` e `perfis_emb.pkl` não são novas fontes de dados. Eles são artefatos
derivados, gerados pelos scripts `scripts/gerar_embeddings.py` e
`scripts/gerar_embeddings_perfis.py` a partir dos JSONs acima, usando o modelo
`paraphrase-multilingual-MiniLM-L12-v2`.

# EduAcesso ESG

Dashboard analítico para monitoramento de indicadores educacionais municipais, com foco em equidade (ODS 4 e 10).

**Status:** TP4 — LLM local (HuggingFace) integrado à API FastAPI e ao app.

## Estrutura

```
.
├── app/
│   ├── main.py             # Entrada: menu de navegação (st.navigation)
│   ├── common.py           # Cache, estado de sessão e utilidades compartilhadas
│   └── views/              # Páginas: home, indicadores, noticias, dados, api, llm
├── api/                    # API FastAPI: main.py (dados) e llm_routes.py (LLM local)
├── data/
│   ├── raw/                # Coleta web + amostras de fallback
│   ├── processed/          # Corpus para LLM, registros enviados pela API
│   └── external/
├── docs/
│   ├── project_charter/    # Project Charter (v1 → v3)
│   ├── data_summary/       # Data Summary Report (v1 → v3)
│   ├── api/                # Documentação da API
│   └── llm/                # Escolha e integração do modelo LLM (TP4)
├── src/
│   ├── data/               # collect.py, scrape_web.py (Beautiful Soup), store.py (camada de dados)
│   └── llm/                # engine.py (LLM local), corpus.py, prompts.py, local_models.py
├── tests/                  # pytest: API e páginas do app
├── requirements.txt / requirements-dev.txt / requirements-llm.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Executar

Todos os comandos na raiz do projeto.

```bash
# 1) (opcional) coletar dados reais da web — sem isso o app usa as amostras
python src/data/scrape_web.py

# 2) (opcional) regenerar o corpus para LLM com os dados coletados
python src/llm/corpus.py

# 3) app (menu lateral com as 6 páginas)
streamlit run app/main.py

# 4) API, em outro terminal — Swagger em http://localhost:8000/docs
uvicorn api.main:app --reload
```

### LLM local (TP4)

```bash
pip install -r requirements-llm.txt        # transformers, torch, sentencepiece
python src/llm/local_models.py download    # baixa os modelos (~2,8 GB, uma vez)
python src/llm/local_models.py benchmark   # opcional: mede tempo e resultados
```

Com a API no ar, `POST /processar_texto` analisa um texto (sentimento ou resumo) e a página
**LLM local** do app usa essa rota. Sem os modelos instalados a API continua funcionando e a rota
responde 503 com a explicação. Detalhes em [`docs/llm/escolha_do_modelo.md`](docs/llm/escolha_do_modelo.md).

O app funciona sem a API no ar. A página **API FastAPI** traz um testador que chama a API
local; registros enviados por POST aparecem no app após "Recarregar base" em
**Upload / Download**. Detalhes das rotas em [`docs/api/api_documentation.md`](docs/api/api_documentation.md).

## Testes

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```

## Git

```bash
git checkout -b tp4
git add .
git commit -m "TP4 - LLM local com HuggingFace integrado ao FastAPI"
git push -u origin tp4
```

## Deploy

O app (`app/main.py`) é compatível com o Streamlit Community Cloud. A API FastAPI e os modelos de LLM rodam
localmente; o app não depende deles para funcionar (a página "LLM local" avisa quando a API está fora do ar).

## Metodologia

CRISP-DM + TDSP, documentadas em `docs/project_charter/` e `docs/data_summary/`.

# LLM local no EduAcesso ESG (TP4)

Meu projeto já tinha notícias sobre educação coletadas e indicadores municipais. Para o TP4 escolhi
duas tarefas que fazem sentido com esses dados: **classificar o sentimento das manchetes** e
**resumir textos**. Os dois modelos rodam na minha máquina com a biblioteca Transformers, sem nenhuma
API de nuvem.

## Modelos escolhidos

| Tarefa | Modelo (HuggingFace) | Tamanho | Licença |
|---|---|---|---|
| Sentimento | `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | 135 milhões de parâmetros (~0,5 GB) | Apache 2.0 |
| Resumo | `csebuetnlp/mT5_multilingual_XLSum` | arquivo de 2,3 GB (mT5-base) | CC BY-NC-SA 4.0 (uso não comercial, ok para o trabalho acadêmico) |

## Por que esses

**Desempenho**
- *Sentimento:* tem as três classes que preciso (positivo, neutro, negativo) e foi treinado em 12
  idiomas, incluindo português. É uma versão destilada de um modelo maior (mDeBERTa); o card
  registra 88,29% de concordância entre os dois.
- *Resumo:* foi treinado no XL-Sum, que cobre 45 idiomas. Para português, o card informa ROUGE-1/2/L de
  37,17 / 15,90 / 28,56.

**Custo computacional local** (estimativas a partir do tamanho dos arquivos, não medições)

| | Sentimento | Resumo |
|---|---|---|
| Disco | ~0,5 GB | ~2,3 GB |
| RAM | ~1–2 GB | ~4 GB |
| Hardware | CPU basta | CPU funciona, mas é mais lento; GPU é opcional |

O `python src/llm/local_models.py benchmark` mede o tempo real de cada tarefa na máquina de quem rodar.

**Acessibilidade local**
- Os dois estão no Hub e carregam com `AutoTokenizer` e `AutoModel...`.
- O download acontece uma vez e fica em cache (`~/.cache/huggingface`). Depois disso dá para rodar sem
  internet com `HF_HUB_OFFLINE=1`.

## O que descartei

| Modelo | Motivo |
|---|---|
| `nlptown/bert-base-multilingual-uncased-sentiment` | Devolve de 1 a 5 estrelas (feito para avaliação de produto) e não tem classe neutra |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Maior e treinado em tweets; o escolhido é mais leve e serve igual para manchetes |
| `facebook/bart-large-cnn` | Resume só em inglês |
| BERTimbau (puro) | É um modelo base: precisaria de fine-tuning para sentimento ou resumo |

## Como integrei

```
Streamlit (página "LLM local")  ──HTTP──►  FastAPI  ──►  src/llm/engine.py  ──►  Transformers (local)
```

- `src/llm/engine.py` carrega cada modelo **na primeira chamada** e o mantém em memória. A API sobe rápido
  e, se faltar biblioteca ou modelo, responde com erro claro em vez de cair.
- `api/llm_routes.py` define a rota e os Response Models.
- Os modelos podem ser trocados por variável de ambiente (`EDUACESSO_SENTIMENT_MODEL`,
  `EDUACESSO_SUMMARY_MODEL`), sem mexer no código.

### Rota `POST /processar_texto`

Entrada:

```json
{"texto": "Taxa de abandono escolar cai pela primeira vez em três anos.", "tipo_analise": "sentimento"}
```

`tipo_analise` é `"sentimento"` (padrão) ou `"resumo"`. O resumo aceita `max_palavras_resumo` (10 a 200)
e exige pelo menos 20 palavras no texto.

Resposta (`RespostaProcessamento`):

```json
{
  "tipo_analise": "sentimento",
  "texto_processado": "Taxa de abandono escolar cai pela primeira vez em três anos.",
  "modelo": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
  "tempo_ms": 42.7,
  "sentimento": {
    "rotulo": "positivo",
    "confianca": 0.81,
    "probabilidades": {"positivo": 0.81, "neutro": 0.14, "negativo": 0.05}
  }
}
```

Para `"resumo"`, o campo `sentimento` dá lugar a `"resumo": {"resumo": "...", "palavras": 12}`. Os valores
do exemplo são ilustrativos. A resposta é validada: confiança entre 0 e 1 e o resultado sempre bate com o
tipo pedido.

Também existe `GET /llm/status`, que mostra quais modelos estão carregados e se `transformers`, `torch`
e `sentencepiece` estão instalados.

## Tratamento de erros

Todo erro do LLM volta no mesmo formato: `{"codigo": "...", "mensagem": "...", "dica": "..."}`.

| Situação | HTTP | `codigo` |
|---|---|---|
| Texto vazio, curto (<3 caracteres), longo (>5000), tipo inválido, resumo com <20 palavras | 422 | (padrão do FastAPI) |
| Bibliotecas não instaladas, modelo não baixado ou sem internet no primeiro uso | 503 | `modelo_indisponivel` |
| Erro durante a inferência | 500 | `erro_inferencia` |
| Inferência passou do tempo limite (60 s; `EDUACESSO_LLM_TIMEOUT`) | 504 | `tempo_esgotado` |

Uma falha de carregamento **não fica guardada**: se eu resolver o problema (instalar, baixar), a próxima
chamada já funciona sem reiniciar a API. A página do Streamlit mostra a mensagem e a dica de cada erro.

## Como rodar

```bash
pip install -r requirements-llm.txt          # transformers, torch, sentencepiece
python src/llm/local_models.py download      # baixa os dois modelos (~2,8 GB, uma vez)
uvicorn api.main:app --reload                # http://localhost:8000/docs
streamlit run app/main.py                    # página "LLM local"
```

```bash
curl -X POST http://localhost:8000/processar_texto -H "Content-Type: application/json" \
     -d '{"texto": "Município amplia internet nas escolas rurais.", "tipo_analise": "sentimento"}'
```

## Limitações

- O XL-Sum gera resumos bem curtos (em geral uma frase), no estilo de notícia.
- O sentimento foi treinado com textos gerais, não com educação: vale conferir algumas manchetes à mão.
- O limite de tempo devolve o 504, mas não interrompe a inferência que já começou, e as chamadas
  seguintes esperam ela terminar.
- Os 39 testes automatizados usam modelos de mentira para não depender de download. A inferência real se
  confere com `python src/llm/local_models.py benchmark`.

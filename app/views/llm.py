import pandas as pd
import plotly.express as px
import streamlit as st

import common
from src.llm.corpus import retrieve
from src.llm.prompts import PROMPTS, render

common.init_state()
common.header("LLM local")
st.write(
    "Sentimento e resumo rodam com modelos HuggingFace **na própria máquina**, chamados pela API "
    "(`POST /processar_texto`). Q&A e geração de relatório continuam como prompts preparados."
)

corpus = common.load_corpus()
TIPOS = {"sentimento": "Sentimento", "resumo": "Resumo"}

# --- Execução com o LLM local ---------------------------------------------------
common.section("Analisar com o LLM local (via API)")
api_url = st.text_input("URL da API", common.API_URL_DEFAULT, key="llm_api_url")
st.caption("Precisa da API no ar (`uvicorn api.main:app --reload`) e dos modelos instalados "
           "(`pip install -r requirements-llm.txt`). A primeira chamada carrega o modelo e demora mais.")

tab_texto, tab_noticias = st.tabs(["Analisar um texto", "Sentimento das notícias"])

with tab_texto:
    tipo = st.radio("Análise", list(TIPOS), format_func=TIPOS.get, horizontal=True, key="llm_tipo")
    exemplo = corpus.loc[corpus["tipo"] == "texto", "texto"].iloc[0] if (corpus["tipo"] == "texto").any() else ""
    texto = st.text_area("Texto", value=exemplo, height=160, key="llm_texto")
    if st.button("Processar", key="btn_processar"):
        with st.spinner("Processando..."):
            ok, dados = common.chamar_api_llm(api_url, {"texto": texto, "tipo_analise": tipo})
        if not ok:
            st.error(dados["mensagem"])
            if dados.get("dica"):
                st.info(dados["dica"])
        else:
            st.caption(f"Modelo: `{dados['modelo']}` · {dados['tempo_ms']:.0f} ms")
            if tipo == "sentimento":
                s = dados["sentimento"]
                c1, c2 = st.columns(2)
                c1.metric("Sentimento", s["rotulo"].capitalize())
                c2.metric("Confiança", f"{s['confianca']:.0%}")
                st.bar_chart(pd.Series(s["probabilidades"], name="probabilidade"))
            else:
                st.success(dados["resumo"]["resumo"])

with tab_noticias:
    news, _ = common.load_news_source()
    n = st.slider("Quantas manchetes classificar", 1, min(len(news), 30), min(len(news), 10), key="llm_n")
    if st.button("Classificar manchetes", key="btn_noticias"):
        linhas, barra = [], st.progress(0.0)
        for i, titulo in enumerate(news["titulo"].head(n), start=1):
            ok, dados = common.chamar_api_llm(api_url, {"texto": titulo, "tipo_analise": "sentimento"})
            if not ok:
                st.error(dados["mensagem"])
                if dados.get("dica"):
                    st.info(dados["dica"])
                break
            linhas.append({"manchete": titulo, "sentimento": dados["sentimento"]["rotulo"],
                           "confiança": round(dados["sentimento"]["confianca"], 3)})
            barra.progress(i / n)
        if linhas:
            st.session_state.sentimento_noticias = pd.DataFrame(linhas)
    resultado = st.session_state.get("sentimento_noticias")
    if resultado is not None and len(resultado):
        st.dataframe(resultado, use_container_width=True, hide_index=True)
        contagem = resultado["sentimento"].value_counts().reset_index()
        st.plotly_chart(px.bar(contagem, x="sentimento", y="count", height=300,
                               color_discrete_sequence=["#2e7d5e"],
                               labels={"sentimento": "Sentimento", "count": "Manchetes"}),
                        use_container_width=True)
        st.download_button("⬇️ Baixar resultado (CSV)", resultado.to_csv(index=False).encode("utf-8-sig"),
                           file_name="sentimento_noticias.csv", mime="text/csv")

# --- Corpus ---------------------------------------------------------------------
common.section("Corpus unificado")
c1, c2, c3 = st.columns(3)
counts = corpus["tipo"].value_counts()
c1.metric("Textos (Wikipédia)", int(counts.get("texto", 0)))
c2.metric("Notícias", int(counts.get("noticia", 0)))
c3.metric("Indicadores (frases)", int(counts.get("indicador", 0)))
filtro = st.selectbox("Filtrar por tipo", ["todos", *sorted(corpus["tipo"].unique())])
view = corpus if filtro == "todos" else corpus[corpus["tipo"] == filtro]
st.dataframe(view, use_container_width=True, hide_index=True)
st.download_button("⬇️ Baixar corpus (CSV)", corpus.to_csv(index=False).encode("utf-8-sig"),
                   file_name="corpus_llm.csv", mime="text/csv")

# --- Prompts das outras tarefas ---------------------------------------------------
common.section("Tarefas e prompts")
for key, t in PROMPTS.items():
    with st.expander(f"{t['titulo']} — tipos: {', '.join(t['tipos'])}"):
        st.write(t["descricao"])
        st.code(t["template"], language="text")

common.section("Teste de prompt (sem LLM)")
tarefa = st.selectbox("Tarefa", list(PROMPTS), format_func=lambda k: PROMPTS[k]["titulo"])
if tarefa == "qa":
    pergunta = st.text_input("Pergunta", "Qual o IDEB de Três Lagoas?")
    ctx = retrieve(pergunta, corpus, k=3)
    st.caption("Contexto recuperado por sobreposição de palavras (baseline sem embeddings):")
    st.dataframe(ctx[["doc_id", "tipo", "texto", "score"]] if len(ctx) else ctx, use_container_width=True, hide_index=True)
    prompt = render("qa", contexto="\n".join(ctx["texto"]) or "(nenhum documento encontrado)", pergunta=pergunta)
else:
    opcoes = corpus[corpus["tipo"].isin(PROMPTS[tarefa]["tipos"])]
    doc = st.selectbox("Documento", opcoes["doc_id"])
    prompt = render(tarefa, texto=opcoes.set_index("doc_id").loc[doc, "texto"])
st.code(prompt, language="text")

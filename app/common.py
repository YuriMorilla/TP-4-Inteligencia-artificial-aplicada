"""Funções compartilhadas entre as páginas do app (dados, cache, estado, estilo)."""

import os
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402
from wordcloud import WordCloud  # noqa: E402

from src.data import store  # noqa: E402
from src.llm.corpus import build_corpus  # noqa: E402

REQUIRED_COLUMNS = store.COLUMNS
API_URL_DEFAULT = os.getenv("EDUACESSO_API_URL", "http://localhost:8000")

STOPWORDS_PT = {
    "a", "o", "e", "de", "do", "da", "em", "um", "uma", "os", "as", "dos", "das",
    "no", "na", "nos", "nas", "para", "com", "por", "que", "se", "ao", "aos", "à",
    "às", "é", "são", "foi", "ser", "sua", "seu", "suas", "seus", "mais", "como",
    "entre", "este", "esta", "isso", "essa", "esse", "pelo", "pela", "também",
    "já", "não", "sobre", "mas", "ou", "há", "cada", "the", "of", "and", "in",
}

CSS = """
<style>
    .ods-badge {display:inline-block;background:#2e7d5e;color:white;border-radius:20px;
                padding:4px 14px;font-size:13px;font-weight:600;margin-right:8px;}
    .section-title {font-size:1.1rem;font-weight:700;color:#1a3c2e;margin-top:1rem;}
</style>
"""


def header(titulo: str) -> None:
    """Cabeçalho padrão das páginas."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(titulo)
    st.markdown(
        '<span class="ods-badge">ODS 4 — Educação de Qualidade</span>'
        '<span class="ods-badge">ODS 10 — Redução das Desigualdades</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")


def section(texto: str) -> None:
    st.markdown(f'<p class="section-title">{texto}</p>', unsafe_allow_html=True)


# --- Estado de sessão -------------------------------------------------------
def init_state() -> None:
    """Base da sessão = amostra + registros enviados pela API."""
    if "base_df" not in st.session_state:
        st.session_state.base_df = store.load_all()
    st.session_state.setdefault("uploads_count", 0)


# --- Dados cacheados ----------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_text_source() -> tuple[str, bool]:
    return store.load_text()


@st.cache_data(show_spinner=False)
def load_news_source() -> tuple[pd.DataFrame, bool]:
    return store.load_news()


@st.cache_data(show_spinner=False)
def load_corpus() -> pd.DataFrame:
    return build_corpus()


@st.cache_data(show_spinner=False)
def word_frequencies(text: str, top_n: int = 20) -> pd.DataFrame:
    words = [w.lower().strip(".,;:!?()\"'") for w in text.split()]
    words = [w for w in words if len(w) > 2 and w not in STOPWORDS_PT]
    return pd.DataFrame(Counter(words).most_common(top_n), columns=["palavra", "frequência"])


@st.cache_resource(show_spinner=False)
def build_wordcloud(text: str):
    return WordCloud(width=900, height=420, background_color="white",
                     stopwords=STOPWORDS_PT, colormap="Greens", collocations=False).generate(text)


# --- Cliente da API de LLM ------------------------------------------------------
def chamar_api_llm(base_url: str, payload: dict, timeout: float = 180) -> tuple[bool, dict]:
    """POST /processar_texto. Retorna (ok, dados): JSON de sucesso, ou {'mensagem', 'dica'} em caso de erro."""
    try:
        resp = requests.post(f"{base_url.rstrip('/')}/processar_texto", json=payload, timeout=timeout)
    except requests.ConnectionError:
        return False, {"mensagem": "A API está fora do ar neste endereço.",
                       "dica": "Suba com: uvicorn api.main:app --reload"}
    except requests.Timeout:
        return False, {"mensagem": "A API demorou demais para responder.",
                       "dica": "Na primeira chamada o modelo é carregado, o que pode levar um tempo. Tente de novo."}
    except requests.RequestException as exc:
        return False, {"mensagem": f"Falha de rede ao chamar a API: {exc}"}

    try:
        dados = resp.json()
    except ValueError:
        return False, {"mensagem": f"Resposta inesperada da API (HTTP {resp.status_code})."}
    if resp.ok:
        return True, dados
    if resp.status_code == 422 and isinstance(dados, dict) and isinstance(dados.get("detail"), list):
        msgs = "; ".join(str(d.get("msg", "")).removeprefix("Value error, ") for d in dados["detail"])
        return False, {"mensagem": f"Entrada inválida: {msgs}"}
    if isinstance(dados, dict) and "mensagem" in dados:
        return False, {"mensagem": dados["mensagem"], "dica": dados.get("dica")}
    return False, {"mensagem": f"Erro HTTP {resp.status_code} da API."}

"""Smoke tests: cada página do app executa sem exceções."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

VIEWS = ["home", "indicadores", "noticias", "dados", "api", "llm"]


@pytest.mark.parametrize("view", VIEWS)
def test_view_roda_sem_erro(view):
    at = AppTest.from_file(str(APP_DIR / "views" / f"{view}.py"), default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]


def test_entrypoint_com_navegacao():
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]


def test_filtro_indicadores_reduz_selecao():
    at = AppTest.from_file(str(APP_DIR / "views" / "indicadores.py"), default_timeout=30).run()
    at.multiselect(key="uf_filter").set_value(["MS"]).run()
    assert not at.exception
    assert at.metric[0].value == "5"  # Campo Grande, Três Lagoas, Dourados, Corumbá, Ponta Porã


# --- Página LLM e cliente da API ------------------------------------------------
def test_llm_sem_api_mostra_erro_amigavel():
    at = AppTest.from_file(str(APP_DIR / "views" / "llm.py"), default_timeout=30).run()
    assert not at.exception
    at.text_input(key="llm_api_url").set_value("http://localhost:59999").run()  # porta sem API
    at.button(key="btn_processar").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.error and "fora do ar" in at.error[0].value


class _Resp:
    def __init__(self, status, corpo):
        self.status_code, self._corpo, self.ok = status, corpo, 200 <= status < 300

    def json(self):
        if self._corpo is None:
            raise ValueError("sem json")
        return self._corpo


@pytest.mark.parametrize("status,corpo,ok,trecho", [
    (200, {"tipo_analise": "sentimento"}, True, None),
    (503, {"codigo": "modelo_indisponivel", "mensagem": "Modelo fora", "dica": "Baixe"}, False, "Modelo fora"),
    (422, {"detail": [{"msg": "Value error, texto curto"}]}, False, "texto curto"),
    (500, None, False, "inesperada"),
])
def test_chamar_api_llm_trata_respostas(monkeypatch, status, corpo, ok, trecho):
    import common
    monkeypatch.setattr(common.requests, "post", lambda *a, **k: _Resp(status, corpo))
    sucesso, dados = common.chamar_api_llm("http://x", {"texto": "abc"})
    assert sucesso is ok
    if trecho:
        assert trecho in dados["mensagem"]


def test_chamar_api_llm_timeout(monkeypatch):
    import common
    def estoura(*a, **k):
        raise common.requests.Timeout()
    monkeypatch.setattr(common.requests, "post", estoura)
    sucesso, dados = common.chamar_api_llm("http://x", {})
    assert not sucesso and "demorou" in dados["mensagem"]

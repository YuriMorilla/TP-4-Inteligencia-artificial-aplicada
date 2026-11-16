"""Testes da rota /processar_texto com modelos falsos (sem baixar nada do HuggingFace)."""
import importlib.util
import time

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.llm import engine

TEXTO_LONGO = " ".join(["A educação básica municipal enfrenta desafios de infraestrutura e conectividade."] * 3)


@pytest.fixture()
def client():
    engine.REGISTRY.clear()
    yield TestClient(app)
    engine.REGISTRY.clear()


def _fake_sentimento(texto):
    return {"positive": 0.7, "neutral": 0.2, "negative": 0.1}


def _fake_resumo(texto, max_palavras):
    return "  Resumo   gerado\n pelo modelo falso. "


def test_sentimento_ok(client):
    engine.REGISTRY.set_runner("sentimento", _fake_sentimento, "modelo-falso")
    r = client.post("/processar_texto", json={"texto": "  Notícia ótima!  "})
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["tipo_analise"] == "sentimento" and corpo["modelo"] == "modelo-falso"
    assert corpo["texto_processado"] == "Notícia ótima!"  # espaços removidos
    assert corpo["sentimento"]["rotulo"] == "positivo" and corpo["sentimento"]["confianca"] == 0.7
    assert set(corpo["sentimento"]["probabilidades"]) == {"positivo", "neutro", "negativo"}
    assert "resumo" not in corpo and corpo["tempo_ms"] >= 0


def test_resumo_ok(client):
    engine.REGISTRY.set_runner("resumo", _fake_resumo, "modelo-falso")
    r = client.post("/processar_texto", json={"texto": TEXTO_LONGO, "tipo_analise": "resumo"})
    assert r.status_code == 200
    assert r.json()["resumo"] == {"resumo": "Resumo gerado pelo modelo falso.", "palavras": 5}


@pytest.mark.parametrize("corpo", [
    {"texto": ""},
    {"texto": "   "},
    {"texto": "ok", "tipo_analise": "sentimento"},
    {"texto": "x" * 5001},
    {"texto": "abc", "tipo_analise": "traducao"},
    {"texto": "texto curto demais", "tipo_analise": "resumo"},
    {"texto": TEXTO_LONGO, "tipo_analise": "resumo", "max_palavras_resumo": 5},
    {},
])
def test_entrada_invalida_422(client, corpo):
    assert client.post("/processar_texto", json=corpo).status_code == 422


def test_modelo_indisponivel_503(client, monkeypatch):
    def falha(tarefa, name):
        raise engine.ModelUnavailableError("Não foi possível carregar o modelo 'x'.", dica="Rode o download.")
    monkeypatch.setattr(engine, "_load", falha)
    r = client.post("/processar_texto", json={"texto": "Texto qualquer"})
    assert r.status_code == 503
    assert r.json() == {"codigo": "modelo_indisponivel", "mensagem": "Não foi possível carregar o modelo 'x'.",
                        "dica": "Rode o download."}


@pytest.mark.skipif(importlib.util.find_spec("transformers") is not None, reason="transformers instalado")
def test_503_real_quando_transformers_nao_instalado(client):
    r = client.post("/processar_texto", json={"texto": "Texto qualquer"})
    assert r.status_code == 503
    assert r.json()["codigo"] == "modelo_indisponivel" and "requirements-llm" in r.json()["dica"]


def test_falha_de_carga_nao_fica_em_cache(client, monkeypatch):
    chamadas = []
    def carrega(tarefa, name):
        chamadas.append(name)
        if len(chamadas) == 1:
            raise engine.ModelUnavailableError("sem internet")
        return _fake_sentimento
    monkeypatch.setattr(engine, "_load", carrega)
    assert client.post("/processar_texto", json={"texto": "Texto qualquer"}).status_code == 503
    assert client.post("/processar_texto", json={"texto": "Texto qualquer"}).status_code == 200


def test_erro_na_inferencia_500(client):
    def quebra(texto):
        raise RuntimeError("tensor com formato errado")
    engine.REGISTRY.set_runner("sentimento", quebra)
    r = client.post("/processar_texto", json={"texto": "Texto qualquer"})
    assert r.status_code == 500
    assert r.json()["codigo"] == "erro_inferencia" and "tensor" in r.json()["mensagem"]


def test_timeout_504(client, monkeypatch):
    monkeypatch.setenv("EDUACESSO_LLM_TIMEOUT", "0.2")
    engine.REGISTRY.set_runner("sentimento", lambda t: (time.sleep(0.8), _fake_sentimento(t))[1])
    r = client.post("/processar_texto", json={"texto": "Texto qualquer"})
    assert r.status_code == 504 and r.json()["codigo"] == "tempo_esgotado"
    time.sleep(0.8)  # deixa a thread presa terminar antes do próximo teste


def test_resumo_vazio_500(client):
    engine.REGISTRY.set_runner("resumo", lambda t, n: "   ")
    r = client.post("/processar_texto", json={"texto": TEXTO_LONGO, "tipo_analise": "resumo"})
    assert r.status_code == 500 and "vazio" in r.json()["mensagem"]


def test_llm_status(client):
    engine.REGISTRY.set_runner("sentimento", _fake_sentimento, "modelo-falso")
    corpo = client.get("/llm/status").json()
    por_tarefa = {m["tarefa"]: m for m in corpo["modelos"]}
    assert por_tarefa["sentimento"]["carregado"] is True and por_tarefa["resumo"]["carregado"] is False
    assert set(corpo["dependencias"]) == {"transformers", "torch", "sentencepiece"}


def test_modelo_configuravel_por_variavel_de_ambiente(monkeypatch):
    monkeypatch.setenv("EDUACESSO_SENTIMENT_MODEL", "outro/modelo")
    assert engine.model_name("sentimento") == "outro/modelo"
    monkeypatch.delenv("EDUACESSO_SENTIMENT_MODEL")
    assert engine.model_name("sentimento").startswith("lxyuan/")


def test_rotulos_desconhecidos_passam_sem_quebrar(client):
    engine.REGISTRY.set_runner("sentimento", lambda t: {"LABEL_0": 0.6, "LABEL_1": 0.4})
    r = client.post("/processar_texto", json={"texto": "Texto qualquer"})
    assert r.status_code == 200 and r.json()["sentimento"]["rotulo"] == "label_0"

"""Motor de LLM local (HuggingFace Transformers) do EduAcesso ESG — TP4.

Dois modelos, carregados sob demanda (na primeira chamada) e mantidos em memória:

- sentimento: classificação positivo / neutro / negativo
- resumo:     resumo abstrativo em português

Este módulo não depende de FastAPI. As bibliotecas pesadas (transformers, torch)
só são importadas na hora de carregar o modelo, então a API sobe mesmo sem elas e
responde 503 explicando o que falta.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable

log = logging.getLogger("eduacesso.llm")

DEFAULT_MODELS = {
    "sentimento": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    "resumo": "csebuetnlp/mT5_multilingual_XLSum",
}
MODEL_ENV_VARS = {
    "sentimento": "EDUACESSO_SENTIMENT_MODEL",
    "resumo": "EDUACESSO_SUMMARY_MODEL",
}
LABELS_PT = {"positive": "positivo", "neutral": "neutro", "negative": "negativo"}
TAREFAS = tuple(DEFAULT_MODELS)


# --- Erros (cada um carrega o código HTTP que a API deve devolver) ---------------
class LLMError(Exception):
    status_code = 500
    codigo = "erro_llm"

    def __init__(self, mensagem: str, dica: str | None = None):
        super().__init__(mensagem)
        self.mensagem = mensagem
        self.dica = dica


class ModelUnavailableError(LLMError):
    status_code = 503
    codigo = "modelo_indisponivel"


class InferenceTimeoutError(LLMError):
    status_code = 504
    codigo = "tempo_esgotado"


class InferenceError(LLMError):
    status_code = 500
    codigo = "erro_inferencia"


# --- Configuração ----------------------------------------------------------------
def model_name(tarefa: str) -> str:
    return os.getenv(MODEL_ENV_VARS[tarefa], DEFAULT_MODELS[tarefa])


def inference_timeout() -> float:
    return float(os.getenv("EDUACESSO_LLM_TIMEOUT", "60"))


# --- Carregamento ----------------------------------------------------------------
def _load_sentiment(name: str) -> Callable[[str], dict[str, float]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()

    def run(texto: str) -> dict[str, float]:
        enc = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits[0], dim=-1).tolist()
        return {model.config.id2label[i]: p for i, p in enumerate(probs)}

    return run


def _load_summary(name: str) -> Callable[[str, int], str]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.eval()

    def run(texto: str, max_palavras: int) -> str:
        limpo = re.sub(r"\s+", " ", texto.strip())  # recomendado no card do modelo
        enc = tokenizer([limpo], return_tensors="pt", truncation=True, max_length=512)
        saida = model.generate(
            **enc, max_new_tokens=int(max_palavras * 1.8), num_beams=4, no_repeat_ngram_size=2
        )[0]
        return tokenizer.decode(saida, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return run


def _load(tarefa: str, name: str) -> Callable:
    """Carrega o modelo da tarefa. Qualquer falha vira ModelUnavailableError (HTTP 503)."""
    try:
        return _load_sentiment(name) if tarefa == "sentimento" else _load_summary(name)
    except ImportError as exc:
        raise ModelUnavailableError(
            f"Bibliotecas de IA não instaladas ({exc.name or exc}).",
            dica="Rode: pip install -r requirements-llm.txt",
        ) from exc
    except OSError as exc:
        raise ModelUnavailableError(
            f"Não foi possível carregar o modelo '{name}'.",
            dica=("Na primeira execução é preciso internet para baixá-lo (python src/llm/local_models.py "
                  "download). Com HF_HUB_OFFLINE=1 o modelo já precisa estar no cache."),
        ) from exc
    except Exception as exc:  # noqa: BLE001 — qualquer falha de carga vira 503 explicativa
        raise ModelUnavailableError(f"Falha ao carregar o modelo '{name}': {exc}") from exc


class _Registry:
    """Guarda os modelos já carregados. Falhas não são guardadas (a próxima chamada tenta de novo)."""

    def __init__(self) -> None:
        self._runners: dict[str, Callable] = {}
        self._names: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, tarefa: str) -> Callable:
        if tarefa in self._runners:
            return self._runners[tarefa]
        with self._lock:
            if tarefa not in self._runners:
                name = model_name(tarefa)
                log.info("Carregando modelo '%s' para %s...", name, tarefa)
                self._runners[tarefa] = _load(tarefa, name)
                self._names[tarefa] = name
            return self._runners[tarefa]

    def name(self, tarefa: str) -> str:
        return self._names.get(tarefa) or model_name(tarefa)

    def is_loaded(self, tarefa: str) -> bool:
        return tarefa in self._runners

    def set_runner(self, tarefa: str, runner: Callable, name: str = "modelo-de-teste") -> None:
        self._runners[tarefa], self._names[tarefa] = runner, name

    def clear(self) -> None:
        self._runners.clear()
        self._names.clear()


REGISTRY = _Registry()
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")  # uma inferência por vez


def _run(fn: Callable, *args):
    """Executa a inferência com limite de tempo; erros viram InferenceError/InferenceTimeoutError."""
    limite = inference_timeout()
    future = _EXECUTOR.submit(fn, *args)
    try:
        return future.result(timeout=limite)
    except FutureTimeout as exc:
        future.cancel()
        raise InferenceTimeoutError(
            f"A inferência passou de {limite:g}s.",
            dica="Envie um texto menor ou aumente EDUACESSO_LLM_TIMEOUT.",
        ) from exc
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("Erro na inferência")
        raise InferenceError(f"Erro durante a inferência: {exc}") from exc


# --- API pública do módulo ---------------------------------------------------------
def analisar_sentimento(texto: str) -> dict:
    """Retorna {'rotulo', 'confianca', 'probabilidades'} (rótulos em português)."""
    bruto = _run(REGISTRY.get("sentimento"), texto)
    probs = {LABELS_PT.get(k.lower(), k.lower()): float(v) for k, v in bruto.items()}
    if not probs:
        raise InferenceError("O modelo não retornou probabilidades.")
    rotulo = max(probs, key=probs.get)
    return {"rotulo": rotulo, "confianca": probs[rotulo], "probabilidades": probs}


def resumir(texto: str, max_palavras: int = 60) -> str:
    resumo = " ".join(str(_run(REGISTRY.get("resumo"), texto, max_palavras)).split())
    if not resumo:
        raise InferenceError("O modelo retornou um resumo vazio.")
    return resumo


def status() -> dict:
    return {
        "modelos": [
            {"tarefa": t, "modelo": REGISTRY.name(t), "carregado": REGISTRY.is_loaded(t)} for t in TAREFAS
        ],
        "dependencias": {
            lib: importlib.util.find_spec(lib) is not None for lib in ("transformers", "torch", "sentencepiece")
        },
        "timeout_segundos": inference_timeout(),
    }


def preload(tarefas: list[str]) -> None:
    """Carrega modelos antecipadamente (uso opcional na subida da API)."""
    for t in tarefas:
        try:
            REGISTRY.get(t)
        except LLMError as exc:
            log.warning("Pré-carga de '%s' falhou: %s", t, exc.mensagem)

"""Rotas de LLM local da API (TP4): POST /processar_texto e GET /llm/status."""

from __future__ import annotations

import time
from typing import Annotated, Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints, model_validator

from src.llm import engine

router = APIRouter(tags=["LLM local"])

MIN_PALAVRAS_RESUMO = 20


# --- Modelos de entrada e saída ----------------------------------------------------
class TextoEntrada(BaseModel):
    texto: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=5000)] = Field(
        ..., examples=["Taxa de abandono escolar cai pela primeira vez em três anos."]
    )
    tipo_analise: Literal["sentimento", "resumo"] = Field(
        "sentimento", description="Análise a executar sobre o texto."
    )
    max_palavras_resumo: int = Field(60, ge=10, le=200, description="Tamanho máximo do resumo (só para 'resumo').")

    @model_validator(mode="after")
    def _resumo_precisa_de_texto_longo(self) -> "TextoEntrada":
        if self.tipo_analise == "resumo" and len(self.texto.split()) < MIN_PALAVRAS_RESUMO:
            raise ValueError(f"Para resumir, envie um texto com pelo menos {MIN_PALAVRAS_RESUMO} palavras.")
        return self


class ResultadoSentimento(BaseModel):
    rotulo: str = Field(..., examples=["positivo"], description="positivo, neutro ou negativo")
    confianca: float = Field(..., ge=0, le=1, description="Probabilidade do rótulo escolhido.")
    probabilidades: dict[str, float] = Field(..., description="Probabilidade de cada rótulo.")


class ResultadoResumo(BaseModel):
    resumo: str = Field(..., min_length=1)
    palavras: int = Field(..., ge=1)


class RespostaProcessamento(BaseModel):
    tipo_analise: Literal["sentimento", "resumo"]
    texto_processado: str
    modelo: str = Field(..., description="Identificador do modelo HuggingFace usado.")
    tempo_ms: float = Field(..., ge=0, description="Tempo de inferência em milissegundos.")
    sentimento: ResultadoSentimento | None = None
    resumo: ResultadoResumo | None = None

    @model_validator(mode="after")
    def _resultado_bate_com_o_tipo(self) -> "RespostaProcessamento":
        esperado = self.sentimento if self.tipo_analise == "sentimento" else self.resumo
        if esperado is None:
            raise ValueError(f"Resultado de '{self.tipo_analise}' ausente na resposta.")
        return self


class ErroResposta(BaseModel):
    codigo: str = Field(..., examples=["modelo_indisponivel"])
    mensagem: str
    dica: str | None = None


class ModeloStatus(BaseModel):
    tarefa: str
    modelo: str
    carregado: bool


class StatusLLM(BaseModel):
    modelos: list[ModeloStatus]
    dependencias: dict[str, bool]
    timeout_segundos: float


ERROS = {
    500: {"model": ErroResposta, "description": "Erro durante a inferência."},
    503: {"model": ErroResposta, "description": "Modelo não carregado ou bibliotecas ausentes."},
    504: {"model": ErroResposta, "description": "Inferência excedeu o tempo limite."},
}


# --- Rotas ---------------------------------------------------------------------------
@router.post("/processar_texto", response_model=RespostaProcessamento, response_model_exclude_none=True,
             responses=ERROS, summary="Analisa um texto com o LLM local (POST)")
def processar_texto(entrada: TextoEntrada) -> RespostaProcessamento:
    """Executa **sentimento** ou **resumo** sobre o texto, usando modelo HuggingFace rodando na própria máquina."""
    inicio = time.perf_counter()
    if entrada.tipo_analise == "sentimento":
        resultado = {"sentimento": engine.analisar_sentimento(entrada.texto)}
    else:
        resumo = engine.resumir(entrada.texto, entrada.max_palavras_resumo)
        resultado = {"resumo": {"resumo": resumo, "palavras": len(resumo.split())}}
    return RespostaProcessamento(
        tipo_analise=entrada.tipo_analise,
        texto_processado=entrada.texto,
        modelo=engine.REGISTRY.name(entrada.tipo_analise),
        tempo_ms=round((time.perf_counter() - inicio) * 1000, 1),
        **resultado,
    )


@router.get("/llm/status", response_model=StatusLLM, summary="Estado dos modelos locais")
def llm_status() -> dict:
    """Mostra quais modelos estão configurados/carregados e se as bibliotecas estão instaladas (não carrega nada)."""
    return engine.status()


async def llm_error_handler(_: Request, exc: engine.LLMError) -> JSONResponse:
    """Converte erros do motor de LLM em respostas HTTP com formato único (ErroResposta)."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"codigo": exc.codigo, "mensagem": exc.mensagem, "dica": exc.dica},
    )

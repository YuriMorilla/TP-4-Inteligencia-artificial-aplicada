"""Utilitários dos modelos locais de LLM.

    python src/llm/local_models.py download [sentimento] [resumo]   # baixa e carrega (uma vez)
    python src/llm/local_models.py benchmark [-n 10]                # mede tempo e distribuição

Depois do download, os modelos ficam em cache (~/.cache/huggingface) e funcionam sem
internet (HF_HUB_OFFLINE=1).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm import engine  # noqa: E402
from src.llm.corpus import build_corpus  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def download(tarefas: list[str]) -> None:
    for tarefa in tarefas:
        print(f"Carregando '{engine.model_name(tarefa)}' ({tarefa})...")
        inicio = time.perf_counter()
        try:
            engine.REGISTRY.get(tarefa)
        except engine.LLMError as exc:
            print(f"  ERRO: {exc.mensagem}" + (f"\n  Dica: {exc.dica}" if exc.dica else ""))
            continue
        print(f"  pronto em {time.perf_counter() - inicio:.1f}s")


def benchmark(n: int, com_resumo: bool = True, saida: Path = REPORTS_DIR / "benchmark_llm.csv") -> pd.DataFrame:
    corpus = build_corpus()
    linhas = []
    for r in corpus[corpus["tipo"] == "noticia"].head(n).itertuples():
        t0 = time.perf_counter()
        res = engine.analisar_sentimento(r.texto)
        linhas.append({"tarefa": "sentimento", "doc_id": r.doc_id, "resultado": res["rotulo"],
                       "confianca": round(res["confianca"], 3), "tempo_ms": round((time.perf_counter() - t0) * 1000, 1)})
    if com_resumo:
        longos = corpus[(corpus["tipo"] == "texto") & (corpus["n_palavras"] >= 20)].head(max(1, n // 3))
        for r in longos.itertuples():
            t0 = time.perf_counter()
            resumo = engine.resumir(r.texto)
            linhas.append({"tarefa": "resumo", "doc_id": r.doc_id, "resultado": resumo, "confianca": None,
                           "tempo_ms": round((time.perf_counter() - t0) * 1000, 1)})
    df = pd.DataFrame(linhas)
    saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(saida, index=False, encoding="utf-8-sig")
    print(df.groupby("tarefa")["tempo_ms"].agg(["count", "mean", "max"]).round(1).to_string())
    if "sentimento" in set(df["tarefa"]):
        print(df[df["tarefa"] == "sentimento"]["resultado"].value_counts().to_string())
    print(f"Detalhes em {saida}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="baixa e carrega os modelos")
    d.add_argument("tarefas", nargs="*", help=f"opções: {', '.join(engine.TAREFAS)} (padrão: todas)")
    b = sub.add_parser("benchmark", help="mede tempo de inferência nos documentos do corpus")
    b.add_argument("-n", type=int, default=10, help="manchetes a classificar")
    b.add_argument("--sem-resumo", action="store_true")
    args = parser.parse_args()
    if args.cmd == "download":
        invalidas = [t for t in args.tarefas if t not in engine.TAREFAS]
        if invalidas:
            parser.error(f"tarefa(s) desconhecida(s): {', '.join(invalidas)}. Use: {', '.join(engine.TAREFAS)}")
        download(args.tarefas or list(engine.TAREFAS))
    else:
        benchmark(args.n, com_resumo=not args.sem_resumo)


if __name__ == "__main__":
    main()

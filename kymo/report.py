"""Estatísticas consolidadas a partir dos cruzamentos registrados."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tracking import Cruzamento
from .video import formatar_tempo


@dataclass
class Resumo:
    total: int
    inicio: float
    fim: float
    duracao: float
    vazao_media: float           # pacotes/min
    intervalo_medio: float
    intervalo_mediano: float
    intervalo_p95: float
    intervalo_min: float
    intervalo_max: float
    paradas: list[tuple[float, float]]   # (instante do gap, duração)
    por_minuto: list[tuple[int, int]]
    pico_por_minuto: tuple[int, int] | None

    def como_dict(self) -> dict:
        return {
            "total_pacotes": self.total,
            "trecho": {
                "inicio": formatar_tempo(self.inicio),
                "fim": formatar_tempo(self.fim),
                "duracao_s": round(self.duracao, 3),
            },
            "vazao_media_por_min": round(self.vazao_media, 2),
            "intervalo_s": {
                "medio": round(self.intervalo_medio, 3),
                "mediano": round(self.intervalo_mediano, 3),
                "p95": round(self.intervalo_p95, 3),
                "min": round(self.intervalo_min, 3),
                "max": round(self.intervalo_max, 3),
            },
            "paradas": [
                {"em": formatar_tempo(t), "duracao_s": round(d, 2)}
                for t, d in self.paradas
            ],
            "pacotes_por_minuto": [
                {"minuto": m, "pacotes": c} for m, c in self.por_minuto
            ],
        }


def resumir(
    eventos: list[Cruzamento],
    inicio: float,
    fim: float,
    limite_parada: float = 20.0,
) -> Resumo:
    tempos = sorted(e.tempo for e in eventos)
    duracao = max(1e-6, fim - inicio)
    gaps = np.diff(tempos) if len(tempos) >= 2 else np.array([])

    paradas = [
        (tempos[i + 1], float(g)) for i, g in enumerate(gaps) if g > limite_parada
    ]

    contagem: dict[int, int] = {}
    for t in tempos:
        contagem[int(t // 60)] = contagem.get(int(t // 60), 0) + 1
    por_minuto = sorted(contagem.items())
    pico = max(por_minuto, key=lambda kv: kv[1]) if por_minuto else None

    z = lambda f, d=0.0: float(f) if len(gaps) else d
    return Resumo(
        total=len(tempos),
        inicio=inicio, fim=fim, duracao=duracao,
        vazao_media=len(tempos) / duracao * 60.0,
        intervalo_medio=z(np.mean(gaps) if len(gaps) else 0),
        intervalo_mediano=z(np.median(gaps) if len(gaps) else 0),
        intervalo_p95=z(np.percentile(gaps, 95) if len(gaps) else 0),
        intervalo_min=z(gaps.min() if len(gaps) else 0),
        intervalo_max=z(gaps.max() if len(gaps) else 0),
        paradas=paradas,
        por_minuto=por_minuto,
        pico_por_minuto=pico,
    )


def imprimir(resumo: Resumo, limite_parada: float = 20.0) -> None:
    r = resumo
    largura = 62
    print()
    print("=" * largura)
    print("  ANÁLISE DE PACOTES NA ESTEIRA".center(largura))
    print("=" * largura)
    print(f"  Trecho analisado ....... {formatar_tempo(r.inicio)} → "
          f"{formatar_tempo(r.fim)}  ({r.duracao:.1f}s)")
    print(f"  Total de pacotes ....... {r.total}")
    print(f"  Vazão média ............ {r.vazao_media:.2f} pacotes/min")
    if r.pico_por_minuto:
        m, c = r.pico_por_minuto
        print(f"  Melhor minuto .......... minuto {m} com {c} pacotes")
    print()
    print("  Intervalo entre pacotes (segundos)")
    print(f"    mínimo {r.intervalo_min:7.2f}   mediano {r.intervalo_mediano:7.2f}"
          f"   médio {r.intervalo_medio:7.2f}")
    print(f"    p95    {r.intervalo_p95:7.2f}   máximo  {r.intervalo_max:7.2f}")

    if r.paradas:
        print()
        print(f"  Paradas acima de {limite_parada:.0f}s ({len(r.paradas)})")
        for t, d in r.paradas[:15]:
            print(f"    {formatar_tempo(t)}   {d:6.1f}s sem pacote")
        if len(r.paradas) > 15:
            print(f"    ... e mais {len(r.paradas) - 15}")

    if r.por_minuto:
        print()
        print("  Pacotes por minuto")
        pico = max(c for _, c in r.por_minuto) or 1
        for m, c in r.por_minuto:
            barra = "█" * int(round(c / pico * 34))
            print(f"    min {m:>3}  {c:>4}  {barra}")
    print("=" * largura)


def salvar_json(resumo: Resumo, caminho: str | Path) -> None:
    Path(caminho).write_text(
        json.dumps(resumo.como_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

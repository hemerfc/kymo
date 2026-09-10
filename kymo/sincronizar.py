#!/usr/bin/env python3
"""Descobre o atraso entre o vídeo analisado e o vídeo MINI.

    # tenta pelo áudio, e diz por que não deu quando não dá
    python sincronizar.py "VIDEO.mp4"

    # você viu o mesmo evento nos dois: aos 45,2 s no principal e 12,7 s no MINI
    python sincronizar.py "VIDEO.mp4" --em-principal 45.2 --em-mini 12.7

    # confere um offset candidato: gera pares de quadros lado a lado
    python sincronizar.py "VIDEO.mp4" --offset -32.5 --previa previa.png

O resultado é o valor de `analisar.py render --mini-offset`, na convenção:
o instante `t` do principal corresponde a `t + offset` no MINI.

Fica fora de `analisar.py` de propósito: sincronizar é preparação de material,
não medição. O resultado entra na análise como um número.

Sobre o material deste projeto, para não se repetir a investigação:

- **o áudio não serve** — os vídeos principais foram gravados sem som (RMS
  exatamente zero em todos os cinco). Só os MINI têm faixa audível, e
  correlacionar som com silêncio não produz nada;
- **os metadados não servem** — nenhum dos arquivos tem `creation_time`; a
  edição que gerou os MINI apagou as tags de gravação;
- **correlacionar imagem não serve** — o MINI é outra câmera, de outro ângulo,
  e é câmera de mão que caminha pelo galpão. Medida a diferença entre quadros
  consecutivos, ela não fica parada em nenhum momento: o trecho mais estável do
  MINI do TESTE 02 dura 4,5 s. Não há região comum e fixa para casar.

Sobra o alinhamento por evento comum, que é o que os modos `--em-*` e
`--previa` servem. Um pacote de aparência distinta chegando ao ponto de
medição, ou um operador atravessando o quadro, bastam.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

TAXA = 8000            # Hz do áudio extraído; sobra para o envelope
PASSO_ENV = 0.005      # s por amostra do envelope (200 Hz)
JANELA_ENV = 0.020     # s de RMS por amostra
CONFIANCA_MINIMA = 6.0


def caminho_mini(principal: str) -> str:
    p = Path(principal)
    return str(p.with_name(f"{p.stem} - MINI{p.suffix}"))


# ----------------------------------------------------------------- áudio ---
def _audio_mono(caminho: str) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", caminho,
           "-vn", "-ac", "1", "-ar", str(TAXA), "-f", "f32le", "-"]
    bruto = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(bruto, dtype=np.float32)


def envelope(sinal: np.ndarray) -> np.ndarray:
    """Energia em janelas curtas — o 'quando' do som, sem a fase.

    A fase não se preserva entre microfones diferentes, a distâncias
    diferentes, gravando em taxas diferentes. O instante do evento, sim.
    """
    passo = max(1, int(PASSO_ENV * TAXA))
    janela = max(passo, int(JANELA_ENV * TAXA))
    n = max(0, (len(sinal) - janela) // passo + 1)
    if n == 0:
        return np.zeros(0)
    idx = np.arange(n) * passo
    soma = np.concatenate([[0.0], np.cumsum(sinal.astype(np.float64) ** 2)])
    env = np.sqrt((soma[idx + janela] - soma[idx]) / janela)
    # Log comprime os impactos fortes: sem isso um estrondo perto de um dos
    # microfones domina a correlação inteira e crava o pico nele.
    return np.log1p(env * 1e3)


def _normalizar(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    dp = x.std()
    return x / dp if dp > 1e-12 else x


def por_audio(principal: str, mini: str) -> tuple[float, float] | None:
    """Offset por correlação dos envelopes, ou None se algum lado é mudo."""
    a_p, a_m = _audio_mono(principal), _audio_mono(mini)
    for nome, a in ((principal, a_p), (mini, a_m)):
        if len(a) == 0:
            print(f"  {Path(nome).name}: sem faixa de áudio")
            return None
        rms = float(np.sqrt((a.astype(np.float64) ** 2).mean()))
        if rms < 1e-6:
            print(f"  {Path(nome).name}: faixa de áudio muda (rms {rms:.0e})")
            return None
    env_p, env_m = _normalizar(envelope(a_p)), _normalizar(envelope(a_m))
    n = 1 << int(np.ceil(np.log2(len(env_p) + len(env_m))))
    corr = np.fft.irfft(np.fft.rfft(env_p, n) * np.conj(np.fft.rfft(env_m, n)), n)
    corr = np.concatenate([corr[-(len(env_m) - 1):], corr[: len(env_p)]])
    pico = int(np.argmax(corr))
    # Confiança = quantos desvios o melhor alinhamento se destaca dos demais.
    confianca = float((corr[pico] - corr.mean()) / (corr.std() or 1e-12))
    return (pico - (len(env_m) - 1)) * PASSO_ENV, confianca


# ------------------------------------------------------------ conferência ---
def _quadro(caminho: str, t: float, largura: int = 480) -> np.ndarray | None:
    if t < 0:
        return None
    saida = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", caminho,
         "-frames:v", "1", "-vf", f"scale={largura}:-2", "-f", "image2pipe",
         "-vcodec", "png", "-"],
        capture_output=True).stdout
    if not saida:
        return None
    return cv2.imdecode(np.frombuffer(saida, np.uint8), cv2.IMREAD_COLOR)


def previa(principal: str, mini: str, offset: float, saida: str,
           instantes: list[float] | None = None) -> Path:
    """Pares (principal, MINI) em vários instantes, para julgar o alinhamento.

    Julgar um offset é comparar o que as duas câmeras mostram no mesmo
    instante; uma imagem com os pares empilhados resolve isso em segundos,
    enquanto abrir os dois vídeos lado a lado e caçar o quadro certo não.
    """
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", principal], capture_output=True, text=True).stdout)
    if instantes is None:
        instantes = list(np.linspace(dur * 0.15, dur * 0.85, 4))
    linhas = []
    for t in instantes:
        a = _quadro(principal, t)
        b = _quadro(mini, t + offset)
        if a is None:
            continue
        if b is None:
            b = np.zeros_like(a)
            cv2.putText(b, "fora do MINI", (12, b.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
        alt = min(a.shape[0], b.shape[0])
        a, b = a[:alt], b[:alt]
        cv2.putText(a, f"principal {t:7.2f}s", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(b, f"MINI {t + offset:7.2f}s", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        linhas.append(np.hstack([a, b]))
    if not linhas:
        raise SystemExit("nenhum quadro pôde ser lido")
    larg = max(l.shape[1] for l in linhas)
    linhas = [np.pad(l, ((0, 4), (0, larg - l.shape[1]), (0, 0))) for l in linhas]
    cv2.imwrite(saida, np.vstack(linhas))
    return Path(saida)


def _relatar(offset: float) -> None:
    sinal = "+" if offset >= 0 else "-"
    print(f"\noffset ...... {offset:+.3f} s")
    print(f"  o instante t do principal corresponde a t {sinal} {abs(offset):.3f}s no MINI")
    print(f"\n  --mini-offset {offset:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("principal")
    p.add_argument("mini", nargs="?", default=None,
                   help='padrao: o mesmo nome com " - MINI" antes da extensao')
    p.add_argument("--em-principal", dest="em_principal", type=float, default=None,
                   help="instante (s) de um evento no video principal")
    p.add_argument("--em-mini", dest="em_mini", type=float, default=None,
                   help="instante (s) do MESMO evento no MINI")
    p.add_argument("--offset", type=float, default=None,
                   help="offset a conferir, em vez de calcular")
    p.add_argument("--previa", default=None,
                   help="arquivo PNG com pares de quadros para conferir o offset")
    a = p.parse_args()

    mini = a.mini or caminho_mini(a.principal)
    for f in (a.principal, mini):
        if not Path(f).exists():
            sys.exit(f"'{f}' nao existe")

    if a.em_principal is not None or a.em_mini is not None:
        if a.em_principal is None or a.em_mini is None:
            sys.exit("--em-principal e --em-mini andam juntos")
        offset = a.em_mini - a.em_principal
        print(f"evento aos {a.em_principal:.3f}s no principal "
              f"e {a.em_mini:.3f}s no MINI")
        _relatar(offset)
    elif a.offset is not None:
        offset = a.offset
        print(f"conferindo offset {offset:+.3f}s")
    else:
        print("tentando pelo audio...")
        r = por_audio(a.principal, mini)
        if r is None:
            print("\nAudio nao serve para estes arquivos.")
            print("Use --em-principal/--em-mini com um evento visto nos dois,")
            print("ou --offset X --previa p.png para conferir um palpite.")
            sys.exit(1)
        offset, confianca = r
        print(f"confianca ... {confianca:.1f} sigma", end="")
        print("  (abaixo de 6 nao confie)" if confianca < CONFIANCA_MINIMA else "")
        _relatar(offset)

    if a.previa:
        caminho = previa(a.principal, mini, offset, a.previa)
        print(f"\nprevia: {caminho}")


if __name__ == "__main__":
    main()

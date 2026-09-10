"""Panorama slit-scan: a esteira desenrolada, para conferir a contagem no olho.

O gate é um sensor pontual — o CSV diz *quando* algo passou, mas não *o quê*.
Quando a contagem automática discorda da contagem manual, é preciso ver a fila
inteira, e o vídeo não serve: cada pacote aparece em dezenas de frames e sai do
quadro antes que dê para conferir o conjunto.

A solução é a de uma câmera de varredura linear. A cada frame recorta-se uma
fatia estreita sobre o ponto de medição, com largura igual ao deslocamento da
esteira entre dois frames, e concatenam-se as fatias. Como cada pedaço da
esteira cruza o ponto uma única vez, cada pacote aparece uma única vez no
resultado — contar objetos vira contar caixas numa foto.

Foi assim que apareceu o trecho de sacos plásticos brancos do TESTE 01, que o
critério de cor de papelão não enxergava: no sinal do gate ele era um vão longo,
indistinguível de esteira vazia; no panorama era uma fila cheia de sacos.
"""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np

from .config import Configuracao
from .cor import mascara_cor
from .video import LeitorFrames, sondar

# Diferença média (níveis de cinza) abaixo da qual dois frames são o mesmo
# quadro. Medido no TESTE 04, que é 60 fps com conteúdo de 30: as duplicatas
# ficam em 0,3-1,0 e os quadros novos em 15-20 — a folga é enorme.
_LIMIAR_DUPLICATA = 2.0


def _mascara_faixa(cfg: Configuracao, forma: tuple[int, int]) -> np.ndarray:
    """Faixa da esteira como máscara; o retângulo da ROI se não houver polígono."""
    poli = cfg.roi.poligono
    m = np.zeros(forma, np.uint8)
    if poli:
        cv2.fillPoly(m, [np.array(poli, dtype=np.int32)], 255)
    else:
        x, y, w, h = cfg.roi.como_tupla()
        cv2.rectangle(m, (x, y), (x + w, y + h), 255, -1)
    return m


def _recorte_da_faixa(faixa: np.ndarray) -> tuple[slice, slice]:
    ys, xs = np.nonzero(faixa)
    if ys.size == 0:
        raise ValueError("a faixa da esteira não intersecta o quadro")
    return (slice(int(ys.min()), int(ys.max()) + 1),
            slice(int(xs.min()), int(xs.max()) + 1))


def _media_temporal(cfg: Configuracao, rec: tuple[slice, slice], info,
                    frames: int = 300) -> np.ndarray:
    """Fração do tempo em que cada pixel é material, num trecho inicial.

    Subtrair esta média do perfil apaga o que não se move. A faixa marcada no
    setup costuma ser mais larga que a esteira, e o que sobra dentro dela é
    estrutura: no TESTE 04 entra um corrimão amarelo, que tem b* de papelão,
    cobre mais pixels que a carga e ancoraria a correlação em deslocamento
    zero. Filtrar por frequência não serviria — numa esteira cheia como a do
    TESTE 01 um pixel da própria esteira fica coberto mais da metade do tempo,
    e o corte que remove estrutura removeria a carga junto.
    """
    leitor = LeitorFrames(cfg.video, info=info, inicio=cfg.inicio,
                          duracao=cfg.duracao, escala=1.0)
    soma = np.zeros((rec[0].stop - rec[0].start, rec[1].stop - rec[1].start),
                    dtype=np.float32)
    total = 0
    for _indice, _tempo, frame in leitor:
        soma += (mascara_cor(cfg.cor, frame) > 0)[rec]
        total += 1
        if total >= frames:
            break
    return soma / max(total, 1)


def _melhor_deslocamento(a: np.ndarray, b: np.ndarray, maximo: int) -> int:
    """Deslocamento inteiro que melhor alinha dois perfis, por produto interno."""
    a = a - a.mean()
    b = b - b.mean()
    # O perfil transversal é tão curto quanto a faixa é estreita — 137 px no
    # TESTE 02 — e deslocar além disso deixa a sobreposição vazia.
    maximo = min(maximo, len(a) - 1)
    if maximo < 1:
        return 0
    melhor, pontos = -np.inf, 0
    for d in range(-maximo, maximo + 1):
        if d > 0:
            s = float(np.dot(a[d:], b[: len(b) - d]))
        elif d < 0:
            s = float(np.dot(a[:d], b[-d:]))
        else:
            s = float(np.dot(a, b))
        if s > melhor:
            melhor, pontos = s, d
    return pontos


def estimar_movimento(
    cfg: Configuracao, amostras: int = 600, maximo: int = 150
) -> tuple[float, float]:
    """Deslocamento da esteira em px por frame novo, como vetor `(dx, dy)`.

    Mede o transporte real em vez de pedir a velocidade ao usuário: é ela que
    define a largura da fatia, e uma largura errada estica ou comprime o
    panorama até um mesmo pacote aparecer duas ou nenhuma vez.

    Quatro armadilhas, cada uma capaz de zerar a medida sozinha, todas
    encontradas nos vídeos deste repositório:

    - correlacionar a imagem devolve zero, porque os roletes formam um padrão
      periódico estático que domina o produto interno. Daí a máscara de cor;
    - estrutura fixa dentro da faixa também devolve zero, e no TESTE 04 ela
      cobre mais pixels que a carga. Daí subtrair a média temporal;
    - frame duplicado anda zero de verdade. O TESTE 04 é 60 fps com conteúdo de
      30, e metade dos pares é duplicata. Daí pular quadro repetido, aqui e na
      montagem — que é o que mantém as duas coerentes;
    - trecho de esteira vazia não tem o que correlacionar e devolve zero com
      toda a razão: no TESTE 04 nada passa nos primeiros 17,6 s. Daí descartar
      os pares que não acusam movimento nenhum.

    `cv2.phaseCorrelate` foi testado e descartado: sobre a máscara binária
    acertava menos de um décimo dos pares, com confiança reportada tão alta nos
    erros quanto nos acertos.
    """
    info = sondar(cfg.video)
    faixa = _mascara_faixa(cfg, (info.altura, info.largura)) > 0
    rec = _recorte_da_faixa(faixa)
    dentro = faixa[rec]
    media = _media_temporal(cfg, rec, info)

    leitor = LeitorFrames(cfg.video, info=info, inicio=cfg.inicio,
                          duracao=cfg.duracao, escala=1.0)
    quadro_anterior: np.ndarray | None = None
    perfil_anterior: tuple[np.ndarray, np.ndarray] | None = None
    dxs: list[int] = []
    dys: list[int] = []
    for _indice, _tempo, frame in leitor:
        recorte = frame[rec].astype(np.int16)
        if _e_duplicata(recorte, quadro_anterior):
            continue
        m = (mascara_cor(cfg.cor, frame) > 0)[rec].astype(np.float32)
        desvio = (m - media) * dentro
        atual = (desvio.sum(axis=0), desvio.sum(axis=1))
        if perfil_anterior is not None:
            dx = _melhor_deslocamento(perfil_anterior[0], atual[0], maximo)
            dy = _melhor_deslocamento(perfil_anterior[1], atual[1], maximo)
            if dx != 0 or dy != 0:
                dxs.append(dx)
                dys.append(dy)
        perfil_anterior = atual
        quadro_anterior = recorte
        if len(dxs) >= amostras:
            break

    if len(dxs) < 30:
        raise ValueError(
            "não foi possível medir o deslocamento: nada se move dentro da "
            "faixa no trecho. Ou a esteira está parada, ou a faixa pegou só "
            "estrutura fixa, ou o critério de cor não enxerga o que passa nela. "
            "Passe --passo se souber o valor."
        )
    # Mediana, não média: um par que correlacionou com o pacote errado erra por
    # um espaçamento inteiro, e a média carregaria esse erro.
    return float(np.median(dxs)), float(np.median(dys))


def _e_duplicata(recorte: np.ndarray, anterior: np.ndarray | None) -> bool:
    """Quadro repetido, ou esteira parada — nos dois casos não há esteira nova.

    Emitir a fatia mesmo assim repetiria o mesmo pedaço de esteira no panorama,
    e um pacote parado sobre o ponto de medição viraria uma faixa esticada.
    """
    if anterior is None:
        return False
    return bool(np.abs(recorte - anterior).mean() < _LIMIAR_DUPLICATA)


def gerar_panorama(
    cfg: Configuracao,
    saida: str | Path,
    passo: float | None = None,
    altura_fator: float = 1.6,
    marcas: list[float] | None = None,
    segundos_por_tira: float = 10.0,
    escala: float = 1.0,
) -> tuple[Path, float]:
    """Escreve o panorama da esteira, quebrado em tiras empilhadas.

    `marcas` são instantes (s) desenhados como linha vermelha numerada — passe
    os tempos do CSV para ver se cada objeto recebeu exatamente uma contagem.
    Devolve o caminho e o passo usado, em px por frame novo.
    """
    info = sondar(cfg.video)
    dx, dy = (passo, 0.0) if passo is not None else estimar_movimento(cfg)
    modulo = float(np.hypot(dx, dy))
    if modulo < 1.0:
        raise ValueError(
            f"deslocamento medido ({modulo:.1f} px/frame) é pequeno demais para "
            "montar um panorama. Passe --passo se souber o valor."
        )
    passo_px = max(1, int(round(modulo)))

    # Gira o quadro até o movimento ficar horizontal: assim a fatia é um
    # retângulo, e não um paralelogramo que precisaria de reamostragem própria.
    # Só a direção da reta importa, não o sentido — normalizar o ângulo para
    # (-90, 90] evita virar o panorama de cabeça para baixo quando o fluxo corre
    # para a esquerda.
    cx = (cfg.linha.x1 + cfg.linha.x2) / 2.0
    cy = (cfg.linha.y1 + cfg.linha.y2) / 2.0
    angulo = float(np.degrees(np.arctan2(dy, dx)))
    if angulo > 90.0:
        angulo -= 180.0
    elif angulo <= -90.0:
        angulo += 180.0
    M = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)

    comprimento = float(np.hypot(cfg.linha.x2 - cfg.linha.x1,
                                 cfg.linha.y2 - cfg.linha.y1))
    meia_altura = max(20, int(comprimento * altura_fator / 2))
    y0 = max(0, int(round(cy)) - meia_altura)
    x0 = max(0, int(round(cx)) - passo_px // 2)
    rec = _recorte_da_faixa(_mascara_faixa(cfg, (info.altura, info.largura)) > 0)

    leitor = LeitorFrames(cfg.video, info=info, inicio=cfg.inicio,
                          duracao=cfg.duracao, escala=1.0)
    fatias: list[np.ndarray] = []
    tempos: list[float] = []
    quadro_anterior: np.ndarray | None = None
    for _indice, tempo, frame in leitor:
        recorte = frame[rec].astype(np.int16)
        if _e_duplicata(recorte, quadro_anterior):
            continue
        quadro_anterior = recorte
        rot = cv2.warpAffine(frame, M, (info.largura, info.altura))
        fatias.append(rot[y0: y0 + 2 * meia_altura, x0: x0 + passo_px].copy())
        tempos.append(tempo)
    if not fatias:
        raise ValueError("nenhum frame lido no trecho")

    # As fatias entram em ordem de tempo, mesmo quando o fluxo corre para a
    # esquerda; nesse caso a esteira sai espelhada como um todo, o que não
    # atrapalha contar e mantém o eixo x sendo o tempo.
    pano = np.hstack(fatias)
    altura = pano.shape[0]

    # O eixo x é o índice da fatia, não o tempo direto: quadros repetidos foram
    # pulados, e a relação entre um e outro deixou de ser proporcional.
    def x_do_tempo(t: float) -> int:
        i = bisect_left(tempos, t)
        if i >= len(tempos):
            i = len(tempos) - 1
        elif i > 0 and (t - tempos[i - 1]) < (tempos[i] - t):
            i -= 1
        return i * passo_px

    if marcas:
        for i, t in enumerate(marcas, 1):
            x = x_do_tempo(t)
            if 0 <= x < pano.shape[1]:
                cv2.line(pano, (x, 0), (x, altura), (0, 0, 255), 3)
                cv2.putText(pano, str(i), (x + 5, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    largura_tira = max(1, int(round(len(fatias) * passo_px * segundos_por_tira
                                    / max(tempos[-1] - tempos[0], 1e-6))))
    tiras: list[np.ndarray] = []
    for x in range(0, pano.shape[1], largura_tira):
        tira = pano[:, x: x + largura_tira]
        if tira.shape[1] < largura_tira:
            tira = np.pad(tira, ((0, 0), (0, largura_tira - tira.shape[1]), (0, 0)))
        rotulo = tempos[min(x // passo_px, len(tempos) - 1)]
        cv2.putText(tira, f"{rotulo:.0f}s", (4, altura - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if escala != 1.0:
            tira = cv2.resize(tira, (int(largura_tira * escala), int(altura * escala)))
        tiras.append(tira)
        tiras.append(np.zeros((4, tira.shape[1], 3), np.uint8))

    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(saida), np.vstack(tiras))
    return saida, modulo

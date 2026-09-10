"""Sensor de gate: ocupação do material transportado numa janela sobre a linha.

Por que este é o método principal, e não o rastreamento de blobs: nesta esteira
os pacotes viajam ENCOSTADOS, em fila contínua. Segmentar por cor devolve um
único blob para a fila toda, e dividi-lo pelo comprimento típico faz as
fronteiras artificiais escorregarem entre frames, o que quebra a identidade dos
objetos e produz cruzamentos fantasma.

O que é limpo nesta cena é o vão entre pacotes. Medido em um trecho de dois
minutos, a ocupação da janela do gate cai a exatamente zero entre um pacote e o
seguinte, com cadência de ~8,25 s e variação abaixo de 0,2 s. Cada subida de
"vazio" para "ocupado" é a borda dianteira de um pacote chegando ao ponto de
medição — é isso que se cronometra, do mesmo jeito que faria uma fotocélula.

O sinal é coletado para todo o trecho e só depois transformado em eventos: com a
distribuição inteira em mãos, o nível de referência sai de um percentil dos
próprios dados, em vez de depender de um limiar absoluto calibrado à mão.

O que conta como "ocupado" vem inteiro de `cor.mascara_cor`: se o critério de
cor não enxerga um tipo de embalagem, o gate não a vê passar e o vão aparece no
sinal como se a esteira estivesse vazia. Conferir o critério contra a cena é
parte de aceitar a contagem, não um detalhe de ajuste fino.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Configuracao
from .cor import mascara_cor
from .tracking import Cruzamento


@dataclass
class ParametrosGate:
    """Ajustes do sensor. Os limiares são FRAÇÕES do nível de referência."""

    # Espessura da janela na direção do fluxo, em px do vídeo original.
    # Estreita demais fica sensível a ruído; larga demais funde o vão entre
    # dois pacotes próximos e perde a contagem.
    largura: int = 20
    suavizacao_frames: int = 5
    limiar_ocupado: float = 0.45
    limiar_vazio: float = 0.25
    # Tempos mínimos em cada estado, para não contar tremulação do sinal.
    min_vazio_s: float = 0.20
    min_ocupado_s: float = 0.20
    # O nível de referência ("gate cheio") vem deste percentil do sinal.
    referencia_percentil: float = 85.0
    nivel_referencia: float | None = None   # fixa o nível, se quiser
    # Conta na borda de entrada do pacote ("subida") ou na de saída ("descida").
    borda: str = "subida"


class SensorGate:
    """Mede, frame a frame, a fração de material dentro da janela do gate."""

    def __init__(self, cfg: Configuracao, params: ParametrosGate | None = None):
        self.cfg = cfg
        self.p = params or ParametrosGate()
        self.escala = cfg.deteccao.escala_processamento
        self._preparar_janela()

    def _preparar_janela(self) -> None:
        """Retângulo com o segmento da linha como eixo e `largura` de espessura."""
        x1, y1, x2, y2 = self.cfg.linha.escalada(self.escala)
        dx, dy = x2 - x1, y2 - y1
        norma = (dx * dx + dy * dy) ** 0.5 or 1.0
        # Perpendicular à linha = direção do fluxo, onde a janela tem espessura.
        ux, uy = dy / norma, -dx / norma
        meia = max(1.0, self.p.largura * self.escala / 2.0)

        cantos = np.array([
            [x1 + ux * meia, y1 + uy * meia],
            [x2 + ux * meia, y2 + uy * meia],
            [x2 - ux * meia, y2 - uy * meia],
            [x1 - ux * meia, y1 - uy * meia],
        ])
        x0 = int(np.floor(cantos[:, 0].min())) - 1
        y0 = int(np.floor(cantos[:, 1].min())) - 1
        x3 = int(np.ceil(cantos[:, 0].max())) + 1
        y3 = int(np.ceil(cantos[:, 1].max())) + 1
        self.recorte = (max(0, x0), max(0, y0), x3 - x0, y3 - y0)

        rx, ry, rw, rh = self.recorte
        mascara = np.zeros((rh, rw), np.uint8)
        pts = np.array(
            [[[int(round(cx)) - rx, int(round(cy)) - ry] for cx, cy in cantos]],
            dtype=np.int32,
        )
        cv2.fillPoly(mascara, pts, 255)

        # A faixa da esteira também limita o gate: sem isso, uma esteira que
        # passa atrás no plano da imagem entraria na mesma medição.
        poli = self.cfg.roi.poligono
        if poli:
            faixa = np.zeros((rh, rw), np.uint8)
            pts_faixa = np.array(
                [[[int(round(px * self.escala)) - rx,
                   int(round(py * self.escala)) - ry] for px, py in poli]],
                dtype=np.int32,
            )
            cv2.fillPoly(faixa, pts_faixa, 255)
            mascara = cv2.bitwise_and(mascara, faixa)

        self.mascara = mascara > 0
        self.n_pixels = int(self.mascara.sum())
        self.cantos = cantos

    def medir(self, frame: np.ndarray) -> float:
        if self.n_pixels == 0:
            return 0.0
        rx, ry, rw, rh = self.recorte
        h, w = frame.shape[:2]
        recorte = frame[ry:min(h, ry + rh), rx:min(w, rx + rw)]
        if recorte.size == 0:
            return 0.0
        m = self.mascara[: recorte.shape[0], : recorte.shape[1]]
        pap = mascara_cor(self.cfg.cor, recorte) > 0
        return float((pap & m).sum()) / self.n_pixels


def _suavizar(sinal: np.ndarray, janela: int) -> np.ndarray:
    if janela <= 1 or len(sinal) < janela:
        return sinal.astype(np.float64)
    nucleo = np.ones(janela) / janela
    # 'same' com bordas replicadas: sem isso o começo e o fim do trecho
    # afundariam artificialmente e inventariam um vão inexistente.
    pad = janela // 2
    estendido = np.concatenate(
        [np.full(pad, sinal[0]), sinal.astype(np.float64), np.full(pad, sinal[-1])]
    )
    return np.convolve(estendido, nucleo, mode="same")[pad: pad + len(sinal)]


def _cruzamento_interpolado(
    sinal: np.ndarray, tempos: np.ndarray, i: int, limiar: float
) -> float:
    """Instante em que o sinal cruza `limiar` entre os frames i-1 e i."""
    if i <= 0:
        return float(tempos[0])
    a, b = sinal[i - 1], sinal[i]
    if b == a:
        return float(tempos[i])
    frac = (limiar - a) / (b - a)
    frac = min(1.0, max(0.0, frac))
    return float(tempos[i - 1] + (tempos[i] - tempos[i - 1]) * frac)


@dataclass
class ResultadoGate:
    eventos: list[Cruzamento]
    sinal: np.ndarray            # sinal suavizado, por frame
    tempos: np.ndarray
    nivel_referencia: float
    limiar_ocupado: float
    limiar_vazio: float


def eventos_do_sinal(
    sinal_bruto: list[float] | np.ndarray,
    tempos: list[float] | np.ndarray,
    p: ParametrosGate,
    fps: float = 30.0,
) -> ResultadoGate:
    """Converte o sinal de ocupação em cruzamentos cronometrados."""
    sinal = np.asarray(sinal_bruto, dtype=np.float64)
    tempos = np.asarray(tempos, dtype=np.float64)
    if len(sinal) == 0:
        return ResultadoGate([], sinal, tempos, 0.0, 0.0, 0.0)

    s = _suavizar(sinal, p.suavizacao_frames)
    nivel = (
        p.nivel_referencia
        if p.nivel_referencia is not None
        else float(np.percentile(s, p.referencia_percentil))
    )
    nivel = max(nivel, 1e-6)
    lim_alto = nivel * p.limiar_ocupado
    lim_baixo = nivel * p.limiar_vazio

    min_vazio = max(1, int(round(p.min_vazio_s * fps)))
    min_ocup = max(1, int(round(p.min_ocupado_s * fps)))

    eventos: list[Cruzamento] = []
    # Começa em "indefinido": só conta um pacote depois de ver um vão de
    # verdade, para não registrar como evento o pacote que já estava sobre a
    # linha quando o trecho começou.
    estado = "ocupado" if s[0] >= lim_alto else "vazio"
    desde = 0
    candidato_subida: int | None = None

    for i in range(1, len(s)):
        v = s[i]
        if estado == "vazio":
            if v >= lim_alto:
                if candidato_subida is None:
                    candidato_subida = i
                # Só confirma depois de `min_ocup` frames acima do limiar: um
                # pico isolado de ruído não vira pacote.
                if i - candidato_subida + 1 >= min_ocup:
                    if (candidato_subida - desde) >= min_vazio:
                        t = _cruzamento_interpolado(s, tempos, candidato_subida, lim_alto)
                        eventos.append(
                            Cruzamento(
                                id_pacote=len(eventos) + 1,
                                id_track=len(eventos) + 1,
                                tempo=t,
                                frame=int(candidato_subida),
                                cx=0.0, cy=0.0,
                                velocidade=0.0,
                                area=float(v),
                                sentido=1,
                            )
                        )
                    estado, desde, candidato_subida = "ocupado", candidato_subida, None
            else:
                candidato_subida = None
        else:  # ocupado
            if v <= lim_baixo:
                if (i - desde) >= min_ocup:
                    estado, desde = "vazio", i
            # Enquanto ocupado, nada a registrar: o pacote já foi contado na
            # borda de entrada.

    if p.borda == "descida":
        eventos = _recontar_na_descida(s, tempos, lim_alto, lim_baixo,
                                       min_vazio, min_ocup)
    return ResultadoGate(eventos, s, tempos, nivel, lim_alto, lim_baixo)


def _recontar_na_descida(s, tempos, lim_alto, lim_baixo, min_vazio, min_ocup):
    """Variante que cronometra a saída do pacote, e não a entrada."""
    eventos: list[Cruzamento] = []
    estado = "ocupado" if s[0] >= lim_alto else "vazio"
    desde = 0
    for i in range(1, len(s)):
        v = s[i]
        if estado == "ocupado" and v <= lim_baixo:
            if (i - desde) >= min_ocup:
                t = _cruzamento_interpolado(s, tempos, i, lim_baixo)
                eventos.append(
                    Cruzamento(
                        id_pacote=len(eventos) + 1, id_track=len(eventos) + 1,
                        tempo=t, frame=int(i), cx=0.0, cy=0.0,
                        velocidade=0.0, area=float(v), sentido=-1,
                    )
                )
                estado, desde = "vazio", i
        elif estado == "vazio" and v >= lim_alto:
            if (i - desde) >= min_vazio:
                estado, desde = "ocupado", i
    return eventos

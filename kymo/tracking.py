"""Rastreamento de pacotes e detecção do instante exato do cruzamento.

Por que rastrear em vez de usar a linha como um sensor de presença: nesta
esteira os pacotes acumulam e param no ponto de medição por vários segundos,
e frequentemente encostam um no outro. Um sensor de ocupação registraria um
único bloco de 7 s onde passaram três pacotes. Rastreando cada blob, o
cruzamento é contado uma vez por objeto, mesmo que ele pare sobre a linha.

O instante do cruzamento é interpolado entre os dois frames vizinhos: a 29,97
fps um frame vale 33,4 ms, e o pedido era milissegundo. A interpolação linear
da distância assinada ao segmento dá precisão bem abaixo do frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


def lado_da_linha(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Distância assinada (em px) do ponto à reta que contém o segmento.

    O sinal indica de que lado o ponto está; o zero é exatamente a linha.
    """
    dx, dy = x2 - x1, y2 - y1
    norma = (dx * dx + dy * dy) ** 0.5
    if norma == 0:
        return 0.0
    return ((px - x1) * dy - (py - y1) * dx) / norma


def projecao_no_segmento(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """Posição do ponto ao longo do segmento, normalizada em 0..1.

    Serve para exigir que o cruzamento aconteça DENTRO do segmento desenhado,
    e não na reta infinita que o prolonga.
    """
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return 0.0
    return ((px - x1) * dx + (py - y1) * dy) / denom


@dataclass
class Deteccao:
    cx: float
    cy: float
    x: int
    y: int
    largura: int
    altura: int
    area: float


@dataclass
class Cruzamento:
    """Um pacote passando pela linha, com o instante interpolado."""

    id_pacote: int
    id_track: int
    tempo: float          # segundos no vídeo original
    frame: int
    cx: float             # ponto do cruzamento (coords de processamento)
    cy: float
    velocidade: float     # px/s de processamento, no momento do cruzamento
    area: float
    sentido: int          # +1 ou -1


@dataclass
class Track:
    id: int
    cx: float
    cy: float
    largura: float
    altura: float
    area: float
    vx: float = 0.0
    vy: float = 0.0
    idade: int = 1
    perdidos: int = 0
    ja_contou: bool = False
    lado: float | None = None
    tempo_anterior: float | None = None
    lado_anterior: float | None = None
    pos_anterior: tuple[float, float] | None = None
    cx0: float = 0.0
    cy0: float = 0.0
    historico: deque = field(default_factory=lambda: deque(maxlen=90))

    def prever(self, dt: float) -> tuple[float, float]:
        return self.cx + self.vx * dt, self.cy + self.vy * dt

    @property
    def deslocamento(self) -> float:
        return ((self.cx - self.cx0) ** 2 + (self.cy - self.cy0) ** 2) ** 0.5

    @property
    def velocidade(self) -> float:
        return (self.vx ** 2 + self.vy ** 2) ** 0.5


class Rastreador:
    """Rastreador por centroide com predição de velocidade e associação gulosa.

    A associação gulosa (menor custo primeiro) basta aqui: os pacotes são bem
    separados no plano da imagem e o movimento entre frames é pequeno, então
    o ganho de uma atribuição ótima não compensaria a dependência extra.
    """

    def __init__(
        self,
        distancia_max: float = 24.0,
        frames_para_perder: int = 20,
        deslocamento_min: float = 10.0,
        idade_min: int = 4,
        velocidade_max: float = 220.0,
        suavizacao_velocidade: float = 0.6,
    ):
        self.distancia_max = distancia_max
        self.frames_para_perder = frames_para_perder
        self.deslocamento_min = deslocamento_min
        self.idade_min = idade_min
        self.velocidade_max = velocidade_max
        self.alfa = suavizacao_velocidade
        self.tracks: dict[int, Track] = {}
        self._proximo_id = 1
        self._t_anterior: float | None = None

    def atualizar(self, deteccoes: list[Deteccao], tempo: float) -> list[Track]:
        dt = 1.0 / 30.0 if self._t_anterior is None else max(1e-6, tempo - self._t_anterior)
        self._t_anterior = tempo

        # Custo de cada par (track previsto, detecção); descarta pares distantes.
        pares: list[tuple[float, int, int]] = []
        ids = list(self.tracks)
        for i, tid in enumerate(ids):
            tr = self.tracks[tid]
            px, py = tr.prever(dt)
            for j, det in enumerate(deteccoes):
                d = ((px - det.cx) ** 2 + (py - det.cy) ** 2) ** 0.5
                if d <= self.distancia_max:
                    pares.append((d, i, j))
        pares.sort()

        tracks_usados: set[int] = set()
        dets_usadas: set[int] = set()
        for d, i, j in pares:
            if i in tracks_usados or j in dets_usadas:
                continue
            tracks_usados.add(i)
            dets_usadas.add(j)
            tr = self.tracks[ids[i]]
            det = deteccoes[j]
            vx_novo = (det.cx - tr.cx) / dt
            vy_novo = (det.cy - tr.cy) / dt
            tr.vx = self.alfa * tr.vx + (1 - self.alfa) * vx_novo
            tr.vy = self.alfa * tr.vy + (1 - self.alfa) * vy_novo
            tr.pos_anterior = (tr.cx, tr.cy)
            tr.cx, tr.cy = det.cx, det.cy
            tr.largura, tr.altura, tr.area = det.largura, det.altura, det.area
            tr.idade += 1
            tr.perdidos = 0
            tr.historico.append((tempo, det.cx, det.cy))

        # Tracks sem detecção: seguem por inércia até estourar o limite.
        for i, tid in enumerate(ids):
            if i in tracks_usados:
                continue
            tr = self.tracks[tid]
            tr.perdidos += 1
            if tr.perdidos > self.frames_para_perder:
                del self.tracks[tid]
            else:
                tr.pos_anterior = (tr.cx, tr.cy)
                tr.cx, tr.cy = tr.prever(dt)

        # Detecções órfãs abrem novos tracks.
        for j, det in enumerate(deteccoes):
            if j in dets_usadas:
                continue
            tr = Track(
                id=self._proximo_id, cx=det.cx, cy=det.cy,
                largura=det.largura, altura=det.altura, area=det.area,
                cx0=det.cx, cy0=det.cy,
            )
            tr.historico.append((tempo, det.cx, det.cy))
            self.tracks[self._proximo_id] = tr
            self._proximo_id += 1

        return list(self.tracks.values())

    def cruzamentos(
        self,
        linha: tuple[float, float, float, float],
        tempo: float,
        frame: int,
        sentido_exigido: str = "ambos",
        proximo_id_pacote: int = 1,
    ) -> list[Cruzamento]:
        """Detecta mudanças de lado desde o frame anterior, com interpolação."""
        x1, y1, x2, y2 = linha
        eventos: list[Cruzamento] = []

        for tr in self.tracks.values():
            lado_atual = lado_da_linha(tr.cx, tr.cy, x1, y1, x2, y2)
            lado_ant = tr.lado
            tr.lado = lado_atual

            if tr.ja_contou or lado_ant is None:
                if tr.tempo_anterior is None:
                    tr.tempo_anterior = tempo
                continue

            t_ant = tr.tempo_anterior if tr.tempo_anterior is not None else tempo
            tr.tempo_anterior = tempo

            # Precisa haver troca de sinal entre os dois frames.
            if lado_ant == 0 or lado_atual == 0 or (lado_ant > 0) == (lado_atual > 0):
                continue

            sentido = 1 if lado_atual > 0 else -1
            if sentido_exigido == "positivo" and sentido != 1:
                continue
            if sentido_exigido == "negativo" and sentido != -1:
                continue

            # Filtros contra ruído: o objeto precisa ter sido visto por alguns
            # frames e ter andado de verdade, não apenas tremido sobre a linha.
            if tr.idade < self.idade_min or tr.deslocamento < self.deslocamento_min:
                continue

            # Velocidade impossível = o rastreador pulou para outro objeto.
            if self.velocidade_max > 0 and tr.velocidade > self.velocidade_max:
                continue

            # Fração do intervalo em que a distância assinada passa por zero.
            fracao = lado_ant / (lado_ant - lado_atual)
            fracao = min(1.0, max(0.0, fracao))
            t_cruzamento = t_ant + (tempo - t_ant) * fracao

            ax, ay = tr.pos_anterior if tr.pos_anterior else (tr.cx, tr.cy)
            cx = ax + (tr.cx - ax) * fracao
            cy = ay + (tr.cy - ay) * fracao

            # O cruzamento tem de cair dentro do segmento desenhado.
            u = projecao_no_segmento(cx, cy, x1, y1, x2, y2)
            if not (-0.05 <= u <= 1.05):
                continue

            tr.ja_contou = True
            eventos.append(
                Cruzamento(
                    id_pacote=proximo_id_pacote + len(eventos),
                    id_track=tr.id,
                    tempo=t_cruzamento,
                    frame=frame,
                    cx=cx, cy=cy,
                    velocidade=tr.velocidade,
                    area=tr.area,
                    sentido=sentido,
                )
            )

        eventos.sort(key=lambda e: e.tempo)
        for i, ev in enumerate(eventos):
            ev.id_pacote = proximo_id_pacote + i
        return eventos

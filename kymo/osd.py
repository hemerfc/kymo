"""Renderização do OSD sobre o vídeo.

Mostra, quadro a quadro: o instante exato de cada pacote que cruza a linha,
a cadência (vazão instantânea e média), o intervalo desde o pacote anterior
com destaque para paradas, e um gráfico de vazão que avança com o vídeo.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from bisect import bisect_right
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .config import Configuracao
from .detect import DetectorPacotes, parametros_gate
from .gate import SensorGate, _suavizar
from .tracking import Cruzamento, Rastreador
from .video import LeitorFrames, formatar_tempo, sondar

FONTE = cv2.FONT_HERSHEY_DUPLEX
FONTE_MONO = cv2.FONT_HERSHEY_SIMPLEX

COR_LINHA = (90, 235, 90)
COR_FLASH = (60, 240, 255)
COR_TEXTO = (245, 245, 245)
COR_FRACA = (170, 170, 170)
COR_DESTAQUE = (80, 225, 255)
COR_ALERTA = (70, 90, 255)
COR_OK = (120, 230, 140)
COR_TRACK = (255, 170, 60)
COR_MELHOR = (205, 150, 255)   # janela de pico: distinta da media e da capacidade
COR_PAINEL = (18, 18, 20)


def _painel(img: np.ndarray, x: int, y: int, w: int, h: int, alfa: float = 0.78) -> None:
    """Retângulo escuro translúcido — mantém o texto legível sobre a cena."""
    x, y = max(0, x), max(0, y)
    x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x2 <= x or y2 <= y:
        return
    regiao = img[y:y2, x:x2]
    fundo = np.full_like(regiao, COR_PAINEL)
    cv2.addWeighted(fundo, alfa, regiao, 1 - alfa, 0, regiao)
    cv2.rectangle(img, (x, y), (x2 - 1, y2 - 1), (70, 70, 76), 1)


def _texto(img, txt, org, escala=0.6, cor=COR_TEXTO, espessura=1, fonte=FONTE):
    cv2.putText(img, txt, org, fonte, escala, (0, 0, 0), espessura + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, fonte, escala, cor, espessura, cv2.LINE_AA)


def _passo_bonito(topo: float) -> float:
    """Passo de grade legível para um eixo que vai de 0 a `topo`."""
    alvo = topo / 4.0
    for passo in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if passo >= alvo:
            return float(passo)
    return float(topo)


def _passo_tempo(span: float) -> float:
    """Espaçamento das marcas do eixo do tempo: ~6 a 10 marcas no trecho."""
    for passo in (10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600):
        if span / passo <= 10:
            return float(passo)
    return span / 8.0


# Espaçamento mínimo entre caixas, em centímetros: abaixo disso a passagem
# conta como erro de gap. Especificação da linha, não medida do vídeo — 8,89 cm
# são as 3,5 polegadas do equipamento. Hoje o total vem por argumento do
# `render`; medir isso no vídeo exige converter px para cm, o que pede a escala
# do plano da esteira e ainda não existe aqui.
GAP_MINIMO_CM = 8.89

# Janela do pico de vazão, em segundos. Um minuto porque é a unidade em que a
# operação fala ("quantos pacotes por minuto a linha faz no melhor momento").
JANELA_PICO = 60.0


def _melhor_janela(
    tempos: np.ndarray, t_inicio: float, t_fim: float, largura: float
) -> tuple[float, float, int] | None:
    """Janela de `largura` segundos com mais pacotes: `(vazao, inicio, n)`.

    Contagem em janela retangular, e não o pico da curva do gráfico. A curva
    usa kernel triangular, que é o certo para desenhar — pesos contínuos não
    serrilham — mas o valor que sai dele não é contável: ninguém confere 142,5
    no CSV. Já "139 pacotes entre 7,6 s e 67,6 s" qualquer um reaudita depois,
    que é o que o relatório precisa.

    As bordas só andam sobre instantes de evento: entre dois eventos a contagem
    da janela não muda, então varrer o tempo em passos finos só custaria mais
    para achar o mesmo máximo.

    Cuidado ao ler o número em trecho curto: com 67,6 s de vídeo a janela de um
    minuto desliza 7,6 s e cobre 98% dos eventos — ali "melhor minuto" e "média
    do período" medem quase a mesma coisa, e a diferença entre 139 e 126 é
    posição de janela, não um minuto de fato melhor. O campo só ganha sentido
    de pico quando o trecho é bem mais longo que a janela.
    """
    if len(tempos) == 0 or (t_fim - t_inicio) < largura:
        return None
    melhor: tuple[float, float, int] | None = None
    for inicio in np.concatenate(([t_inicio], tempos)):
        if inicio < t_inicio or inicio + largura > t_fim:
            continue
        n = int(((tempos >= inicio) & (tempos < inicio + largura)).sum())
        vazao = n * 60.0 / largura
        if melhor is None or n > melhor[2]:
            melhor = (vazao, float(inicio), n)
    return melhor


def _linha_tracejada(img, x1, y, x2, cor, tam: int = 7, vao: int = 5) -> None:
    x = x1
    while x < x2:
        cv2.line(img, (x, y), (min(x + tam, x2), y), cor, 1)
        x += tam + vao


class JanelaMini:
    """Segundo vídeo, alinhado no tempo, para desenhar como quadro no OSD.

    O MINI é outra câmera — outro ângulo, outra taxa de quadros e, no material
    deste projeto, câmera de mão que caminha pelo galpão. Nada nele pode ser
    derivado do vídeo principal, nem o instante em que começa: o alinhamento
    entra pronto, em `offset`, medido fora da análise.

    Avança em passo próprio porque as taxas diferem (60 fps contra 30). Guardar
    o último quadro lido e só avançar quando o tempo alvo passa dele mantém a
    leitura sequencial — dar seek a cada frame seria ordens de grandeza mais
    caro, e o ffmpeg já entrega os quadros em ordem.
    """

    def __init__(self, caminho: str, offset: float, largura: int):
        self.caminho = caminho
        self.offset = float(offset)
        self.info = sondar(caminho)
        escala = largura / self.info.largura
        # Começa no primeiro instante do MINI que o principal chega a mostrar.
        self.inicio = max(0.0, self.offset)
        self._leitor = LeitorFrames(caminho, info=self.info,
                                    inicio=self.inicio, escala=escala)
        self._iter = iter(self._leitor)
        self._quadro: np.ndarray | None = None
        self._t: float = -1e18
        self._fim = False
        self.largura = self._leitor.largura
        self.altura = self._leitor.altura

    def em(self, tempo: float) -> np.ndarray | None:
        """Quadro do MINI correspondente a `tempo` do vídeo principal."""
        alvo = tempo + self.offset
        if alvo < self.inicio - 1e-9:
            return None                      # o MINI ainda não começou
        if alvo > self.info.duracao:
            # O MINI acabou. Sem isto o último quadro dele ficaria congelado
            # até o fim do render, parecendo uma câmera travada em vez de um
            # vídeo mais curto — no TESTE 02 seriam 5,6 s de imagem parada.
            return None
        while not self._fim and self._t < alvo:
            try:
                _i, t, f = next(self._iter)
            except StopIteration:
                self._fim = True
                break
            self._quadro, self._t = f, t
        return self._quadro


class Renderizador:
    def __init__(
        self,
        cfg: Configuracao,
        eventos: list[Cruzamento],
        escala_saida: float | None = None,
        janela_vazao: float = 60.0,
        mostrar_tracks: bool = True,
        mostrar_grafico: bool = True,
        mostrar_faixa: bool = True,
        bin_grafico: float = 0.0,
        janela_pico: float = JANELA_PICO,
        erros_gap: int | None = None,
        mini: str | None = None,
        mini_offset: float = 0.0,
    ):
        self.cfg = cfg
        self.t_inicio = cfg.inicio
        self.t_fim = (
            cfg.inicio + cfg.duracao if cfg.duracao else sondar(cfg.video).duracao
        )
        # Só os eventos DENTRO do trecho renderizado. Sem este filtro, um
        # render de 3 min carregava os 475 eventos do CSV inteiro e a média
        # saía em 158/min — a contagem do vídeo todo dividida pelo trecho.
        self.eventos = [
            e for e in sorted(eventos, key=lambda e: e.tempo)
            if self.t_inicio <= e.tempo <= self.t_fim
        ]
        self.tempos = [e.tempo for e in self.eventos]
        self.escala_proc = cfg.deteccao.escala_processamento
        self.escala_saida = escala_saida or self.escala_proc
        self.fator = self.escala_saida / self.escala_proc
        self.janela_pico = janela_pico
        # None = não informado, e a linha nem aparece: um "0 erros" que ninguém
        # contou seria pior que a ausência do campo.
        self.erros_gap = erros_gap
        self.mini_caminho = mini
        self.mini_offset = mini_offset
        self._mini: JanelaMini | None = None
        self.mostrar_tracks = mostrar_tracks
        self.mostrar_grafico = mostrar_grafico
        self.mostrar_faixa = mostrar_faixa
        # Uma só janela móvel para o número do topo e para a curva do gráfico:
        # com janelas diferentes, o painel e o gráfico mostravam valores que não
        # se explicavam um pelo outro. `--grafico-bin` manda; `--janela-vazao`
        # continua aceito como fallback.
        span_total = max(1e-6, self.t_fim - self.t_inicio)
        if bin_grafico and bin_grafico > 0:
            self.janela_movel = float(bin_grafico)
        elif janela_vazao and janela_vazao > 0:
            self.janela_movel = float(janela_vazao)
        else:
            self.janela_movel = max(30.0, min(90.0, span_total / 10.0))
        self._graf: dict | None = None
        self.modo = cfg.deteccao.modo
        # ~4 s de sinal do sensor, para o traço tipo osciloscópio.
        self.hist_sensor: deque[float] = deque(maxlen=120)
        self.nivel_ref = 1.0
        self.lim_alto = 0.45
        self.lim_baixo = 0.25
        self.flash_ate = -1.0
        self.flash_texto = ""
        self.pos_flash: tuple[int, int] | None = None
        self._idx_evento = 0

    # -- geometria ---------------------------------------------------------
    def _linha_saida(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.cfg.linha.escalada(self.escala_saida)
        return int(x1), int(y1), int(x2), int(y2)

    def _p(self, x: float, y: float) -> tuple[int, int]:
        """Coordenada de processamento -> coordenada de saída."""
        return int(round(x * self.fator)), int(round(y * self.fator))

    # -- desenho -----------------------------------------------------------
    def _desenhar_faixa(self, img: np.ndarray) -> None:
        """Contorno da faixa monitorada — deixa claro o que entra na contagem."""
        if not self.mostrar_faixa or not self.cfg.roi.poligono:
            return
        pts = np.array(
            [[int(round(x * self.escala_saida)), int(round(y * self.escala_saida))]
             for x, y in self.cfg.roi.poligono],
            dtype=np.int32,
        )
        cv2.polylines(img, [pts], True, (150, 200, 150), 1, cv2.LINE_AA)

    def _desenhar_linha(self, img: np.ndarray, tempo: float) -> None:
        x1, y1, x2, y2 = self._linha_saida()
        ativo = tempo <= self.flash_ate
        cor = COR_FLASH if ativo else COR_LINHA
        espessura = 5 if ativo else 3
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), espessura + 3, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x2, y2), cor, espessura, cv2.LINE_AA)
        for p in ((x1, y1), (x2, y2)):
            cv2.circle(img, p, 6, cor, -1, cv2.LINE_AA)
        rot = self.cfg.linha.nome.upper()
        _texto(img, rot, (x1 + 8, y1 - 12), 0.55, cor, 1)

    def _desenhar_tracks(self, img: np.ndarray, tracks) -> None:
        if not self.mostrar_tracks:
            return
        for tr in tracks:
            if tr.perdidos > 0:
                continue
            w = int(tr.largura * self.fator)
            h = int(tr.altura * self.fator)
            cx, cy = self._p(tr.cx, tr.cy)
            x, y = cx - w // 2, cy - h // 2
            cor = COR_OK if tr.ja_contou else COR_TRACK
            cv2.rectangle(img, (x, y), (x + w, y + h), cor, 2)
            _texto(img, f"#{tr.id}", (x + 3, y - 6), 0.45, cor, 1, FONTE_MONO)
            # Rastro dos últimos frames: mostra o caminho e o sentido.
            pts = [self._p(hx, hy) for _, hx, hy in list(tr.historico)[-18:]]
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, cor, 1, cv2.LINE_AA)

    def _desenhar_flash(self, img: np.ndarray, tempo: float) -> None:
        if tempo > self.flash_ate or self.pos_flash is None:
            return
        restante = (self.flash_ate - tempo) / 0.8
        raio = int(18 + (1 - restante) * 46)
        cv2.circle(img, self.pos_flash, raio, COR_FLASH, 2, cv2.LINE_AA)
        cv2.circle(img, self.pos_flash, 6, COR_FLASH, -1, cv2.LINE_AA)
        px, py = self.pos_flash
        largura = int(11 * len(self.flash_texto) * 0.62) + 22
        _painel(img, px + 16, py - 44, largura, 34, 0.78)
        _texto(img, self.flash_texto, (px + 27, py - 20), 0.6, COR_FLASH, 1)

    def _desenhar_gate(self, img: np.ndarray, sensor: "SensorGate") -> None:
        """Contorno da janela onde a ocupação é medida."""
        pts = np.array(
            [[[int(round(cx * self.fator)), int(round(cy * self.fator))]
              for cx, cy in sensor.cantos]], dtype=np.int32
        )
        cv2.polylines(img, pts, True, (200, 240, 200), 1, cv2.LINE_AA)

    def _painel_sensor(self, img: np.ndarray) -> None:
        """Medidor + traço do sinal: mostra POR QUE cada pacote foi contado."""
        if not self.hist_sensor:
            return
        H, W = img.shape[:2]
        pw, ph = 268, 118
        x, y = W - pw - 18, 18
        _painel(img, x, y, pw, ph)
        _texto(img, "GATE", (x + 16, y + 24), 0.46, COR_FRACA, 1, FONTE_MONO)

        atual = self.hist_sensor[-1]
        ocupado = atual >= self.lim_alto
        cor = COR_DESTAQUE if ocupado else COR_OK
        _texto(img, "OCUPADO" if ocupado else "VAZIO", (x + pw - 92, y + 24),
               0.46, cor, 1, FONTE_MONO)

        # Traço do sinal, normalizado pelo nível de referência.
        gx, gy = x + 14, y + 36
        gw, gh = pw - 28, ph - 52
        cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (60, 60, 66), 1)
        topo = max(self.nivel_ref * 1.35, 1e-6)

        def py(v: float) -> int:
            return int(gy + gh - min(1.0, max(0.0, v / topo)) * gh)

        for lim, c in ((self.lim_alto, (90, 170, 250)), (self.lim_baixo, (90, 120, 90))):
            yy = py(lim)
            cv2.line(img, (gx + 1, yy), (gx + gw - 1, yy), c, 1)

        vals = list(self.hist_sensor)
        passo = gw / max(1, len(vals) - 1)
        pts = [(int(gx + i * passo), py(v)) for i, v in enumerate(vals)]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, cor, 1, cv2.LINE_AA)
        _texto(img, f"{atual / max(self.nivel_ref, 1e-6) * 100:3.0f}%",
               (gx + gw - 44, gy + gh - 5), 0.42, COR_FRACA, 1, FONTE_MONO)

    def _painel_principal(self, img: np.ndarray, tempo: float, total_ate: int) -> None:
        H, W = img.shape[:2]
        self._preparar_grafico()
        melhor = (self._graf or {}).get("melhor")

        # Cada linha é (valor, cor, rótulo). Montar a lista antes de desenhar
        # mantém largura, altura e posições coerentes: as linhas aparecem e
        # somem conforme o que existe para mostrar, e nenhuma medida fica
        # dependendo de um `if` repetido em três lugares.
        linhas: list[tuple[str, tuple[int, int, int], str]] = []

        # Mesmo valor que a curva do gráfico mostra neste instante — os dois são
        # lidos do mesmo array, então não podem divergir.
        movel = self._vazao_movel(tempo)
        linhas.append((f"{movel:.1f}/min", COR_TEXTO,
                       f"media {self.janela_movel:.0f}s"))

        if melhor:
            # Pico e projeção horária ficam aqui, e não no gráfico, porque são
            # números do trecho inteiro: não mudam com o instante, e no painel
            # quem assiste os lê sem procurar. A cor os liga à faixa desenhada
            # no gráfico, que é onde se vê QUANDO a janela cai.
            linhas.append((f"{melhor[0]:.1f}/min", COR_MELHOR,
                           f"melhor {self.janela_pico:.0f}s"))
            # Extrapolação da melhor janela para uma hora — a unidade em que o
            # alvo contratado costuma vir. Continua sendo projeção: pressupõe a
            # linha sustentando o melhor minuto por sessenta minutos seguidos,
            # o que nenhum trecho de um minuto demonstra. O rótulo já não diz
            # isso, então a ressalva tem de viver na aba Método do relatório.
            por_hora = f"{melhor[0] * 60.0:,.0f}/h".replace(",", ".")
            linhas.append((por_hora, COR_MELHOR, "throughput calculado"))

        if self.erros_gap is not None:
            cor_erro = COR_OK if self.erros_gap == 0 else COR_ALERTA
            linhas.append((f"{self.erros_gap}", cor_erro, "gap errors"))

        # Grade de duas colunas: valores alinhados à DIREITA, rótulos numa
        # coluna fixa. Antes o rótulo vinha logo depois do valor, e como a
        # largura do valor varia ("8.340/h" contra "1"), cada rótulo começava
        # num x diferente — quatro linhas, quatro margens.
        esc_val, esc_rot = 0.58, 0.46
        larg_val = max(cv2.getTextSize(v, FONTE, esc_val, 1)[0][0]
                       for v, _c, _r in linhas)
        larg_rot = max(cv2.getTextSize(r, FONTE, esc_rot, 1)[0][0]
                       for _v, _c, r in linhas)
        x_val_fim = 32 + larg_val
        x_rot = x_val_fim + 16

        cabecalho = 32 + cv2.getTextSize(f"{total_ate}", FONTE, 1.5, 2)[0][0] \
            + 12 + cv2.getTextSize("pacotes", FONTE, 0.5, 1)[0][0]
        pw = max(x_rot + larg_rot + 28, cabecalho + 28, int(W * 0.24))
        ph = 106 + 30 * len(linhas)
        _painel(img, 18, 18, pw, ph)

        # Cronômetro e contagem na mesma linha: são a identidade do frame, e
        # separá-los em duas linhas gastava 40px de painel sem ganhar leitura.
        _texto(img, formatar_tempo(tempo), (32, 58), 0.95, COR_TEXTO, 2)
        _texto(img, f"{total_ate}", (32, 108), 1.5, COR_DESTAQUE, 2)
        largura_num = cv2.getTextSize(f"{total_ate}", FONTE, 1.5, 2)[0][0]
        _texto(img, "pacotes", (32 + largura_num + 12, 108), 0.5, COR_FRACA, 1)

        for i, (valor, cor, rot) in enumerate(linhas):
            ly = 142 + i * 30
            larg = cv2.getTextSize(valor, FONTE, esc_val, 1)[0][0]
            _texto(img, valor, (x_val_fim - larg, ly), esc_val, cor, 1)
            _texto(img, rot, (x_rot, ly), esc_rot, COR_FRACA, 1)

    def _painel_mini(self, img: np.ndarray, tempo: float) -> None:
        """Quadro do segundo vídeo, encostado no canto inferior direito."""
        if self._mini is None:
            return
        quadro = self._mini.em(tempo)
        H, W = img.shape[:2]
        larg = self._mini.largura
        alt = self._mini.altura
        x = W - larg - 18
        y = H - alt - 18
        if quadro is None:
            # Antes do início do MINI (offset negativo) ou depois do fim dele:
            # a moldura fica, para o quadro não saltar de tamanho no meio.
            _painel(img, x, y, larg, alt, alfa=0.9)
            _texto(img, "MINI fora do trecho", (x + 14, y + alt // 2), 0.5,
                   COR_FRACA, 1, FONTE_MONO)
        else:
            recorte = quadro[:alt, :larg]
            img[y: y + recorte.shape[0], x: x + recorte.shape[1]] = recorte
        cv2.rectangle(img, (x, y), (x + larg, y + alt), (90, 90, 96), 1)

    def _preparar_grafico(self) -> None:
        """Calcula bins e referências uma única vez.

        Antes isto rodava a cada frame; num render de 34 mil frames era o
        mesmo laço sobre todos os eventos, 34 mil vezes, para um resultado que
        nunca muda. Só o cursor depende do instante.
        """
        if self._graf is not None:
            return
        span = max(1e-6, self.t_fim - self.t_inicio)
        # Cada barra agrega uma janela de tempo, nunca frames isolados: com
        # barras de meio segundo, um único pacote lia como 120/min e o eixo
        # ficava dominado por um pico que não corresponde a cadência alguma.
        # Janela móvel em vez de barras discretas: com barras estreitas a
        # contagem por bin fica quantizada (0, 1, 2 pacotes viram 0, 12, 24/min
        # e o traço serrilha); a janela deslizante dá uma curva contínua e
        # mostra a tendência real sem esconder as quedas.
        janela = self.janela_movel
        n = 480                                  # amostras da curva
        eixo = self.t_inicio + np.arange(n) * (span / max(1, n - 1))
        tempos = np.asarray(self.tempos)
        vazao = np.zeros(n)
        if len(tempos):
            # Kernel triangular, não janela retangular: contar eventos dentro de
            # uma janela dura devolve um inteiro, e com cadência de 2,1 s a
            # contagem alterna entre 7 e 8 por janela de 15 s — a curva vira
            # dente de serra por quantização, não por variação da esteira. Os
            # pesos contínuos do triângulo eliminam isso.
            h = janela / 2.0
            for i, tc in enumerate(eixo):
                d = np.abs(tempos - tc)
                dentro = d < h
                if not dentro.any():
                    continue
                pesos = 1.0 - d[dentro] / h
                # Normaliza pela parte do kernel que cai dentro do trecho: sem
                # isso as pontas afundariam por falta de dados, não por queda.
                a, b = max(tc - h, self.t_inicio), min(tc + h, self.t_fim)
                area = h - ((tc - h - self.t_inicio) ** 2 / (2 * h) if tc - h < self.t_inicio else 0.0)                          - ((tc + h - self.t_fim) ** 2 / (2 * h) if tc + h > self.t_fim else 0.0)
                if area <= 1e-6:
                    continue
                vazao[i] = pesos.sum() * 60.0 / area

        # Trechos em parada, para pintar ao fundo.
        limite = self.cfg.limite_gap_alerta
        faixas_parada = [
            (a, b) for a, b in zip(self.tempos, self.tempos[1:]) if b - a > limite
        ]

        # Cadência do slot -> capacidade nominal da esteira. É a referência que
        # separa "a linha está lenta" de "a linha está recebendo menos".
        cadencia = None
        if len(self.tempos) >= 4:
            gaps = np.diff(self.tempos)
            mediana = float(np.median(gaps))
            if mediana > 1e-6:
                cadencia = 60.0 / mediana

        media = len(self.tempos) / span * 60.0
        melhor = _melhor_janela(tempos, self.t_inicio, self.t_fim,
                                self.janela_pico)
        topo = max(float(vazao.max()) if n else 1.0, cadencia or 0.0,
                   melhor[0] if melhor else 0.0, 1.0) * 1.12

        self._graf = {
            "janela": janela, "n": n, "eixo": eixo, "vazao": vazao,
            "faixas_parada": faixas_parada, "cadencia": cadencia, "media": media,
            "melhor": melhor, "topo": topo, "span": span,
        }

    def _vazao_movel(self, tempo: float) -> float:
        """Valor da curva de média móvel no instante pedido."""
        self._preparar_grafico()
        g = self._graf
        if g is None or not len(g["eixo"]):
            return 0.0
        return float(np.interp(tempo, g["eixo"], g["vazao"]))

    def _grafico(self, img: np.ndarray, tempo: float) -> None:
        if not self.mostrar_grafico:
            return
        self._preparar_grafico()
        g = self._graf
        H, W = img.shape[:2]
        # Com o quadro do MINI ocupando o canto inferior direito, o gráfico
        # encolhe e encosta à esquerda; sem ele, segue centralizado como antes.
        if self._mini is not None:
            gw, gh = int(W * 0.52), 172
            x, y = 18, H - gh - 18
        else:
            gw, gh = int(W * 0.62), 172
            x, y = (W - gw) // 2, H - gh - 18
        _painel(img, x, y, gw, gh)

        # Área de plotagem, com margem à esquerda para os rótulos do eixo Y.
        px, py = x + 56, y + 32
        pw, ph = gw - 70, gh - 62
        topo = g["topo"]

        def alt(v: float) -> int:
            return int(py + ph - min(1.0, max(0.0, v / topo)) * ph)

        # A janela da média móvel já aparece no painel principal; repeti-la
        # aqui era a mesma informação em dois lugares. Fica só o que o gráfico
        # tem de próprio: a unidade e o que os tiques da base significam.
        _texto(img, "VAZAO  pacotes/min", (x + 14, y + 21), 0.44, COR_FRACA, 1,
               FONTE_MONO)
        _texto(img, "tiques = pacotes", (x + 208, y + 21), 0.38, COR_FRACA, 1,
               FONTE_MONO)

        # Legenda no cabeçalho: assim as linhas de referência ficam sem rótulo
        # em cima delas, onde colidiam com a grade.
        itens = [(f"media {g['media']:.1f}", (150, 230, 170))]
        if g["melhor"]:
            # Sem o valor: ele mora no painel principal agora. Aqui fica só a
            # chave de cor, que diz de quem é a faixa desenhada sobre a curva.
            itens.append((f"melhor {self.janela_pico:.0f}s", COR_MELHOR))
        if g["cadencia"]:
            itens.append((f"capacidade {g['cadencia']:.1f}", (120, 200, 255)))
        if g["faixas_parada"]:
            itens.append(("parada", COR_ALERTA))
        lx = x + gw - 22
        for texto_item, cor_item in reversed(itens):
            lw = cv2.getTextSize(texto_item, FONTE_MONO, 0.38, 1)[0][0]
            lx -= lw
            _texto(img, texto_item, (lx, y + 21), 0.38, cor_item, 1, FONTE_MONO)
            lx -= 10
            cv2.line(img, (lx - 12, y + 17), (lx, y + 17), cor_item, 2)
            lx -= 22

        # Grade horizontal com rótulos em passos redondos.
        passo = _passo_bonito(topo)
        v = passo
        while v < topo:
            yy = alt(v)
            cv2.line(img, (px, yy), (px + pw, yy), (56, 56, 62), 1)
            _texto(img, f"{v:g}", (x + 14, yy + 4), 0.38, (130, 130, 136), 1, FONTE_MONO)
            v += passo
        cv2.line(img, (px, py + ph), (px + pw, py + ph), (90, 90, 96), 1)

        def col(t: float) -> int:
            return px + int(min(1.0, max(0.0, (t - self.t_inicio) / g["span"])) * pw)

        # Paradas ao fundo, antes da curva.
        for a, b in g["faixas_parada"]:
            xa, xb = col(a), col(b)
            if xb - xa < 1:
                xb = xa + 1
            sub = img[py:py + ph, xa:xb]
            if sub.size:
                cv2.addWeighted(sub, 0.55, np.full_like(sub, COR_ALERTA), 0.45, 0, sub)

        # Curva de vazão: área preenchida no que já passou, linha fina adiante.
        cx = col(tempo)
        pts = [(col(t), alt(v)) for t, v in zip(g["eixo"], g["vazao"])]
        passado = [p for p in pts if p[0] <= cx]
        if len(passado) >= 2:
            poli = np.array(
                [[passado[0][0], py + ph]] + [list(p) for p in passado]
                + [[passado[-1][0], py + ph]], dtype=np.int32
            )
            sombra = img.copy()
            cv2.fillPoly(sombra, [poli], COR_DESTAQUE)
            cv2.addWeighted(sombra, 0.30, img, 0.70, 0, img)
        for a, b in zip(pts, pts[1:]):
            cor = COR_DESTAQUE if b[0] <= cx else (110, 110, 118)
            cv2.line(img, a, b, cor, 2 if b[0] <= cx else 1, cv2.LINE_AA)

        # Um tique por pacote, na base: o detalhe individual que a curva suaviza.
        base_tique = py + ph
        for t in self.tempos:
            tx = col(t)
            passou = t <= tempo
            cv2.line(img, (tx, base_tique - 1), (tx, base_tique - 7),
                     COR_OK if passou else (95, 95, 100), 1)

        # Referências: capacidade nominal e média do período.
        if g["cadencia"]:
            _linha_tracejada(img, px, alt(g["cadencia"]), px + pw, (120, 200, 255))
        _linha_tracejada(img, px, alt(g["media"]), px + pw, (150, 230, 170),
                         tam=4, vao=4)

        # A melhor janela vem com a extensão que ocupa no eixo, não só com o
        # valor: sem ver onde ela cai, não dá para julgar se o pico é um momento
        # da operação ou quase o trecho inteiro.
        if g["melhor"]:
            _, ini_melhor, _n = g["melhor"]
            mx0, mx1 = col(ini_melhor), col(ini_melhor + self.janela_pico)
            ym = alt(g["melhor"][0])
            _linha_tracejada(img, mx0, ym, mx1, COR_MELHOR, tam=6, vao=4)
            for mx in (mx0, mx1):
                cv2.line(img, (mx, ym - 4), (mx, ym + 4), COR_MELHOR, 1)

        # Cursor do instante atual.
        cx = px + int(min(1.0, max(0.0, (tempo - self.t_inicio) / g["span"])) * pw)
        cv2.line(img, (cx, py), (cx, py + ph), COR_FLASH, 1, cv2.LINE_AA)

        # Eixo do tempo, em marcas redondas.
        passo_t = _passo_tempo(g["span"])
        t_marca = self.t_inicio - (self.t_inicio % passo_t) + passo_t
        while t_marca < self.t_fim:
            mx = px + int((t_marca - self.t_inicio) / g["span"] * pw)
            cv2.line(img, (mx, py + ph), (mx, py + ph + 4), (110, 110, 116), 1)
            # mm:ss basta: as marcas caem em passos redondos de segundos, e
            # os milissegundos do formato completo eram três dígitos de zero
            # repetidos em cada rótulo do eixo.
            rot_t = formatar_tempo(t_marca).split(".")[0]
            larg_t = cv2.getTextSize(rot_t, FONTE_MONO, 0.36, 1)[0][0]
            _texto(img, rot_t, (mx - larg_t // 2, y + gh - 6), 0.36,
                   (130, 130, 136), 1, FONTE_MONO)
            t_marca += passo_t
        _texto(img, formatar_tempo(self.t_inicio).split(".")[0],
               (px - 24, y + gh - 6), 0.36,
               COR_FRACA, 1, FONTE_MONO)

    # -- laço principal ----------------------------------------------------
    def renderizar(
        self,
        saida: str | Path,
        crf: int = 20,
        preset: str = "medium",
        silencioso: bool = False,
    ) -> Path:
        info = sondar(self.cfg.video)
        leitor = LeitorFrames(
            self.cfg.video, info=info, inicio=self.cfg.inicio,
            duracao=self.cfg.duracao, escala=self.escala_proc,
        )
        usa_gate = self.modo == "gate"
        sensor = SensorGate(self.cfg, parametros_gate(self.cfg)) if usa_gate else None
        detector = None if usa_gate else DetectorPacotes(self.cfg)
        rastreador = None if usa_gate else Rastreador(
            distancia_max=self.cfg.deteccao.distancia_max_associacao,
            frames_para_perder=self.cfg.deteccao.frames_para_perder,
            deslocamento_min=self.cfg.deteccao.deslocamento_min_para_contar,
            idade_min=self.cfg.deteccao.idade_min_para_contar,
            velocidade_max=self.cfg.deteccao.velocidade_max_para_contar,
        )
        linha_proc = self.cfg.linha.escalada(self.escala_proc)
        if usa_gate:
            self._carregar_referencia_gate(saida)

        largura_saida = int(round(info.largura * self.escala_saida)) // 2 * 2
        altura_saida = int(round(info.altura * self.escala_saida)) // 2 * 2

        # O MINI é aberto aqui, e não no construtor: o tamanho do quadro sai da
        # largura de saída, que só é conhecida agora, e abrir um pipe de ffmpeg
        # que talvez nunca seja lido custa um processo à toa.
        if self.mini_caminho:
            self._mini = JanelaMini(
                self.mini_caminho, self.mini_offset,
                largura=int(largura_saida * 0.34) // 2 * 2,
            )
        escritor = _EscritorVideo(
            saida, largura_saida, altura_saida, info.fps_float, crf, preset
        )

        total = leitor.total_frames_estimado()
        t0 = time.time()
        n = 0
        try:
            for indice, tempo, frame in leitor:
                n = indice + 1
                tracks = []
                if usa_gate:
                    self.hist_sensor.append(sensor.medir(frame))
                else:
                    deteccoes, _ = detector.detectar(frame)
                    tracks = rastreador.atualizar(deteccoes, tempo)
                    rastreador.cruzamentos(
                        linha_proc, tempo, indice,
                        sentido_exigido=self.cfg.linha.sentido,
                        proximo_id_pacote=1,
                    )

                if self.fator != 1.0:
                    img = cv2.resize(frame, (largura_saida, altura_saida),
                                     interpolation=cv2.INTER_AREA)
                else:
                    img = frame.copy()

                # Consome os eventos do CSV que já aconteceram até aqui.
                while (self._idx_evento < len(self.eventos)
                       and self.eventos[self._idx_evento].tempo <= tempo):
                    ev = self.eventos[self._idx_evento]
                    self.flash_ate = tempo + 0.8
                    self.flash_texto = f"#{ev.id_pacote}  {formatar_tempo(ev.tempo)}"
                    self.pos_flash = (
                        self._p(ev.cx, ev.cy) if (ev.cx or ev.cy)
                        else self._centro_linha()
                    )
                    self._idx_evento += 1

                total_ate = bisect_right(self.tempos, tempo)

                self._desenhar_faixa(img)
                if usa_gate:
                    self._desenhar_gate(img, sensor)
                else:
                    self._desenhar_tracks(img, tracks)
                self._desenhar_linha(img, tempo)
                self._desenhar_flash(img, tempo)
                self._painel_principal(img, tempo, total_ate)
                self._painel_mini(img, tempo)
                if usa_gate:
                    self._painel_sensor(img)
                self._grafico(img, tempo)

                escritor.escrever(img)

                if not silencioso and indice % 30 == 0:
                    passado = time.time() - t0
                    fps = n / passado if passado else 0
                    resta = (total - n) / fps if fps else 0
                    print(f"\r  render {n}/{total} · {fps:5.1f} fps · "
                          f"resta {resta/60:4.1f} min", end="", file=sys.stderr,
                          flush=True)
        finally:
            escritor.fechar()
            if not silencioso:
                print(file=sys.stderr)
        return Path(saida)

    def _carregar_referencia_gate(self, saida) -> None:
        """Reusa os limiares que o `detect` calculou, para o OSD não divergir."""
        p = parametros_gate(self.cfg)
        caminho = Path(saida).with_name("sinal_gate.npy")
        nivel = None
        if caminho.exists():
            try:
                dados = np.load(caminho)
                sinal = dados[1]
                nivel = (
                    p.nivel_referencia
                    if p.nivel_referencia is not None
                    else float(np.percentile(sinal, p.referencia_percentil))
                )
            except Exception:
                nivel = None
        if nivel is None:
            nivel = p.nivel_referencia if p.nivel_referencia is not None else 0.25
        self.nivel_ref = max(nivel, 1e-6)
        self.lim_alto = self.nivel_ref * p.limiar_ocupado
        self.lim_baixo = self.nivel_ref * p.limiar_vazio

    def _centro_linha(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self._linha_saida()
        return (x1 + x2) // 2, (y1 + y2) // 2


class _EscritorVideo:
    """Escreve o vídeo por um pipe do ffmpeg (H.264 + faststart)."""

    def __init__(self, caminho, largura, altura, fps, crf=20, preset="medium"):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg não encontrado no PATH")
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [
                ffmpeg, "-v", "error", "-y", "-nostdin",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{largura}x{altura}", "-r", f"{fps:.6f}",
                "-i", "-",
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(caminho),
            ],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def escrever(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def fechar(self) -> None:
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        erro = self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else ""
        self.proc.wait()
        if self.proc.returncode not in (0, None) and erro.strip():
            print(f"\nffmpeg: {erro.strip()}", file=sys.stderr)

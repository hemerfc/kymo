"""Ferramenta interativa para marcar a linha de contagem sobre o vídeo.

Abre uma janela com o frame, você navega no tempo, clica dois pontos para
desenhar a linha por onde os pacotes passam, ajusta a ROI e calibra a cor do
papelão amostrando pixels. Salva tudo em um JSON reutilizável.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import (
    Configuracao,
    FaixaCor,
    LinhaContagem,
    ParametrosDeteccao,
    RegiaoInteresse,
)
from .detect import DetectorPacotes, parametros_gate
from .gate import SensorGate
from .video import formatar_tempo, ler_frame_unico, sondar

AJUDA = [
    "CLIQUE ESQ x2      desenha a linha de contagem",
    "G + cliques        faixa da esteira (poligono); ESPACO fecha, X apaga",
    "R + cliques x2     redefine a ROI (retangulo de busca)",
    "P + cliques x2     mede o comprimento tipico de um pacote",
    "CLIQUE DIR         amostra a cor do papelao (calibra LAB)",
    "SETA <- / ->       -1s / +1s        A / D    -10s / +10s",
    "W / S              -60s / +60s      HOME     inicio",
    "M                  alterna previa da mascara + janela do gate",
    "N                  alterna previa das deteccoes (caixas)",
    "F                  inverte o sentido valido do cruzamento",
    "C                  limpa a calibracao de cor",
    "ENTER              salva a configuracao        ESC / Q  sai",
]

VERDE = (0, 220, 0)
AMARELO = (0, 230, 255)
CIANO = (255, 220, 0)
MAGENTA = (255, 0, 220)
BRANCO = (255, 255, 255)


class _Estado:
    def __init__(self, info, escala_tela: float):
        self.escala_tela = escala_tela
        self.tempo = 0.0
        self.modo = "linha"          # linha | roi | comprimento
        self.pontos: list[tuple[int, int]] = []
        self.linha: LinhaContagem | None = None
        self.roi: RegiaoInteresse | None = None
        self.poligono: list[tuple[int, int]] = []      # em coords do original
        self.sentido = "ambos"
        self.comprimento_pacote = 0
        self.amostras_hsv: list[tuple[int, int, int]] = []
        self.mostrar_mascara = False
        self.mostrar_deteccoes = False
        self.mensagem = "Clique dois pontos para desenhar a linha de contagem."
        self.info = info

    def para_original(self, x: int, y: int) -> tuple[int, int]:
        return int(round(x / self.escala_tela)), int(round(y / self.escala_tela))

    def para_tela(self, x: float, y: float) -> tuple[int, int]:
        return int(round(x * self.escala_tela)), int(round(y * self.escala_tela))


def _faixa_das_amostras(amostras: list[tuple[int, int, int]]) -> FaixaCor:
    """Faixa LAB a partir dos pixels amostrados (valores já em L, a*, b* reais).

    Percentis em vez de mínimo/máximo: um clique que pegue de raspão a borda
    da caixa não deve esticar a faixa até o metal da esteira. O limite inferior
    de b* nunca desce abaixo de 1 — abaixo disso entra o cinza azulado dos
    roletes e a máscara vaza pela esteira toda.
    """
    arr = np.array(amostras, dtype=np.int16)
    ll, aa, bb = arr[:, 0], arr[:, 1], arr[:, 2]
    p = lambda a, q: int(np.percentile(a, q))
    return FaixaCor(
        modo="lab",
        b_min=max(1, p(bb, 10) - 1),
        b_max=min(127, p(bb, 98) + 12),
        a_min=p(aa, 5) - 8,
        a_max=p(aa, 95) + 10,
        l_min=max(20, p(ll, 5) - 25),
        l_max=252,
    )


def _config_parcial(est: _Estado, video: str) -> Configuracao | None:
    if est.linha is None:
        return None
    if len(est.poligono) >= 3:
        roi = RegiaoInteresse.do_poligono(
            [[x, y] for x, y in est.poligono], folga=8,
            limite=(est.info.largura, est.info.altura),
        )
    else:
        roi = est.roi or _roi_automatica(est)
    cor = _faixa_das_amostras(est.amostras_hsv) if len(est.amostras_hsv) >= 20 else FaixaCor()
    params = ParametrosDeteccao()
    if est.comprimento_pacote > 0:
        params.comprimento_pacote = int(
            round(est.comprimento_pacote * params.escala_processamento)
        )
    return Configuracao(video=video, linha=est.linha, roi=roi, cor=cor, deteccao=params)


def _roi_automatica(est: _Estado) -> RegiaoInteresse:
    """ROI padrão: uma faixa em volta da linha, alongada no sentido do fluxo.

    Buscar em toda a imagem convidaria falsos positivos (paletes, pilhas de
    papelão no fundo); uma faixa em volta da linha basta para rastrear o
    pacote chegando e saindo.
    """
    ln = est.linha
    assert ln is not None
    dx, dy = ln.x2 - ln.x1, ln.y2 - ln.y1
    comprimento = max(1.0, (dx * dx + dy * dy) ** 0.5)
    # Perpendicular à linha = direção do fluxo.
    ux, uy = dy / comprimento, -dx / comprimento
    alcance = comprimento * 2.0
    xs = [ln.x1, ln.x2, ln.x1 + ux * alcance, ln.x2 + ux * alcance,
          ln.x1 - ux * alcance, ln.x2 - ux * alcance]
    ys = [ln.y1, ln.y2, ln.y1 + uy * alcance, ln.y2 + uy * alcance,
          ln.y1 - uy * alcance, ln.y2 - uy * alcance]
    folga = comprimento * 0.25
    x0 = max(0, int(min(xs) - folga))
    y0 = max(0, int(min(ys) - folga))
    x1 = min(est.info.largura, int(max(xs) + folga))
    y1 = min(est.info.altura, int(max(ys) + folga))
    return RegiaoInteresse(x=x0, y=y0, largura=x1 - x0, altura=y1 - y0)


def _desenhar_painel(tela: np.ndarray, linhas: list[str], x: int, y: int,
                     cor=BRANCO, escala=0.46) -> None:
    altura_linha = int(20 * escala / 0.46)
    largura = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)[0][0]
                  for t in linhas) + 16
    fundo = tela[y - 14: y + altura_linha * len(linhas), x - 8: x + largura]
    if fundo.size:
        cv2.addWeighted(fundo, 0.25, np.zeros_like(fundo), 0.75, 0, fundo)
    for i, t in enumerate(linhas):
        cv2.putText(tela, t, (x, y + i * altura_linha),
                    cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 1, cv2.LINE_AA)


def _desenhar_overlay(tela: np.ndarray, est: _Estado) -> None:
    # Faixa poligonal da esteira
    if est.poligono:
        pts = np.array(
            [est.para_tela(x, y) for x, y in est.poligono], dtype=np.int32
        )
        fechado = est.modo != "poligono" and len(est.poligono) >= 3
        if len(pts) >= 3:
            sombra = tela.copy()
            cv2.fillPoly(sombra, [pts], (40, 130, 40))
            cv2.addWeighted(sombra, 0.22, tela, 0.78, 0, tela)
        cv2.polylines(tela, [pts], fechado, VERDE, 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            cv2.circle(tela, tuple(p), 4, VERDE, -1, cv2.LINE_AA)
            cv2.putText(tela, str(i + 1), (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, VERDE, 1, cv2.LINE_AA)

    # ROI retangular (só quando não há faixa desenhada)
    roi = None
    if len(est.poligono) < 3:
        roi = est.roi or (_roi_automatica(est) if est.linha else None)
    if roi:
        x, y = est.para_tela(roi.x, roi.y)
        x2, y2 = est.para_tela(roi.x + roi.largura, roi.y + roi.altura)
        cv2.rectangle(tela, (x, y), (x2, y2), CIANO, 1)
        cv2.putText(tela, "ROI", (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, CIANO, 1, cv2.LINE_AA)

    # Linha de contagem e seta do sentido positivo
    if est.linha:
        p1 = est.para_tela(est.linha.x1, est.linha.y1)
        p2 = est.para_tela(est.linha.x2, est.linha.y2)
        cv2.line(tela, p1, p2, VERDE, 3, cv2.LINE_AA)
        for p in (p1, p2):
            cv2.circle(tela, p, 5, VERDE, -1, cv2.LINE_AA)
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        n = max(1.0, (dx * dx + dy * dy) ** 0.5)
        # Sentido positivo do produto vetorial usado em tracking.lado_da_linha.
        ux, uy = dy / n, -dx / n
        if est.sentido != "negativo":
            cv2.arrowedLine(tela, (mx, my), (int(mx + ux * 55), int(my + uy * 55)),
                            AMARELO, 2, tipLength=0.3)
        if est.sentido != "positivo":
            cv2.arrowedLine(tela, (mx, my), (int(mx - ux * 55), int(my - uy * 55)),
                            AMARELO, 2, tipLength=0.3)

    # Pontos em construção
    for p in est.pontos:
        cv2.circle(tela, p, 6, MAGENTA, 2, cv2.LINE_AA)
    if len(est.pontos) == 1:
        cv2.circle(tela, est.pontos[0], 12, MAGENTA, 1, cv2.LINE_AA)


def rodar_setup(
    video: str,
    saida: str = "config.json",
    tempo_inicial: float = 0.0,
    escala_tela: float = 0.0,
) -> Configuracao | None:
    """Loop interativo. Devolve a configuração salva, ou None se cancelado."""
    info = sondar(video)
    print(f"Vídeo: {info.descricao()}")

    if escala_tela <= 0:
        # Cabe numa tela comum sem perder detalhe para clicar.
        escala_tela = min(0.42, 1600 / info.largura)

    est = _Estado(info, escala_tela)
    est.tempo = max(0.0, min(tempo_inicial, info.duracao - 0.1))

    janela = "Marcar linha de contagem  —  ENTER salva, ESC sai"
    cv2.namedWindow(janela, cv2.WINDOW_AUTOSIZE)

    def ao_clicar(evento, x, y, flags, _):
        if evento == cv2.EVENT_RBUTTONDOWN:
            _amostrar_cor(est, x, y)
            return
        if evento != cv2.EVENT_LBUTTONDOWN:
            return
        if est.modo == "poligono":
            est.poligono.append(est.para_original(x, y))
            est.mensagem = (
                f"faixa: {len(est.poligono)} vertices · ESPACO fecha, X apaga o ultimo"
            )
            return
        est.pontos.append((x, y))
        if len(est.pontos) < 2:
            est.mensagem = "Agora clique o segundo ponto."
            return
        (ax, ay), (bx, by) = est.pontos[0], est.pontos[1]
        oa, ob = est.para_original(ax, ay), est.para_original(bx, by)
        if est.modo == "linha":
            est.linha = LinhaContagem(oa[0], oa[1], ob[0], ob[1], sentido=est.sentido)
            est.mensagem = (
                f"Linha: ({oa[0]},{oa[1]}) -> ({ob[0]},{ob[1]})  "
                f"· {est.linha.comprimento():.0f} px"
            )
        elif est.modo == "roi":
            x0, y0 = min(oa[0], ob[0]), min(oa[1], ob[1])
            est.roi = RegiaoInteresse(x0, y0, abs(ob[0] - oa[0]), abs(ob[1] - oa[1]))
            est.mensagem = f"ROI: {est.roi.largura}x{est.roi.altura} em ({x0},{y0})"
            est.modo = "linha"
        elif est.modo == "comprimento":
            comp = ((ob[0] - oa[0]) ** 2 + (ob[1] - oa[1]) ** 2) ** 0.5
            est.comprimento_pacote = int(round(comp))
            est.mensagem = f"Comprimento do pacote: {est.comprimento_pacote} px (original)"
            est.modo = "linha"
        est.pontos.clear()

    cv2.setMouseCallback(janela, ao_clicar)

    frame_cache: dict[float, np.ndarray] = {}
    frame_atual: np.ndarray | None = None
    tempo_carregado = None

    while True:
        if tempo_carregado != est.tempo:
            if est.tempo in frame_cache:
                frame_atual = frame_cache[est.tempo]
            else:
                frame_atual = ler_frame_unico(video, est.tempo, escala_tela)
                if len(frame_cache) > 24:
                    frame_cache.clear()
                if frame_atual is not None:
                    frame_cache[est.tempo] = frame_atual
            tempo_carregado = est.tempo
        if frame_atual is None:
            est.tempo = max(0.0, est.tempo - 1.0)
            tempo_carregado = None
            continue

        est.frame_atual = frame_atual
        tela = frame_atual.copy()

        if (est.mostrar_mascara or est.mostrar_deteccoes) and est.linha:
            _previa_deteccao(tela, est, video)

        _desenhar_overlay(tela, est)

        cabecalho = [
            f"t = {formatar_tempo(est.tempo)}   ({est.tempo:.2f}s de {formatar_tempo(info.duracao)})",
            f"modo: {est.modo}   sentido: {est.sentido}   "
            f"cor: {len(est.amostras_hsv)} amostras   "
            f"pacote: {est.comprimento_pacote or '-'} px",
            est.mensagem,
        ]
        _desenhar_painel(tela, cabecalho, 14, 26, AMARELO, 0.52)
        _desenhar_painel(tela, AJUDA, 14, tela.shape[0] - 20 * len(AJUDA) - 6,
                         BRANCO, 0.42)

        cv2.imshow(janela, tela)
        tecla = cv2.waitKey(20) & 0xFFFFFF

        if tecla in (27, ord("q"), ord("Q")):
            cv2.destroyAllWindows()
            print("Cancelado — nada foi salvo.")
            return None
        if tecla in (13, 10):
            cfg = _config_parcial(est, video)
            if cfg is None:
                est.mensagem = "Desenhe a linha de contagem antes de salvar."
                continue
            cfg.linha.sentido = est.sentido
            cfg.salvar(saida)
            cv2.destroyAllWindows()
            print(f"\nConfiguração salva em {saida}")
            print(f"  linha .......... {cfg.linha.como_tupla()} sentido={cfg.linha.sentido}")
            print(f"  ROI ............ x={cfg.roi.x} y={cfg.roi.y} "
                  f"{cfg.roi.largura}x{cfg.roi.altura}"
                  + (f"  (faixa de {len(cfg.roi.poligono)} vertices)"
                     if cfg.roi.poligono else "  (retangulo)"))
            print(f"  cor ({cfg.cor.modo}) ..... b* {cfg.cor.b_min}..{cfg.cor.b_max}  "
                  f"a* {cfg.cor.a_min}..{cfg.cor.a_max}  L {cfg.cor.l_min}..{cfg.cor.l_max}")
            print(f"  comprimento .... {cfg.deteccao.comprimento_pacote} px "
                  f"(escala de processamento)")
            return cfg

        _tratar_navegacao(est, tecla, info)

    return None


def _tratar_navegacao(est: _Estado, tecla: int, info) -> None:
    passos = {
        ord("a"): -10.0, ord("A"): -10.0, ord("d"): 10.0, ord("D"): 10.0,
        ord("w"): -60.0, ord("W"): -60.0, ord("s"): 60.0, ord("S"): 60.0,
        2: -1.0, 3: 1.0,                      # setas no backend do macOS
        63234: -1.0, 63235: 1.0,
        81: -1.0, 83: 1.0,                    # setas no backend GTK/Qt
    }
    if tecla in passos:
        est.tempo = min(max(0.0, est.tempo + passos[tecla]), info.duracao - 0.1)
        return
    if tecla in (ord("g"), ord("G")):
        if est.modo == "poligono":
            est.modo = "linha"
            est.mensagem = "faixa mantida; de volta ao modo linha"
        else:
            est.modo, est.pontos = "poligono", []
            est.mensagem = (
                "Clique os vertices da faixa acompanhando a esteira alvo. "
                "ESPACO fecha."
            )
        return
    if tecla == 32:                      # espaço fecha a faixa
        if est.modo == "poligono" and len(est.poligono) >= 3:
            est.modo = "linha"
            est.mensagem = f"faixa fechada com {len(est.poligono)} vertices"
        elif est.modo == "poligono":
            est.mensagem = "a faixa precisa de pelo menos 3 vertices"
        return
    if tecla in (ord("x"), ord("X")):
        if est.poligono:
            est.poligono.pop()
            est.mensagem = f"vertice removido ({len(est.poligono)} restantes)"
        return
    if tecla in (ord("r"), ord("R")):
        est.modo, est.pontos = "roi", []
        est.mensagem = "Clique dois cantos opostos da ROI."
    elif tecla in (ord("p"), ord("P")):
        est.modo, est.pontos = "comprimento", []
        est.mensagem = "Clique as duas pontas de UM pacote, no sentido do fluxo."
    elif tecla in (ord("m"), ord("M")):
        est.mostrar_mascara = not est.mostrar_mascara
        est.mensagem = f"prévia da máscara: {'on' if est.mostrar_mascara else 'off'}"
    elif tecla in (ord("n"), ord("N")):
        est.mostrar_deteccoes = not est.mostrar_deteccoes
        est.mensagem = f"prévia das detecções: {'on' if est.mostrar_deteccoes else 'off'}"
    elif tecla in (ord("f"), ord("F")):
        ordem = {"ambos": "positivo", "positivo": "negativo", "negativo": "ambos"}
        est.sentido = ordem[est.sentido]
        if est.linha:
            est.linha.sentido = est.sentido
        est.mensagem = f"sentido válido do cruzamento: {est.sentido}"
    elif tecla in (ord("c"), ord("C")):
        est.amostras_hsv.clear()
        est.mensagem = "calibração de cor limpa (volta ao padrão do papelão)"
    elif tecla == 63273 or tecla == 80:       # Home
        est.tempo = 0.0


def _amostrar_cor(est: _Estado, x: int, y: int) -> None:
    frame = getattr(est, "frame_atual", None)
    if frame is None:
        return
    r = 5
    h, w = frame.shape[:2]
    recorte = frame[max(0, y - r):min(h, y + r), max(0, x - r):min(w, x + r)]
    if recorte.size == 0:
        return
    lab = cv2.cvtColor(recorte, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.int16)
    lab[:, 1] -= 128
    lab[:, 2] -= 128
    est.amostras_hsv.extend(map(tuple, lab))
    faixa = _faixa_das_amostras(est.amostras_hsv)
    est.mensagem = (
        f"cor amostrada ({len(est.amostras_hsv)} px) · "
        f"b* {faixa.b_min}..{faixa.b_max}  a* {faixa.a_min}..{faixa.a_max}  "
        f"L>{faixa.l_min}"
    )


def _previa_deteccao(tela: np.ndarray, est: _Estado, video: str) -> None:
    """Roda o detector no frame exibido, para conferir a calibração na hora."""
    cfg = _config_parcial(est, video)
    if cfg is None:
        return
    # A prévia opera na própria escala da tela; sem histórico de movimento não
    # há como aplicar a subtração de fundo em um frame isolado.
    cfg.deteccao.escala_processamento = est.escala_tela
    cfg.deteccao.usar_subtracao_fundo = False
    fator = est.escala_tela / 0.5
    cfg.deteccao.area_min = max(120, int(cfg.deteccao.area_min * fator * fator))
    if cfg.deteccao.comprimento_pacote:
        cfg.deteccao.comprimento_pacote = int(est.comprimento_pacote * est.escala_tela)

    detector = DetectorPacotes(cfg)
    deteccoes, mascara = detector.detectar(getattr(est, "frame_atual"))

    if est.mostrar_mascara:
        rx, ry, rw, rh = detector.roi
        h, w = tela.shape[:2]
        rx, ry = max(0, rx), max(0, ry)
        alvo = tela[ry:min(h, ry + rh), rx:min(w, rx + rw)]
        m = mascara[: alvo.shape[0], : alvo.shape[1]]
        colorida = np.zeros_like(alvo)
        colorida[..., 1] = m
        cv2.addWeighted(alvo, 0.6, colorida, 0.4, 0, alvo)

    # Janela do gate e ocupação atual: é a leitura que decide a contagem.
    try:
        cfg_gate = _config_parcial(est, video)
        cfg_gate.deteccao.escala_processamento = est.escala_tela
        sensor = SensorGate(cfg_gate, parametros_gate(cfg_gate))
        if sensor.n_pixels > 0:
            pts = np.array(
                [[[int(round(cx)), int(round(cy))] for cx, cy in sensor.cantos]],
                dtype=np.int32,
            )
            cv2.polylines(tela, pts, True, AMARELO, 2, cv2.LINE_AA)
            ocup = sensor.medir(getattr(est, "frame_atual"))
            cv2.putText(tela, f"gate: {ocup * 100:.0f}% papelao",
                        (int(pts[0][:, 0].mean()) - 60, int(pts[0][:, 1].min()) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMARELO, 1, cv2.LINE_AA)
    except Exception:
        pass

    if est.mostrar_deteccoes:
        for d in deteccoes:
            cv2.rectangle(tela, (d.x, d.y), (d.x + d.largura, d.y + d.altura),
                          MAGENTA, 2)
            cv2.circle(tela, (int(d.cx), int(d.cy)), 4, MAGENTA, -1)
        cv2.putText(tela, f"{len(deteccoes)} deteccoes",
                    (14, tela.shape[0] - 20 * len(AJUDA) - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAGENTA, 1, cv2.LINE_AA)

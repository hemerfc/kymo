"""Segmentação dos pacotes e passagem de detecção sobre o vídeo."""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .config import Configuracao
from .cor import mascara_cor
from .gate import ParametrosGate, ResultadoGate, SensorGate, eventos_do_sinal
from .tracking import Cruzamento, Deteccao, Rastreador
from .video import LeitorFrames, decompor_tempo, formatar_tempo, sondar


class DetectorPacotes:
    """Encontra pacotes de papelão dentro da ROI.

    Duas evidências combinadas, e ambas são necessárias:
      1. cor — no LAB o papelão é amarelado e o metal dos roletes é azulado,
         o que separa pacote de esteira de forma limpa;
      2. movimento — cor NÃO distingue papelão de madeira: paletes têm b*
         praticamente igual ao das caixas. É a subtração de fundo que descarta
         o que faz parte do cenário parado.

    O ponto delicado é a fila de caixas encostadas, que a segmentação por cor
    entrega como um blob único. `_dividir` corta esses blobs pelo comprimento
    típico de um pacote, ao longo do eixo do fluxo.
    """

    def __init__(self, cfg: Configuracao):
        self.cfg = cfg
        p = cfg.deteccao
        self.escala = p.escala_processamento
        self.roi = cfg.roi.escalada(self.escala)
        self._mascara_roi: np.ndarray | None = None
        self._poligono = cfg.roi.poligono_escalado(self.escala)
        self._k_abertura = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.abertura, p.abertura))
            if p.abertura > 1 else None
        )
        self._k_fechamento = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.fechamento, p.fechamento))
            if p.fechamento > 1 else None
        )
        self._fundo = (
            cv2.createBackgroundSubtractorMOG2(
                history=p.historico_fundo, varThreshold=p.limiar_fundo,
                detectShadows=False,
            )
            if p.usar_subtracao_fundo else None
        )
        # Guarda por quantos frames recentes cada pixel se moveu: um pacote que
        # para sobre a linha continua sendo um pacote, mesmo sem movimento agora.
        self._memoria_movimento: np.ndarray | None = None
        self._decaimento = p.decaimento_movimento
        self._limiar_memoria = p.limiar_memoria
        self._k_memoria = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (p.dilatacao_memoria, p.dilatacao_memoria)
        )

        # Eixo do fluxo = perpendicular à linha de contagem.
        x1, y1, x2, y2 = cfg.linha.escalada(self.escala)
        dx, dy = x2 - x1, y2 - y1
        norma = (dx * dx + dy * dy) ** 0.5 or 1.0
        self.eixo_fluxo = (dy / norma, -dx / norma)

    def _recortar(self, frame: np.ndarray) -> np.ndarray:
        rx, ry, rw, rh = self.roi
        h, w = frame.shape[:2]
        rx, ry = max(0, rx), max(0, ry)
        return frame[ry:min(h, ry + rh), rx:min(w, rx + rw)]

    def mascara(self, frame: np.ndarray) -> np.ndarray:
        recorte = self._recortar(frame)
        m = mascara_cor(self.cfg.cor, recorte)

        if self._fundo is not None:
            fg = self._fundo.apply(recorte)
            fg = cv2.medianBlur(fg, 5)
            atual = (fg > 0).astype(np.float32)
            if self._memoria_movimento is None or self._memoria_movimento.shape != atual.shape:
                self._memoria_movimento = atual.copy()
            else:
                self._memoria_movimento *= self._decaimento
                np.maximum(self._memoria_movimento, atual, out=self._memoria_movimento)
            # Dilata a memória: a caixa se move, mas seu interior uniforme
            # gera pouco sinal de fundo — a vizinhança cobre isso.
            mem = cv2.dilate(
                (self._memoria_movimento > self._limiar_memoria).astype(np.uint8) * 255,
                self._k_memoria,
            )
            m = cv2.bitwise_and(m, mem)

        if self._k_abertura is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._k_abertura)
        if self._k_fechamento is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._k_fechamento)

        # A faixa poligonal entra por último: aplicada antes da morfologia, o
        # fechamento reconstruiria material que acabou de ser removido na borda.
        if self._poligono is not None:
            if self._mascara_roi is None or self._mascara_roi.shape != m.shape:
                self._mascara_roi = np.zeros(m.shape, np.uint8)
                cv2.fillPoly(self._mascara_roi, self._poligono, 255)
            m = cv2.bitwise_and(m, self._mascara_roi)
        return m

    def detectar(self, frame: np.ndarray) -> tuple[list[Deteccao], np.ndarray]:
        m = self.mascara(frame)
        rx, ry = self.roi[0], self.roi[1]
        p = self.cfg.deteccao

        n, etiquetas, stats, centroides = cv2.connectedComponentsWithStats(m, 8)
        deteccoes: list[Deteccao] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < p.area_min or area > p.area_max:
                continue
            cx, cy = centroides[i]
            base = Deteccao(
                cx=cx + rx, cy=cy + ry,
                x=x + rx, y=y + ry, largura=w, altura=h, area=float(area),
            )
            deteccoes.extend(self._dividir(base, etiquetas == i, rx, ry))
        return deteccoes, m

    def _dividir(
        self, det: Deteccao, mascara_blob: np.ndarray, rx: int, ry: int
    ) -> list[Deteccao]:
        """Divide um blob que é claramente uma fila de pacotes encostados."""
        p = self.cfg.deteccao
        L = p.comprimento_pacote
        if L <= 0:
            return [det]

        ys, xs = np.nonzero(mascara_blob)
        if len(xs) == 0:
            return [det]
        ux, uy = self.eixo_fluxo
        proj = xs * ux + ys * uy
        extensao = float(proj.max() - proj.min())
        if extensao <= L * p.tolerancia_divisao:
            return [det]

        partes = max(2, int(round(extensao / L)))
        bordas = np.linspace(proj.min(), proj.max(), partes + 1)
        saida: list[Deteccao] = []
        for k in range(partes):
            sel = (proj >= bordas[k]) & (proj <= bordas[k + 1])
            if sel.sum() < p.area_min * 0.5:
                continue
            sx, sy = xs[sel], ys[sel]
            saida.append(
                Deteccao(
                    cx=float(sx.mean()) + rx, cy=float(sy.mean()) + ry,
                    x=int(sx.min()) + rx, y=int(sy.min()) + ry,
                    largura=int(sx.max() - sx.min() + 1),
                    altura=int(sy.max() - sy.min() + 1),
                    area=float(sel.sum()),
                )
            )
        return saida or [det]


def parametros_gate(cfg: Configuracao) -> ParametrosGate:
    d = cfg.deteccao
    return ParametrosGate(
        largura=d.gate_largura,
        suavizacao_frames=d.gate_suavizacao,
        limiar_ocupado=d.gate_limiar_ocupado,
        limiar_vazio=d.gate_limiar_vazio,
        min_vazio_s=d.gate_min_vazio_s,
        min_ocupado_s=d.gate_min_ocupado_s,
        referencia_percentil=d.gate_referencia_percentil,
        nivel_referencia=d.gate_nivel_referencia,
        borda=d.gate_borda,
    )


def executar_deteccao_gate(
    cfg: Configuracao,
    saida_csv: str | Path,
    saida_json: str | Path | None = None,
    silencioso: bool = False,
) -> tuple[list[Cruzamento], ResultadoGate]:
    """Mede a ocupação do gate em todo o trecho e depois extrai os eventos."""
    info = sondar(cfg.video)
    leitor = LeitorFrames(
        cfg.video, info=info, inicio=cfg.inicio, duracao=cfg.duracao,
        escala=cfg.deteccao.escala_processamento,
    )
    sensor = SensorGate(cfg, parametros_gate(cfg))
    if sensor.n_pixels == 0:
        raise ValueError(
            "A janela do gate ficou vazia. A linha de contagem provavelmente "
            "está fora da faixa da esteira — refaça o setup."
        )

    sinal: list[float] = []
    tempos: list[float] = []
    total = leitor.total_frames_estimado()
    t0 = time.time()

    for indice, tempo, frame in leitor:
        sinal.append(sensor.medir(frame))
        tempos.append(tempo)
        if not silencioso and indice % 120 == 0:
            passado = time.time() - t0
            fps = (indice + 1) / passado if passado > 0 else 0
            resta = (total - indice - 1) / fps if fps > 0 else 0
            print(f"\r  {indice + 1}/{total} frames · {fps:5.1f} fps · "
                  f"resta {resta / 60:4.1f} min", end="", file=sys.stderr, flush=True)
    if not silencioso:
        print(file=sys.stderr)

    resultado = eventos_do_sinal(
        sinal, tempos, parametros_gate(cfg), fps=info.fps_float
    )
    gravar_csv(resultado.eventos, saida_csv)
    if saida_json:
        gravar_json(resultado.eventos, cfg, saida_json, len(sinal))
    # O sinal vai para o disco: o OSD desenha o traço da fotocélula sem ter
    # de reprocessar o vídeo, e dá para conferir os limiares depois.
    np.save(Path(saida_csv).with_name("sinal_gate.npy"),
            np.vstack([resultado.tempos, resultado.sinal]))
    return resultado.eventos, resultado


def executar_deteccao(
    cfg: Configuracao,
    saida_csv: str | Path,
    saida_json: str | Path | None = None,
    silencioso: bool = False,
) -> list[Cruzamento]:
    """Percorre o trecho configurado e grava os cruzamentos com timestamps."""
    if cfg.deteccao.modo == "gate":
        eventos, _ = executar_deteccao_gate(cfg, saida_csv, saida_json, silencioso)
        return eventos
    info = sondar(cfg.video)
    leitor = LeitorFrames(
        cfg.video, info=info, inicio=cfg.inicio, duracao=cfg.duracao,
        escala=cfg.deteccao.escala_processamento,
    )
    detector = DetectorPacotes(cfg)
    rastreador = Rastreador(
        distancia_max=cfg.deteccao.distancia_max_associacao,
        frames_para_perder=cfg.deteccao.frames_para_perder,
        deslocamento_min=cfg.deteccao.deslocamento_min_para_contar,
        idade_min=cfg.deteccao.idade_min_para_contar,
        velocidade_max=cfg.deteccao.velocidade_max_para_contar,
    )
    linha = cfg.linha.escalada(cfg.deteccao.escala_processamento)

    eventos: list[Cruzamento] = []
    total = leitor.total_frames_estimado()
    inicio_relogio = time.time()
    n_frames = 0

    for indice, tempo, frame in leitor:
        n_frames = indice + 1
        deteccoes, _ = detector.detectar(frame)
        rastreador.atualizar(deteccoes, tempo)
        novos = rastreador.cruzamentos(
            linha, tempo, indice,
            sentido_exigido=cfg.linha.sentido,
            proximo_id_pacote=len(eventos) + 1,
        )
        eventos.extend(novos)

        if not silencioso and indice % 60 == 0:
            passado = time.time() - inicio_relogio
            fps_proc = n_frames / passado if passado > 0 else 0
            restante = (total - n_frames) / fps_proc if fps_proc > 0 else 0
            print(
                f"\r  {n_frames}/{total} frames · {fps_proc:5.1f} fps · "
                f"pacotes: {len(eventos):4d} · resta {restante/60:4.1f} min",
                end="", file=sys.stderr, flush=True,
            )

    if not silencioso:
        print(file=sys.stderr)

    gravar_csv(eventos, saida_csv)
    if saida_json:
        gravar_json(eventos, cfg, saida_json, n_frames)
    return eventos


def gravar_csv(eventos: list[Cruzamento], caminho: str | Path) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id_pacote", "minuto", "segundo", "milissegundo", "tempo_mm_ss_mmm",
            "tempo_segundos", "frame", "gap_anterior_s", "velocidade_px_s",
            "ocupacao_ou_area", "sentido", "id_track",
        ])
        anterior = None
        for ev in eventos:
            minuto, segundo, ms = decompor_tempo(ev.tempo)
            gap = "" if anterior is None else f"{ev.tempo - anterior:.3f}"
            w.writerow([
                ev.id_pacote, minuto, segundo, ms, formatar_tempo(ev.tempo),
                f"{ev.tempo:.3f}", ev.frame, gap, f"{ev.velocidade:.1f}",
                f"{ev.area:.4f}", ev.sentido, ev.id_track,
            ])
            anterior = ev.tempo


def gravar_json(
    eventos: list[Cruzamento], cfg: Configuracao, caminho: str | Path, n_frames: int
) -> None:
    Path(caminho).write_text(
        json.dumps(
            {
                "video": cfg.video,
                "trecho": {"inicio": cfg.inicio, "duracao": cfg.duracao},
                "frames_processados": n_frames,
                "total_pacotes": len(eventos),
                "eventos": [
                    {**asdict(ev), "tempo_formatado": formatar_tempo(ev.tempo)}
                    for ev in eventos
                ],
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def carregar_csv(caminho: str | Path) -> list[Cruzamento]:
    """Recarrega os eventos gravados (usado pelo render e pelo report)."""
    eventos: list[Cruzamento] = []
    with Path(caminho).open(encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            eventos.append(
                Cruzamento(
                    id_pacote=int(linha["id_pacote"]),
                    id_track=int(linha["id_track"]),
                    tempo=float(linha["tempo_segundos"]),
                    frame=int(linha["frame"]),
                    cx=0.0, cy=0.0,
                    velocidade=float(linha["velocidade_px_s"] or 0),
                    area=float(
                        linha.get("ocupacao_ou_area") or linha.get("area_px") or 0
                    ),
                    sentido=int(linha["sentido"]),
                )
            )
    return eventos

"""Leitura de frames via ffmpeg com timestamps exatos.

Decodificar 4K/HEVC pelo `cv2.VideoCapture` é lento e o seek é impreciso.
Aqui o ffmpeg faz o trabalho pesado (seek, decode, escala) e entrega frames
crus por um pipe, enquanto o `ffprobe` fornece o PTS real do primeiro frame
do intervalo — é o que garante o milissegundo correto quando se processa um
trecho no meio do vídeo.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction


class FFmpegAusente(RuntimeError):
    pass


def _bin(nome: str) -> str:
    caminho = shutil.which(nome)
    if not caminho:
        raise FFmpegAusente(
            f"'{nome}' não encontrado no PATH. Instale com: brew install ffmpeg"
        )
    return caminho


@dataclass(frozen=True)
class InfoVideo:
    largura: int
    altura: int
    fps: Fraction
    duracao: float
    n_frames: int
    codec: str

    @property
    def fps_float(self) -> float:
        return float(self.fps)

    def descricao(self) -> str:
        return (
            f"{self.largura}x{self.altura} · {self.fps_float:.3f} fps · "
            f"{formatar_tempo(self.duracao)} · {self.codec} · ~{self.n_frames} frames"
        )


def sondar(caminho: str) -> InfoVideo:
    """Extrai metadados do vídeo com ffprobe."""
    saida = subprocess.run(
        [
            _bin("ffprobe"), "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name",
            "-show_entries", "format=duration",
            "-of", "json", caminho,
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    dados = json.loads(saida)
    fluxo = dados["streams"][0]
    num, den = fluxo["r_frame_rate"].split("/")
    fps = Fraction(int(num), int(den))
    duracao = float(dados["format"]["duration"])
    n_frames = int(fluxo.get("nb_frames") or round(duracao * float(fps)))
    return InfoVideo(
        largura=int(fluxo["width"]),
        altura=int(fluxo["height"]),
        fps=fps,
        duracao=duracao,
        n_frames=n_frames,
        codec=fluxo.get("codec_name", "?"),
    )


# Quantos segundos antes do ponto pedido o seek rápido cai, para depois o
# ffmpeg descartar o excedente com precisão de frame.
_FOLGA_SEEK = 5.0


def pts_inicial(caminho: str, inicio: float, fps: float = 30.0) -> float:
    """PTS real do primeiro frame em/após `inicio`.

    Um seek por keyframe pode cair bem antes do ponto pedido — neste vídeo,
    `-ss 300` entregava o frame de 299,299 s, e todos os timestamps saíam
    0,7 s adiantados. Aqui se pergunta ao ffprobe qual é o primeiro frame com
    PTS >= inicio; `LeitorFrames` usa seek preciso para começar nesse mesmo
    frame, e os dois ficam alinhados por construção.
    """
    if inicio <= 0:
        return 0.0
    janela = max(1.0, 3.0 / max(fps, 1.0) + 0.5)
    saida = subprocess.run(
        [
            _bin("ffprobe"), "-v", "error",
            "-select_streams", "v:0",
            "-read_intervals", f"{max(0.0, inicio - _FOLGA_SEEK)}%+{_FOLGA_SEEK + janela}",
            "-show_entries", "frame=best_effort_timestamp_time",
            "-of", "csv=p=0", caminho,
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    candidatos = []
    for linha in saida.splitlines():
        linha = linha.strip().rstrip(",")
        if not linha or linha == "N/A":
            continue
        try:
            candidatos.append(float(linha))
        except ValueError:
            continue
    # Tolerância de meio frame: PTS é racional e pode ficar um epsilon abaixo.
    limite = inicio - 0.5 / max(fps, 1.0)
    posteriores = sorted(v for v in candidatos if v >= limite)
    if posteriores:
        return posteriores[0]
    return inicio


class LeitorFrames:
    """Itera sobre `(indice, tempo_segundos, frame_bgr)`.

    `tempo_segundos` é sempre relativo ao início do vídeo original, mesmo
    quando se processa apenas um trecho — é esse valor que vai para o CSV.
    """

    def __init__(
        self,
        caminho: str,
        info: InfoVideo | None = None,
        inicio: float = 0.0,
        duracao: float | None = None,
        escala: float = 1.0,
    ):
        self.caminho = caminho
        self.info = info or sondar(caminho)
        self.inicio = max(0.0, float(inicio))
        self.duracao = duracao
        self.escala = escala
        self.largura = int(round(self.info.largura * escala)) // 2 * 2
        self.altura = int(round(self.info.altura * escala)) // 2 * 2
        self.t0 = pts_inicial(caminho, self.inicio, self.info.fps_float)
        self._proc: subprocess.Popen | None = None

    @property
    def bytes_por_frame(self) -> int:
        return self.largura * self.altura * 3

    def _comando(self) -> list[str]:
        cmd = [_bin("ffmpeg"), "-v", "error", "-nostdin"]
        if self.inicio > 0:
            # Seek em duas etapas: o `-ss` antes de `-i` pula rápido até o
            # keyframe anterior; o `-ss` depois de `-i` descarta o excedente
            # com precisão de frame. Só o primeiro seria rápido mas impreciso;
            # só o segundo seria preciso mas decodificaria o vídeo inteiro.
            bruto = max(0.0, self.inicio - _FOLGA_SEEK)
            cmd += ["-ss", f"{bruto:.6f}", "-i", self.caminho,
                    "-ss", f"{self.inicio - bruto:.6f}"]
        else:
            cmd += ["-i", self.caminho]
        if self.duracao is not None:
            cmd += ["-t", f"{self.duracao:.6f}"]
        if self.escala != 1.0:
            cmd += ["-vf", f"scale={self.largura}:{self.altura}:flags=area"]
        cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "-an", "-sn", "-"]
        return cmd

    def __iter__(self):
        import numpy as np

        self._proc = subprocess.Popen(
            self._comando(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=self.bytes_por_frame * 4,
        )
        passo = 1.0 / self.info.fps_float
        indice = 0
        try:
            while True:
                bruto = self._proc.stdout.read(self.bytes_por_frame)
                if len(bruto) < self.bytes_por_frame:
                    break
                frame = np.frombuffer(bruto, np.uint8).reshape(
                    (self.altura, self.largura, 3)
                )
                yield indice, self.t0 + indice * passo, frame
                indice += 1
        finally:
            self.fechar()

    def fechar(self) -> None:
        if self._proc is None:
            return
        for fluxo in (self._proc.stdout, self._proc.stderr):
            try:
                if fluxo:
                    fluxo.close()
            except Exception:
                pass
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def total_frames_estimado(self) -> int:
        span = self.duracao if self.duracao is not None else max(
            0.0, self.info.duracao - self.inicio
        )
        return max(1, int(span * self.info.fps_float))


def ler_frame_unico(caminho: str, tempo: float, escala: float = 1.0):
    """Devolve um único frame no instante pedido (usado pela ferramenta de setup)."""
    import numpy as np

    info = sondar(caminho)
    largura = int(round(info.largura * escala)) // 2 * 2
    altura = int(round(info.altura * escala)) // 2 * 2
    cmd = [_bin("ffmpeg"), "-v", "error", "-nostdin"]
    if tempo > 0:
        bruto = max(0.0, tempo - _FOLGA_SEEK)
        cmd += ["-ss", f"{bruto:.6f}", "-i", caminho, "-ss", f"{tempo - bruto:.6f}"]
    else:
        cmd += ["-i", caminho]
    cmd += ["-frames:v", "1"]
    if escala != 1.0:
        cmd += ["-vf", f"scale={largura}:{altura}:flags=area"]
    cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "-an", "-"]
    bruto = subprocess.run(cmd, capture_output=True, check=True).stdout
    esperado = largura * altura * 3
    if len(bruto) < esperado:
        return None
    return np.frombuffer(bruto[:esperado], np.uint8).reshape((altura, largura, 3)).copy()


def decompor_tempo(segundos: float) -> tuple[int, int, int]:
    """Divide em (minutos, segundos, milissegundos).

    O arredondamento vem PRIMEIRO, em milissegundos inteiros, e só depois a
    divisão. Fazendo o contrário, 299,9997 s virava "04:60.000": o resto
    59,9997 arredondava para 60,000 dentro do minuto 4.
    """
    if segundos < 0:
        segundos = 0.0
    total_ms = int(round(segundos * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    return total_s // 60, total_s % 60, ms


def formatar_tempo(segundos: float) -> str:
    """`mm:ss.mmm` — o formato pedido para registrar cada pacote."""
    minutos, seg, ms = decompor_tempo(segundos)
    return f"{minutos:02d}:{seg:02d}.{ms:03d}"


def formatar_tempo_longo(segundos: float) -> str:
    """`hh:mm:ss.mmm`, para vídeos com mais de uma hora."""
    minutos, seg, ms = decompor_tempo(segundos)
    return f"{minutos // 60:02d}:{minutos % 60:02d}:{seg:02d}.{ms:03d}"

"""Configuração da análise: linha de contagem, ROI e parâmetros de detecção.

Todas as coordenadas são armazenadas em pixels do vídeo ORIGINAL (4K aqui),
não da resolução de processamento. Assim a mesma configuração continua válida
se você mudar `escala_processamento`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class LinhaContagem:
    """Segmento que os pacotes atravessam, em coordenadas do vídeo original."""

    x1: int
    y1: int
    x2: int
    y2: int
    # Sentido válido do cruzamento: "ambos", "positivo" ou "negativo".
    # O sinal é dado pelo produto vetorial (lado da linha); a ferramenta de
    # setup mostra a seta do sentido positivo para você conferir.
    sentido: str = "ambos"
    nome: str = "gate"

    def como_tupla(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def escalada(self, escala: float) -> tuple[float, float, float, float]:
        return self.x1 * escala, self.y1 * escala, self.x2 * escala, self.y2 * escala

    def comprimento(self) -> float:
        return ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2) ** 0.5


@dataclass
class RegiaoInteresse:
    """Onde os pacotes são procurados, em coordenadas do vídeo original.

    O retângulo é o mínimo; o que realmente resolve esta cena é o `poligono`.
    Vista de cima com lente grande-angular, várias esteiras se sobrepõem na
    imagem: um retângulo em volta do ponto de medição pega também a esteira
    que passa atrás, e as duas contagens se misturam. Desenhando uma faixa
    que acompanha só a esteira de interesse, o resto da cena é ignorado.

    `poligono` é uma lista de vértices [[x, y], ...]. Quando presente, o
    retângulo passa a ser apenas a caixa envolvente usada para recortar.
    """

    x: int
    y: int
    largura: int
    altura: int
    poligono: list[list[int]] | None = None

    def escalada(self, escala: float) -> tuple[int, int, int, int]:
        return (
            int(round(self.x * escala)),
            int(round(self.y * escala)),
            int(round(self.largura * escala)),
            int(round(self.altura * escala)),
        )

    def poligono_escalado(self, escala: float):
        """Vértices em coordenadas do recorte já escalado (para cv2.fillPoly)."""
        if not self.poligono:
            return None
        import numpy as np

        rx, ry, _, _ = self.escalada(escala)
        pts = [
            [int(round(px * escala)) - rx, int(round(py * escala)) - ry]
            for px, py in self.poligono
        ]
        return np.array([pts], dtype=np.int32)

    @staticmethod
    def do_poligono(pontos: list[list[int]], folga: int = 0,
                    limite: tuple[int, int] | None = None) -> "RegiaoInteresse":
        xs = [p[0] for p in pontos]
        ys = [p[1] for p in pontos]
        x0 = max(0, min(xs) - folga)
        y0 = max(0, min(ys) - folga)
        x1 = max(xs) + folga
        y1 = max(ys) + folga
        if limite:
            x1 = min(limite[0], x1)
            y1 = min(limite[1], y1)
        return RegiaoInteresse(
            x=x0, y=y0, largura=x1 - x0, altura=y1 - y0,
            poligono=[[int(p[0]), int(p[1])] for p in pontos],
        )


@dataclass
class FaixaCor:
    """Critério de cor que distingue o papelão do resto da cena.

    O padrão é LAB, não HSV. Nesta esteira o metal dos roletes é levemente
    azulado (b* fica em torno de -4) e o papelão é amarelado (b* acima de 2),
    o que dá separação limpa. Em HSV o mesmo papelão perde saturação na sombra
    do galpão e escapa da faixa — medido nas amostras: S>45 pegava só metade
    dos pixels da caixa, enquanto b*>2 pega a caixa inteira.

    Cor não distingue papelão de madeira (paletes têm b* praticamente igual);
    é a subtração de fundo, em `ParametrosDeteccao`, que descarta o cenário
    parado.

    b* também não enxerga embalagem que não seja de papelão — daí a segunda
    faixa, `incluir_claros`, para sacos plásticos e envelopes brancos. Quem
    aplica as duas a um frame é `cor.mascara_cor`.
    """

    modo: str = "lab"          # "lab" (recomendado) ou "hsv"

    # LAB — b* é o eixo azul(-) / amarelo(+); L exclui sombras muito escuras.
    b_min: int = 2
    b_max: int = 70
    a_min: int = -14
    a_max: int = 28
    l_min: int = 45
    l_max: int = 252

    # HSV — mantido para o modo alternativo.
    h_min: int = 8
    h_max: int = 32
    s_min: int = 25
    s_max: int = 255
    v_min: int = 70
    v_max: int = 255

    # --- segunda faixa: material claro (sacos plásticos, envelopes) ---
    # Nem toda esteira transporta só papelão, e b* não serve para o resto. Nos
    # sacos plásticos brancos do TESTE 01 b* fica em -3, praticamente igual ao
    # rolete metálico (-2): o eixo amarelo/azul não os separa de jeito nenhum.
    # O que separa é o brilho — medido no trecho 55-65 s, L (escala OpenCV,
    # 0-255) passa de 190 em 28% dos pixels de saco contra 0,1% dos pixels de
    # rolete. Sem esta faixa um trecho de sacos simplesmente some da contagem:
    # o TESTE 01 fechava em 107 pacotes onde a contagem manual dava 141.
    # Fica desligada por padrão: onde só há papelão, ligar isto só adiciona
    # risco de pegar reflexo de metal polido.
    incluir_claros: bool = False
    claro_l_min: int = 190
    claro_l_max: int = 255
    # Faixa cromática estreita em volta do neutro: o critério é "claro E sem
    # cor". Amarelo brilhante (corrimão, colete) tem b* alto e já pertence — ou
    # não — à faixa do papelão; não deve entrar por aqui também.
    claro_a_min: int = -12
    claro_a_max: int = 12
    claro_b_min: int = -12
    claro_b_max: int = 12

    def limites_hsv(self):
        import numpy as np

        baixo = np.array([self.h_min, self.s_min, self.v_min], np.uint8)
        alto = np.array([self.h_max, self.s_max, self.v_max], np.uint8)
        return baixo, alto

    def limites_lab(self):
        """Limites no espaço OpenCV, onde a* e b* são deslocados em +128."""
        import numpy as np

        baixo = np.array(
            [self.l_min, max(0, self.a_min + 128), max(0, self.b_min + 128)], np.uint8
        )
        alto = np.array(
            [self.l_max, min(255, self.a_max + 128), min(255, self.b_max + 128)],
            np.uint8,
        )
        return baixo, alto

    def limites_claros(self):
        """Limites LAB da faixa de material claro, no espaço OpenCV."""
        import numpy as np

        baixo = np.array(
            [self.claro_l_min, max(0, self.claro_a_min + 128),
             max(0, self.claro_b_min + 128)], np.uint8
        )
        alto = np.array(
            [self.claro_l_max, min(255, self.claro_a_max + 128),
             min(255, self.claro_b_max + 128)], np.uint8
        )
        return baixo, alto


@dataclass
class ParametrosDeteccao:
    # "gate"  — sensor de ocupação sobre a linha, como uma fotocélula. É o
    #           padrão: funciona com os pacotes encostados em fila, que é o
    #           caso desta esteira, e produz o instante da passagem com
    #           precisão de milissegundos.
    # "blobs" — segmenta e rastreia cada pacote. Dá caixas e trajetórias no
    #           OSD, mas só é confiável quando os pacotes vêm separados.
    modo: str = "gate"

    # Escala de processamento aplicada ao frame antes de detectar.
    # 0.5 em um vídeo 4K = processa em 1920x1080: rápido e com resolução sobrando.
    escala_processamento: float = 0.5

    # --- modo "gate" ---
    gate_largura: int = 20              # espessura da janela, px do original
    gate_suavizacao: int = 5            # frames
    gate_limiar_ocupado: float = 0.45   # fração do nível de referência
    gate_limiar_vazio: float = 0.25
    gate_min_vazio_s: float = 0.20
    gate_min_ocupado_s: float = 0.20
    gate_referencia_percentil: float = 85.0
    gate_nivel_referencia: float | None = None
    gate_borda: str = "subida"          # "subida" = entrada do pacote no gate

    # Área mínima/máxima do blob, em pixels da resolução de PROCESSAMENTO.
    area_min: int = 900
    area_max: int = 200_000

    # Fechamento/abertura morfológica (pixels, resolução de processamento).
    abertura: int = 3
    fechamento: int = 7

    # Comprimento típico de um pacote ao longo do fluxo, em pixels de
    # processamento. Blobs muito mais longos que isso são fila de caixas
    # encostadas e são divididos — sem isso a contagem funde pacotes.
    comprimento_pacote: int = 0          # 0 = não divide
    tolerancia_divisao: float = 1.45     # divide quando len > tol * comprimento

    # Exige que o pixel também tenha se movido em algum momento (subtração de
    # fundo), o que descarta papelão parado no cenário — paletes, pilhas ao fundo.
    usar_subtracao_fundo: bool = True
    historico_fundo: int = 500
    limiar_fundo: int = 16
    # A esteira medida fica PARADA 77% do tempo (é de acumulação: enche, libera,
    # enche de novo). Por isso a memória de movimento tem decaimento quase nulo:
    # ela funciona como um mapa de "por onde já passou pacote", que descarta o
    # cenário fixo (paletes, pilhas ao fundo) sem apagar uma caixa que fica
    # meio minuto imóvel sobre a linha. Com decaimento rápido, uma caixa parada
    # 5 s simplesmente saía da máscara.
    decaimento_movimento: float = 0.9995   # meia-vida ~46 s
    limiar_memoria: float = 0.10
    dilatacao_memoria: int = 31

    # Rastreamento.
    # Os pacotes andam devagar: medido nesta esteira, 20 a 90 px/s em 4K, ou
    # seja 1,5 px por frame na escala 0,5. Uma tolerância de associação larga
    # (o valor inicial era 90 px) fazia o rastreador trocar a identidade entre
    # caixas vizinhas da fila e gerar cruzamentos falsos com velocidade
    # absurda. Mantenha esta folga bem acima do movimento por frame, mas bem
    # abaixo do espaçamento entre pacotes.
    distancia_max_associacao: float = 24.0   # px de processamento
    frames_para_perder: int = 20
    deslocamento_min_para_contar: float = 10.0  # evita contar ruído estático
    idade_min_para_contar: int = 4               # frames observados
    # Descarta cruzamentos com velocidade impossível: são salto de associação,
    # não pacote. 0 desliga o filtro.
    velocidade_max_para_contar: float = 220.0   # px/s de processamento


@dataclass
class Configuracao:
    video: str
    linha: LinhaContagem
    roi: RegiaoInteresse
    cor: FaixaCor = field(default_factory=FaixaCor)
    deteccao: ParametrosDeteccao = field(default_factory=ParametrosDeteccao)
    # Trecho analisado (segundos). `duracao=None` = até o fim.
    inicio: float = 0.0
    duracao: float | None = None
    # Limite de gap (s) acima do qual o OSD marca parada/gargalo.
    limite_gap_alerta: float = 20.0
    # Segundo vídeo do ensaio, desenhado como quadro no OSD. `None` = procura o
    # mesmo nome com " - MINI". O alinhamento mora aqui, e não só na linha de
    # comando, porque é propriedade do ensaio: descobri-lo custa trabalho manual
    # (ver `sincronizar.py`) e o valor não muda entre um render e outro.
    # Convenção: o instante `t` do principal corresponde a `t + mini_offset`
    # no MINI.
    mini: str | None = None
    mini_offset: float = 0.0

    def salvar(self, caminho: str | Path) -> None:
        Path(caminho).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def carregar(caminho: str | Path) -> "Configuracao":
        dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
        return Configuracao(
            video=dados["video"],
            linha=LinhaContagem(**dados["linha"]),
            roi=RegiaoInteresse(**dados["roi"]),
            cor=FaixaCor(**dados.get("cor", {})),
            deteccao=ParametrosDeteccao(**dados.get("deteccao", {})),
            inicio=dados.get("inicio", 0.0),
            duracao=dados.get("duracao"),
            limite_gap_alerta=dados.get("limite_gap_alerta", 20.0),
            mini=dados.get("mini"),
            mini_offset=dados.get("mini_offset", 0.0),
        )

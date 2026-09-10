#!/usr/bin/env python3
"""Linha de comando do KYMO.

    python -m kymo setup  VIDEO [--saida config.json] [--em 300]
    python -m kymo detect [--config config.json] [--inicio 300] [--duracao 120]
    python -m kymo render [--config config.json] [--saida saida/osd.mp4]
    python -m kymo report [--eventos saida/pacotes.csv]
    python -m kymo panorama [--config config.json]
    python -m kymo relatorio --alvo 1950

O `setup` é interativo: você marca a linha de contagem clicando sobre o vídeo.
Os demais comandos leem essa configuração.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Configuracao
from .detect import carregar_csv, executar_deteccao
from .osd import Renderizador
from .report import imprimir, resumir, salvar_json
from .video import formatar_tempo, sondar

PADRAO_CONFIG = "config.json"
PADRAO_SAIDA = Path("saida")


def _tempo(valor: str) -> float:
    """Aceita segundos (`305.4`) ou `mm:ss` / `hh:mm:ss`."""
    if ":" not in valor:
        return float(valor)
    partes = [float(p) for p in valor.split(":")]
    total = 0.0
    for p in partes:
        total = total * 60 + p
    return total


def _carregar(caminho: str, inicio=None, duracao=None) -> Configuracao:
    p = Path(caminho)
    if not p.exists():
        sys.exit(
            f"Configuração '{caminho}' não existe.\n"
            f"Rode primeiro:  python analisar.py setup \"SEU_VIDEO.MP4\""
        )
    cfg = Configuracao.carregar(p)
    if inicio is not None:
        cfg.inicio = inicio
    if duracao is not None:
        cfg.duracao = duracao
    return cfg


def cmd_setup(a) -> None:
    from .setup_gate import rodar_setup

    rodar_setup(a.video, saida=a.saida, tempo_inicial=a.em, escala_tela=a.escala_tela)


def cmd_detect(a) -> None:
    cfg = _carregar(a.config, a.inicio, a.duracao)
    if a.escala:
        cfg.deteccao.escala_processamento = a.escala
    PADRAO_SAIDA.mkdir(exist_ok=True)
    csv_saida = a.saida or PADRAO_SAIDA / "pacotes.csv"
    json_saida = Path(csv_saida).with_suffix(".json")

    info = sondar(cfg.video)
    fim = cfg.inicio + cfg.duracao if cfg.duracao else info.duracao
    print(f"Vídeo ....... {info.descricao()}")
    print(f"Trecho ...... {formatar_tempo(cfg.inicio)} → {formatar_tempo(fim)}")
    print(f"Linha ....... {cfg.linha.como_tupla()}  sentido={cfg.linha.sentido}")
    print(f"Escala ...... {cfg.deteccao.escala_processamento} "
          f"({int(info.largura*cfg.deteccao.escala_processamento)}px)")
    print("Detectando...")

    eventos = executar_deteccao(cfg, csv_saida, json_saida)
    print(f"\n{len(eventos)} pacotes registrados em {csv_saida}")
    if eventos:
        resumo = resumir(eventos, cfg.inicio, fim, cfg.limite_gap_alerta)
        imprimir(resumo, cfg.limite_gap_alerta)
        salvar_json(resumo, Path(csv_saida).with_name("resumo.json"))


def cmd_render(a) -> None:
    cfg = _carregar(a.config, a.inicio, a.duracao)
    eventos_csv = Path(a.eventos or PADRAO_SAIDA / "pacotes.csv")
    if not eventos_csv.exists():
        sys.exit(f"'{eventos_csv}' não existe. Rode 'detect' primeiro.")
    eventos = carregar_csv(eventos_csv)
    saida = a.saida or PADRAO_SAIDA / "osd.mp4"

    # Ordem de precedência do segundo vídeo: argumento, depois `config.json`,
    # depois o mesmo nome com " - MINI". Assim um ensaio já calibrado não
    # precisa de argumento nenhum, e um teste pontual não exige editar o
    # config.
    mini = a.mini or cfg.mini
    if mini is None and not a.sem_mini:
        candidato = Path(cfg.video)
        candidato = candidato.with_name(
            f"{candidato.stem} - MINI{candidato.suffix}")
        mini = str(candidato) if candidato.exists() else None
    if a.sem_mini:
        mini = None
    if mini and not Path(mini).exists():
        sys.exit(f"'{mini}' nao existe")
    # `--mini-offset` não passado é None, não zero: sem essa distinção não dá
    # para saber se o usuário pediu zero ou não pediu nada, e o valor do config
    # seria descartado silenciosamente em todo render.
    offset = a.mini_offset if a.mini_offset is not None else cfg.mini_offset
    if mini:
        origem = "argumento" if a.mini_offset is not None else "config"
        print(f"MINI ........ {Path(mini).name}  "
              f"offset {offset:+.3f}s ({origem})")

    rend = Renderizador(
        cfg, eventos,
        escala_saida=a.escala_saida or None,
        janela_vazao=a.janela_vazao,
        mostrar_tracks=not a.sem_tracks,
        mostrar_grafico=not a.sem_grafico,
        mostrar_faixa=not a.sem_faixa,
        bin_grafico=a.grafico_bin,
        janela_pico=a.pico_janela,
        erros_gap=a.erros_gap,
        mini=mini,
        mini_offset=offset,
    )
    print(f"Renderizando OSD com {len(eventos)} eventos → {saida}")
    rend.renderizar(saida, crf=a.crf, preset=a.preset)
    print(f"Pronto: {saida}")


def cmd_relatorio(a) -> None:
    eventos_csv = Path(a.eventos or PADRAO_SAIDA / "pacotes.csv")
    if not eventos_csv.exists():
        sys.exit(f"'{eventos_csv}' não existe. Rode 'detect' primeiro.")
    eventos = carregar_csv(eventos_csv)
    if len(eventos) < 2:
        sys.exit("São necessários pelo menos 2 pacotes.")

    cfg = _carregar(a.config) if Path(a.config).exists() else None
    tempos = [e.tempo for e in eventos]

    # Janela analisada: o trecho realmente medido, gravado ao lado do CSV.
    t_ini, t_fim = None, None
    par = eventos_csv.with_suffix(".json")
    if par.exists():
        try:
            d = json.loads(par.read_text(encoding="utf-8"))
            tr = d.get("trecho") or {}
            t_ini = tr.get("inicio")
            if t_ini is not None and tr.get("duracao"):
                t_fim = t_ini + tr["duracao"]
        except (OSError, ValueError, TypeError):
            pass
    if a.de is not None:
        t_ini = a.de
    if a.ate is not None:
        t_fim = a.ate
    if t_ini is None:
        t_ini = cfg.inicio if cfg else min(tempos)
    if t_fim is None:
        t_fim = max(tempos)

    from .relatorio_excel import gerar

    saida = a.saida or PADRAO_SAIDA / "relatorio.xlsx"
    caminho, m = gerar(
        eventos=eventos,
        video=cfg.video if cfg else str(eventos_csv),
        alvo_hora=a.alvo,
        t_ini=t_ini, t_fim=t_fim,
        reg_ini=a.regime_de, reg_fim=a.regime_ate,
        limite_parada=cfg.limite_gap_alerta if cfg else 20.0,
        saida=saida,
        notas=a.nota or [],
    )
    from .relatorio_excel import _num

    atg = m.realizado_hora / m.alvo_hora if m.alvo_hora else 0
    print(f"Relatório gerado: {caminho}")
    print(f"  alvo ................... {_num(m.alvo_hora)} pacotes/h")
    print(f"  realizado (regime) ..... {_num(m.realizado_hora)} pacotes/h "
          f"({atg*100:.1f}% do alvo)")
    print(f"  capacidade na cadencia . {_num(m.capacidade_hora)} pacotes/h")
    print(f"  cadencia observada ..... {m.cadencia:.3f}s  (alvo exige "
          f"{m.cadencia_alvo:.3f}s)")
    print(f"  ocupacao ............... {m.ocupacao*100:.1f}%  "
          f"({m.slots_vazios} de {m.slots} posicoes vazias)")
    print(f"  desvio por cadencia .... {_num(m.gap_cadencia)} pacotes/h")
    print(f"  desvio por ocupacao .... {_num(m.gap_ocupacao)} pacotes/h")


def cmd_panorama(a) -> None:
    from .panorama import gerar_panorama

    cfg = _carregar(a.config, a.inicio, a.duracao)
    marcas = None
    if not a.sem_marcas:
        eventos_csv = Path(a.eventos or PADRAO_SAIDA / "pacotes.csv")
        if eventos_csv.exists():
            marcas = [e.tempo for e in carregar_csv(eventos_csv)]
            print(f"Marcando ..... {len(marcas)} eventos de {eventos_csv}")

    saida = a.saida or PADRAO_SAIDA / "panorama.png"
    print("Montando panorama...")
    caminho, passo = gerar_panorama(
        cfg, saida, passo=a.passo, marcas=marcas,
        segundos_por_tira=a.tira, escala=a.escala_saida,
    )
    print(f"Passo ........ {passo:.1f} px/frame"
          f"{'' if a.passo is not None else ' (medido por correlação)'}")
    print(f"Pronto: {caminho}")


def cmd_report(a) -> None:
    eventos_csv = Path(a.eventos or PADRAO_SAIDA / "pacotes.csv")
    if not eventos_csv.exists():
        sys.exit(f"'{eventos_csv}' não existe. Rode 'detect' primeiro.")
    eventos = carregar_csv(eventos_csv)
    if not eventos:
        sys.exit("Nenhum pacote registrado no CSV.")
    cfg = _carregar(a.config) if Path(a.config).exists() else None
    limite = cfg.limite_gap_alerta if cfg else a.limite_parada

    # O trecho vem do JSON gravado junto do CSV, não do config.json: `detect`
    # aceita --inicio/--duracao sem reescrever a configuração, e usar o
    # intervalo do config aqui daria uma vazão média de outro trecho.
    inicio = fim = None
    par = Path(eventos_csv).with_suffix(".json")
    if par.exists():
        try:
            dados = json.loads(par.read_text(encoding="utf-8"))
            trecho = dados.get("trecho") or {}
            inicio = trecho.get("inicio")
            if inicio is not None and trecho.get("duracao"):
                fim = inicio + trecho["duracao"]
        except (OSError, ValueError, TypeError):
            inicio = fim = None
    if inicio is None:
        inicio = cfg.inicio if cfg else min(e.tempo for e in eventos)
    if fim is None:
        fim = max(e.tempo for e in eventos)

    # Recorte do intervalo sem reprocessar o vídeo. Serve para descartar
    # trechos que não valem como medição — no vídeo desta esteira, os
    # primeiros ~40 s são a câmera sendo posicionada, e incluí-los rebaixa a
    # vazão média de um período em que a operação nem havia começado.
    if a.de is not None:
        inicio = max(inicio, a.de)
    if a.ate is not None:
        fim = min(fim, a.ate)
    if a.de is not None or a.ate is not None:
        eventos = [e for e in eventos if inicio <= e.tempo <= fim]
        if not eventos:
            sys.exit("Nenhum pacote no intervalo pedido.")
    resumo = resumir(eventos, inicio, fim, limite)
    imprimir(resumo, limite)
    if a.json:
        salvar_json(resumo, a.json)
        print(f"\nResumo salvo em {a.json}")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m kymo",
        description="Registra o instante (mm:ss.mmm) em que cada pacote cruza "
                    "um ponto da esteira e gera vídeo com OSD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="comando", required=True)

    s = sub.add_parser("setup", help="marcar a linha de contagem (interativo)")
    s.add_argument("video")
    s.add_argument("--saida", default=PADRAO_CONFIG)
    s.add_argument("--em", type=_tempo, default=0.0,
                   help="instante inicial exibido (s ou mm:ss)")
    s.add_argument("--escala-tela", dest="escala_tela", type=float, default=0.0,
                   help="0 = automático")
    s.set_defaults(func=cmd_setup)

    d = sub.add_parser("detect", help="detectar e cronometrar os pacotes")
    d.add_argument("--config", default=PADRAO_CONFIG)
    d.add_argument("--inicio", type=_tempo, default=None)
    d.add_argument("--duracao", type=_tempo, default=None)
    d.add_argument("--escala", type=float, default=None,
                   help="escala de processamento (sobrepõe a do config)")
    d.add_argument("--saida", default=None, help="CSV de saída")
    d.set_defaults(func=cmd_detect)

    r = sub.add_parser("render", help="gerar vídeo com OSD")
    r.add_argument("--config", default=PADRAO_CONFIG)
    r.add_argument("--eventos", default=None)
    r.add_argument("--saida", default=None)
    r.add_argument("--inicio", type=_tempo, default=None)
    r.add_argument("--duracao", type=_tempo, default=None)
    r.add_argument("--escala-saida", dest="escala_saida", type=float, default=0.0)
    r.add_argument("--janela-vazao", dest="janela_vazao", type=float, default=0.0,
                   help="apelido antigo de --grafico-bin (use aquele)")
    r.add_argument("--crf", type=int, default=20)
    r.add_argument("--preset", default="medium")
    r.add_argument("--sem-tracks", dest="sem_tracks", action="store_true")
    r.add_argument("--sem-grafico", dest="sem_grafico", action="store_true")
    r.add_argument("--sem-faixa", dest="sem_faixa", action="store_true")
    r.add_argument("--mini", default=None,
                   help='segundo video no canto inferior direito; o padrao e o '
                        'mesmo nome com " - MINI"')
    r.add_argument("--mini-offset", dest="mini_offset", type=float, default=None,
                   help="deslocamento do MINI em segundos: o instante t do "
                        "principal corresponde a t+offset no MINI. O padrao e "
                        "o `mini_offset` do config (use kymo-sincronizar para "
                        "descobrir)")
    r.add_argument("--sem-mini", dest="sem_mini", action="store_true",
                   help="ignora o video MINI mesmo que exista")
    r.add_argument("--erros-gap", dest="erros_gap", type=int, default=None,
                   help="total de passagens com espacamento abaixo do minimo "
                        "(8,89 cm). Contado fora daqui por enquanto; sem o "
                        "argumento o campo nao aparece no OSD")
    r.add_argument("--pico-janela", dest="pico_janela", type=_tempo, default=60.0,
                   help="janela da melhor vazao mostrada no grafico, em segundos "
                        "(padrao 60; some se o trecho for menor que ela)")
    r.add_argument("--grafico-bin", dest="grafico_bin", type=float, default=0.0,
                   help="janela da media movel, em segundos: vale para o numero "
                        "do topo e para a curva do grafico (0 = automatico, "
                        "30s a 90s conforme o trecho)")
    r.set_defaults(func=cmd_render)

    n = sub.add_parser(
        "panorama",
        help="esteira desenrolada (slit-scan) para conferir a contagem no olho",
    )
    n.add_argument("--config", default=PADRAO_CONFIG)
    n.add_argument("--eventos", default=None)
    n.add_argument("--saida", default=None)
    n.add_argument("--inicio", type=_tempo, default=None)
    n.add_argument("--duracao", type=_tempo, default=None)
    n.add_argument("--passo", type=float, default=None,
                   help="px por frame; o padrão é medir por correlação")
    n.add_argument("--tira", type=float, default=10.0,
                   help="segundos de esteira por tira empilhada")
    n.add_argument("--escala-saida", dest="escala_saida", type=float, default=0.4)
    n.add_argument("--sem-marcas", dest="sem_marcas", action="store_true",
                   help="não desenhar os eventos detectados sobre o panorama")
    n.set_defaults(func=cmd_panorama)

    t = sub.add_parser("report", help="estatísticas do CSV")
    t.add_argument("--eventos", default=None)
    t.add_argument("--config", default=PADRAO_CONFIG)
    t.add_argument("--json", default=None)
    t.add_argument("--limite-parada", dest="limite_parada", type=float, default=20.0)
    t.add_argument("--de", type=_tempo, default=None,
                   help="recorta o início do intervalo analisado (s ou mm:ss)")
    t.add_argument("--ate", type=_tempo, default=None,
                   help="recorta o fim do intervalo analisado (s ou mm:ss)")
    t.set_defaults(func=cmd_report)

    x = sub.add_parser("relatorio", help="gerar relatorio Excel")
    x.add_argument("--alvo", type=float, required=True,
                   help="alvo de throughput, em pacotes/hora")
    x.add_argument("--eventos", default=None)
    x.add_argument("--config", default=PADRAO_CONFIG)
    x.add_argument("--saida", default=None)
    x.add_argument("--de", type=_tempo, default=None,
                   help="inicio da janela analisada (s ou mm:ss)")
    x.add_argument("--ate", type=_tempo, default=None,
                   help="fim da janela analisada (s ou mm:ss)")
    x.add_argument("--regime-de", dest="regime_de", type=_tempo, default=None,
                   help="inicio do regime usado na comparacao com o alvo")
    x.add_argument("--regime-ate", dest="regime_ate", type=_tempo, default=None,
                   help="fim do regime usado na comparacao com o alvo")
    x.add_argument("--nota", action="append", default=None,
                   help="observacao livre na aba de metodo (repetivel)")
    x.set_defaults(func=cmd_relatorio)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

"""Segmentação por cor do material transportado.

Vive fora de `config.py` porque `config.py` só guarda e interpreta parâmetros —
quem aplica o critério a um frame é este módulo, usado tanto pelo sensor de
gate quanto pela segmentação de blobs, sem que um precise importar o outro.

O critério tem duas faixas independentes, unidas por OU:

- **papelão** — b* do LAB acima de zero, o eixo que separa o papelão amarelado
  do rolete metálico levemente azulado;
- **material claro** — L alto e cromaticidade próxima do neutro, para sacos
  plásticos e envelopes brancos, cujo b* é indistinguível do rolete.

A segunda faixa só entra quando `incluir_claros` está ligada.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import FaixaCor


def mascara_cor(faixa: FaixaCor, bgr: np.ndarray) -> np.ndarray:
    """Máscara binária (0/255) do material transportado dentro de `bgr`."""
    if faixa.modo == "lab":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        m = cv2.inRange(lab, *faixa.limites_lab())
    else:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, *faixa.limites_hsv())
        lab = None

    if faixa.incluir_claros:
        # Mesmo no modo HSV a faixa de claros é definida em LAB: é o L que
        # separa saco branco de rolete, e V do HSV mistura brilho com cor.
        if lab is None:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        m = cv2.bitwise_or(m, cv2.inRange(lab, *faixa.limites_claros()))
    return m

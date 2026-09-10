"""KYMO — vazão de esteiras medida a partir de vídeo, sem instrumentar a linha.

O nome vem do grego κῦμα (*kŷma*), "onda", e γράφειν (*gráphein*), "escrever":
o quimógrafo de Carl Ludwig, 1847, registrava pressão arterial numa agulha
encostada num cilindro de papel que girava sozinho. Como o cilindro se move, o
tempo vira distância, e o traçado mostra a história inteira de uma vez.

Aqui a ideia aparece duas vezes: o gate é a agulha — um ponto fixo que espera a
esteira passar, sem acompanhar pacote nenhum — e o `panorama` é o papel, uma
fatia por frame concatenada até a esteira sair desenrolada, cada pacote uma
única vez.

    kymo setup VIDEO --em 05:00
    kymo detect
    kymo render
    kymo panorama
    kymo report
    kymo relatorio --alvo 1950
    kymo-sincronizar VIDEO --em-principal 26 --em-mini 22
"""

__version__ = "0.2.0"

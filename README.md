# KYMO

Mede vazão de esteiras a partir de vídeo, sem instrumentar a linha. Registra o
instante — `mm:ss.mmm` — em que cada pacote passa por um ponto escolhido, e gera
um vídeo com esses dados exibidos sobre a própria imagem (OSD), estatísticas e
relatório Excel.

Ferramenta de campo, escrita por Hemerson Camargo.

## De onde vem o nome

Do grego **κῦμα** (*kŷma*), "onda", e **γράφειν** (*gráphein*), "escrever" —
literalmente *escritor de ondas*.

O quimógrafo foi um instrumento de laboratório do século XIX, criado por Carl
Ludwig em 1847 para registrar a pressão arterial. Um cilindro coberto de papel
enfumaçado girava em velocidade constante enquanto uma agulha encostada nele
marcava o que estava sendo medido. O truque é o que sobra no papel: como o
cilindro se move sozinho, **o tempo vira distância**. Quem olha o traçado depois
não vê um instante — vê a história inteira, de uma vez.

É exatamente o que esta ferramenta faz duas vezes:

- **o gate é a agulha.** Um ponto fixo sobre a esteira, medindo ocupação frame a
  frame. Não acompanha pacote nenhum; espera a esteira passar por ele, como uma
  fotocélula — ou como a agulha encostada no cilindro;
- **o `panorama` é o papel.** A cada frame recorta-se uma fatia estreita sobre
  esse ponto e as fatias são concatenadas. Como cada pedaço da esteira cruza o
  ponto uma única vez, cada pacote aparece uma única vez na imagem final. A
  esteira sai desenrolada, e contar pacotes vira contar caixas numa foto.

O nome moderno sobreviveu na biologia celular, onde *kymograph* é precisamente
essa técnica: empilhar uma linha da imagem ao longo do tempo, para que o que se
move vire uma faixa diagonal cuja inclinação é a velocidade. Foi assim que
apareceu, num dos ensaios, um trecho inteiro de sacos plásticos que o critério
de cor não enxergava — no sinal do sensor era um vão, indistinguível de esteira
vazia; no papel do quimógrafo era uma fila cheia.

## Instalação

```bash
python3 -m venv .venv
.venv/bin/pip install -e .     # opencv-python, numpy, openpyxl
```

Requer `ffmpeg` e `ffprobe` no PATH. Instalado assim, os comandos `kymo` e
`kymo-sincronizar` ficam disponíveis de qualquer diretório — os dados de cada
ensaio moram em `projetos/<ensaio>/`, fora da árvore de código, e é de lá que se
roda a análise.

Precisa de `ffmpeg` e `ffprobe` no PATH (`brew install ffmpeg`).

## Uso

```bash
# 1. marcar o ponto de medição (interativo) — abre uma janela com o vídeo
kymo setup "ensaio.mp4" --em 05:00

# 2. cronometrar os pacotes -> saida/pacotes.csv + resumo no terminal
kymo detect --inicio 05:00 --duracao 02:00

# 3. vídeo com OSD -> saida/osd.mp4
kymo render

# 4. reimprimir as estatísticas a partir do CSV
kymo report --json saida/resumo.json

# 5. conferir a contagem: a esteira desenrolada, com um traço por pacote contado
kymo panorama
```

Sem `--inicio`/`--duracao`, vale o trecho gravado no `config.json`; para o vídeo
inteiro, use `--inicio 0` e não passe `--duracao`.

### Na tela do `setup`

| tecla | ação |
|---|---|
| clique esq. ×2 | desenha a **linha de contagem** |
| `G` + cliques | desenha a **faixa da esteira** (polígono); `ESPAÇO` fecha, `X` apaga vértice |
| `R` + cliques ×2 | ROI retangular (alternativa à faixa) |
| `P` + cliques ×2 | mede o comprimento típico de um pacote |
| clique dir. | amostra a cor do papelão (calibra LAB) |
| `←` `→` / `A` `D` / `W` `S` | ±1 s / ±10 s / ±60 s |
| `M` / `N` | prévia da máscara + janela do gate / prévia das detecções |
| `F` | alterna o sentido válido do cruzamento |
| `ENTER` / `ESC` | salva / cancela |

**Desenhe a faixa.** Vista de cima com lente grande-angular, várias esteiras se
sobrepõem na imagem. Sem a faixa, um retângulo em volta do ponto de medição
captura também a esteira que passa atrás, e as duas contagens se misturam — foi
exatamente o que aconteceu nos primeiros testes deste vídeo.

## Saídas

- `saida/pacotes.csv` — uma linha por pacote: `id`, `minuto`, `segundo`,
  `milissegundo`, `mm:ss.mmm`, tempo em segundos, frame, intervalo desde o
  anterior, ocupação no instante do disparo.
- `saida/pacotes.json` — o mesmo, estruturado.
- `saida/resumo.json` — totais, vazão, distribuição dos intervalos, paradas.
- `saida/sinal_gate.npy` — o sinal do sensor, frame a frame (`[tempos, sinal]`),
  para reconferir os limiares sem reprocessar o vídeo.
- `saida/osd.mp4` — o vídeo com timecode, contador, vazão instantânea e média,
  intervalo entre pacotes com alerta de parada, gráfico de vazão e o traço do
  sensor.

### O gráfico de vazão

- **curva** — média móvel de 30 a 90 s (ajustável com `--grafico-bin SEGUNDOS`),
  preenchida no que já passou e em linha fina no que está por vir. Usa kernel
  triangular, não janela retangular: contar eventos numa janela dura devolve um
  inteiro, e com cadência de 2,1 s a contagem alterna entre 7 e 8 por janela de
  15 s — a curva viraria dente de serra por quantização, não por variação real
  da esteira.
- **tiques na base** — um por pacote, verde no que já passou. É o detalhe
  individual que a média móvel suaviza; os vãos aparecem a olho nu.
- **linha da capacidade** — `60 / cadência mediana`, o teto da esteira no ritmo
  em que ela opera. Comparar a curva com ela é o que separa "a linha está lenta"
  de "a linha está recebendo menos".
- **linha da média** e **faixas vermelhas** nos trechos com parada acima de
  `limite_gap_alerta`.
- eixo Y com grade rotulada e eixo do tempo em marcas redondas.

O gráfico e a média cobrem apenas o trecho renderizado — quando o render começa
depois do zero, o contador passa a dizer `PACOTES NO TRECHO`, enquanto os IDs
mantêm a numeração global do CSV.

## Como a medição funciona

**Modo `gate` (padrão).** Uma fotocélula virtual: mede, frame a frame, a fração
de papelão dentro de uma janela estreita sobre a linha. Cada subida de "vazio"
para "ocupado" é a borda dianteira de um pacote chegando ao ponto. O instante
sai por interpolação linear entre os dois frames vizinhos — a 29,97 fps um frame
vale 33,4 ms, e o pedido era milissegundo.

Escolhido depois de o rastreamento de blobs falhar nesta cena: os pacotes viajam
**encostados em fila**, então a segmentação por cor devolve um blob único para a
fila toda, e dividi-lo pelo comprimento típico faz as fronteiras artificiais
escorregarem entre frames — a identidade dos objetos se embaralha e aparecem
cruzamentos fantasma. O vão entre pacotes, ao contrário, é um sinal limpo: a
ocupação do gate cai a zero entre um pacote e o seguinte.

**Modo `blobs`.** Segmenta e rastreia cada pacote, com caixas, IDs e rastros no
OSD. Use quando os pacotes vierem separados. Ative com `"modo": "blobs"` em
`config.json`.

**Cor.** O critério é LAB, não HSV: o metal dos roletes é levemente azulado
(b\* ≈ −4) e o papelão é amarelado (b\* > 2), o que separa de forma limpa. Em HSV
o papelão perde saturação na sombra do galpão e escapa da faixa — medido neste
vídeo, `S>45` pegava metade dos pixels da caixa, enquanto `b*>2` pega a caixa
inteira. Cor **não** distingue papelão de madeira: paletes têm b\* praticamente
igual ao das caixas, e é a faixa poligonal que os mantém fora.

## Ao medir um vídeo novo

Na ordem em que os problemas costumam aparecer. O `CLAUDE.md` traz o detalhe e
os números que motivaram cada passo.

1. **Ache onde há carga.** Vídeo longo não quer dizer ensaio longo — já houve
   gravação de 22 minutos com 52 segundos de operação. Amostre a ocupação da
   faixa e olhe a *amplitude* por minuto: carga faz o sinal oscilar, estrutura
   fixa não.
2. **Verifique a estabilidade da câmera** antes de confiar em qualquer número.
   O ponto de medição é fixo em coordenadas de imagem; se a câmera derivar, o
   gate sai da esteira. Correlação de fase sobre estrutura fixa resolve. Em
   ensaios com câmera de mão, os primeiros segundos costumam ser o operador
   posicionando o equipamento.
3. **Confira o critério de cor contra o que a esteira leva.** Embalagem que a
   cor não enxerga não vira erro: vira vão no sinal, indistinguível de esteira
   vazia. Sacos plásticos e envelopes brancos exigem `cor.incluir_claros`.
4. **Meça o trecho de carga, não o vídeo inteiro.** O nível de referência do
   gate sai de um percentil do próprio sinal; minutos vazios derrubam o
   percentil e mudam os limiares.
5. **Confira no panorama.** `kymo panorama` desenha um traço por pacote contado
   sobre a esteira desenrolada: cada objeto deve receber exatamente uma marca.
6. **Confira a quantização dos intervalos.** Múltiplos inteiros da cadência é o
   esperado. Gap bem abaixo da cadência é contagem dupla.
7. **Confirme a unidade do alvo** antes de montar relatório. Alvo em
   pacotes/hora é o padrão em intralogística, mas isso muda toda a conclusão.

## Cadência e ocupação

O relatório separa o desvio em relação ao alvo em duas causas independentes, que
somam exatamente o total:

- **cadência** — intervalo entre posições consecutivas da esteira. Define o
  teto: nenhuma vazão passa de `3600 / cadência` por hora.
- **ocupação** — fração dessas posições que chega com pacote. Depende de quem
  alimenta a linha, não da linha.

A separação se sustenta porque os intervalos observados são múltiplos inteiros
da cadência. Verifique essa quantização na aba *Intervalos* antes de usar o
argumento.

A cadência medida é a cadência **em que a linha estava operando**, não o limite
do equipamento. Afirmar capacidade máxima exige ensaio com alimentação saturada.

## Desempenho

Em 4K com `escala_processamento: 0.5`, o modo `gate` mede a ~150 fps e o render
sai a ~90 fps. Os 21m40s completos levam cerca de 4 min para medir e 7 min para
renderizar.

## Estrutura

```
kymo/cli.py            CLI: setup | detect | render | panorama | report | relatorio
kymo/video.py          leitura via ffmpeg, seek preciso, formatação de tempo
kymo/config.py         linha, faixa, cor, parâmetros (JSON)
kymo/cor.py            critério de cor: papelão (b*) + material claro (L)
kymo/gate.py           sensor de ocupação — método principal
kymo/detect.py         segmentação e passagem de detecção
kymo/tracking.py       rastreamento por centroide (modo blobs)
kymo/osd.py            renderização do OSD, incluindo o quadro do vídeo MINI
kymo/panorama.py       slit-scan da esteira, para conferir a contagem
kymo/report.py         estatísticas
kymo/relatorio_excel.py  pasta de trabalho com resumo, desvio, séries, método
kymo/setup_gate.py     ferramenta interativa de marcação
kymo/sincronizar.py    alinhamento do vídeo MINI (fora do fluxo de medição)

exemplos/              config genérico, versionado
projetos/              dados de ensaio — fora do versionamento
```

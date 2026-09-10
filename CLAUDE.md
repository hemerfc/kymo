# KYMO

Mede vazão de esteiras a partir de vídeo, sem instrumentar a linha. Registra o
instante — `mm:ss.mmm` — em que cada pacote passa por um ponto escolhido, e
produz vídeo com OSD, estatísticas e relatório Excel comparando o realizado com
um alvo contratado. Ferramenta de campo, escrita por Hemerson Camargo.

O nome vem do quimógrafo (grego *kŷma*, onda + *gráphein*, escrever): o
instrumento de 1847 em que uma agulha fixa registrava num cilindro de papel
girante, fazendo o tempo virar distância. O gate é a agulha e o `panorama` é o
papel — ver o README. É o que transforma "confie na contagem" em "confira a
contagem".

## Comandos

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
# requer ffmpeg e ffprobe no PATH

kymo setup  VIDEO --em 05:00            # marca o ponto (janela OpenCV)
kymo detect --inicio 0                  # cronometra -> saida/pacotes.csv
kymo render --grafico-bin 15            # vídeo com OSD
kymo panorama                           # esteira desenrolada, para conferir
kymo report --de 03:00 --ate 17:00
kymo relatorio --alvo 1950 --regime-de 03:00 --regime-ate 17:00

kymo-sincronizar VIDEO --em-principal 45.2 --em-mini 12.7   # offset do MINI
```

Instalado em modo editável, o comando roda de qualquer diretório — que é o que
a ferramenta precisa: os dados de cada ensaio moram fora da árvore de código, e
é de dentro da pasta do ensaio que se roda a análise.

Tempos aceitam segundos (`305.4`) ou `mm:ss` / `hh:mm:ss`. `--inicio`/`--duracao`
no `detect` não reescrevem o `config.json`; o trecho realmente medido fica no
`pacotes.json` ao lado do CSV, e é dele que `report` e `relatorio` leem a janela.

## Arquitetura

| módulo | responsabilidade |
|---|---|
| `kymo/cli.py` | CLI: `setup`, `detect`, `render`, `panorama`, `report`, `relatorio` |
| `kymo/video.py` | leitura via pipe do ffmpeg, seek preciso, formatação de tempo |
| `kymo/config.py` | linha de contagem, faixa/ROI, cor, parâmetros — serializados em JSON |
| `kymo/cor.py` | aplica o critério de cor a um frame (papelão + material claro) |
| `kymo/gate.py` | sensor de ocupação sobre a linha — **método principal de contagem** |
| `kymo/detect.py` | segmentação por cor, passagem de detecção, escrita do CSV/JSON |
| `kymo/tracking.py` | rastreador por centroide (modo `blobs`, alternativo) |
| `kymo/osd.py` | renderização do OSD, incluindo o quadro do vídeo MINI |
| `kymo/panorama.py` | slit-scan: esteira desenrolada, conferência da contagem |
| `kymo/report.py` | estatísticas de terminal |
| `kymo/relatorio_excel.py` | pasta de trabalho com resumo, desvio, séries e método |
| `kymo/setup_gate.py` | ferramenta interativa de marcação |
| `kymo/sincronizar.py` | alinhamento do vídeo MINI (fora do fluxo de medição) |

Os dados de ensaio ficam em `projetos/<ensaio>/`, fora da árvore de código e
fora do versionamento: vídeo, `config-*.json` e `saida/`. `exemplos/` guarda um
config genérico, versionado, para partir dele.

Coordenadas de linha e faixa são sempre em **pixels do vídeo original**, nunca da
resolução de processamento — assim a mesma configuração continua válida se
`escala_processamento` mudar.

## Convenções

Código e comentários em **português**; termos técnicos consagrados mantêm a forma
original (`gate`, `blob`, `frame`, `bin`). Comentário explica **por que**, não o
que o código faz — e vários comentários deste repositório registram uma medição
que motivou a escolha. Não os remova ao refatorar: sem eles a decisão parece
arbitrária e tende a ser revertida.

## Decisões que vieram de medição, não de suposição

Reverter qualquer uma delas exige medir de novo e mostrar número.

**Contagem por gate, não por rastreamento.** Pacotes em esteira de acumulação
viajam encostados em fila. A segmentação por cor devolve um blob único para a
fila toda, e dividi-lo pelo comprimento típico faz as fronteiras artificiais
escorregarem entre frames — a identidade dos objetos se embaralha e surgem
cruzamentos fantasma (velocidades 10× acima do plausível). O vão entre pacotes,
ao contrário, é limpo: a ocupação do gate cai a zero. O modo `blobs` continua no
código para cenas com pacotes separados, onde ele dá caixas e trajetórias.

**Cor pelo eixo b\* do LAB, não HSV.** O metal dos roletes é levemente azulado
(b\* ≈ −4) e o papelão amarelado (b\* > 2). Em HSV o papelão perde saturação na
sombra do galpão: `S>45` pegava metade dos pixels da caixa, `b*>2` pega a caixa
inteira. Cor **não** distingue papelão de madeira — paletes têm b\* praticamente
igual; é a faixa poligonal que os mantém fora.

**Duas faixas de cor: papelão (b\*) e material claro (L).** O critério de b\* não
enxerga saco plástico branco: medido no TESTE 01, o saco tem b\* ≈ −3 e o rolete
metálico b\* ≈ −2 — o eixo amarelo/azul não os separa. O que separa é o brilho:
L (escala OpenCV) acima de 190 cobre 28% dos pixels de saco e 0,1% dos pixels de
rolete. Sem a segunda faixa o trecho de sacos some da contagem inteiro, e o
sinal do gate mostra um vão longo onde a esteira estava cheia — o TESTE 01
fechava em 107 pacotes onde a contagem manual dava 141 (com a faixa ligada, 142).
A faixa vem desligada por padrão: onde só passa papelão, ligá-la só acrescenta
risco de pegar reflexo de metal polido.

**Faixa poligonal, não retângulo.** Vista de cima com lente grande-angular,
várias esteiras se sobrepõem na imagem. Um retângulo em volta do ponto de medição
captura também a esteira que passa atrás, e as duas contagens se misturam.

**Seek de duas etapas no ffmpeg.** `-ss` antes de `-i` cai no keyframe anterior:
pedir 300 s entregava o frame de 299,299 s e adiantava todos os timestamps em
0,7 s. O `-ss` depois de `-i` descarta o excedente com precisão de frame.

**Arredondamento antes da divisão em `decompor_tempo`.** Dividir primeiro produzia
`04:60.000` para 299,9997 s.

**Kernel triangular na curva de vazão do OSD.** Contar eventos numa janela
retangular devolve um inteiro; com cadência de ~2 s a contagem alterna entre 7 e
8 por janela de 15 s e a curva serrilha por quantização, não por variação real.

**Melhor janela por contagem retangular, não pelo pico da curva.** O campo
*melhor 60s* do OSD conta pacotes numa janela deslizante; a curva ao lado usa
kernel triangular. São cálculos diferentes de propósito: o triangular é o certo
para desenhar, mas seu pico não é contável — no TESTE 01 ele dá 142,5/min, um
número que ninguém confere no CSV, enquanto "139 pacotes entre 5,98 s e 65,98 s"
qualquer um reaudita. Espere os dois discordarem; não é bug.

Em trecho curto o campo mede pouco: com 67,6 s de vídeo a janela de um minuto
desliza 7,6 s e cobre 98% dos eventos, e a distância entre os 139/min do "melhor
minuto" e os 126,1/min de média é posição de janela, não um minuto melhor. Ele
some sozinho quando o trecho é menor que a janela; `--pico-janela` ajusta.

**Erro de gap não se mede por vídeo — não tente de novo.** Conta como erro a
passagem com espaçamento abaixo de **101,6 mm** (4"). A 1.166 mm/s um quadro de
60 fps vale 19,4 mm de esteira, 19% do vão mínimo, e a janela do gate consome
outra parte dele. Medido numa série de cinco ensaios: quatro formas igualmente
defensáveis de extrair o vão do mesmo sinal deram de 5% a 20%. Quando o método
move a resposta nessa amplitude, o que se está medindo é o método.

A decisão foi tirar esse critério do escopo da análise por vídeo: ele é
verificado **na esteira, por medição direta**. O total entra no OSD por
`render --erros-gap N`, rotulado como *informado*; sem o argumento o campo não é
desenhado, porque um "0 erros" que ninguém contou seria pior que a ausência do
campo.

**Cuidado com unidade de velocidade, e verifique a coerência.** Uma velocidade
informada em unidade trocada passa despercebida e contamina tudo que dela
depende. Duas verificações baratas, com v em mm/s: o tempo de gate ocupado vira
comprimento do produto no sentido do fluxo, e a cadência vira o **passo** da
linha — numa série de cinco ensaios o passo deu 428 a 466 mm, com três dos cinco
em exatamente 428 mm. O passo é propriedade da linha e deve repetir entre
ensaios; se não repetir, a velocidade ou a geometria está errada.

**Sincronismo do MINI entra como número, não é descoberto pelo programa.**
O segundo vídeo (`" - MINI"`) é outra câmera. Os três caminhos automáticos foram
testados neste material e nenhum serve: os vídeos principais foram gravados
**sem som** (RMS exatamente zero nos cinco), então correlação de áudio não tem
com o que trabalhar; nenhum arquivo tem `creation_time`; e casar imagem não
funciona porque o MINI é câmera de mão que caminha pelo galpão — ORB entre um
quadro do principal e o MINI inteiro rendeu no máximo 14 inliers em 400 pares,
que é ruído, e o trecho mais estável do MINI dura 4,5 s. Sobra o alinhamento
por evento comum: `sincronizar.py --em-principal T --em-mini T2`, conferido com
`--offset X --previa p.png`. Se um dia os vídeos vierem com áudio, o modo
automático já está lá e volta a valer.

**Média móvel única.** O número do topo do OSD e a curva do gráfico são lidos do
mesmo array (`_vazao_movel` interpola sobre ele). Havia dois cálculos com janelas
diferentes, e os valores não se explicavam um pelo outro.

## Ao medir um vídeo novo

1. **Verifique a estabilidade da câmera** antes de confiar em qualquer número. O
   ponto de medição é fixo em coordenadas de imagem; se a câmera derivar, o gate
   sai da esteira. Correlação de fase sobre uma região de estrutura fixa resolve.
   Em ensaios com câmera de mão, os primeiros segundos costumam ser o operador
   posicionando o equipamento — descarte-os com `report --de`.
2. **Confirme que o ponto tem pacotes em movimento.** Um mapa de "papelão que se
   move" (máscara de cor ∩ subtração de fundo, acumulada) mostra onde vale medir
   e evita colocar o gate sobre paletes estáticos ou estrutura.
3. **Confira o critério de cor contra o que a esteira realmente leva.** Cor é o
   que define "ocupado"; embalagem que ela não enxerga não é contada e nem
   aparece como erro — vira vão no sinal. Sacos plásticos e envelopes brancos
   exigem `cor.incluir_claros`. O comando `panorama` desenrola a esteira num
   slit-scan — uma fatia estreita por frame, concatenada — e desenha os eventos
   do CSV sobre ela: cada objeto aparece uma única vez e deve receber
   exatamente uma marca. É a contagem de referência mais barata que existe
   aqui, e conferir na imagem custa menos que recontar o vídeo.
4. **Cuidado com amarelo em primeiro plano.** Escadas, grades e uniformes têm b\*
   alto e entram no critério de papelão.
5. **Inspecione o sinal do gate** (`saida/sinal_gate.npy`, `[tempos, sinal]`)
   antes de aceitar a contagem. Sinal bom sobe a um platô e cai a zero entre
   pacotes. Se não cair, o gate está sobre duas esteiras ou pegou algo estático.
6. **Confirme a unidade do alvo** antes de montar relatório. Alvo em pacotes/hora
   é o padrão em intralogística, mas isso muda toda a conclusão.

## Cadência e ocupação

O relatório separa o desvio em relação ao alvo em duas causas independentes, que
somam exatamente o total:

- **cadência** — intervalo entre posições consecutivas da esteira. Define o teto:
  nenhuma vazão passa de `3600 / cadência` por hora.
- **ocupação** — fração dessas posições que chega com pacote. Depende de quem
  alimenta a linha, não da linha.

A separação se sustenta porque os intervalos observados são **múltiplos inteiros
da cadência** — a esteira mantém ritmo fixo e o que varia é quantas posições vêm
vazias. Verifique essa quantização na aba *Intervalos* antes de usar o argumento;
se os intervalos não forem quantizados, a decomposição não se aplica e a variação
é de outra natureza.

Ao redigir conclusão: a cadência medida é a cadência **em que a linha estava
operando**, não o limite do equipamento. Pode ser setpoint, velocidade ajustada
ou limite imposto a montante. Afirmar capacidade máxima exige ensaio com
alimentação saturada. Essa ressalva está na aba *Método* e não deve ser removida —
ela protege quem apresenta o relatório.

## Direção: ferramenta gráfica multiprojeto

O projeto vai evoluir para uma GUI que atende vários vídeos de projetos
diferentes. O que isso pede de quem mexe no código agora:

- **Não introduza estado global nem `cwd` implícito.** Hoje `config.json` e
  `saida/` são defaults do CLI; devem virar propriedades de um *projeto*
  (configuração, vídeo, resultados, alvo), com vários deles coexistindo.
- **Mantenha o núcleo livre de I/O de terminal.** `detect` e `render` imprimem
  progresso em `stderr` direto no laço; isso precisa virar callback de progresso
  para a GUI mostrar barra e permitir cancelamento. É a principal dívida
  arquitetural hoje.
- **`setup_gate.py` é uma implementação de marcação, não a única.** A GUI vai
  substituir a janela do OpenCV; preserve a separação entre *marcar geometria* e
  *interpretar geometria* (`config.py` já é puro).
- **Tudo que é ajustável mora em `config.py`** e é serializável. Não espalhe
  constantes de ajuste pelo código: a GUI precisa expor esses valores.
- **Preserve a rastreabilidade.** CSV com o instante de cada pacote e o sinal
  bruto do sensor são o que permite reauditar um relatório meses depois. Nenhuma
  otimização vale perdê-los.

## Como um ensaio costuma correr

O caminho que os ensaios percorreram, na ordem em que os problemas aparecem.
Nenhum destes passos é opcional se o número vai para um relatório.

1. **Ache onde há carga antes de medir qualquer coisa.** Um dos vídeos tinha
   22 minutos e **52 segundos** de operação; o resto era esteira vazia. Amostrar
   a ocupação da faixa a 2 fps e olhar a *amplitude* (p95−p05) por minuto separa
   carga de estrutura na hora: carga faz o sinal oscilar (0,38), estrutura fixa
   não (0,01). A média sozinha engana — ela fica alta e constante quando a faixa
   pegou corrimão.
2. **Corte o desperdício, de preferência a partir de zero.** `-c copy` corta em
   um instante e não recodifica. Cortando a partir de 0 os timestamps não mudam
   e CSV, config e medição anterior continuam válidos — verificado: diferença
   máxima de 0,0 ms. Cortando do meio, **anote o offset**: o TESTE 03 começa em
   t=41,500 s da gravação original.
3. **Confira o critério de cor contra o que a esteira leva.** Ver a seção de
   decisões: material que a cor não enxerga vira vão no sinal, não erro.
4. **Meça o trecho de carga, não o vídeo inteiro.** O nível de referência do
   gate sai de um percentil do próprio sinal; incluir minutos vazios derruba o
   percentil e muda os limiares. Foi o que separou 109 de 118 pacotes no mesmo
   vídeo.
5. **Confira no panorama antes de aceitar.** Um traço por pacote, um pacote por
   objeto. É o passo que pega o que os números não mostram.
6. **Confira a quantização dos intervalos.** Múltiplos inteiros da cadência é o
   esperado. Gap muito abaixo da cadência é contagem dupla; um vídeo tinha 15
   gaps de 0,10 s contra cadência de 0,372 s.

## Calibração: o que ajustar, e o que os números querem dizer

- **Não existe um jogo de limiares que sirva para todos os vídeos.** Tentado e
  medido: aplicar o mesmo par aos cinco ensaios de uma mesma série derrubou um deles
  de 142 para 106 pacotes. Varra por vídeo e escolha o **platô**.
- **Qual limiar domina depende da taxa de quadros.** A 60 fps o que manda é
  `gate_limiar_ocupado` (0,45 perdia sacos achatados; 0,25 os recupera) e o de
  vazio é quase indiferente. A 30 fps inverte: há metade das amostras dentro do
  vão, o sinal não chega tão fundo, e `gate_limiar_vazio` baixo funde pacotes
  vizinhos — 0,25 dá 142, 0,10 dá 106, com o mesmo sinal.
- **`gate_limiar_ocupado`** — fração do nível de referência para contar como
  ocupado. Procure **platô**: se a contagem fica igual num intervalo largo, o
  valor é robusto; se cai continuamente, você está cortando pacotes reais e o
  ensaio merece conferência extra.
- **`gate_min_vazio_s`** — quanto tempo o vão precisa durar. Sobe para matar
  contagem dupla vinda de entalhe no platô; **0,060 s** resolveu sem custo
  perceptível a 60 fps.
- **Ordem de grandeza que ajuda a farejar erro:** nesta linha a cadência é
  ~0,375 s e a esteira anda ~1 m/s (30 px/frame a 30 fps, 1280 px de largura).
  Um pacote cruza o gate em poucos frames.

## Vídeo MINI e sincronismo

O segundo vídeo (`"<nome> - MINI"`) entra como quadro no canto inferior direito
do OSD e desloca o gráfico para a esquerda. Convenção do alinhamento: o instante
`t` do principal corresponde a `t + offset` no MINI.

O offset mora em **`mini_offset` no config do ensaio**, não na linha de comando:
descobri-lo custa trabalho manual e o valor não muda entre um render e outro.
`--mini-offset` continua existindo e sobrepõe o config, para teste pontual; o
terminal informa de onde o valor veio. Mesma coisa para o caminho: `mini` no
config, `--mini` no argumento, e por último o nome com `" - MINI"`.

Ver a decisão sobre por que o alinhamento não é descoberto sozinho.

## Dados de ensaio

Uma medição identifica a operação em que foi feita, e **nome de arquivo e
caminho já são parte disso**. Por isso `projetos/` inteiro fica fora do
versionamento: vídeo, config e `saida/` de cada ensaio moram lá, um ensaio por
pasta.

O código não nomeia a operação medida, e não deve passar a nomear — nem em
comentário, nem em nome de arquivo, nem em valor padrão. Ao descrever uma
decisão que veio de medição, o vídeo é "o TESTE 02", e só.

A mesma regra vale para o histórico do git: um arquivo de resultado nomeado
pela operação continua legível em `git log` muito depois de sair da árvore.
`git log --diff-filter=A --name-only` mostra o que entrou e quando.

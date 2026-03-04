# OMRChecker — Guia de Configuração de Cartão de Respostas

> Do cartão físico ao `template.json` funcionando, usando **uv** (Astral) como executor.

---

## 1. Como o OMRChecker Funciona

O OMRChecker recebe uma imagem do cartão de respostas e executa o seguinte pipeline:

```
Foto/Scan → Detecta Marcadores de Canto → Warp (alinha perspectiva) → Detecta Bolhas → CSV de Resultados
```

O passo de **Warp** é feito pelo preprocessador `CropOnMarkers`: ele encontra os 4 marcadores de canto, corrige rotação e perspectiva, e entrega uma imagem com dimensões fixas (`pageDimensions`). **Todas as coordenadas do `template.json` são relativas a essa imagem já alinhada**, não à foto original.

---

## 2. Requisitos do Cartão de Respostas

### 2.1 Marcadores de Canto (OBRIGATÓRIO)

> ⚠️ Este é o requisito mais crítico. Sem os 4 marcadores o OMRChecker não consegue alinhar a imagem.

- **4 quadrados pretos sólidos**, um em cada canto do cartão
- Tamanho recomendado: **15–25mm** (visíveis mesmo em fotos de baixa resolução)
- **Fundo branco** ao redor de cada marcador — a margem de contraste é essencial
- Todos os 4 marcadores devem ser **idênticos** em tamanho e forma
- Devem estar dentro da área imprimível, não nas bordas extremas

✅ Funciona: quadrado preto sólido com margem branca de pelo menos 5mm ao redor  
❌ Não funciona: marcadores circulares, com bordas decorativas, ou de cores diferentes entre si

### 2.2 Bolhas

- Círculos vazios de tamanho **uniforme** em toda a folha
- Diâmetro mínimo recomendado: **6–8mm**
- Espaçamento **uniforme** entre bolhas na horizontal (`bubblesGap`) e vertical (`labelsGap`)
- Fundo branco atrás das bolhas — sem linhas ou sombras sobrepostas
- Máximo de 5 alternativas por questão quando usar `QTYPE_MCQ5`

### 2.3 Qualidade de Impressão e Preenchimento

- Papel branco ou levemente creme (não amarelado)
- Impressão em **preto sobre fundo branco**
- Resolução mínima para scan: **200 DPI**; para foto de celular: câmera de **8MP+**
- Preenchimento com **caneta esferográfica preta ou azul escura**, cobrindo o círculo completamente

---

## 3. Estrutura de Pastas

```
OMRChecker/
└── samples/
    └── SEU_EXAM/
        ├── template.json       ← configuração das bolhas (você cria manualmente)
        ├── omr_marker.jpg      ← imagem do marcador de canto (extraída do cartão)
        ├── config.json         ← configurações opcionais de processamento
        └── inputs/
            ├── cartao_01.jpg   ← fotos/scans dos cartões preenchidos
            ├── cartao_02.jpg
            └── ...
```

---

## 4. Extraindo o `omr_marker.jpg`

O marcador deve ser **extraído da imagem de referência** do cartão escaneado, não desenhado manualmente. Isso garante que o template matching funcione corretamente.

### Por que o marcador precisa de margem branca?

O `CropOnMarkers` usa template matching com processamento de erosão. Se o marcador não tiver a margem branca ao redor do quadrado preto, o matching falha por falta de contraste contextual.

### Script de extração

```python
import cv2
import numpy as np

IMAGE_PATH  = "samples/SEU_EXAM/image.png"   # cartão em branco escaneado
OUTPUT_PATH = "samples/SEU_EXAM/omr_marker.jpg"
MARGIN      = 10   # px de margem branca ao redor do quadrado preto

img  = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# Recortar canto superior esquerdo
corner = gray[0:120, 0:120]
_, thresh = cv2.threshold(corner, 50, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Pegar o maior contorno (o marcador)
c = max(contours, key=cv2.contourArea)
x, y, cw, ch = cv2.boundingRect(c)

# Extrair com margem
x1 = max(0, x - MARGIN)
y1 = max(0, y - MARGIN)
x2 = min(120, x + cw + MARGIN)
y2 = min(120, y + ch + MARGIN)
marker = gray[y1:y2, x1:x2]

cv2.imwrite(OUTPUT_PATH, marker)
print(f"Marcador salvo: {marker.shape[1]}x{marker.shape[0]}px")
print(f"sheetToMarkerWidthRatio = {w} / {marker.shape[1]} = {w / marker.shape[1]:.1f}")
```

### Calculando o `sheetToMarkerWidthRatio`

```
sheetToMarkerWidthRatio = largura_imagem_original / largura_do_omr_marker.jpg
```

Exemplo: imagem original `1241px` de largura, marcador `60px` → ratio = `1241 / 60 ≈ 21`

Arredonde para o inteiro mais próximo. Se o `CropOnMarkers` falhar, tente `±1`.

---

## 5. Criando o `template.json` Manualmente

### 5.1 Conceito fundamental

O `origin` de cada bloco é o **centro da bolha A da primeira questão** daquele bloco, em coordenadas da imagem **após o warp** (tamanho = `pageDimensions`).

```
origin = [x_da_bolha_A_da_Q1, y_da_bolha_A_da_Q1]
```

### 5.2 Como medir as coordenadas

**Opção A — pelo --setLayout (método iterativo):**
1. Crie um `template.json` aproximado com os valores que você acha que são os certos
2. Rode `uv run main.py -i samples/SEU_EXAM --setLayout`
3. Uma janela abre com quadrados cinzas sobrepostos às bolhas
4. Observe o deslocamento e ajuste os valores no JSON
5. Repita até os quadrados ficarem centrados nas bolhas

**Opção B — medindo na imagem warped:**
1. Abra a imagem do cartão em qualquer editor de imagem (GIMP, Paint, etc.)
2. Aplique o warp manualmente com um script ou use a imagem de saída do `--setLayout`
3. Leia as coordenadas de pixel da bolha A da Q1 diretamente

### 5.3 Estrutura completa do template.json

```json
{
  "pageDimensions": [LARGURA, ALTURA],
  "bubbleDimensions": [18, 18],
  "preProcessors": [
    {
      "name": "CropOnMarkers",
      "options": {
        "relativePath": "omr_marker.jpg",
        "sheetToMarkerWidthRatio": 21
      }
    }
  ],
  "fieldBlocks": {
    "nome_do_bloco": {
      "fieldType": "QTYPE_MCQ5",
      "origin": [X, Y],
      "bubblesGap": 24,
      "labelsGap": 27,
      "bubbleDimensions": [18, 18],
      "fieldLabels": ["q1", "q2", "q3", "q4", "q5"]
    }
  }
}
```

### 5.4 Descrição de cada campo

| Campo | Exemplo | Descrição |
|---|---|---|
| `pageDimensions` | `[1013, 1499]` | Dimensões da imagem após o warp |
| `bubbleDimensions` | `[18, 18]` | Tamanho da janela de detecção de cada bolha |
| `fieldType` | `QTYPE_MCQ5` | MCQ5 = 5 alternativas, MCQ4 = 4, etc. |
| `origin` | `[45, 1020]` | Centro da bolha A da Q1 do bloco (em pixels) |
| `bubblesGap` | `24` | Distância em pixels entre centros das bolhas A→B→C... |
| `labelsGap` | `27` | Distância em pixels entre centros das linhas Q1→Q2→Q3... |
| `bubbleDimensions` | `[18, 18]` | Janela de leitura individual de cada bolha |
| `fieldLabels` | `["q1","q2"...]` | Nomes das colunas no CSV de resultado |

### 5.5 Fórmulas para verificar antes de salvar

Antes de salvar, verifique que nenhum bloco transborda a página:

```
# Verificar overflow horizontal
origin[0] + (num_opcoes - 1) × bubblesGap + bubbleDimensions[0]  <  pageDimensions[0]

# Verificar overflow vertical  
origin[1] + (num_questoes - 1) × labelsGap + bubbleDimensions[1]  <  pageDimensions[1]

# bubbleDimensions ideal
bubbleDimensions = bubblesGap × 0.75  (arredondado)
```

---

## 6. Validando com --setLayout

```bash
uv run main.py -i samples/SEU_EXAM --setLayout
```

Uma janela abre mostrando a imagem do cartão com **quadrados cinzas sobrepostos** representando onde o OMRChecker vai ler cada bolha. O objetivo é que cada quadrado fique **centrado em cima da bolha correspondente**.

### O que ajustar quando está errado

| Sintoma no --setLayout | O que ajustar |
|---|---|
| Todos os blocos deslocados para baixo | Diminuir `origin[1]` de todos os blocos |
| Todos os blocos deslocados para cima | Aumentar `origin[1]` de todos os blocos |
| Todos os blocos deslocados para a direita | Diminuir `origin[0]` de todos os blocos |
| Bolhas se acumulam nas últimas questões | `labelsGap` pequeno demais — aumentar |
| Bolhas se acumulam nas últimas opções (A-E) | `bubblesGap` pequeno demais — aumentar |
| Só um bloco deslocado | Ajustar `origin` só daquele bloco |
| Erro `Overflowing field block` | Reduzir `origin[0]` ou `bubblesGap` do bloco indicado |

### Ciclo de ajuste recomendado

```
1. Editar template.json
2. uv run main.py -i samples/SEU_EXAM --setLayout
3. Observar deslocamento
4. Voltar ao passo 1 até alinhar
```

---

## 7. Processando os Cartões

Com o template validado, coloque as imagens dos cartões preenchidos em `inputs/` e execute:

```bash
uv run main.py -i samples/SEU_EXAM
```

Os resultados aparecem em `outputs/` como CSV com uma linha por cartão e uma coluna por questão.

---

## 8. Foto de Celular vs Scanner

O template é criado com um cartão **escaneado** (imagem plana e uniforme), mas o OMRChecker processa fotos de celular também, pois o `CropOnMarkers` corrige a perspectiva.

**Foto de celular funciona se:**
- Os 4 marcadores de canto estão visíveis e não cortados
- Iluminação uniforme, sem sombra forte sobre as bolhas
- Câmera de 8MP ou mais, foto nítida
- Papel sem dobras ou amassados
- Bolha preenchida com caneta (não lápis)

**Foto de celular falha se:**
- Algum marcador está cortado ou tampado com o dedo
- Há sombra sobre as bolhas
- O papel está muito amassado causando reflexo
- A foto está desfocada ou tremida

### Preprocessadores extras para robustez com celular

Adicione antes do `CropOnMarkers` no `template.json`:

```json
"preProcessors": [
  {
    "name": "GaussianBlur",
    "options": { "kSize": [3, 3], "sigmaX": 0 }
  },
  {
    "name": "Levels",
    "options": { "low": 50, "high": 220 }
  },
  {
    "name": "CropOnMarkers",
    "options": {
      "relativePath": "omr_marker.jpg",
      "sheetToMarkerWidthRatio": 21
    }
  }
]
```

O `GaussianBlur` reduz ruído de câmera e o `Levels` aumenta o contraste entre bolha preenchida e fundo.

---

## 9. Problemas Comuns

| Problema | Causa | Solução |
|---|---|---|
| Imagem aparece cortada ou virada | Marcador extraído sem margem branca | Regenerar `omr_marker.jpg` com `MARGIN = 10` no script |
| `Overflowing field block` | Bloco ultrapassa `pageDimensions` | Verificar fórmula de overflow e reduzir `origin[0]` ou `bubblesGap` |
| Nenhuma bolha detectada | `pageDimensions` errado | Medir a imagem warped real e atualizar o valor |
| Muitos erros aleatórios de detecção | `bubbleDimensions` grande demais | Reduzir para ~75% do `bubblesGap` |
| `CropOnMarkers` falha silenciosamente | `sheetToMarkerWidthRatio` errado | Testar `ratio ± 1` até funcionar |
| Foto de celular não processa | Marcador não detectado | Verificar se os 4 cantos estão visíveis na foto |
| CSV com respostas erradas mesmo alinhado | Threshold de detecção inadequado | Ajustar `config.json` — parâmetro `threshold_circles` |
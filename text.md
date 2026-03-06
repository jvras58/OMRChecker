## Visão Geral do Fluxo

```
Imagem → Pré-processamento → Leitura das Bolhas → Avaliação → CSV de Resultados
```

---

## 1. Ponto de Entrada

**main.py**
- Faz o parsing dos argumentos CLI (`--setLayout`, `--inputDir`, `--outputDir`, `--debug`)
- Chama `entry_point_for_args(args)` que itera sobre os diretórios de entrada

---

## 2. Orquestração de Diretórios

**entry.py**

| Função | Responsabilidade |
|---|---|
| `entry_point()` | Valida se o diretório existe e chama `process_dir` |
| `process_dir()` | Encontra `template.json`, `config.json`, recursão em subpastas |
| `show_template_layouts()` | Modo `--setLayout`: exibe o layout visual sem processar |
| `process_files()` | Itera sobre cada imagem OMR e chama o pipeline completo |
| `print_stats()` | Exibe resumo ao final |

**Fluxo interno do `process_dir`:**
```
Lê config.json local
    → Carrega Template
        → Encontra imagens .jpg/.png
            → Se --setLayout: mostra layout
            → Senão: process_files()
                → Recursão nos subdiretórios
```

---

## 3. Configuração e Template

**template.py** — Classe `Template`

Carrega e valida o `template.json`:

```
template.json
    ├── pageDimensions     → dimensões da imagem após warp
    ├── bubbleDimensions   → tamanho da janela de leitura de cada bolha
    ├── preProcessors      → lista de pré-processadores
    ├── fieldBlocks        → blocos de questões (origin, gaps, labels)
    └── customLabels       → rótulos customizados para o CSV
```

Classes auxiliares em template.py:
- `Template` — configuração geral
- `FieldBlock` — um grupo de questões adjacentes
- `Bubble` — uma bolha individual (tem `x`, `y`, `field_label`, `field_value`)

**config.py** — Valores padrão de configuração (dimensões, threshold, alinhamento, outputs)

---

## 4. Validação de Schemas

**template_schema.py**
**config_schema.py**
**evaluation_schema.py**

Validam os JSONs de entrada usando `jsonschema` antes de processar.

---

## 5. Pré-processadores de Imagem

**ImagePreprocessor.py** — Interface base

**manager.py** — Registro de processadores disponíveis

| Processador | Arquivo | O que faz |
|---|---|---|
| `CropPage` | CropPage.py | Detecta borda da página via contornos e faz warp |
| `CropOnMarkers` | CropOnMarkers.py | Detecta 4 marcadores de canto via template matching e faz warp |
| `FeatureBasedAlignment` | FeatureBasedAlignment.py | Alinha via ORB keypoints + homografia |
| `GaussianBlur` | builtins.py | Reduz ruído com blur gaussiano |
| `MedianBlur` | builtins.py | Reduz ruído com blur mediano |
| `Levels` | builtins.py | Ajuste de contraste |

**Fluxo do `CropOnMarkers`** (o mais comum):
```
Imagem original
    → Erode + Normaliza
    → Divide em 4 quadrantes
    → Template matching em cada quadrante (testa múltiplas escalas)
    → Encontra 4 centros dos marcadores
    → four_point_transform → imagem warped com pageDimensions fixas
```

---

## 6. Leitura das Bolhas (Coração do Sistema)

**core.py** — Classe `ImageInstanceOps`

| Método | Responsabilidade |
|---|---|
| `apply_preprocessors()` | Executa a cadeia de pré-processadores na imagem |
| `read_omr_response()` | **Pipeline principal** de leitura das bolhas |
| `draw_template_layout()` | Desenha os retângulos do template sobre a imagem |
| `get_global_threshold()` | Calcula o threshold global pela maior variação de intensidade |
| `get_local_threshold()` | Threshold local por strip de questão |

**Fluxo do `read_omr_response()`:**

```
1. Redimensiona para pageDimensions
2. Normaliza intensidade
3. (Opcional) Auto-alinhamento morfológico por coluna
4. Coleta intensidade média de cada bolha:
   └── Para cada FieldBlock
       └── Para cada strip de questão (linha de bolhas)
           └── Para cada bolha → cv2.mean() na janela [y:y+h, x:x+w]
5. Calcula threshold global (get_global_threshold)
   └── Ordena valores, encontra maior "salto" → linha divisória preenchido/vazio
6. Para cada bolha: compara valor com threshold
   └── Se abaixo do threshold → bolha marcada
7. Monta omr_response = { "q1": "A", "q2": "C", ... }
8. Gera imagem anotada (final_marked)
```

---

## 7. Utilitários de Imagem

**image.py** — Classe `ImageUtils` (só métodos estáticos)

| Método | O que faz |
|---|---|
| `resize_util()` | Redimensiona mantendo proporção |
| `normalize_util()` | Normaliza para [0, 255] |
| `four_point_transform()` | Transforma perspectiva dados 4 pontos |
| `adjust_gamma()` | Correção de gama |
| `auto_canny()` | Detecção de bordas adaptativa |

---

## 8. Avaliação e Resultados

**evaluation.py**
- Compara `omr_response` com gabarito do `evaluation.json`
- Calcula pontuação com marcação customizada (parcial, negativa, etc.)

**file.py**
- `setup_outputs_for_template()` — cria estrutura de pastas de saída
- `Paths` — gerencia caminhos: `CheckedOMRs/`, `Results/`, `Manual/`, `Evaluation/`

**Saídas geradas:**
```
outputs/
    ├── CheckedOMRs/        ← imagens anotadas com respostas detectadas
    ├── Results/
    │   └── Results_HH.csv  ← uma linha por cartão com todas as respostas
    ├── Evaluation/         ← CSVs com scores calculados
    └── Manual/
        ├── MultiMarkedFiles.csv ← cartões com múltiplas marcações
        └── ErrorFiles.csv       ← cartões com falha no processamento
```

---

## 9. Parsing e Constantes

**parsing.py** — Abre JSONs com defaults, faz merge de configurações

**common.py** — `FIELD_TYPES` (QTYPE_MCQ4, QTYPE_INT, etc.), códigos de erro, regex

**image_processing.py** — Constantes de visão computacional (kernels, cores, thresholds)

---

## Diagrama Resumido

```
main.py
  └── entry.py (process_dir / process_files)
        ├── template.py (Template + FieldBlock + Bubble)
        │     └── processors/ (CropOnMarkers, CropPage, FeatureBasedAlignment...)
        ├── core.py (ImageInstanceOps.read_omr_response)
        │     └── utils/image.py (ImageUtils)
        ├── evaluation.py (calcula score)
        └── utils/file.py (salva CSVs e imagens)
```
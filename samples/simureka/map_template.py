import sys
import cv2
import json
import numpy as np
from pathlib import Path

# Garante que a raiz do projeto está no sys.path (necessário ao rodar via uv run)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Importa diretamente do OMRChecker — mesmos parâmetros usados no processamento real
from src.utils.image import ImageUtils
from src.constants.image_processing import (
    DEFAULT_GAUSSIAN_BLUR_PARAMS_MARKER,
    DEFAULT_NORMALIZE_PARAMS,
    EROSION_PARAMS,
    QUADRANT_DIVISION,
)

# ── Configurações ────────────────────────────────────────────────────────────
IMAGE_PATH = Path("samples/simureka/image.png")
MARKER_PATH = Path("samples/simureka/omr_marker.jpg")
OUTPUT_PATH = Path("samples/simureka/template.json")
DISPLAY_WIDTH = 800  # largura da janela
SCREEN_HEIGHT = 900  # altura máxima da janela (cabe em 1080p)

# ── Estado global ─────────────────────────────────────────────────────────────
blocks = []
current_block = {}
mode = "origin"
scale = 1.0
scroll_y = 0  # posição atual do scroll em pixels do canvas
img_full = None  # imagem redimensionada completa
img_original = None
mouse_pos = (0, 0)

# Preview interativo (modo confirm) — ajustável com A/D/I/K ou arrastar
preview_rows = 15  # número de questões para preview
preview_cols = 5  # número de opções para preview
bubbles_gap_adj = 0  # ajuste fino sobre o valor medido
labels_gap_adj = 0  # ajuste fino sobre o valor medido

# Estado de drag
drag_target = None  # 'grid' | 'origin' | 'end_bubble' | 'next_label'
drag_offset = [0, 0]  # offset (original) entre ponto clicado e a origem do grid
DRAG_THRESHOLD = 18  # distância em px de tela para "pegar" um marcador


def apply_crop_on_markers(
    img_gray, marker_path, rescale_range=(35, 100), rescale_steps=10
):
    """
    Replica fiel do CropOnMarkers do OMRChecker usando os mesmos parâmetros e constantes.
    Retorna (warped_color, warped_gray) no espaço de coordenadas do template.
    """
    # ── Prepara marcador (igual a CropOnMarkers.load_marker) ─────────────
    marker_raw = cv2.imread(str(marker_path), cv2.IMREAD_GRAYSCALE)
    if marker_raw is None:
        raise FileNotFoundError(f"Marcador não encontrado: {marker_path}")

    marker = cv2.GaussianBlur(
        marker_raw,
        DEFAULT_GAUSSIAN_BLUR_PARAMS_MARKER["kernel_size"],
        DEFAULT_GAUSSIAN_BLUR_PARAMS_MARKER["sigma_x"],
    )
    marker = cv2.normalize(
        marker,
        None,
        alpha=DEFAULT_NORMALIZE_PARAMS["alpha"],
        beta=DEFAULT_NORMALIZE_PARAMS["beta"],
        norm_type=cv2.NORM_MINMAX,
    )
    marker = marker.astype(np.uint8)
    marker -= cv2.erode(
        marker,
        kernel=np.ones(EROSION_PARAMS["kernel_size"]),
        iterations=EROSION_PARAMS["iterations"],
    )

    # ── Prepara imagem (igual a CropOnMarkers.apply_filter, apply_erode_subtract=True) ──
    # Com apply_erode_subtract=True (padrão), a imagem é apenas normalizada
    img_es = ImageUtils.normalize_util(img_gray)

    h1, w1 = img_es.shape[:2]
    # ⚠️ usa os fatores reais do OMRChecker: height//3, width//2
    midh = h1 // QUADRANT_DIVISION["height_factor"]
    midw = w1 // QUADRANT_DIVISION["width_factor"]

    quads = [
        img_es[0:midh, 0:midw],
        img_es[0:midh, midw:w1],
        img_es[midh:h1, 0:midw],
        img_es[midh:h1, midw:w1],
    ]
    origins = [[0, 0], [midw, 0], [0, midh], [midw, midh]]

    # ── Encontra a melhor escala (igual a getBestMatch) ───────────────────
    _h, _w = marker.shape[:2]
    best_scale, all_max_t = None, 0.0
    descent = max(1, (rescale_range[1] - rescale_range[0]) // rescale_steps)

    for r0 in range(rescale_range[1], rescale_range[0], -descent):
        s = r0 / 100.0
        scaled = ImageUtils.resize_util_h(marker, u_height=max(1, int(_h * s)))
        res = cv2.matchTemplate(img_es, scaled, cv2.TM_CCOEFF_NORMED)
        max_t = float(res.max())
        if max_t > all_max_t:
            best_scale, all_max_t = s, max_t

    if best_scale is None or all_max_t < 0.3:
        raise RuntimeError(
            f"Marcador não detectado (melhor match={all_max_t:.3f}). "
            "Verifique o arquivo omr_marker.jpg."
        )

    opt_marker = ImageUtils.resize_util_h(marker, u_height=max(1, int(_h * best_scale)))
    mh, mw = opt_marker.shape[:2]

    # ── Localiza centro em cada quadrante ─────────────────────────────────
    centres = []
    for k, quad in enumerate(quads):
        res = cv2.matchTemplate(quad, opt_marker, cv2.TM_CCOEFF_NORMED)
        pt = np.argwhere(res == res.max())[0]  # [row, col]
        cx = pt[1] + origins[k][0] + mw / 2
        cy = pt[0] + origins[k][1] + mh / 2
        centres.append([cx, cy])
        print(
            f"  Quadrante {k + 1}: match={res.max():.3f}  centro=({cx:.1f}, {cy:.1f})"
        )

    print(f"  Escala ótima: {best_scale}  (QUADRANT height={midh}px width={midw}px)")

    # ── Warp ──────────────────────────────────────────────────────────────
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    warped_color = ImageUtils.four_point_transform(img_color, np.array(centres))
    warped_gray = ImageUtils.four_point_transform(img_gray, np.array(centres))
    return warped_color, warped_gray


def load_image():
    global img_original, img_full, scale
    raw = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Imagem não encontrada: {IMAGE_PATH}")

    print("\n🔍 Aplicando CropOnMarkers para obter o espaço de coordenadas correto...")
    warped_color, warped_gray = apply_crop_on_markers(raw, MARKER_PATH)
    print(f"  Imagem após warp: {warped_color.shape[1]}x{warped_color.shape[0]}px")

    img_original = warped_gray  # coordenadas originais = espaço pós-warp
    h, w = warped_color.shape[:2]
    scale = DISPLAY_WIDTH / w
    img_full = cv2.resize(warped_color, (DISPLAY_WIDTH, int(h * scale)))
    return img_full.copy()


def to_original(x, y):
    """Coordenada da janela → coordenada da imagem original."""
    return int(x / scale), int((y + scroll_y) / scale)


def to_display(x, y):
    """Coordenada original → coordenada da janela (com scroll)."""
    return int(x * scale), int(y * scale) - scroll_y


def get_viewport(canvas):
    """Recorta a região visível da imagem (scroll)."""
    h = canvas.shape[0]
    y1 = scroll_y
    y2 = scroll_y + SCREEN_HEIGHT
    y2 = min(y2, h)
    return canvas[y1:y2].copy()


def draw_block_circles(
    canvas, origin, bubbles_gap, labels_gap, num_rows, num_cols, color, thickness=1
):
    """Desenha círculos VAZADOS para um bloco. Permite ver o alinhamento com as bolhas reais."""
    ox = int(origin[0] * scale)
    oy = int(origin[1] * scale)
    # Raio = metade do gap entre bolhas (estimativa do tamanho real da bolha)
    r_ = max(3, int(bubbles_gap * scale / 2))
    for r in range(num_rows):
        for c in range(num_cols):
            cx = ox + int(c * bubbles_gap * scale)
            cy = oy + int(r * labels_gap * scale)
            cv2.circle(canvas, (cx, cy), r_, color, thickness)


def draw_state():
    """Desenha blocos e pontos sobre a imagem completa e retorna o viewport."""
    canvas = img_full.copy()

    # Blocos finalizados — círculos VAZADOS verdes
    for i, b in enumerate(blocks):
        ox = int(b["origin"][0] * scale)
        oy = int(b["origin"][1] * scale)
        draw_block_circles(
            canvas,
            b["origin"],
            b["bubblesGap"],
            b["labelsGap"],
            len(b["fieldLabels"]),
            b["num_options"],
            color=(0, 230, 0),
            thickness=2,
        )
        cv2.putText(
            canvas,
            f"Bloco {i + 1}: {b['name']}",
            (ox, oy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 230, 0),
            1,
        )

    # Preview do bloco em construção (modo confirm) — círculos azuis VAZADOS
    if mode == "confirm" and "origin" in current_block:
        o = current_block["origin"]
        e = current_block["end_bubble"]
        n = current_block["next_label"]
        bg_preview = abs(e[0] - o[0]) + bubbles_gap_adj
        lg_preview = abs(n[1] - o[1]) + labels_gap_adj
        # Usa num_options e num_rows do último input, ou padrões
        draw_block_circles(
            canvas,
            o,
            bg_preview,
            lg_preview,
            preview_rows,
            preview_cols,
            color=(255, 180, 0),
            thickness=2,
        )
        # Info de ajuste fino no topo
        adj_text = (
            f"gaps: bubbles={bg_preview}px  labels={lg_preview}px  "
            f"  [A/D] ajusta bolhas  [I/K] ajusta questoes  ENTER=confirmar"
        )
        cv2.rectangle(
            canvas, (0, scroll_y + 44), (DISPLAY_WIDTH, scroll_y + 64), (20, 20, 60), -1
        )
        cv2.putText(
            canvas,
            adj_text,
            (6, scroll_y + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 200, 255),
            1,
        )

    # Marcadores dos 3 pontos clicados
    colors_markers = {
        "origin": (0, 0, 255),  # vermelho
        "end_bubble": (255, 80, 0),  # azul
        "next_label": (0, 140, 255),  # laranja
    }
    for key, color in colors_markers.items():
        if key in current_block:
            px = int(current_block[key][0] * scale)
            py = int(current_block[key][1] * scale)
            cv2.drawMarker(canvas, (px, py), color, cv2.MARKER_CROSS, 24, 2)

    # HUD linha 1 — coordenadas do mouse
    orig_x, orig_y = to_original(*mouse_pos)
    hud = f"Original: ({orig_x}, {orig_y})  |  Scroll: {scroll_y}px  |  Modo: {mode}  |  Blocos: {len(blocks)}"
    cv2.rectangle(
        canvas, (0, scroll_y), (DISPLAY_WIDTH, scroll_y + 22), (30, 30, 30), -1
    )
    cv2.putText(
        canvas,
        hud,
        (6, scroll_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 255),
        1,
    )

    # HUD linha 2 — instrução atual
    instructions = {
        "origin": "PASSO 1: Clique no CENTRO da bolha A da Q1 do bloco",
        "end_bubble": "PASSO 2: Clique no CENTRO da bolha B da mesma Q1",
        "next_label": "PASSO 3: Clique no CENTRO da bolha A da Q2",
        "confirm": "PASSO 4: Ajuste com A/D/I/K se necessario, depois ENTER para confirmar",
    }
    cv2.rectangle(
        canvas, (0, scroll_y + 22), (DISPLAY_WIDTH, scroll_y + 44), (50, 50, 50), -1
    )
    cv2.putText(
        canvas,
        instructions.get(mode, ""),
        (6, scroll_y + 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 100),
        1,
    )

    return get_viewport(canvas)


def _hit_test(x, y):
    """
    Verifica o que o clique atingiu no modo confirm.
    Retorna: 'origin' | 'end_bubble' | 'next_label' | 'grid' | None
    """
    if not current_block:
        return None

    # 1. Marcadores individuais (cruzes) têm prioridade
    for key in ("origin", "end_bubble", "next_label"):
        if key not in current_block:
            continue
        px = int(current_block[key][0] * scale)
        py = int(current_block[key][1] * scale)
        if abs(px - x) <= DRAG_THRESHOLD and abs(py - y) <= DRAG_THRESHOLD:
            return key

    # 2. Qualquer clique dentro do bounding-box do grid move o grid inteiro
    if (
        "origin" not in current_block
        or "end_bubble" not in current_block
        or "next_label" not in current_block
    ):
        return None
    o = current_block["origin"]
    e = current_block["end_bubble"]
    n = current_block["next_label"]
    bg = abs(e[0] - o[0]) + bubbles_gap_adj
    lg = abs(n[1] - o[1]) + labels_gap_adj
    cols = preview_cols
    rows = preview_rows

    gx1 = int(o[0] * scale) - DRAG_THRESHOLD
    gy1 = int(o[1] * scale) - DRAG_THRESHOLD
    gx2 = int(o[0] * scale) + int((cols - 1) * bg * scale) + DRAG_THRESHOLD
    gy2 = int(o[1] * scale) + int((rows - 1) * lg * scale) + DRAG_THRESHOLD

    if gx1 <= x <= gx2 and gy1 <= y <= gy2:
        return "grid"

    return None


def mouse_callback(event, x, y, flags, param):
    global mode, current_block, mouse_pos, drag_target, drag_offset

    mouse_pos = (x, y)

    # ── Arrastar no modo confirm ─────────────────────────────────────────────
    if mode == "confirm":
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = _hit_test(x, y)
            if hit:
                drag_target = hit
                ox, oy = to_original(x, y)
                if hit == "grid":
                    # offset entre clique e origem do grid
                    drag_offset = [
                        ox - current_block["origin"][0],
                        oy - current_block["origin"][1],
                    ]
                return  # consome o evento

        elif event == cv2.EVENT_MOUSEMOVE and drag_target:
            ox, oy = to_original(x, y)
            if drag_target == "grid":
                # desloca origem; end_bubble e next_label seguem proporcionalmente
                prev_o = current_block["origin"][:]
                new_o = [ox - drag_offset[0], oy - drag_offset[1]]
                dx = new_o[0] - prev_o[0]
                dy = new_o[1] - prev_o[1]
                current_block["origin"] = new_o
                current_block["end_bubble"] = [
                    current_block["end_bubble"][0] + dx,
                    current_block["end_bubble"][1] + dy,
                ]
                current_block["next_label"] = [
                    current_block["next_label"][0] + dx,
                    current_block["next_label"][1] + dy,
                ]
            else:
                current_block[drag_target] = [ox, oy]
            return

        elif event == cv2.EVENT_LBUTTONUP and drag_target:
            ox, oy = to_original(x, y)
            if drag_target == "grid":
                print(
                    f"  🖱 grid movido → origem=({current_block['origin'][0]},{current_block['origin'][1]})"
                )
            else:
                current_block[drag_target] = [ox, oy]
                print(f"  🖱 {drag_target} → original=({ox},{oy})")
            drag_target = None
            return

    if event == cv2.EVENT_LBUTTONDOWN:
        ox, oy = to_original(x, y)
        print(f"  📍 tela=({x},{y}+scroll{scroll_y}) → original=({ox},{oy})")

        if mode == "origin":
            current_block = {"origin": [ox, oy]}
            mode = "end_bubble"
        elif mode == "end_bubble":
            current_block["end_bubble"] = [ox, oy]
            mode = "next_label"
        elif mode == "next_label":
            current_block["next_label"] = [ox, oy]
            mode = "confirm"


def ask_block_info():
    print("\n── Informações do bloco ──────────────────────────────")
    name = input("  Nome do bloco (ex: Questoes_1_15): ").strip()

    raw = input(f"  Quantas questões? [{preview_rows}]: ").strip()
    num_q = int(raw) if raw else preview_rows

    raw = input(f"  Quantas opções? (A-E=5) [{preview_cols}]: ").strip()
    num_options = int(raw) if raw else preview_cols

    start_q = int(input("  Número da 1ª questão (ex: 1): ").strip())
    labels = [f"q{i}" for i in range(start_q, start_q + num_q)]
    return name, labels, num_options


def compute_block(name, labels, num_options):
    o = current_block["origin"]
    e = current_block["end_bubble"]
    n = current_block["next_label"]

    bubbles_gap = abs(e[0] - o[0]) + bubbles_gap_adj
    labels_gap = abs(n[1] - o[1]) + labels_gap_adj
    # Diâmetro da bolha ≈ 80% do gap (com mínimo de 8px)
    bubble_dim = max(8, int(bubbles_gap * 0.8))

    return {
        "name": name,
        "origin": o,
        "bubblesGap": bubbles_gap,
        "labelsGap": labels_gap,
        "bubbleDimensions": [bubble_dim, bubble_dim],
        "fieldLabels": labels,
        "num_options": num_options,
    }


def save_template():
    # Dimensões = imagem já warped (espaço pós-CropOnMarkers)
    h, w = img_original.shape[:2]
    field_blocks = {}
    for b in blocks:
        field_blocks[b["name"]] = {
            "fieldType": f"QTYPE_MCQ{b['num_options']}",
            "origin": b["origin"],
            "bubblesGap": b["bubblesGap"],
            "labelsGap": b["labelsGap"],
            "bubbleDimensions": b["bubbleDimensions"],
            "fieldLabels": b["fieldLabels"],
        }

    template = {
        "pageDimensions": [w, h],
        "bubbleDimensions": blocks[0]["bubbleDimensions"] if blocks else [20, 20],
        "preProcessors": [
            {
                "name": "CropOnMarkers",
                "options": {
                    "relativePath": MARKER_PATH.name,
                },
            }
        ],
        "fieldBlocks": field_blocks,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Template salvo em: {OUTPUT_PATH}")
    print(json.dumps(template, indent=2, ensure_ascii=False))


def main():
    global scroll_y, mode, current_block
    global \
        preview_rows, \
        preview_cols, \
        bubbles_gap_adj, \
        labels_gap_adj, \
        drag_target, \
        drag_offset

    load_image()
    img_h = img_full.shape[0]
    max_scroll = max(0, img_h - SCREEN_HEIGHT)
    scroll_step = 40

    cv2.namedWindow("OMR Template Mapper", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("OMR Template Mapper", DISPLAY_WIDTH, SCREEN_HEIGHT)
    cv2.setMouseCallback("OMR Template Mapper", mouse_callback)

    print("\n🗺️  OMR Template Mapper")
    print("=" * 60)
    print("  W / S ou ↑↓     →  scroll")
    print("  ENTER           →  confirmar bloco (após 3 cliques)")
    print("  R               →  refazer passo atual")
    print("  A / D           →  diminuir/aumentar gap entre bolhas (ajuste fino)")
    print("  I / K           →  diminuir/aumentar gap entre questões (ajuste fino)")
    print("  ARRASTAR grid  →  mover o bloco inteiro")
    print("  ARRASTAR cruz  →  ajustar ponto individual (origem/bolha/questão)")
    print("  Q               →  salvar e sair")
    print("=" * 60)

    while True:
        frame = draw_state()
        cv2.imshow("OMR Template Mapper", frame)
        key = cv2.waitKey(20) & 0xFF

        # Scroll
        if key in (ord("s"), 84):
            scroll_y = min(scroll_y + scroll_step, max_scroll)
        elif key in (ord("w"), 82):
            scroll_y = max(scroll_y - scroll_step, 0)

        # Ajuste fino dos gaps (só no modo confirm)
        elif key == ord("d") and mode == "confirm":
            bubbles_gap_adj += 1
            print(f"  ↔ bubblesGap adj: +{bubbles_gap_adj}px")
        elif key == ord("a") and mode == "confirm":
            bubbles_gap_adj -= 1
            print(f"  ↔ bubblesGap adj: {bubbles_gap_adj}px")
        elif key == ord("k") and mode == "confirm":
            labels_gap_adj += 1
            print(f"  ↕ labelsGap adj: +{labels_gap_adj}px")
        elif key == ord("i") and mode == "confirm":
            labels_gap_adj -= 1
            print(f"  ↕ labelsGap adj: {labels_gap_adj}px")

        # ENTER — confirmar bloco
        elif key == 13 and mode == "confirm":
            name, labels, num_options = ask_block_info()
            preview_rows = len(labels)
            preview_cols = num_options
            block = compute_block(name, labels, num_options)
            blocks.append(block)
            print(f"\n  ✅ Bloco '{name}' adicionado! ({len(blocks)} bloco(s))")
            print(
                f"     bubblesGap={block['bubblesGap']}px  labelsGap={block['labelsGap']}px  bubbleDim={block['bubbleDimensions']}"
            )
            current_block = {}
            mode = "origin"
            bubbles_gap_adj = 0
            labels_gap_adj = 0
            drag_target = None

        # R — refazer
        elif key == ord("r"):
            current_block = {}
            mode = "origin"
            bubbles_gap_adj = 0
            labels_gap_adj = 0
            drag_target = None
            print("\n  🔄 Resetado.")

        # Q — sair
        elif key == ord("q"):
            if blocks:
                save_template()
            else:
                print("\n  ⚠️  Nenhum bloco mapeado.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

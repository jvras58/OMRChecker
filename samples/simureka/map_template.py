"""
OMR Template Mapper - versao corrigida
=======================================
PROBLEMA DA VERSAO ANTERIOR:
  O script calibrava as coordenadas na imagem ORIGINAL (antes do CropOnMarkers).
  O template.json precisa das coordenadas na imagem JA ALINHADA pelo warp.

SOLUCAO:
  Este script aplica o CropOnMarkers ANTES de mostrar a imagem.
  As coordenadas coletadas ja sao corretas para o template.json.

USO:
  1. Ajuste IMAGE_PATH e MARKER_PATH abaixo
  2. python3 omr_template_mapper.py
  3. Siga as instrucoes na janela

CONTROLES:
  W / S    scroll vertical
  A / D    gap entre bolhas +-1px  (modo confirm)
  I / K    gap entre questoes +-1px (modo confirm)
  ARRASTAR mover grid ou ponto individual
  ENTER    confirmar bloco
  R        refazer bloco atual
  Q        salvar template.json e sair
"""

import cv2
import json
import numpy as np
from pathlib import Path

# Configuracoes - AJUSTE AQUI
IMAGE_PATH = Path("samples/simureka/image.png")
MARKER_PATH = Path("samples/simureka/omr_marker.jpg")
OUTPUT_PATH = Path("samples/simureka/template.json")

DISPLAY_WIDTH = 900
SCREEN_HEIGHT = 900
TARGET_W = 1013  # pageDimensions width
TARGET_H = 1499  # pageDimensions height
SHEET_TO_MARKER_RATIO = 21

MIN_MATCH_THRESHOLD = 0.3
MAX_MATCH_VARIATION = 0.41
MARKER_RESCALE_RANGE = (35, 100)
MARKER_RESCALE_STEPS = 10

# Estado global
blocks = []
current_block = {}
mode = "origin"
scale = 1.0
scroll_y = 0
img_warped = None
img_display = None
mouse_pos = (0, 0)
preview_rows = 15
preview_cols = 5
bubbles_gap_adj = 0
labels_gap_adj = 0
drag_target = None
drag_offset = [0, 0]
DRAG_THRESHOLD = 18


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts, tw, th):
    rect = order_points(np.array(pts, dtype="float32"))
    dst = np.array(
        [[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype="float32"
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (tw, th))


def apply_crop_on_markers(image, marker):
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3
        else image.copy()
    )
    mg = (
        cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
        if len(marker.shape) == 3
        else marker.copy()
    )
    mg = cv2.GaussianBlur(mg, (5, 5), 0)
    mg = cv2.normalize(mg, None, 0, 255, cv2.NORM_MINMAX)
    mg -= cv2.erode(mg, np.ones((5, 5), np.uint8), iterations=5)

    eroded = cv2.erode(gray, np.ones((5, 5), np.uint8), iterations=5)
    ip = cv2.normalize(gray - eroded, None, 0, 255, cv2.NORM_MINMAX)

    hi, wi = ip.shape
    mh2, mw2 = hi // 3, wi // 2
    quads = {
        0: ip[0:mh2, 0:mw2],
        1: ip[0:mh2, mw2:wi],
        2: ip[mh2:hi, 0:mw2],
        3: ip[mh2:hi, mw2:wi],
    }
    origins = [[0, 0], [mw2, 0], [0, mh2], [mw2, mh2]]

    mh, mw = mg.shape[:2]
    desc = (MARKER_RESCALE_RANGE[1] - MARKER_RESCALE_RANGE[0]) // MARKER_RESCALE_STEPS
    best_scale, all_max_t = None, 0
    for r0 in range(MARKER_RESCALE_RANGE[1], MARKER_RESCALE_RANGE[0], -desc):
        s = r0 / 100.0
        res_m = cv2.resize(mg, (int(mw * s), int(mh * s)))
        if res_m.shape[0] >= hi or res_m.shape[1] >= wi:
            continue
        res = cv2.matchTemplate(ip, res_m, cv2.TM_CCOEFF_NORMED)
        t = float(res.max())
        if t > all_max_t:
            all_max_t, best_scale = t, s

    if best_scale is None or all_max_t < MIN_MATCH_THRESHOLD:
        print(
            f"  AVISO: marcador nao encontrado (max_t={all_max_t:.3f}), usando fallback."
        )
        return None

    optimal = cv2.resize(mg, (int(mw * best_scale), int(mh * best_scale)))
    oh, ow = optimal.shape[:2]
    centres = []
    for k in range(4):
        res = cv2.matchTemplate(quads[k], optimal, cv2.TM_CCOEFF_NORMED)
        t = float(res.max())
        if t < MIN_MATCH_THRESHOLD or abs(all_max_t - t) >= MAX_MATCH_VARIATION:
            print(f"  AVISO: quadrante {k + 1} falhou (t={t:.3f}), usando fallback.")
            return None
        pt = np.argwhere(res == res.max())[0]
        centres.append([pt[1] + origins[k][0] + ow / 2, pt[0] + origins[k][1] + oh / 2])

    print(f"  OK: escala={best_scale:.2f} max_t={all_max_t:.3f}")
    print(f"  Centros: {[(round(c[0]), round(c[1])) for c in centres]}")
    return four_point_transform(image, centres, TARGET_W, TARGET_H)


def to_original(x, y):
    return int(x / scale), int((y + scroll_y) / scale)


def draw_circles(canvas, origin, bg, lg, rows, cols, color, thickness=1):
    r_ = max(3, int(bg * scale / 2))
    for r in range(rows):
        for c in range(cols):
            cx = int(origin[0] * scale) + int(c * bg * scale)
            cy = int(origin[1] * scale) + int(r * lg * scale)
            cv2.circle(canvas, (cx, cy), r_, color, thickness)


def get_viewport(canvas):
    y1, y2 = scroll_y, min(scroll_y + SCREEN_HEIGHT, canvas.shape[0])
    return canvas[y1:y2].copy()


def draw_state():
    canvas = img_display.copy()

    for i, b in enumerate(blocks):
        draw_circles(
            canvas,
            b["origin"],
            b["bubblesGap"],
            b["labelsGap"],
            len(b["fieldLabels"]),
            b["num_options"],
            (0, 230, 0),
            2,
        )
        cv2.putText(
            canvas,
            f"Bloco {i + 1}: {b['name']}",
            (int(b["origin"][0] * scale), int(b["origin"][1] * scale) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 230, 0),
            1,
        )

    if mode == "confirm" and "origin" in current_block:
        o = current_block["origin"]
        bg = abs(current_block["end_bubble"][0] - o[0]) + bubbles_gap_adj
        lg = abs(current_block["next_label"][1] - o[1]) + labels_gap_adj
        draw_circles(canvas, o, bg, lg, preview_rows, preview_cols, (0, 165, 255), 2)
        info = f"bolhas={bg}px  questoes={lg}px  | A/D=bolhas  I/K=questoes  ENTER=confirmar"
        cv2.rectangle(
            canvas, (0, scroll_y + 44), (DISPLAY_WIDTH, scroll_y + 64), (20, 20, 60), -1
        )
        cv2.putText(
            canvas,
            info,
            (6, scroll_y + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 200, 255),
            1,
        )

    for key, color in [
        ("origin", (0, 0, 255)),
        ("end_bubble", (255, 80, 0)),
        ("next_label", (0, 140, 255)),
    ]:
        if key in current_block:
            cv2.drawMarker(
                canvas,
                (
                    int(current_block[key][0] * scale),
                    int(current_block[key][1] * scale),
                ),
                color,
                cv2.MARKER_CROSS,
                24,
                2,
            )

    ox, oy = to_original(*mouse_pos)
    hud = f"warped=({ox},{oy})  scroll={scroll_y}  modo={mode}  blocos={len(blocks)}"
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

    instrs = {
        "origin": "PASSO 1: clique no centro da bolha A da primeira questao do bloco",
        "end_bubble": "PASSO 2: clique no centro da bolha B da mesma questao",
        "next_label": "PASSO 3: clique no centro da bolha A da segunda questao",
        "confirm": "PASSO 4: ajuste A/D/I/K se necessario, depois ENTER para confirmar",
    }
    cv2.rectangle(
        canvas, (0, scroll_y + 22), (DISPLAY_WIDTH, scroll_y + 44), (50, 50, 50), -1
    )
    cv2.putText(
        canvas,
        instrs.get(mode, ""),
        (6, scroll_y + 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 100),
        1,
    )
    return get_viewport(canvas)


def _hit_test(x, y):
    for key in ("origin", "end_bubble", "next_label"):
        if key not in current_block:
            continue
        px, py = int(current_block[key][0] * scale), int(current_block[key][1] * scale)
        if abs(px - x) <= DRAG_THRESHOLD and abs(py - y) <= DRAG_THRESHOLD:
            return key
    if all(k in current_block for k in ("origin", "end_bubble", "next_label")):
        o = current_block["origin"]
        bg = abs(current_block["end_bubble"][0] - o[0]) + bubbles_gap_adj
        lg = abs(current_block["next_label"][1] - o[1]) + labels_gap_adj
        gx1 = int(o[0] * scale) - DRAG_THRESHOLD
        gy1 = int(o[1] * scale) - DRAG_THRESHOLD
        gx2 = int(o[0] * scale) + int((preview_cols - 1) * bg * scale) + DRAG_THRESHOLD
        gy2 = int(o[1] * scale) + int((preview_rows - 1) * lg * scale) + DRAG_THRESHOLD
        if gx1 <= x <= gx2 and gy1 <= y <= gy2:
            return "grid"
    return None


def mouse_callback(event, x, y, flags, param):
    global mode, current_block, mouse_pos, drag_target, drag_offset
    mouse_pos = (x, y)

    if mode == "confirm":
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = _hit_test(x, y)
            if hit:
                drag_target = hit
                ox, oy = to_original(x, y)
                if hit == "grid":
                    drag_offset = [
                        ox - current_block["origin"][0],
                        oy - current_block["origin"][1],
                    ]
                return
        elif event == cv2.EVENT_MOUSEMOVE and drag_target:
            ox, oy = to_original(x, y)
            if drag_target == "grid":
                prev = current_block["origin"][:]
                no = [ox - drag_offset[0], oy - drag_offset[1]]
                dx, dy = no[0] - prev[0], no[1] - prev[1]
                current_block["origin"] = no
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
            if drag_target != "grid":
                current_block[drag_target] = [ox, oy]
            drag_target = None
            return

    if event == cv2.EVENT_LBUTTONDOWN:
        ox, oy = to_original(x, y)
        print(f"  clicou warped=({ox},{oy})")
        if mode == "origin":
            current_block = {"origin": [ox, oy]}
            mode = "end_bubble"
        elif mode == "end_bubble":
            current_block["end_bubble"] = [ox, oy]
            mode = "next_label"
        elif mode == "next_label":
            current_block["next_label"] = [ox, oy]
            mode = "confirm"


def save_template():
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
        "pageDimensions": [TARGET_W, TARGET_H],
        "bubbleDimensions": blocks[0]["bubbleDimensions"] if blocks else [18, 18],
        "preProcessors": [
            {
                "name": "CropOnMarkers",
                "options": {
                    "relativePath": "omr_marker.jpg",
                    "sheetToMarkerWidthRatio": SHEET_TO_MARKER_RATIO,
                },
            }
        ],
        "fieldBlocks": field_blocks,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    print(f"\nTemplate salvo em: {OUTPUT_PATH}")
    print(json.dumps(template, indent=2, ensure_ascii=False))


def main():
    global img_warped, img_display, scale, scroll_y, mode, current_block
    global preview_rows, preview_cols, bubbles_gap_adj, labels_gap_adj
    global drag_target, drag_offset

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {IMAGE_PATH}")

    marker = cv2.imread(str(MARKER_PATH))
    if marker is None:
        raise FileNotFoundError(f"Marcador nao encontrado: {MARKER_PATH}")

    print("\nAplicando CropOnMarkers...")
    warped = apply_crop_on_markers(image, marker)
    img_warped = (
        warped if warped is not None else cv2.resize(image, (TARGET_W, TARGET_H))
    )
    print(f"Imagem warped: {img_warped.shape[1]}x{img_warped.shape[0]}")

    h_w, w_w = img_warped.shape[:2]
    scale = DISPLAY_WIDTH / w_w
    img_display = cv2.resize(img_warped, (DISPLAY_WIDTH, int(h_w * scale)))
    max_scroll = max(0, img_display.shape[0] - SCREEN_HEIGHT)

    cv2.namedWindow("OMR Template Mapper", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("OMR Template Mapper", DISPLAY_WIDTH, SCREEN_HEIGHT)
    cv2.setMouseCallback("OMR Template Mapper", mouse_callback)

    print(
        "\nOMR Template Mapper - coordenadas = imagem WARPED (corretas para template.json)"
    )
    print(
        "W/S=scroll  A/D=gap bolhas  I/K=gap questoes  ENTER=confirmar  R=refazer  Q=salvar e sair"
    )

    while True:
        frame = draw_state()
        cv2.imshow("OMR Template Mapper", frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("s"), 84):
            scroll_y = min(scroll_y + 40, max_scroll)
        elif key in (ord("w"), 82):
            scroll_y = max(scroll_y - 40, 0)
        elif key == ord("d") and mode == "confirm":
            bubbles_gap_adj += 1
        elif key == ord("a") and mode == "confirm":
            bubbles_gap_adj -= 1
        elif key == ord("k") and mode == "confirm":
            labels_gap_adj += 1
        elif key == ord("i") and mode == "confirm":
            labels_gap_adj -= 1
        elif key == 13 and mode == "confirm":
            print("\nInformacoes do bloco:")
            name = input("  Nome (ex: questoes_1_15): ").strip()
            raw = input(f"  Qtd questoes? [{preview_rows}]: ").strip()
            num_q = int(raw) if raw else preview_rows
            raw = input(f"  Qtd opcoes? [{preview_cols}]: ").strip()
            num_opt = int(raw) if raw else preview_cols
            start = int(input("  Numero da 1a questao: ").strip())
            labels = [f"q{i}" for i in range(start, start + num_q)]

            o = current_block["origin"]
            e = current_block["end_bubble"]
            n = current_block["next_label"]
            bg = abs(e[0] - o[0]) + bubbles_gap_adj
            lg = abs(n[1] - o[1]) + labels_gap_adj
            bd = max(8, int(bg * 0.8))
            block = {
                "name": name,
                "origin": o,
                "bubblesGap": bg,
                "labelsGap": lg,
                "bubbleDimensions": [bd, bd],
                "fieldLabels": labels,
                "num_options": num_opt,
            }
            blocks.append(block)
            print(f"  Bloco '{name}' adicionado! bubblesGap={bg} labelsGap={lg}")

            current_block = {}
            mode = "origin"
            bubbles_gap_adj = 0
            labels_gap_adj = 0
            drag_target = None
            preview_rows = num_q
            preview_cols = num_opt
        elif key == ord("r"):
            current_block = {}
            mode = "origin"
            bubbles_gap_adj = 0
            labels_gap_adj = 0
            drag_target = None
            print("  Resetado.")
        elif key == ord("q"):
            if blocks:
                save_template()
            else:
                print("  Nenhum bloco mapeado.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

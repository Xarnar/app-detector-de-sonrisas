"""
capture_dataset.py — Captura de recortes para fine-tuning del clasificador de sonrisas.

Usa EXACTAMENTE el mismo detector (YuNet) y el mismo preprocesado (crop_face + preprocess)
que app_sonrisa.py, garantizando que los recortes nuevos sean coherentes con el dataset
de entrenamiento (mismo tamaño 64×64, misma escala de grises, mismo margen).

Teclas:
  s         → guarda el recorte actual en  mis_fotos/sonrie/
  n         → guarda el recorte actual en  mis_fotos/no_sonrie/
  q         → salir
  +  /  -   → ajustar margen del recorte (igual que en app_sonrisa.py)

Uso:
  python capture_dataset.py
  python capture_dataset.py --margin -0.10 --source 0
"""

import argparse
import os
import time

import cv2
import numpy as np

# ── Importamos las funciones YA probadas de app_sonrisa.py ───────────────────
# Así garantizamos que el preprocesado es IDÉNTICO al del entrenamiento.
from app_sonrisa import (
    YUNET_FILE,
    IMG_SIZE,
    load_detector,
    detect_faces,
    crop_face,
    preprocess,
)

# ── Carpetas de destino ───────────────────────────────────────────────────────
DIR_SONRIE    = os.path.join("mis_fotos", "sonrie")
DIR_NO_SONRIE = os.path.join("mis_fotos", "no_sonrie")

WHITE = (255, 255, 255)
GREEN = (60, 200, 60)
RED   = (50, 50, 230)
CYAN  = (200, 200, 0)


def save_crop(preprocessed_crop: np.ndarray, folder: str, label: str) -> str:
    """Guarda el recorte preprocesado (64×64 float32 0-255) como PNG en uint8."""
    os.makedirs(folder, exist_ok=True)
    n_existing = len([f for f in os.listdir(folder) if f.endswith(".png")])
    filename = f"{label}_{int(time.time())}_{n_existing:04d}.png"
    path = os.path.join(folder, filename)
    # Convertimos de float32 a uint8 para guardarlo como imagen estándar
    img_uint8 = preprocessed_crop[..., 0].astype(np.uint8)
    cv2.imwrite(path, img_uint8)
    return path


def pick_largest_face(boxes: list) -> tuple | None:
    """Devuelve la caja (x,y,w,h) de la cara más grande, o None si no hay."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


def draw_hud(frame: np.ndarray, margin: float, cnt_s: int, cnt_n: int,
             has_face: bool, msg: str) -> None:
    """Dibuja el HUD con fondo semitransparente (igual que en app_sonrisa.py)."""
    lines = [
        f"margen: {margin:.2f}  |  +/- para ajustar",
        f"sonrie: {cnt_s}   no_sonrie: {cnt_n}",
        f"[s] guardar sonrie  [n] guardar no sonrie  [q] salir",
    ]
    if msg:
        lines.append(msg)

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    line_h = 22
    pad = 8
    max_w = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines)
    h_box = line_h * len(lines) + pad * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (4, 4), (max_w + pad * 2, h_box), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, line in enumerate(lines):
        color = WHITE
        if "[s]" in line:
            color = (180, 230, 180)   # verde claro para la ayuda de teclas
        cv2.putText(frame, line, (pad, pad + line_h * (i + 1) - 4),
                    font, scale, color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(
        description="Captura recortes 64×64 para fine-tuning."
    )
    ap.add_argument("--source", default="0",
                    help="Índice de cámara (0, 1 …) o ruta a un video.")
    ap.add_argument("--margin", type=float, default=-0.10,
                    help="Fracción del recorte (calibrada en Tarea 3: -0.10).")
    ap.add_argument("--det-score", type=float, default=0.7,
                    help="Umbral de confianza para YuNet.")
    ap.add_argument("--no-mirror", action="store_true",
                    help="Desactiva el espejo horizontal de la cámara.")
    args = ap.parse_args()

    margin = args.margin

    # ── Detector (el mismo YuNet de app_sonrisa.py) ──────────────────────────
    detector = load_detector(YUNET_FILE, args.det_score)

    # ── Cámara ───────────────────────────────────────────────────────────────
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente: {args.source}")

    # Contadores de imágenes guardadas
    cnt_sonrie, cnt_no_sonrie = 0, 0
    msg, msg_until = "", 0.0   # mensaje de feedback temporal en pantalla

    print("Listo. Colócate frente a la cámara.")
    print("  s → guardar como 'sonríe'")
    print("  n → guardar como 'no sonríe'")
    print("  +/- → ajustar margen del recorte")
    print("  q → salir")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not args.no_mirror and isinstance(src, int):
            frame = cv2.flip(frame, 1)

        # ── Detección ────────────────────────────────────────────────────────
        boxes  = detect_faces(detector, frame)
        target = pick_largest_face(boxes)   # cara principal = la más grande

        # Preparamos el recorte de la cara principal (si existe)
        current_crop = None   # recorte preprocesado listo para guardar
        if target is not None:
            raw = crop_face(frame, target, margin)
            if raw is not None:
                current_crop = preprocess(raw)   # (64,64,1) float32 0-255

        # ── Dibujo de cajas ──────────────────────────────────────────────────
        for box in boxes:
            x, y, w, h = box
            color = CYAN if box == target else WHITE
            thickness = 2 if box == target else 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        if target is not None:
            x, y, w, h = target
            cv2.putText(frame, "ACTIVA", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)

        # ── HUD principal ────────────────────────────────────────────────────
        now = time.time()
        active_msg = msg if now < msg_until else ""
        draw_hud(frame, margin, cnt_sonrie, cnt_no_sonrie,
                 current_crop is not None, active_msg)

        cv2.imshow("Captura de dataset", frame)

        # ── Ventana de preview del recorte (lo que vería el modelo) ─────────
        if current_crop is not None:
            # Escalamos 64×64 → 192×192 para verlo bien
            preview = cv2.resize(
                current_crop[..., 0].astype(np.uint8),
                (192, 192),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow("Recorte (lo que ve el modelo)", preview)
        else:
            # Sin cara: muestra imagen negra con texto
            blank = np.zeros((192, 192), np.uint8)
            cv2.putText(blank, "sin cara", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 128, 1, cv2.LINE_AA)
            cv2.imshow("Recorte (lo que ve el modelo)", blank)

        # ── Teclas ───────────────────────────────────────────────────────────
        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break

        elif k in (ord("+"), ord("=")):
            margin = round(min(0.6, margin + 0.02), 2)

        elif k in (ord("-"), ord("_")):
            margin = round(max(-0.3, margin - 0.02), 2)

        elif k == ord("s"):
            if current_crop is not None:
                path = save_crop(current_crop, DIR_SONRIE, "sonrie")
                cnt_sonrie += 1
                msg = f"✓ Guardado en {path}"
                msg_until = time.time() + 2.0
                print(f"[sonrie]    {path}  (total: {cnt_sonrie})")
            else:
                msg = "⚠ No se detecta ninguna cara"
                msg_until = time.time() + 1.5

        elif k == ord("n"):
            if current_crop is not None:
                path = save_crop(current_crop, DIR_NO_SONRIE, "no_sonrie")
                cnt_no_sonrie += 1
                msg = f"✓ Guardado en {path}"
                msg_until = time.time() + 2.0
                print(f"[no_sonrie] {path}  (total: {cnt_no_sonrie})")
            else:
                msg = "⚠ No se detecta ninguna cara"
                msg_until = time.time() + 1.5

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSesión terminada. Imágenes guardadas:")
    print(f"  sonrie     → {DIR_SONRIE}    ({cnt_sonrie} fotos)")
    print(f"  no_sonrie  → {DIR_NO_SONRIE}  ({cnt_no_sonrie} fotos)")


if __name__ == "__main__":
    main()


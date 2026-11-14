#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Práctica 6: Detección de Rostros con Ventana Deslizante + Pirámide de Imágenes
==============================================================================

Pipeline clásico-convolucional:
  1. Pirámide de imágenes (multiescala, conserva la relación de aspecto).
  2. Ventana deslizante de tamaño fijo sobre cada nivel de la pirámide.
  3. Cada ventana se clasifica con una red convolucional PREENTRENADA
     (ResNet-10 SSD de Caffe, cargada con cv2.dnn.readNetFromCaffe), en lotes.
  4. Mapeo inverso de las ventanas aceptadas a coordenadas de la imagen original.
  5. Supresión de No Máximos (NMS) para dejar un único recuadro por rostro.

Uso por consola:
    python practica_6_deteccion_rostros.py ruta/imagen.jpg --step 16 --scale 1.25

Requisitos: numpy, opencv-python (serie 4.x: la serie 5.x eliminó el importador Caffe).
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Modelo preentrenado (pesos oficiales de OpenCV)
# --------------------------------------------------------------------------
PROTO_NAME = "deploy.prototxt"
MODEL_NAME = "res10_300x300_ssd_iter_140000.caffemodel"
PROTO_URL = ("https://raw.githubusercontent.com/opencv/opencv/4.x/"
             "samples/dnn/face_detector/deploy.prototxt")
MODEL_URL = ("https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
             "dnn_samples_face_detector_20170830/" + MODEL_NAME)
MEAN_BGR = (104.0, 177.0, 123.0)   # media con la que fue entrenada la red
NET_INPUT = (300, 300)             # tamaño de entrada fijo del SSD


def descargar_modelo(directorio="."):
    """Descarga deploy.prototxt y los pesos .caffemodel si no existen."""
    os.makedirs(directorio, exist_ok=True)
    rutas = []
    for nombre, url in ((PROTO_NAME, PROTO_URL), (MODEL_NAME, MODEL_URL)):
        ruta = os.path.join(directorio, nombre)
        if not os.path.exists(ruta) or os.path.getsize(ruta) < 1000:
            print(f"[modelo] descargando {nombre} ...")
            urllib.request.urlretrieve(url, ruta)
        rutas.append(ruta)
    return rutas[0], rutas[1]


def cargar_red(directorio="."):
    """Carga la red ResNet-10 SSD con pesos oficiales (sin inicialización aleatoria)."""
    if not hasattr(cv2.dnn, "readNetFromCaffe"):
        raise RuntimeError(
            f"OpenCV {cv2.__version__} no incluye el importador Caffe. "
            "Instala la serie 4.x: pip install 'opencv-python-headless<5'")
    proto, pesos = descargar_modelo(directorio)
    return cv2.dnn.readNetFromCaffe(proto, pesos)


# --------------------------------------------------------------------------
# Desafío 2: variabilidad de tamaño -> Pirámide de imágenes
# --------------------------------------------------------------------------
def image_pyramid(image, scale=1.25, min_size=(96, 96)):
    """
    Generador de la pirámide. En el nivel l la imagen se reduce por scale**l
    (mismo factor en ancho y alto => se preserva el aspect ratio).
    Devuelve (level, scale_factor, resized_image), con scale_factor = scale**level
    respecto a la imagen original. Se detiene cuando algún lado < min_size.
    """
    h0, w0 = image.shape[:2]
    yield 0, 1.0, image
    level = 0
    while True:
        level += 1
        s = scale ** level                      # factor acumulado desde el original
        w, h = int(round(w0 / s)), int(round(h0 / s))
        if w < min_size[0] or h < min_size[1]:
            break
        yield level, s, cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# Desafío 1: variabilidad de ubicación -> Ventana deslizante
# --------------------------------------------------------------------------
def sliding_window(image, step_size, window_size):
    """Recorre la imagen con una ventana fija. Devuelve (x, y, patch)."""
    win_w, win_h = window_size
    H, W = image.shape[:2]
    for y in range(0, H - win_h + 1, step_size):
        for x in range(0, W - win_w + 1, step_size):
            yield x, y, image[y:y + win_h, x:x + win_w]


# --------------------------------------------------------------------------
# Clasificación por lotes con la red convolucional
# --------------------------------------------------------------------------
def puntuar_lote(net, patches, min_cover=0.60, context=0.50):
    """
    Evalúa un lote de parches (ventanas cuadradas) con UN solo net.forward().

    1) Contexto: cada parche se rodea con un margen neutro (color medio del
       entrenamiento, = 0 tras restar la media) de `context` x lado de la ventana
       por cada lado. Sin él, un parche de 96 px ampliado a 300x300 deja un
       "primer plano" fuera de distribución y el SSD alucina rostros en objetos
       (p. ej. ~0.98 de confianza sobre una cuchara en la imagen del café).
    2) Puntaje de ventana: el SSD devuelve cajas de rostro con su probabilidad.
       Cada caja se convierte a cuadrada (lado = sqrt(ancho*alto), mismo centro)
       y se mide su ALINEACIÓN = IoU(caja cuadrada, ventana completa). Solo
       cuentan las cajas con alineación >= min_cover; el puntaje de la ventana es
       la máxima confianza entre ellas. Así, una ventana desplazada o de tamaño
       distinto al rostro no hereda su confianza y el recuadro se ajusta al rostro.
    Devuelve (confianzas, alineaciones), ambos de longitud len(patches).
    Las confianzas son las probabilidades reales de la red, continuas en [0, 1].
    """
    win = patches[0].shape[0]
    pad = int(round(context * win))
    canvas = win + 2 * pad
    if pad > 0:
        patches = [cv2.copyMakeBorder(p, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                      value=MEAN_BGR) for p in patches]
    blob = cv2.dnn.blobFromImages(patches, 1.0, NET_INPUT, MEAN_BGR,
                                  swapRB=False, crop=False)
    net.setInput(blob)
    dets = net.forward()[0, 0]                  # (N_det, 7): id, clase, conf, x1, y1, x2, y2
    scores = np.zeros(len(patches), dtype=np.float32)
    aligns = np.zeros(len(patches), dtype=np.float32)
    if dets.shape[0] == 0:
        return scores, aligns
    ids = dets[:, 0].astype(int)
    conf = dets[:, 2]
    # cajas normalizadas del lienzo -> coordenadas relativas a la ventana [0,1]
    b = (dets[:, 3:7] * canvas - pad) / float(win)
    w = np.clip(b[:, 2] - b[:, 0], 0, None)
    h = np.clip(b[:, 3] - b[:, 1], 0, None)
    side = np.sqrt(w * h)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    iw = np.clip(np.minimum(cx + side / 2, 1) - np.maximum(cx - side / 2, 0), 0, None)
    ih = np.clip(np.minimum(cy + side / 2, 1) - np.maximum(cy - side / 2, 0), 0, None)
    inter = iw * ih
    align = inter / (1.0 + side * side - inter + 1e-9)
    ok = (ids >= 0) & (ids < len(patches)) & (conf > 0) & (align >= min_cover)
    np.maximum.at(scores, ids[ok], conf[ok])
    mejor = ok & (conf >= scores[np.clip(ids, 0, len(patches) - 1)])
    aligns[ids[mejor]] = align[mejor]
    return scores, aligns


# --------------------------------------------------------------------------
# Supresión de No Máximos (vectorizada)
# --------------------------------------------------------------------------
def non_max_suppression(boxes, scores, overlap_thresh=0.3, contain_thresh=None):
    """
    NMS sobre cajas [x1, y1, x2, y2]. Ordena por score descendente, calcula el
    IoU de la mejor caja contra todas las restantes (vectorizado) y elimina las
    que se solapan más de overlap_thresh. Devuelve los índices conservados.

    Extensión opcional (contain_thresh): también elimina las cajas cuya
    intersección con la mejor caja supera esa fracción de SU PROPIA área, es
    decir, cajas pequeñas contenidas dentro de una mayor (el IoU de una caja de
    96 px dentro de otra de 190 px es ~0.25 y el NMS clásico no las quitaría).
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64)
    if boxes.shape[0] == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    orden = np.argsort(-scores)
    keep = []
    while orden.size > 0:
        i = orden[0]
        keep.append(i)
        resto = orden[1:]
        xx1 = np.maximum(x1[i], x1[resto])
        yy1 = np.maximum(y1[i], y1[resto])
        xx2 = np.minimum(x2[i], x2[resto])
        yy2 = np.minimum(y2[i], y2[resto])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[resto] - inter + 1e-9)
        suprimir = iou > overlap_thresh
        if contain_thresh is not None:
            suprimir |= inter / (np.minimum(areas[i], areas[resto]) + 1e-9) > contain_thresh
        orden = resto[~suprimir]
    return np.array(keep, dtype=int)


# --------------------------------------------------------------------------
# Detector completo
# --------------------------------------------------------------------------
def detect_faces(image_bgr, net, window_size=(96, 96), step_size=16, scale=1.25,
                 conf_threshold=0.90, overlap_thresh=0.3, batch_size=32,
                 min_cover=0.60, context=0.50, contain_thresh=0.40, min_size=None):
    """
    Ejecuta pirámide + ventana deslizante + red + mapeo inverso + NMS.
    Los parches de TODOS los niveles se acumulan en lotes de `batch_size`.
    """
    win_w, win_h = window_size
    # un nivel menor que la ventana no puede contener ni una ventana completa
    min_size = (max(win_w, (min_size or (0, 0))[0]), max(win_h, (min_size or (0, 0))[1]))
    H0, W0 = image_bgr.shape[:2]

    t0 = time.perf_counter()
    raw_boxes, raw_scores, raw_rank = [], [], []
    niveles, ventanas_por_nivel = 0, []
    total = 0
    buf_patches, buf_meta = [], []

    def vaciar_lote():
        if not buf_patches:
            return
        sc, al = puntuar_lote(net, buf_patches, min_cover, context)
        for (x, y, s), p, a in zip(buf_meta, sc, al):
            if p >= conf_threshold:
                # Mapeo inverso a la imagen original (factor acumulado s)
                x1o, y1o = int(round(x * s)), int(round(y * s))
                x2o, y2o = int(round((x + win_w) * s)), int(round((y + win_h) * s))
                raw_boxes.append([max(0, x1o), max(0, y1o), min(W0, x2o), min(H0, y2o)])
                raw_scores.append(float(p))
                raw_rank.append(float(p * a))   # prioridad en NMS: confianza x alineación
        buf_patches.clear()
        buf_meta.clear()

    for level, s, resized in image_pyramid(image_bgr, scale, min_size):
        niveles += 1
        n_nivel = 0
        for x, y, patch in sliding_window(resized, step_size, window_size):
            buf_patches.append(patch)
            buf_meta.append((x, y, s))
            n_nivel += 1
            if len(buf_patches) >= batch_size:
                vaciar_lote()
        ventanas_por_nivel.append(n_nivel)
        total += n_nivel
    vaciar_lote()

    raw_boxes = np.array(raw_boxes, dtype=int).reshape(-1, 4)
    raw_scores = np.array(raw_scores, dtype=float)
    raw_rank = np.array(raw_rank, dtype=float)
    # NMS ordena por confianza x alineación (desempata a favor de la ventana mejor
    # centrada cuando varias tienen confianza ~1.0); se reporta la confianza real.
    keep = non_max_suppression(raw_boxes, raw_rank, overlap_thresh, contain_thresh)
    elapsed = time.perf_counter() - t0
    return {
        "boxes": raw_boxes[keep], "scores": raw_scores[keep],
        "raw_boxes": raw_boxes, "raw_scores": raw_scores,
        "niveles": niveles, "ventanas_por_nivel": ventanas_por_nivel,
        "total_ventanas": total, "detecciones_brutas": int(len(raw_boxes)),
        "detecciones_finales": int(len(keep)), "tiempo_s": elapsed,
    }


# --------------------------------------------------------------------------
# Visualización
# --------------------------------------------------------------------------
def draw_detections(image_bgr, boxes, scores, color=(0, 255, 0), thickness=3):
    """Dibuja recuadros verdes (3 px) con el texto 'Rostro: XX.X%'."""
    out = image_bgr.copy()
    fs = max(0.5, min(out.shape[:2]) / 600.0)
    for (x1, y1, x2, y2), sc in zip(boxes, scores):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        txt = f"Rostro: {sc * 100:.1f}%"
        (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        # etiqueta justo encima del borde (sin tapar el trazo); si no cabe, debajo
        ty = int(y1) - thickness - base - 2 if y1 - th - base - thickness - 8 > 0 else int(y2) + th + thickness + 4
        tx = int(min(max(x1, 0), max(out.shape[1] - tw - 2, 0)))
        cv2.rectangle(out, (tx, ty - th - 4), (tx + tw, ty + base), (0, 0, 0), -1)
        cv2.putText(out, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# Línea de comandos
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Detección de rostros: ventana deslizante + pirámide")
    ap.add_argument("imagen", help="ruta de la imagen a procesar")
    ap.add_argument("--step", type=int, default=16, help="paso de la ventana (8, 16, 24)")
    ap.add_argument("--scale", type=float, default=1.25, help="factor de la pirámide (1.15-1.40)")
    ap.add_argument("--ventana", type=int, default=96, help="lado de la ventana en px")
    ap.add_argument("--conf", type=float, default=0.90, help="umbral de confianza (>= 0.80)")
    ap.add_argument("--nms", type=float, default=0.30, help="umbral de solape del NMS")
    ap.add_argument("--alineacion", type=float, default=0.60,
                    help="alineación mínima caja-ventana (IoU) para aceptar una ventana")
    ap.add_argument("--contencion", type=float, default=0.40,
                    help="suprime cajas contenidas en otra mejor (fracción de su área)")
    ap.add_argument("--batch", type=int, default=32, help="tamaño de lote")
    ap.add_argument("--contexto", type=float, default=0.50,
                    help="margen de contexto alrededor de cada ventana (fracción del lado)")
    ap.add_argument("--max-lado", type=int, default=800,
                    help="reduce imágenes más grandes (ahorra tiempo); 0 = no reducir")
    ap.add_argument("--modelo-dir", default="modelo", help="carpeta del modelo Caffe")
    ap.add_argument("--salida", default="resultado_deteccion.png", help="imagen de salida")
    ap.add_argument("--json", default=None, help="guardar detecciones en JSON (opcional)")
    a = ap.parse_args(argv)

    img = cv2.imread(a.imagen)
    if img is None:
        sys.exit(f"No se pudo leer la imagen: {a.imagen}")
    factor = 1.0
    if a.max_lado and max(img.shape[:2]) > a.max_lado:
        factor = a.max_lado / max(img.shape[:2])
        img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        print(f"[info] imagen reducida x{factor:.3f} para acelerar el escaneo")

    net = cargar_red(a.modelo_dir)
    r = detect_faces(img, net, (a.ventana, a.ventana), a.step, a.scale, a.conf,
                     a.nms, a.batch, min_cover=a.alineacion, context=a.contexto,
                     contain_thresh=a.contencion)
    print(f"Niveles: {r['niveles']} | Ventanas: {r['total_ventanas']} | "
          f"Brutas: {r['detecciones_brutas']} | Finales: {r['detecciones_finales']} | "
          f"Tiempo: {r['tiempo_s']:.1f} s")
    for b, s in zip(r["boxes"], r["scores"]):
        print(f"  Rostro {s * 100:.1f}%  caja={b.tolist()}")
    cv2.imwrite(a.salida, draw_detections(img, r["boxes"], r["scores"]))
    print(f"Resultado guardado en {a.salida}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"boxes": r["boxes"].tolist(), "scores": r["scores"].tolist(),
                       "factor_reduccion": factor}, f, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Práctica 6: Detección de Rostros con Ventana Deslizante + Pirámide de Imágenes
==============================================================================

Propósito del módulo
--------------------
Pipeline clásico-convolucional de detección de rostros:
  1. Pirámide de imágenes (multiescala, conserva la relación de aspecto).
  2. Ventana deslizante de tamaño fijo (96×96 px por defecto) sobre cada nivel.
  3. Cada ventana se clasifica en lotes con la red ResNet-10 SSD de Caffe,
     cargada con cv2.dnn.readNetFromCaffe.
  4. Mapeo inverso de las ventanas aceptadas a coordenadas de la imagen original.
  5. Supresión de No Máximos (NMS) para dejar un único recuadro por rostro.

Cómo ejecutarlo
---------------
    python practica_6_deteccion_rostros.py ruta/imagen.jpg --step 16 --scale 1.25

Dependencias
------------
    numpy, opencv-python (serie 4.x; la serie 5.x eliminó el importador Caffe).

Archivos que necesita
---------------------
    models/deploy.prototxt                        – arquitectura de la red (se descarga si falta)
    models/res10_300x300_ssd_iter_140000.caffemodel – pesos de la red (se descarga si falta)
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
MEAN_BGR = (104.0, 177.0, 123.0)   # media de los canales BGR con la que fue entrenada la red
NET_INPUT = (300, 300)             # tamaño de entrada fijo requerido por el SSD


def resolver_directorio_modelo(directorio="models"):
    """Localiza la carpeta que contiene el archivo deploy.prototxt.

    Busca primero en el directorio indicado y luego en las rutas habituales
    ('models', 'modelo', '.').  Si no encuentra nada, devuelve el directorio
    indicado por defecto para que la descarga lo cree.

    Args:
        directorio (str): Carpeta preferida donde buscar el modelo.

    Returns:
        str: Ruta al directorio donde existe (o se creará) el modelo.
    """
    if directorio and os.path.exists(os.path.join(directorio, PROTO_NAME)):
        return directorio
    for d in ("models", "modelo", "."):
        if os.path.exists(os.path.join(d, PROTO_NAME)):
            return d
    return directorio or "models"


def descargar_modelo(directorio="models"):
    """Descarga deploy.prototxt y los pesos .caffemodel si no existen o están incompletos.

    Args:
        directorio (str): Carpeta donde guardar los archivos descargados.

    Returns:
        tuple[str, str]: Rutas (proto, pesos) de los archivos descargados/existentes.
    """
    directorio = resolver_directorio_modelo(directorio)
    os.makedirs(directorio, exist_ok=True)
    rutas = []
    for nombre, url in ((PROTO_NAME, PROTO_URL), (MODEL_NAME, MODEL_URL)):
        ruta = os.path.join(directorio, nombre)
        if not os.path.exists(ruta) or os.path.getsize(ruta) < 1000:
            print(f"[modelo] descargando {nombre} ...")
            urllib.request.urlretrieve(url, ruta)
        rutas.append(ruta)
    return rutas[0], rutas[1]


def cargar_red(directorio="models"):
    """Carga la red ResNet-10 SSD con pesos oficiales (sin inicialización aleatoria).

    Verifica que la versión de OpenCV soporte el importador Caffe (serie 4.x).

    Args:
        directorio (str): Carpeta donde están (o se descargarán) los archivos del modelo.

    Returns:
        cv2.dnn_Net: Red neuronal lista para inferencia.

    Raises:
        RuntimeError: Si OpenCV no incluye el importador Caffe (p.ej. serie 5.x).
    """
    directorio = resolver_directorio_modelo(directorio)
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
    """Generador de pirámide de imágenes multiescala.

    En el nivel ``l`` la imagen se reduce por ``scale**l`` (mismo factor en
    ancho y alto => se preserva el aspect ratio).  Se detiene cuando algún
    lado de la imagen reducida es menor que ``min_size``.

    El factor de escala acumulado ``s = scale**level`` permite después
    mapear las coordenadas del nivel reducido a la imagen original
    multiplicando por ``s`` (ver ``detect_faces``).

    Args:
        image (np.ndarray): Imagen BGR original.
        scale (float): Factor de reducción entre niveles (p.ej. 1.25 → cada
            nivel es 1/1.25 del anterior en cada dimensión).
        min_size (tuple[int, int]): Tamaño mínimo (ancho, alto) para seguir
            generando niveles.  Cuando algún lado cae por debajo de este valor
            se detiene la pirámide.

    Yields:
        tuple[int, float, np.ndarray]: (nivel, factor_escala, imagen_reducida).
            ``factor_escala`` es ``scale**nivel`` respecto a la imagen original.
    """
    h0, w0 = image.shape[:2]
    # Nivel 0: imagen original sin reducir; factor = 1.0
    yield 0, 1.0, image
    level = 0
    while True:
        level += 1
        s = scale ** level                      # factor acumulado desde el original
        w, h = int(round(w0 / s)), int(round(h0 / s))
        # Condición de parada: si algún lado cae bajo el mínimo, se termina la pirámide
        if w < min_size[0] or h < min_size[1]:
            break
        yield level, s, cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# Desafío 1: variabilidad de ubicación -> Ventana deslizante
# --------------------------------------------------------------------------
def sliding_window(image, step_size, window_size):
    """Recorre la imagen con una ventana de tamaño fijo, generando parches.

    La ventana siempre es de 96×96 px (controlado por ``window_size``).
    El paso ``step_size`` controla cuántos píxeles avanza en cada iteración:
    un paso pequeño (p.ej. 8-16) da más cobertura a costa de más ventanas.

    Args:
        image (np.ndarray): Imagen (nivel de la pirámide) a recorrer.
        step_size (int): Desplazamiento en píxeles entre ventanas consecutivas
            (tanto en X como en Y). Configurable vía --step en la CLI.
        window_size (tuple[int, int]): (ancho, alto) de la ventana en píxeles.
            Normalmente (96, 96).

    Yields:
        tuple[int, int, np.ndarray]: (x, y, parche) donde (x, y) es la esquina
            superior-izquierda de la ventana dentro de ``image``.
    """
    win_w, win_h = window_size
    H, W = image.shape[:2]
    for y in range(0, H - win_h + 1, step_size):
        for x in range(0, W - win_w + 1, step_size):
            yield x, y, image[y:y + win_h, x:x + win_w]


# --------------------------------------------------------------------------
# Clasificación por lotes con la red convolucional
# --------------------------------------------------------------------------
def puntuar_lote(net, patches, min_cover=0.60, context=0.50):
    """Evalúa un lote de parches (ventanas cuadradas) con UN solo net.forward().

    Estrategia de contexto
    ~~~~~~~~~~~~~~~~~~~~~~
    Cada parche se rodea con un margen neutro (píxeles iguales a la media del
    entrenamiento, que equivale a 0 tras restar ``MEAN_BGR``) de ``context`` x
    lado de la ventana por cada lado.  Sin él, un parche de 96 px ampliado a
    300x300 queda "fuera de distribución" y el SSD produce falsos positivos
    (p.ej. detecta un rostro con ~0.98 de confianza en una cuchara).  El contexto
    rodea el parche de colores neutros para que la red "vea" el parche como si
    estuviera en el centro de una imagen más grande.

    Cálculo del puntaje de cada ventana y alineación IoU
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. El SSD devuelve cajas de rostro [x1, y1, x2, y2] con su probabilidad.
    2. Cada caja se convierte a cuadrada (lado = sqrt(ancho x alto), mismo centro).
    3. Se mide la ALINEACIÓN = IoU(caja cuadrada, ventana completa [0,1]x[0,1]).
    4. Solo cuentan las cajas con alineación >= ``min_cover``; el puntaje de la
       ventana es la máxima confianza entre ellas.  Así, una ventana desplazada
       o cuyo tamaño no coincide con el del rostro no hereda su confianza y el
       recuadro final se ajusta al verdadero tamaño del rostro.

    Args:
        net (cv2.dnn_Net): Red SSD ya cargada (misma instancia para todos los lotes).
        patches (list[np.ndarray]): Lista de parches BGR de tamaño ``window_size``.
        min_cover (float): Alineación mínima (IoU entre la caja del SSD y la
            ventana) para aceptar una detección.
        context (float): Fracción del lado de la ventana que se añade como
            margen neutro alrededor del parche antes de pasarlo a la red.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - confianzas (float32, len=len(patches)): probabilidad de rostro
              de la red, continua en [0, 1].
            - alineaciones (float32, len=len(patches)): IoU de la mejor caja
              aceptada con la ventana completa (0 si ninguna cumple min_cover).
    """
    win = patches[0].shape[0]
    pad = int(round(context * win))
    canvas = win + 2 * pad
    if pad > 0:
        # Añadir margen neutro (color medio del entrenamiento) para contextualizar cada parche
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
    # Convertir cajas normalizadas del lienzo a coordenadas relativas a la ventana [0, 1]
    b = (dets[:, 3:7] * canvas - pad) / float(win)
    w = np.clip(b[:, 2] - b[:, 0], 0, None)
    h = np.clip(b[:, 3] - b[:, 1], 0, None)
    # Convertir a caja cuadrada: mismo centro, lado = sqrt(ancho * alto)
    side = np.sqrt(w * h)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    # Calcular IoU entre la caja cuadrada y la ventana completa [0,1]x[0,1]
    iw = np.clip(np.minimum(cx + side / 2, 1) - np.maximum(cx - side / 2, 0), 0, None)
    ih = np.clip(np.minimum(cy + side / 2, 1) - np.maximum(cy - side / 2, 0), 0, None)
    inter = iw * ih
    # IoU = intersección / (1 + área_caja_cuadrada - intersección); el +1 es el área de la ventana
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
    """Supresión de No Máximos (NMS) sobre cajas [x1, y1, x2, y2].

    Ordena las cajas por puntuación descendente.  Para cada caja de máxima
    puntuación calcula el IoU (Intersection over Union) con las restantes
    de forma vectorizada y elimina las que se solapan más de ``overlap_thresh``.

    Extensión opcional (``contain_thresh``): también elimina las cajas cuya
    intersección con la mejor caja supera esa fracción de SU PROPIA área.
    Esto elimina cajas pequeñas contenidas dentro de otra mayor (p.ej. una
    caja de 96 px dentro de otra de 190 px tendría IoU ≈ 0.25, que el NMS
    clásico no quitaría).

    Args:
        boxes (array-like): Cajas de forma (N, 4) con columnas [x1, y1, x2, y2].
        scores (array-like): Puntuaciones de cada caja (mayor = mejor).
        overlap_thresh (float): Umbral de IoU por encima del cual se suprime
            una caja solapada.
        contain_thresh (float | None): Fracción de la propia área de la caja
            más pequeña que, si se supera por intersección con la mayor, la
            elimina.  None desactiva esta extensión.

    Returns:
        np.ndarray: Índices (int) de las cajas conservadas, ordenados por puntuación.
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
        # Intersección vectorizada: la mejor caja contra todas las restantes
        xx1 = np.maximum(x1[i], x1[resto])
        yy1 = np.maximum(y1[i], y1[resto])
        xx2 = np.minimum(x2[i], x2[resto])
        yy2 = np.minimum(y2[i], y2[resto])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        # IoU = intersección / unión
        iou = inter / (areas[i] + areas[resto] - inter + 1e-9)
        suprimir = iou > overlap_thresh
        # Extensión: suprimir cajas pequeñas contenidas en la mejor
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
    """Ejecuta el pipeline completo: pirámide + ventana deslizante + red + NMS.

    Los parches de TODOS los niveles de la pirámide se acumulan en lotes de
    ``batch_size`` antes de llamar a la red, minimizando el número de llamadas
    a ``net.forward()``.

    Mapeo inverso a la imagen original
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Cuando una ventana de coordenadas ``(x, y)`` en el nivel ``l`` (escala ``s``)
    se acepta, sus coordenadas reales en la imagen original son:
        x_orig = x * s,  y_orig = y * s.
    Esto se debe a que el nivel ``l`` fue creado dividiendo la imagen original
    por ``s``, por lo que multiplicar por ``s`` deshace esa reducción.

    Args:
        image_bgr (np.ndarray): Imagen BGR de entrada.
        net (cv2.dnn_Net): Red SSD ya cargada.
        window_size (tuple[int, int]): Tamaño de la ventana deslizante en px.
            Por defecto (96, 96).
        step_size (int): Paso de la ventana deslizante en px.
        scale (float): Factor de reducción entre niveles de la pirámide.
        conf_threshold (float): Umbral de confianza mínimo para aceptar una ventana.
        overlap_thresh (float): Umbral de IoU del NMS.
        batch_size (int): Número de ventanas enviadas a la red de una sola vez.
        min_cover (float): Alineación mínima caja-ventana requerida en ``puntuar_lote``.
        context (float): Margen de contexto alrededor de cada ventana (ver ``puntuar_lote``).
        contain_thresh (float): Umbral de contención del NMS (ver ``non_max_suppression``).
        min_size (tuple[int, int] | None): Tamaño mínimo de imagen en la pirámide;
            si None se usa ``window_size``.

    Returns:
        dict: Diccionario con las siguientes claves:
            - ``boxes`` (np.ndarray int (K,4)): Cajas finales [x1,y1,x2,y2] en coords. orig.
            - ``scores`` (np.ndarray float (K,)): Confianzas finales.
            - ``raw_boxes``, ``raw_scores``: Cajas y confianzas antes del NMS.
            - ``niveles`` (int): Número de niveles de la pirámide usados.
            - ``ventanas_por_nivel`` (list[int]): Ventanas por nivel.
            - ``total_ventanas`` (int): Total de ventanas evaluadas.
            - ``detecciones_brutas`` (int): Número de ventanas aceptadas antes del NMS.
            - ``detecciones_finales`` (int): Número de rostros tras el NMS.
            - ``tiempo_s`` (float): Tiempo total en segundos.
    """
    win_w, win_h = window_size
    # Un nivel cuyo lado sea menor que la ventana no puede contener ni una ventana completa
    min_size = (max(win_w, (min_size or (0, 0))[0]), max(win_h, (min_size or (0, 0))[1]))
    H0, W0 = image_bgr.shape[:2]

    t0 = time.perf_counter()
    raw_boxes, raw_scores, raw_rank = [], [], []
    niveles, ventanas_por_nivel = 0, []
    total = 0
    buf_patches, buf_meta = [], []

    def vaciar_lote():
        """Envía el buffer actual a la red y acumula las detecciones aceptadas."""
        if not buf_patches:
            return
        sc, al = puntuar_lote(net, buf_patches, min_cover, context)
        for (x, y, s), p, a in zip(buf_meta, sc, al):
            if p >= conf_threshold:
                # Mapeo inverso a la imagen original: coordenadas del nivel × factor s
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
    # NMS ordena por confianza x alineación (desempata a favor de la ventana mejor centrada
    # cuando varias tienen confianza ~1.0); se reporta la confianza real al exterior.
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
    """Dibuja recuadros verdes (3 px) con el texto 'Rostro: XX.X%'.

    Args:
        image_bgr (np.ndarray): Imagen BGR sobre la que dibujar.
        boxes (array-like): Cajas [x1, y1, x2, y2] de las detecciones.
        scores (array-like): Confianzas correspondientes a cada caja.
        color (tuple[int,int,int]): Color BGR de los recuadros.
        thickness (int): Grosor de los recuadros en píxeles.

    Returns:
        np.ndarray: Copia de la imagen con las anotaciones dibujadas.
    """
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
    """Punto de entrada para uso por consola.

    Args:
        argv (list[str] | None): Argumentos de la línea de comandos.
            None → usa sys.argv.
    """
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
    default_dir = "models" if os.path.isdir("models") else "modelo"
    ap.add_argument("--modelo-dir", default=default_dir, help="carpeta del modelo Caffe")
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

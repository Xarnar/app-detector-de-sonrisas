"""
Núcleo del detector de sonrisas (sin interfaz gráfica).
========================================================

Propósito del módulo
--------------------
Proporciona todos los componentes reutilizables del sistema de detección de
sonrisas: detectores de rostros, clasificador de sonrisa, recorte y
preprocesado de caras, seguimiento entre frames y dibujo de anotaciones.
Es el módulo central que usa la interfaz gráfica (app_gui.py).

Contrato del clasificador
--------------------------
Entrada al modelo: imagen en escala de grises, 64×64 px, float32 valores
0–255, con forma (N, 64, 64, 1).  El reescalado a [0, 1] se hace dentro del
modelo (capa Rescaling).

Dependencias
------------
    numpy, opencv-python (serie 4.x), tensorflow (>=2.x)

Cómo ejecutarlo
---------------
    Este módulo no tiene punto de entrada propio.
    Importarlo desde app_gui.py o app_sonrisa.py.

Archivos que necesita
---------------------
    models/face_detection_yunet_2023mar.onnx  – detector YuNet (se descarga si falta)
    models/deploy.prototxt                    – arquitectura SSD (se descarga si falta)
    models/res10_300x300_ssd_iter_140000.caffemodel – pesos SSD (se descarga si falta)
    models/model_*.keras                      – clasificador de sonrisa (opcional)
    models/*_metrics.json                     – umbral recomendado por modelo (opcional)
"""
import glob
import json
import os
import re
import threading
import time
import urllib.request
from collections import deque

import cv2
import numpy as np

# Tamaño de entrada del clasificador (debe coincidir con el entrenamiento)
IMG_SIZE = 64

def find_file(filename, search_dirs=("models", "modelo", ".")):
    """Busca un archivo en las carpetas comunes y devuelve su ruta si existe.

    Primero comprueba rutas absolutas y relativas directas; después busca en
    cada directorio de ``search_dirs``.  Si no lo encuentra, devuelve la ruta
    canónica dentro de 'models' para que el llamador pueda descargarlo allí.

    Args:
        filename (str): Nombre o ruta del archivo a buscar.
        search_dirs (tuple[str, ...]): Carpetas donde buscar, en orden.

    Returns:
        str: Ruta al archivo encontrado, o ruta sugerida si no existe.
    """
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    if os.path.exists(filename):
        return filename
    for d in search_dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    pref = "models" if os.path.isdir("models") else "."
    return os.path.join(pref, filename) if pref != "." else filename

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")

# Red de la Práctica 6 (ResNet-10 SSD, pesos oficiales de OpenCV)
SSD_DIR = "models" if os.path.exists(os.path.join("models", "deploy.prototxt")) else ("modelo" if os.path.exists("modelo") else "models")
SSD_PROTO = "deploy.prototxt"
SSD_WEIGHTS = "res10_300x300_ssd_iter_140000.caffemodel"
SSD_PROTO_URL = ("https://raw.githubusercontent.com/opencv/opencv/4.x/"
                 "samples/dnn/face_detector/deploy.prototxt")
SSD_WEIGHTS_URL = ("https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
                   "dnn_samples_face_detector_20170830/" + SSD_WEIGHTS)

# Margen inicial del recorte según el detector (cada detector dibuja su caja distinta).
# Son valores de partida: se calibran viendo los recortes de 64x64 en la interfaz.
DEFAULT_MARGIN = {"yunet": 0.10, "ssd": 0.0, "haar": 0.10, "p6roi": 0.10}
# Confianza mínima inicial por detector (en Haar se traduce a minNeighbors; en la Práctica 6 = 0.90)
DEFAULT_SCORE = {"yunet": 0.70, "ssd": 0.70, "haar": 0.60, "p6roi": 0.90}

# Constante de la Práctica 6: lado mínimo del rostro / lado de la ventana con min_cover=0.60
FACTOR_ROSTRO_MIN = 0.775

GREEN, RED, WHITE = (60, 200, 60), (50, 50, 230), (255, 255, 255)


def p6_min_face_size(upsample_levels=0, scale=1.40, window_size=96):
    """Devuelve el tamaño mínimo de rostro detectable (en píxeles) según upsample_levels.

    Basado en el criterio de alineación de la Práctica 6 (min_cover=0.60, lo que
    exige que el lado del rostro cubra al menos una fracción ≈ sqrt(0.60) = 0.775
    del lado de la ventana).
    Al usar niveles de ampliación (``upsample_levels > 0``), el menor factor de
    escala en la pirámide es ``s_min = scale**(-upsample_levels)``, por lo que el
    tamaño mínimo detectable en píxeles de la imagen original es:
        tam_min = FACTOR_ROSTRO_MIN * window_size * s_min

    Args:
        upsample_levels (int): Niveles de ampliación de la pirámide (0 por defecto).
        scale (float): Factor entre niveles consecutivos de la pirámide (1.40 por defecto).
        window_size (int): Lado de la ventana fija en píxeles (96 por defecto).

    Returns:
        float: Lado mínimo estimado del rostro detectable en píxeles de la imagen original.
    """
    s_min = float(scale) ** (-max(0, int(upsample_levels)))
    return float(FACTOR_ROSTRO_MIN * window_size * s_min)


# ------------------------------------------------------------------ descargas
def _download(url, path, min_bytes):
    """Descarga ``path`` si no existe o es demasiado pequeño (p.ej. un puntero Git LFS).

    Args:
        url (str): URL de descarga del archivo.
        path (str): Ruta local donde guardar el archivo.
        min_bytes (int): Tamaño mínimo en bytes que debe tener el archivo descargado.

    Returns:
        str: Ruta local del archivo (mismo que ``path``).

    Raises:
        RuntimeError: Si la descarga falla o el archivo sigue siendo demasiado pequeño.
    """
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        raise RuntimeError(f"No se pudo descargar {os.path.basename(path)} ({e}). "
                           f"Bájalo a mano de {url} y guárdalo en {path}")
    if os.path.getsize(path) < min_bytes:
        raise RuntimeError(f"El archivo {path} descargado es demasiado pequeño; "
                           f"descárgalo a mano de {url}")
    return path


# ------------------------------------------------------------------ detectores
class YuNetDetector:
    """Detector de rostros YuNet (ONNX): rápido, una sola pasada sobre el frame.

    Descarga automáticamente el modelo si no existe en disco.
    """
    name = "yunet"

    def __init__(self, path=None):
        """Inicializa el detector descargando el modelo si es necesario.

        Args:
            path (str | None): Ruta al archivo .onnx.  Si None, se busca
                en las carpetas estándar y se descarga si falta.
        """
        if path is None:
            path = find_file(YUNET_FILE)
        _download(YUNET_URL, path, 100_000)
        self.det = cv2.FaceDetectorYN.create(path, "", (320, 320), 0.7, 0.3, 10)

    def detect(self, frame, score=0.7):
        """Detecta rostros en el frame y devuelve cajas (x, y, w, h) enteras.

        Args:
            frame (np.ndarray): Frame BGR de entrada.
            score (float): Confianza mínima para aceptar una detección.

        Returns:
            list[tuple[int, int, int, int]]: Lista de cajas (x, y, w, h).
        """
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        self.det.setScoreThreshold(float(score))
        _, faces = self.det.detect(frame)
        if faces is None:
            return []
        return [tuple(int(v) for v in f[:4]) for f in faces]


def load_ssd_net(folder=None):
    """Carga la red ResNet-10 SSD (Práctica 6) verificando compatibilidad con OpenCV 4.x.

    Args:
        folder (str | None): Carpeta del modelo.  Si None, usa ``SSD_DIR``.

    Returns:
        cv2.dnn_Net: Red lista para inferencia.

    Raises:
        RuntimeError: Si OpenCV no incluye el importador Caffe (serie 5.x).
    """
    if not hasattr(cv2.dnn, "readNetFromCaffe"):
        raise RuntimeError(f"OpenCV {cv2.__version__} no incluye el importador Caffe "
                           "(serie 5.x). Instala: pip install \"opencv-python<5\"")
    if folder is None:
        folder = SSD_DIR
    proto_path = find_file(SSD_PROTO, (folder, "models", "modelo", "."))
    weights_path = find_file(SSD_WEIGHTS, (folder, "models", "modelo", "."))
    proto = _download(SSD_PROTO_URL, proto_path, 1_000)
    weights = _download(SSD_WEIGHTS_URL, weights_path, 1_000_000)
    return cv2.dnn.readNetFromCaffe(proto, weights)


class SSDDetector:
    """La red de la Práctica 6 aplicada al frame completo en UNA sola pasada (tiempo real).

    A diferencia de ``run_practica6`` / ``p6_scan``, no usa pirámide ni ventana
    deslizante: redimensiona el frame completo a 300×300 y hace un único forward.
    """
    name = "ssd"

    def __init__(self, net):
        """Inicializa con una red SSD ya cargada.

        Args:
            net (cv2.dnn_Net): Red SSD cargada con ``load_ssd_net()``.
        """
        self.net = net

    def detect(self, frame, score=0.6):
        """Detecta rostros en el frame completo con la red SSD.

        Args:
            frame (np.ndarray): Frame BGR de entrada.
            score (float): Confianza mínima para aceptar una detección.

        Returns:
            list[tuple[int, int, int, int]]: Lista de cajas (x, y, w, h).
        """
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        out = self.net.forward()[0, 0]
        boxes = []
        for d in out:
            if d[2] < score:
                continue
            x1, y1 = max(0, int(d[3] * W)), max(0, int(d[4] * H))
            x2, y2 = min(W, int(d[5] * W)), min(H, int(d[6] * H))
            if x2 - x1 >= 20 and y2 - y1 >= 20:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes


def run_practica6(frame, net, step=24, scale=1.40, upsample_levels=0):
    """Ejecuta el algoritmo de la Práctica 6 (pirámide + ventana deslizante) sobre un frame.

    Relación con ``p6_scan``: esta función devuelve el resultado final de una sola vez,
    sin narrar el proceso paso a paso.  ``p6_scan`` es su versión generadora que permite
    animar el escaneo en la GUI.  Ambas producen los mismos resultados.

    Lento: ~segundos por frame, no apto para tiempo real.

    Args:
        frame (np.ndarray): Frame BGR a analizar.
        net (cv2.dnn_Net): Red SSD cargada con ``load_ssd_net()``.
        step (int): Paso de la ventana deslizante en píxeles.
        scale (float): Factor de reducción de la pirámide.
        upsample_levels (int): Niveles de ampliación de la pirámide (0 por defecto).

    Returns:
        tuple[list[tuple], int, float]:
            - boxes: Lista de cajas (x, y, w, h) en coordenadas originales.
            - total_ventanas: Número total de ventanas evaluadas.
            - secs: Tiempo de ejecución en segundos.
    """
    import practica_6_deteccion_rostros as p6
    t0 = time.perf_counter()
    r = p6.detect_faces(frame, net, (96, 96), step, scale, 0.90, 0.30, 32, 0.60, 0.50, 0.40,
                        upsample_levels=upsample_levels)
    secs = time.perf_counter() - t0
    raw = np.asarray(r["boxes"]).reshape(-1, 4)
    boxes = [(int(x1), int(y1), int(x2 - x1), int(y2 - y1)) for x1, y1, x2, y2 in raw]
    return boxes, int(r["total_ventanas"]), secs


class HaarDetector:
    """Clásico de Viola-Jones: pirámide + ventana deslizante + cascada de clasificadores livianos.

    OpenCV realiza el proceso interno con ``detectMultiScale``; por eso corre en tiempo real
    aunque usa el mismo concepto de pirámide y ventana deslizante que la Práctica 6.
    """
    name = "haar"

    def __init__(self):
        """Carga la cascada de Haar desde la instalación de OpenCV.

        Raises:
            RuntimeError: Si no se encuentra el archivo XML de la cascada.
        """
        folder = getattr(getattr(cv2, "data", None), "haarcascades", None) \
            or os.path.join(os.path.dirname(cv2.__file__), "data")
        path = os.path.join(folder, "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"No se encontró la cascada de Haar en {path}")

    def detect(self, frame, score=0.6):
        """Detecta rostros con la cascada de Haar.

        Reduce la imagen a 320 px de ancho para acelerar el proceso, luego
        reescala las cajas al tamaño original.  El parámetro ``score`` (0.3–0.95)
        se traduce a ``minNeighbors`` de ``detectMultiScale``.

        Args:
            frame (np.ndarray): Frame BGR de entrada.
            score (float): "Confianza" normalizada: mayor = más restrictivo.
                Se convierte a minNeighbors en rango [3, 10].

        Returns:
            list[tuple[int, int, int, int]]: Lista de cajas (x, y, w, h).
        """
        H, W = frame.shape[:2]
        f = 320.0 / W if W > 320 else 1.0                  # se analiza una versión reducida: más rápido
        small = cv2.resize(frame, (int(W * f), int(H * f))) if f < 1 else frame
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        # Haar no da "confianza": la perilla es minNeighbors (cuántas ventanas vecinas deben coincidir)
        n = int(round(3 + (min(max(score, 0.3), 0.95) - 0.3) / 0.65 * 7))
        found = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=n, minSize=(30, 30))
        return [(int(x / f), int(y / f), int(w / f), int(h / f)) for x, y, w, h in found]


class P6RoiDetector:
    """Algoritmo de la Práctica 6 adaptado a tiempo real mediante detección/seguimiento separados.

    Separación detección / seguimiento
    ------------------------------------
    1) DETECCIÓN: Escaneo completo (pirámide + ventana deslizante sobre toda la imagen)
       en un hilo en segundo plano, una sola vez.  Mientras corre, el video sigue en vivo.
       Con upsample_levels=1, el escaneo completo evalúa bastantes más ventanas (aprox.
       el doble o más según la resolución), permitiendo detectar rostros pequeños a costa
       de mayor tiempo de cómputo en el escaneo inicial y en los re-escaneos periódicos.
    2) SEGUIMIENTO: Una vez detectados los rostros, en cada frame solo desliza la ventana
       96×96 alrededor de las últimas posiciones conocidas (2×2 posiciones en ROI reducida),
       lo que es órdenes de magnitud más rápido que el escaneo completo.

    Cada ventana de seguimiento sigue requiriendo una pasada completa de la red SSD, por
    eso los FPS siguen siendo bajos (modo experimental).
    """
    name = "p6roi"
    MAX_TRACKED = 3   # máximo de caras que se rastrean simultáneamente

    # Intervalo entre re-escaneos periódicos en segundo plano (segundos).
    # Permite descubrir caras nuevas que entraron al encuadre después del escaneo inicial.
    rescan_interval = 10.0

    def __init__(self, net, step=32, upsample_levels=0):
        """Inicializa el detector con la red SSD.

        Args:
            net (cv2.dnn_Net): Red SSD cargada con ``load_ssd_net()``.
            step (int): Paso de la ventana durante el seguimiento.
            upsample_levels (int): Niveles de ampliación para el escaneo inicial
                y el re-escaneo (0 por defecto). Con 1 nivel el escaneo completo
                evalúa bastantes más ventanas. El seguimiento por ROI no cambia.
        """
        self.net, self.step = net, step
        self.upsample_levels = int(upsample_levels)
        self.boxes = []
        self.status = ""
        self._thread, self._done, self._found = None, False, []
        self._scan_net = None             # instancia dedicada para escaneo de fondo (evita colisión de hilos)
        self._lock = threading.Lock()     # salvaguarda adicional de sincronización
        # Estado para el re-escaneo periódico en segundo plano
        self._last_frame  = None          # frame más reciente recibido por detect()
        self._rescan_timer = None         # threading.Timer pendiente
        self._is_rescan   = False         # True cuando _done viene de un re-escaneo

    def detect(self, frame, score=0.9):
        """Detecta o rastrea rostros en el frame.

        En la primera llamada (sin caras conocidas) lanza el escaneo completo en
        un hilo de fondo y devuelve lista vacía.  En llamadas sucesivas, usa el
        modo seguimiento alrededor de las últimas cajas.

        El re-escaneo periódico (cada ``rescan_interval`` segundos) actualiza la
        lista de cajas fusionando las nuevas con las existentes, sin desplazar las
        que ya se están siguiendo.

        Args:
            frame (np.ndarray): Frame BGR actual.
            score (float): Umbral de confianza para aceptar una ventana.

        Returns:
            list[tuple[int, int, int, int]]: Lista de cajas (x, y, w, h).
        """
        # Guardar el frame más reciente para el re-escaneo periódico
        self._last_frame = frame

        if self._done:
            if self._is_rescan:
                # Re-escaneo: fusionar caras nuevas con las existentes
                self.boxes = self._merge_boxes(self.boxes, self._found)
                self._is_rescan = False
            else:
                # Escaneo inicial: reemplazar cajas
                self.boxes = self._found[: self.MAX_TRACKED]
                # Programar el primer re-escaneo periódico
                if self.boxes:
                    self._schedule_rescan()
            self._found = []
            self._done, self._thread = False, None

        if not self.boxes:
            if self._thread is None:
                self.status = "Escaneo completo (Practica 6): iniciando..."
                self._start_scan(frame)
            return []
        new = [nb for nb in (self._search(frame, b, score) for b in self.boxes) if nb is not None]
        self.boxes = new
        self.status = "" if new else "Rostro perdido: reescaneando..."
        return list(new)

    def _start_scan(self, frame, rescan=False):
        """Escaneo completo (pirámide + ventana deslizante, paso 24, escala 1.40) en un hilo aparte.

        Mientras corre, el video se sigue mostrando a la velocidad de la cámara;
        el estado (``self.status``) informa el progreso.

        Args:
            frame (np.ndarray): Frame que se analizará en segundo plano.
            rescan (bool): True para re-escaneo periódico (fusiona resultados);
                           False para escaneo inicial (reemplaza resultados).
        """
        img = frame.copy()
        prefijo = "Re-escaneo" if rescan else "Escaneo completo (Practica 6)"
        self._cancel = threading.Event()

        # Instancia dedicada para el escaneo de fondo (evita condición de carrera en cv2.dnn_Net)
        if self._scan_net is None:
            try:
                self._scan_net = load_ssd_net()
            except Exception:
                self._scan_net = self.net
        net_scan = self._scan_net

        def job():
            try:
                raw = []
                for ev in p6_scan(img, net_scan, 24, 1.40, batch=16, cancel=self._cancel,
                                  upsample_levels=self.upsample_levels):
                    if ev["phase"] == "scored":
                        self.status = (f"{prefijo}: "
                                       f"{ev['done']}/{ev['total']} ventanas")
                    elif ev["phase"] == "done":
                        raw = ev["boxes"]
                    elif ev["phase"] == "cancelled":
                        return
                self._found = [(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                               for x1, y1, x2, y2 in raw]
            except Exception as e:                        # noqa: BLE001
                print("[p6roi]", e)
                self._found = []
            self._is_rescan = rescan
            self._done = True
            # Programar el siguiente re-escaneo tras finalizar éste
            if rescan:
                self._schedule_rescan()

        self._thread = threading.Thread(target=job, daemon=True)
        self._thread.start()

    def _merge_boxes(self, existing, new_boxes, iou_thr=0.30):
        """Fusiona cajas nuevas del re-escaneo con las existentes.

        Solo añade cajas cuyo IoU con todas las existentes sea menor que
        ``iou_thr`` (caras genuinamente nuevas).  Respeta ``MAX_TRACKED``.

        Args:
            existing (list): Cajas actuales (x, y, w, h).
            new_boxes (list): Cajas encontradas en el re-escaneo.
            iou_thr (float): Umbral de IoU para considerar una caja como duplicada.

        Returns:
            list: Lista fusionada, con un máximo de ``MAX_TRACKED`` cajas.
        """
        merged = list(existing)
        for nb in new_boxes:
            if len(merged) >= self.MAX_TRACKED:
                break
            # Solo añadir si no se solapa significativamente con ninguna existente
            if all(iou(nb, eb) < iou_thr for eb in merged):
                merged.append(nb)
        return merged

    def _schedule_rescan(self):
        """Programa un re-escaneo completo en segundo plano tras ``rescan_interval`` segundos.

        Si ya había un timer pendiente, lo cancela primero.  El callback solo
        lanza el hilo si no hay otro escaneo en curso y existe un frame reciente.
        """
        # Cancelar el timer anterior si aún no ha disparado
        if self._rescan_timer is not None:
            self._rescan_timer.cancel()

        def _rescan_callback():
            # Solo lanzar si no hay hilo activo y tenemos un frame reciente
            if self._thread is None and self._last_frame is not None:
                self.status = "Re-escaneo: buscando caras nuevas..."
                self._start_scan(self._last_frame, rescan=True)

        self._rescan_timer = threading.Timer(self.rescan_interval, _rescan_callback)
        self._rescan_timer.daemon = True
        self._rescan_timer.start()

    def close(self):
        """Detiene el escaneo y el timer de re-escaneo en segundo plano (al cambiar de detector)."""
        if getattr(self, "_cancel", None) is not None:
            self._cancel.set()
        if getattr(self, "_rescan_timer", None) is not None:
            self._rescan_timer.cancel()
            self._rescan_timer = None
        self._scan_net = None

    def _search(self, frame, box, conf):
        """Desliza la ventana de 96×96 solo alrededor de ``box``; prueba 3 tamaños si no encuentra.

        Args:
            frame (np.ndarray): Frame BGR actual.
            box (tuple[int, int, int, int]): Caja conocida (x, y, w, h).
            conf (float): Umbral de confianza mínimo.

        Returns:
            tuple[int, int, int, int] | None: Nueva caja (x, y, w, h) o None si se perdió.
        """
        import practica_6_deteccion_rostros as p6
        H, W = frame.shape[:2]
        x, y, w, h = box
        S0, cx, cy = max(w, h), x + w / 2, y + h / 2
        for factor in (1.0, 1.15, 1 / 1.15):
            S = S0 * factor
            sc = S / 96.0                                  # ventana de 96 px <-> S px reales
            R = S * 1.6                                    # región de búsqueda alrededor de la cara
            x0, y0 = max(0, int(round(cx - R / 2))), max(0, int(round(cy - R / 2)))
            x1, y1 = min(W, int(round(cx + R / 2))), min(H, int(round(cy + R / 2)))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            roi = cv2.resize(frame[y0:y1, x0:x1], None, fx=1 / sc, fy=1 / sc, interpolation=cv2.INTER_AREA)
            rh, rw = roi.shape[:2]
            if rh < 96 or rw < 96:
                roi = cv2.copyMakeBorder(roi, 0, max(0, 96 - rh), 0, max(0, 96 - rw), cv2.BORDER_REPLICATE)
            patches, metas = [], []
            for yy in range(0, roi.shape[0] - 96 + 1, self.step):
                for xx in range(0, roi.shape[1] - 96 + 1, self.step):
                    patches.append(roi[yy:yy + 96, xx:xx + 96])
                    metas.append((xx, yy))
            if not patches:
                continue
            with self._lock:
                scores, aligns = p6.puntuar_lote(self.net, patches, 0.60, 0.50)
            ok = scores >= conf
            if not ok.any():
                continue
            i = int(np.argmax(np.where(ok, scores * aligns, -1.0)))
            xx, yy = metas[i]
            return (int(x0 + xx * sc), int(y0 + yy * sc), int(96 * sc), int(96 * sc))
        return None


def p6_scan(frame, net, step=24, scale=1.40, batch=8, cancel=None,
            conf_thr=0.90, nms_thr=0.30, min_cover=0.60, context=0.50, contain=0.40,
            upsample_levels=0):
    """Versión generadora de ``run_practica6``: misma lógica pero narra cada paso.

    Relación con ``run_practica6``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``p6_scan`` produce exactamente los mismos resultados que ``p6.detect_faces``
    (y por tanto que ``run_practica6``), pero en lugar de retornar el resultado
    final de golpe, va cediendo eventos intermedios para que la GUI pueda animar
    el proceso en tiempo real.  Lotes pequeños => animación más fluida.

    Usa las mismas funciones de la práctica: ``image_pyramid``, ``sliding_window``,
    ``puntuar_lote`` y ``non_max_suppression``.

    Args:
        frame (np.ndarray): Frame BGR a analizar.
        net (cv2.dnn_Net): Red SSD cargada.
        step (int): Paso de la ventana deslizante.
        scale (float): Factor de la pirámide.
        batch (int): Tamaño del lote (lotes pequeños => animación más fluida).
        cancel (threading.Event | None): Si se activa, el generador termina
            devolviendo ``{"phase": "cancelled"}``.
        conf_thr (float): Umbral de confianza para aceptar ventanas.
        nms_thr (float): Umbral de IoU del NMS.
        min_cover (float): Alineación mínima caja-ventana para ``puntuar_lote``.
        context (float): Margen de contexto para ``puntuar_lote``.
        contain (float): Umbral de contención para ``non_max_suppression``.
        upsample_levels (int): Niveles de ampliación de la pirámide (0 por defecto,
            comportamiento clásico de solo reducción). Permite detectar rostros
            menores que la ventana de 96x96 px.

    Yields:
        dict: Eventos de las fases "start", "scan", "scored", "nms", "done" y "cancelled".
    """
    import practica_6_deteccion_rostros as p6
    win = 96
    H0, W0 = frame.shape[:2]
    levels = list(p6.image_pyramid(frame, scale, (win, win), upsample_levels))
    info = []
    for lvl, s, im in levels:
        h, w = im.shape[:2]
        info.append(dict(level=lvl, s=s, size=(w, h),
                         n=len(range(0, h - win + 1, step)) * len(range(0, w - win + 1, step))))
    total = sum(i["n"] for i in info)
    t0 = time.perf_counter()
    n_upsample = sum(1 for lvl, _, _ in levels if lvl < 0)
    yield dict(phase="start", levels=info, total=total, step=step, batch=batch,
               level_imgs=[im for _, _, im in levels],
               upsample_levels=upsample_levels, n_upsample=n_upsample)

    raw_boxes, raw_scores, raw_rank, done = [], [], [], 0
    for li, (lvl, s, im) in enumerate(levels):
        windows = list(p6.sliding_window(im, step, (win, win)))
        for k in range(0, len(windows), batch):
            if cancel is not None and cancel.is_set():
                yield dict(phase="cancelled")
                return
            chunk = windows[k:k + batch]
            boxes = [(int(round(x * s)), int(round(y * s)),
                      int(round((x + win) * s)), int(round((y + win) * s))) for x, y, _ in chunk]
            yield dict(phase="scan", li=li, s=s, boxes=boxes, win_lvl=[(x, y) for x, y, _ in chunk],
                       done=done, total=total, elapsed=time.perf_counter() - t0)
            sc, al = p6.puntuar_lote(net, [p for _, _, p in chunk], min_cover, context)
            acc = []
            for b, p, a in zip(boxes, sc, al):
                if p >= conf_thr:
                    bb = [max(0, b[0]), max(0, b[1]), min(W0, b[2]), min(H0, b[3])]
                    raw_boxes.append(bb)
                    raw_scores.append(float(p))
                    raw_rank.append(float(p * a))
                    acc.append((bb, float(p)))
            done += len(chunk)
            yield dict(phase="scored", li=li, boxes=boxes, accepted=acc, done=done,
                       total=total, elapsed=time.perf_counter() - t0)

    rb = np.array(raw_boxes, dtype=int).reshape(-1, 4)
    rs = np.array(raw_scores, dtype=float)
    rr = np.array(raw_rank, dtype=float)
    keep = p6.non_max_suppression(rb, rr, nms_thr, contain)
    yield dict(phase="nms", raw=rb, kept=rb[keep], n_raw=len(rb), done=done, total=total,
               elapsed=time.perf_counter() - t0)
    yield dict(phase="done", boxes=rb[keep], scores=rs[keep], raw_boxes=rb, total=total,
               elapsed=time.perf_counter() - t0, upsample_levels=upsample_levels)


def render_p6(frame, st):
    """Dibuja el estado de la Práctica 6 sobre el frame congelado.

    gris: ventana ya evaluada | amarillo: en evaluación | naranja: aceptada por la red | verde: resultado.

    Args:
        frame (np.ndarray): Frame BGR congelado (el que se analizó).
        st (dict): Estado actual devuelto por ``p6_scan``.

    Returns:
        np.ndarray: Frame anotado con el estado del escaneo.
    """
    out = (frame * 0.55).astype(np.uint8)
    H, W = out.shape[:2]
    ORANGE, YELLOW = (0, 140, 255), (0, 255, 255)
    for (x1, y1, x2, y2) in st["evaluated"]:
        cv2.circle(out, ((x1 + x2) // 2, (y1 + y2) // 2), 2, (170, 150, 120), -1)
    for b, _ in st["accepted"]:
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), ORANGE, 1)
    for b in st["scanning"]:
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), YELLOW, 2)
    for b in st["final"]:
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), GREEN, 3)

    # barra de progreso + tiempos
    total, done, el = max(1, st["total"]), st["done"], st["elapsed"]
    frac = done / total
    cv2.rectangle(out, (0, 0), (W, 22), (30, 30, 30), -1)
    cv2.rectangle(out, (0, 0), (int(W * frac), 22), (138, 195, 76), -1)
    eta = (el / frac - el) if frac > 0.02 else 0
    cv2.putText(out, f"Ventanas {done}/{total} ({frac * 100:.0f}%)   {el:.1f} s   ETA {eta:.0f} s",
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # recuadro de la pirámide: el nivel actual y la ventana FIJA de 96x96 sobre él
    imgs, li = st["level_imgs"], st["li"]
    if st["phase"] in ("scan", "scored") and imgs:
        im = imgs[li]
        ih, iw = im.shape[:2]
        tw = 190
        th = max(1, int(ih * tw / iw))
        thumb = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        k = tw / iw
        for wx, wy in st["win_lvl"][-1:]:
            cv2.rectangle(thumb, (int(wx * k), int(wy * k)), (int((wx + 96) * k), int((wy + 96) * k)), YELLOW, 2)
        x0, y0 = W - tw - 8, 30
        if y0 + th < H - 96:
            out[y0:y0 + th, x0:x0 + tw] = thumb
            cv2.rectangle(out, (x0 - 1, y0 - 1), (x0 + tw, y0 + th), WHITE, 1)
            s_lvl = st["levels"][li]["s"]
            factor_txt = f"imagen x {1.0 / s_lvl:.2f}" if s_lvl < 1.0 else f"imagen / {s_lvl:.2f}"
            cv2.putText(out, f"nivel {li + 1}/{len(imgs)} ({factor_txt})",
                        (x0, y0 + th + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)

    # texto de la fase actual
    n_acc = len(st["accepted"])
    if st["phase"] in ("start", "scan", "scored"):
        lines = [f"1) Piramide: nivel {li + 1}/{len(st['levels'])}     2) Ventana fija 96x96, paso {st['step']}",
                 f"3) La red evalua cada ventana (lotes de {st['batch']})",
                 f"gris: evaluada | amarillo: en curso | naranja: aceptada ({n_acc})"]
    else:
        lines = [f"4) NMS: {st['n_raw']} ventanas aceptadas -> {len(st['final'])} rostro(s) (verde)",
                 "gris: evaluada | naranja: aceptada | verde: resultado final"]
    draw_banner(out, lines)
    return out


# ---------------------------------------------------------------- clasificador
def find_threshold(model_path):
    """Busca el umbral de clasificación recomendado en el JSON de métricas del modelo.

    Por qué se ignoran las corridas quick_test
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Los JSON generados con ``QUICK_TEST = True`` (solo 2 épocas) producen un modelo
    sin convergencia real: su umbral no es representativo del rendimiento final.
    Se ignoran para evitar que la app use un umbral calculado sobre un modelo
    entrenado parcialmente.  Solo se usa el umbral del último JSON de entrenamiento
    real (``quick_test: false``), ordenado por fecha de modificación (más reciente primero).

    Args:
        model_path (str): Ruta al archivo .keras del modelo.

    Returns:
        tuple[float, str | None]:
            - umbral_recomendado: Valor en [0, 1] leído del JSON, o 0.5 por defecto.
            - nombre_archivo: Nombre del JSON usado, o None si no se encontró.
    """
    folder = os.path.dirname(os.path.abspath(model_path))
    base = os.path.basename(model_path)
    patterns = []
    m = re.match(r"model_([A-Za-z0-9]+)", base)
    if m:
        patterns.append(f"{m.group(1)}_metrics*.json")
    patterns.append(os.path.splitext(base)[0] + "*metrics*.json")

    search_folders = [folder]
    for extra in ("models", "modelo", "."):
        abs_extra = os.path.abspath(extra)
        if abs_extra not in search_folders and os.path.isdir(abs_extra):
            search_folders.append(abs_extra)

    for f_dir in search_folders:
        for pat in patterns:
            files = sorted(glob.glob(os.path.join(f_dir, pat)), key=os.path.getmtime, reverse=True)
            for f in files:
                try:
                    data = json.load(open(f, encoding="utf-8"))
                    # Ignorar corridas rápidas: umbral calculado con solo 2 épocas no es fiable
                    if data.get("quick_test"):
                        continue
                    return float(data["umbral_recomendado"]), os.path.basename(f)
                except Exception:
                    continue
    return 0.5, None


class SmileClassifier:
    """Clasificador de sonrisa basado en un modelo Keras (.keras).

    Envuelve el modelo con un grafo compilado (tf.function) para acelerar la
    inferencia, carga el umbral desde el JSON de métricas asociado, y es
    seguro para uso en hilos (threading.Lock interno).
    """

    def __init__(self, path):
        """Carga el modelo, compila el grafo de inferencia y busca el umbral.

        Optimizaciones en ``__init__``
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        1. ``tf.function`` con firma fija: compila el grafo UNA sola vez con un
           batch de tamaño dinámico (``[None, 64, 64, 1]``).  El resultado es
           ~6x más rápido que ejecutar capa por capa en modo eager, con salidas
           idénticas.  El batch dinámico (None) permite procesar 1, 2 o 5 caras
           en el mismo grafo sin recompilar.
        2. Calentamiento: se realiza una predicción con ceros antes de que llegue
           el primer frame real, para que el primer predict real no sea lento.
        3. Umbral: se lee de ``find_threshold(path)`` que busca en el JSON de
           métricas el mejor umbral calculado sobre validación por F1.

        Args:
            path (str): Ruta al archivo .keras del modelo.
        """
        import tensorflow as tf                # import tardío: la interfaz abre rápido sin TF
        from tensorflow import keras
        self.name = os.path.basename(path)
        self.model = keras.models.load_model(path)
        self.lock = threading.Lock()

        # Compilar el grafo de inferencia una sola vez con tf.function.
        # La firma [None, IMG_SIZE, IMG_SIZE, 1] acepta cualquier tamaño de batch.
        @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 1], tf.float32)])
        def _fast(x):
            return self.model(x, training=False)

        self._fast, self._tf = _fast, tf
        # Calentamiento: primera llamada compila el grafo XLA; después es rápida
        self.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 1), np.float32))
        # Umbral recomendado leído del JSON de métricas (ignora quick_test)
        self.threshold, self.metrics_file = find_threshold(path)

    def predict(self, batch):
        """Clasifica un lote de caras y devuelve las probabilidades de sonrisa.

        Thread-safe gracias al lock interno.

        Args:
            batch (np.ndarray): Array de forma (N, 64, 64, 1), float32, valores 0–255.

        Returns:
            np.ndarray: Array 1D de N probabilidades en [0, 1] (1 = sonrisa).
        """
        with self.lock:
            return self._fast(self._tf.constant(batch)).numpy().ravel()


# ---------------------------------------------------------- recorte y preprocesado
def crop_face(frame, box, margin):
    """Recorte cuadrado alrededor de la caja, ampliado por ``margin`` (0.1 = +10% por lado).

    Args:
        frame (np.ndarray): Frame BGR de entrada.
        box (tuple[int, int, int, int]): Caja de la cara (x, y, w, h).
        margin (float): Fracción del lado a añadir alrededor del recorte.
            Negativo permite recortar el padding extra que algunos detectores incluyen.

    Returns:
        np.ndarray | None: Recorte BGR, o None si el recorte resultante es demasiado pequeño.
    """
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * margin)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return frame[y0:y1, x0:x1]


def preprocess(crop):
    """Preprocesa un recorte de cara según el contrato del clasificador.

    Convierte a escala de grises, redimensiona a 64×64, y devuelve float32
    con valores en [0, 255] y forma (64, 64, 1).

    Args:
        crop (np.ndarray): Recorte BGR de la cara.

    Returns:
        np.ndarray: Array (64, 64, 1) float32 con valores 0–255.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32)[..., None]


# ------------------------------------------------------- seguimiento y suavizado
def iou(a, b):
    """Calcula el IoU (Intersection over Union) entre dos cajas (x, y, w, h).

    Args:
        a (tuple[int, int, int, int]): Primera caja (x, y, w, h).
        b (tuple[int, int, int, int]): Segunda caja (x, y, w, h).

    Returns:
        float: IoU en [0, 1].
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class Track:
    """Representa el seguimiento de una cara individual entre frames.

    Args:
        tid (int): Identificador único del track.
        box (tuple): Caja inicial (x, y, w, h).
        n (int): Número de frames para promediar la probabilidad de sonrisa.
    """
    def __init__(self, tid, box, n):
        """Inicializa un objeto Track.

        Args:
            tid (int): Identificador numérico del track.
            box (tuple[int, int, int, int]): Coordenadas de la caja (x, y, w, h).
            n (int): Longitud del búfer circular para el promedio móvil de probabilidades.
        """
        self.id, self.box, self.missed = tid, box, 0
        self.probs = deque(maxlen=n)   # historial de probabilidades (ventana deslizante)

    @property
    def smooth(self):
        """Probabilidad promedio suavizada, o None si no hay historial.

        Returns:
            float | None: Media de las últimas ``n`` probabilidades.
        """
        return float(np.mean(self.probs)) if self.probs else None


class Tracker:
    """Asocia las caras entre frames (IoU) para suavizar la probabilidad por persona.

    Implementa un tracker simple basado en IoU: en cada frame empareja las nuevas
    cajas con las del frame anterior por máximo IoU, crea nuevos tracks para las
    cajas sin pareja, y elimina los tracks que llevan más de ``max_missed`` frames
    sin ser vistos.
    """

    def __init__(self, smooth_n=5, max_missed=10, min_iou=0.25):
        """Inicializa el tracker.

        Args:
            smooth_n (int): Número de frames para promediar la probabilidad.
            max_missed (int): Frames sin detección antes de eliminar un track.
            min_iou (float): IoU mínimo para asociar una caja a un track existente.
        """
        self.tracks, self.next_id = {}, 0
        self.smooth_n, self.max_missed, self.min_iou = smooth_n, max_missed, min_iou

    def set_smooth(self, n):
        """Ajusta la ventana de suavizado en vivo sin perder el historial.

        Args:
            n (int): Nuevo número de frames para promediar.
        """
        self.smooth_n = n
        for t in list(self.tracks.values()):
            t.probs = deque(t.probs, maxlen=n)

    def clear_probs(self):
        """Borra el historial de probabilidades de todos los tracks (al cambiar de modelo)."""
        for t in list(self.tracks.values()):
            t.probs.clear()

    def update(self, boxes):
        """Asocia nuevas cajas con tracks existentes y devuelve los tracks activos.

        Args:
            boxes (list[tuple]): Cajas detectadas en el frame actual (x, y, w, h).

        Returns:
            list[Track]: Tracks asociados a cada caja de entrada (mismo orden).
        """
        for t in self.tracks.values():
            t.missed += 1
        pairs = sorted(((iou(t.box, b), tid, i)
                        for tid, t in self.tracks.items() for i, b in enumerate(boxes)),
                       reverse=True)
        used_t, used_i, assigned = set(), set(), {}
        for score, tid, i in pairs:
            if score < self.min_iou:
                break
            if tid in used_t or i in used_i:
                continue
            used_t.add(tid)
            used_i.add(i)
            assigned[i] = self.tracks[tid]
        for i, b in enumerate(boxes):
            if i not in assigned:
                assigned[i] = self.tracks[self.next_id] = Track(self.next_id, b, self.smooth_n)
                self.next_id += 1
        for i, t in assigned.items():
            t.box, t.missed = boxes[i], 0
        self.tracks = {k: t for k, t in self.tracks.items() if t.missed <= self.max_missed}
        return [assigned[i] for i in range(len(boxes))]


# ------------------------------------------------------------------- pipeline
class Face:
    """Representa una cara detectada con su probabilidad de sonrisa y su ID de track."""
    __slots__ = ("box", "prob", "tid")

    def __init__(self, box, prob, tid):
        """Inicializa una cara.

        Args:
            box (tuple[int, int, int, int]): Caja (x, y, w, h).
            prob (float | None): Probabilidad de sonrisa, o None si no hay clasificador.
            tid (int): ID del track (-1 si sin seguimiento).
        """
        self.box, self.prob, self.tid = box, prob, tid


class Pipeline:
    """Un frame -> caras con probabilidad de sonrisa. Los parámetros se pueden cambiar en vivo."""

    def __init__(self):
        """Inicializa el pipeline con valores por defecto (sin detector ni clasificador)."""
        self.detector = None
        self.classifier = None
        self.det_name = "yunet"
        self.margins = dict(DEFAULT_MARGIN)
        self.scores = dict(DEFAULT_SCORE)
        self.threshold = 0.5
        self.mirror = True
        self.tracker = Tracker(5)

    @property
    def margin(self):
        """Margen de recorte del detector activo.

        Returns:
            float: Margen en fracción del lado de la caja.
        """
        return self.margins.get(self.det_name, 0.1)

    @property
    def det_score(self):
        """Confianza mínima del detector activo (cada detector recuerda la suya).

        Returns:
            float: Confianza mínima.
        """
        return self.scores.get(self.det_name, 0.7)

    @det_score.setter
    def det_score(self, v):
        """Asigna la confianza mínima para el detector activo.

        Args:
            v (float): Nuevo umbral de confianza.
        """
        self.scores[self.det_name] = v

    def process(self, frame):
        """Ejecuta el pipeline completo sobre un frame: detectar + seguir + clasificar.

        Args:
            frame (np.ndarray): Frame BGR de entrada.

        Returns:
            tuple[list[Face], list[np.ndarray], float, float]:
                - faces: Lista de objetos Face con caja, probabilidad e ID.
                - crops: Lista de recortes preprocesados (64, 64, 1) float32.
                - t_det_ms: Tiempo de detección en milisegundos.
                - t_cls_ms: Tiempo de clasificación en milisegundos.
        """
        t0 = time.perf_counter()
        det = self.detector
        boxes = det.detect(frame, self.det_score) if det is not None else []
        t1 = time.perf_counter()
        tracks = self.tracker.update(boxes)
        crops, valid = [], []
        clf = self.classifier
        if clf is not None and tracks:
            for t in tracks:
                c = crop_face(frame, t.box, self.margin)
                if c is not None:
                    crops.append(preprocess(c))
                    valid.append(t)
            if crops:
                for t, p in zip(valid, clf.predict(np.stack(crops))):   # 1 predict por frame
                    t.probs.append(float(p))
        t2 = time.perf_counter()
        faces = [Face(t.box, t.smooth, t.id) for t in tracks]
        return faces, crops, (t1 - t0) * 1000, (t2 - t1) * 1000

    def classify_boxes(self, frame, boxes):
        """Clasifica cajas sueltas (sin seguimiento), p.ej. las de la Práctica 6.

        Args:
            frame (np.ndarray): Frame BGR de donde se recortarán las caras.
            boxes (list[tuple]): Cajas (x, y, w, h) a clasificar.

        Returns:
            list[Face]: Faces con probabilidad (o None si no hay clasificador).
        """
        clf = self.classifier
        faces = [Face(b, None, -1) for b in boxes]
        if clf is None or not boxes:
            return faces
        crops, idx = [], []
        for i, b in enumerate(boxes):
            c = crop_face(frame, b, self.margins.get("p6roi", 0.1))
            if c is not None:
                crops.append(preprocess(c))
                idx.append(i)
        if crops:
            for i, p in zip(idx, clf.predict(np.stack(crops))):
                faces[i].prob = float(p)
        return faces


# --------------------------------------------------------------------- dibujo
def draw_faces(frame, faces, threshold):
    """Dibuja las anotaciones de sonrisa sobre el frame (in-place).

    Args:
        frame (np.ndarray): Frame BGR a anotar (se modifica in-place).
        faces (list[Face]): Caras a dibujar.
        threshold (float): Umbral de probabilidad para decidir si sonríe.

    Returns:
        np.ndarray: El mismo frame modificado.
    """
    for f in faces:
        x, y, w, h = f.box
        if f.prob is None:
            cv2.rectangle(frame, (x, y), (x + w, y + h), WHITE, 2)
            continue
        smiling = f.prob >= threshold
        color = GREEN if smiling else RED
        label = f"{'Sonrie' if smiling else 'No sonrie'} {f.prob * 100:.0f}%"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ytxt = y - 6 if y - th - 10 >= 0 else y + h + th + 6
        cv2.rectangle(frame, (x, ytxt - th - 6), (x + tw + 8, ytxt + 4), color, -1)
        cv2.putText(frame, label, (x + 4, ytxt - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    return frame


def draw_banner(frame, lines):
    """Dibuja una franja oscura inferior con texto (para mostrar resultados de la Práctica 6).

    Args:
        frame (np.ndarray): Frame BGR a anotar (se modifica in-place).
        lines (list[str]): Líneas de texto a mostrar en la franja.

    Returns:
        np.ndarray: El mismo frame modificado.
    """
    h, w = frame.shape[:2]
    bh = 12 + 24 * len(lines)
    cv2.rectangle(frame, (0, h - bh), (w, h), (30, 30, 30), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, h - bh + 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, WHITE, 1, cv2.LINE_AA)
    return frame

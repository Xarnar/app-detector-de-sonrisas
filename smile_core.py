"""
Núcleo del detector de sonrisas (sin interfaz).
Detectores de rostros, clasificador de sonrisa, recorte, seguimiento y dibujo.
Lo usa la interfaz gráfica (app_gui.py).

Contrato del clasificador: gris, 64x64, float32 0-255, forma (N,64,64,1).
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

IMG_SIZE = 64

def find_file(filename, search_dirs=("models", "modelo", ".")):
    """Busca un archivo en las carpetas comunes y devuelve su ruta si existe."""
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

GREEN, RED, WHITE = (60, 200, 60), (50, 50, 230), (255, 255, 255)


# ------------------------------------------------------------------ descargas
def _download(url, path, min_bytes):
    """Descarga `path` si no existe o es demasiado pequeño (p. ej. un puntero Git LFS)."""
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
    name = "yunet"

    def __init__(self, path=None):
        if path is None:
            path = find_file(YUNET_FILE)
        _download(YUNET_URL, path, 100_000)
        self.det = cv2.FaceDetectorYN.create(path, "", (320, 320), 0.7, 0.3, 10)

    def detect(self, frame, score=0.7):
        """Devuelve cajas (x, y, w, h) enteras."""
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        self.det.setScoreThreshold(float(score))
        _, faces = self.det.detect(frame)
        if faces is None:
            return []
        return [tuple(int(v) for v in f[:4]) for f in faces]


def load_ssd_net(folder=None):
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
    """La red de la Práctica 6 aplicada al frame completo en UNA sola pasada (tiempo real)."""
    name = "ssd"

    def __init__(self, net):
        self.net = net

    def detect(self, frame, score=0.6):
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


def run_practica6(frame, net, step=24, scale=1.40):
    """Algoritmo de la Práctica 6 (pirámide + ventana deslizante). Lento: ~segundos por frame."""
    import practica_6_deteccion_rostros as p6
    t0 = time.perf_counter()
    r = p6.detect_faces(frame, net, (96, 96), step, scale, 0.90, 0.30, 32, 0.60, 0.50, 0.40)
    secs = time.perf_counter() - t0
    raw = np.asarray(r["boxes"]).reshape(-1, 4)
    boxes = [(int(x1), int(y1), int(x2 - x1), int(y2 - y1)) for x1, y1, x2, y2 in raw]
    return boxes, int(r["total_ventanas"]), secs


class HaarDetector:
    """Clásico de Viola-Jones: pirámide + ventana deslizante + cascada de clasificadores livianos.
    OpenCV lo hace por dentro (detectMultiScale); por eso corre en tiempo real."""
    name = "haar"

    def __init__(self):
        folder = getattr(getattr(cv2, "data", None), "haarcascades", None) \
            or os.path.join(os.path.dirname(cv2.__file__), "data")
        path = os.path.join(folder, "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"No se encontró la cascada de Haar en {path}")

    def detect(self, frame, score=0.6):
        H, W = frame.shape[:2]
        f = 320.0 / W if W > 320 else 1.0                  # se analiza una versión reducida: más rápido
        small = cv2.resize(frame, (int(W * f), int(H * f))) if f < 1 else frame
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        # Haar no da "confianza": la perilla es minNeighbors (cuántas ventanas vecinas deben coincidir)
        n = int(round(3 + (min(max(score, 0.3), 0.95) - 0.3) / 0.65 * 7))
        found = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=n, minSize=(30, 30))
        return [(int(x / f), int(y / f), int(w / f), int(h / f)) for x, y, w, h in found]


class P6RoiDetector:
    """Algoritmo de la Práctica 6 adaptado a tiempo real (EXPERIMENTAL).
    1) Escaneo completo (pirámide + ventana deslizante) en segundo plano, una sola vez.
    2) Luego, en cada frame, solo desliza la ventana cerca de donde estaba la cara (2x2 posiciones).
    Cada ventana cuesta una pasada completa de la red, por eso son pocos FPS."""
    name = "p6roi"
    MAX_TRACKED = 3

    def __init__(self, net, step=32):
        self.net, self.step = net, step
        self.boxes = []
        self.status = ""
        self._thread, self._done, self._found = None, False, []

    def detect(self, frame, score=0.9):
        if self._done:                                    # llegó el resultado del escaneo completo
            self.boxes, self._found = self._found[: self.MAX_TRACKED], []
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

    def _start_scan(self, frame):
        """Escaneo completo de la Práctica 6 (imagen completa, paso 24, escala 1.40) en un hilo aparte.
        Mientras corre, el video sigue en vivo y el estado muestra el avance."""
        img = frame.copy()
        self._cancel = threading.Event()

        def job():
            try:
                raw = []
                for ev in p6_scan(img, self.net, 24, 1.40, batch=16, cancel=self._cancel):
                    if ev["phase"] == "scored":
                        self.status = (f"Escaneo completo (Practica 6): "
                                       f"{ev['done']}/{ev['total']} ventanas")
                    elif ev["phase"] == "done":
                        raw = ev["boxes"]
                    elif ev["phase"] == "cancelled":
                        return
                self._found = [(int(x1), int(y1), int(x2 - x1), int(y2 - y1)) for x1, y1, x2, y2 in raw]
            except Exception as e:                        # noqa: BLE001
                print("[p6roi]", e)
                self._found = []
            self._done = True

        self._thread = threading.Thread(target=job, daemon=True)
        self._thread.start()

    def close(self):
        """Detiene el escaneo en segundo plano (al cambiar de detector)."""
        if getattr(self, "_cancel", None) is not None:
            self._cancel.set()

    def _search(self, frame, box, conf):
        """Desliza la ventana de 96x96 solo alrededor de `box`; prueba 3 tamaños si no encuentra."""
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
            scores, aligns = p6.puntuar_lote(self.net, patches, 0.60, 0.50)
            ok = scores >= conf
            if not ok.any():
                continue
            i = int(np.argmax(np.where(ok, scores * aligns, -1.0)))
            xx, yy = metas[i]
            return (int(x0 + xx * sc), int(y0 + yy * sc), int(96 * sc), int(96 * sc))
        return None


def p6_scan(frame, net, step=24, scale=1.40, batch=8, cancel=None,
            conf_thr=0.90, nms_thr=0.30, min_cover=0.60, context=0.50, contain=0.40):
    """Misma lógica que p6.detect_faces (mismos resultados), pero como GENERADOR que narra cada paso
    para poder dibujarlo. Usa las funciones de la práctica: image_pyramid, sliding_window,
    puntuar_lote y non_max_suppression. Lotes pequeños => animación más fluida."""
    import practica_6_deteccion_rostros as p6
    win = 96
    H0, W0 = frame.shape[:2]
    levels = list(p6.image_pyramid(frame, scale, (win, win)))
    info = []
    for lvl, s, im in levels:
        h, w = im.shape[:2]
        info.append(dict(level=lvl, s=s, size=(w, h),
                         n=len(range(0, h - win + 1, step)) * len(range(0, w - win + 1, step))))
    total = sum(i["n"] for i in info)
    t0 = time.perf_counter()
    yield dict(phase="start", levels=info, total=total, step=step, batch=batch,
               level_imgs=[im for _, _, im in levels])

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
               elapsed=time.perf_counter() - t0)


def render_p6(frame, st):
    """Dibuja el estado de la Práctica 6 sobre el frame congelado.
    gris: ventana ya evaluada | amarillo: en evaluación | naranja: aceptada por la red | verde: resultado."""
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
            cv2.putText(out, f"nivel {li + 1}/{len(imgs)} (imagen / {st['levels'][li]['s']:.2f})",
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
    """Busca el umbral recomendado en el JSON de métricas del modelo (ignora corridas quick_test)."""
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
                    if data.get("quick_test"):
                        continue
                    return float(data["umbral_recomendado"]), os.path.basename(f)
                except Exception:
                    continue
    return 0.5, None


class SmileClassifier:
    def __init__(self, path):
        import tensorflow as tf                # import tardío: la interfaz abre rápido
        from tensorflow import keras
        self.name = os.path.basename(path)
        self.model = keras.models.load_model(path)
        self.lock = threading.Lock()

        # Compilar el grafo una vez (tf.function) en lugar de ejecutar capa por capa en modo
        # "eager": ~6x más rápido con salidas idénticas. El batch queda dinámico (None), así que
        # sirve igual para 1, 2 o 5 caras sin volver a compilar.
        @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 1], tf.float32)])
        def _fast(x):
            return self.model(x, training=False)

        self._fast, self._tf = _fast, tf
        self.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 1), np.float32))     # calentamiento
        self.threshold, self.metrics_file = find_threshold(path)

    def predict(self, batch):
        with self.lock:
            return self._fast(self._tf.constant(batch)).numpy().ravel()


# ---------------------------------------------------------- recorte y preprocesado
def crop_face(frame, box, margin):
    """Recorte cuadrado alrededor de la caja, ampliado por `margin` (0.1 = +10 % por lado)."""
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
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32)[..., None]


# ------------------------------------------------------- seguimiento y suavizado
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, tid, box, n):
        self.id, self.box, self.missed = tid, box, 0
        self.probs = deque(maxlen=n)

    @property
    def smooth(self):
        return float(np.mean(self.probs)) if self.probs else None


class Tracker:
    """Asocia las caras entre frames (IoU) para suavizar la probabilidad por persona."""

    def __init__(self, smooth_n=5, max_missed=10, min_iou=0.25):
        self.tracks, self.next_id = {}, 0
        self.smooth_n, self.max_missed, self.min_iou = smooth_n, max_missed, min_iou

    def set_smooth(self, n):
        self.smooth_n = n
        for t in list(self.tracks.values()):
            t.probs = deque(t.probs, maxlen=n)

    def clear_probs(self):
        for t in list(self.tracks.values()):
            t.probs.clear()

    def update(self, boxes):
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
    __slots__ = ("box", "prob", "tid")

    def __init__(self, box, prob, tid):
        self.box, self.prob, self.tid = box, prob, tid


class Pipeline:
    """Un frame -> caras con probabilidad de sonrisa. Los parámetros se pueden cambiar en vivo."""

    def __init__(self):
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
        return self.margins.get(self.det_name, 0.1)

    @property
    def det_score(self):                       # cada detector recuerda su propia confianza mínima
        return self.scores.get(self.det_name, 0.7)

    @det_score.setter
    def det_score(self, v):
        self.scores[self.det_name] = v

    def process(self, frame):
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
        """Clasifica cajas sueltas (sin seguimiento), p. ej. las de la Práctica 6."""
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
    """Franja inferior con texto (para mostrar resultados de la Práctica 6)."""
    h, w = frame.shape[:2]
    bh = 12 + 24 * len(lines)
    cv2.rectangle(frame, (0, h - bh), (w, h), (30, 30, 30), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, h - bh + 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, WHITE, 1, cv2.LINE_AA)
    return frame
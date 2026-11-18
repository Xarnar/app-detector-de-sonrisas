"""
Detector de sonrisas en tiempo real (interfaz de consola).
===========================================================

Propósito del módulo
--------------------
Versión ligera sin GUI del detector de sonrisas.  Lee frames de una cámara o
video, detecta rostros con YuNet, los clasifica con un modelo CNN (.keras) y
muestra el resultado en una ventana OpenCV.

Pipeline
--------
    cámara/video -> YuNet (rostros) -> recorte + preprocesado -> CNN (sonríe/no) -> overlay

Cómo ejecutarlo
---------------
    python app_sonrisa.py                             # cámara 0, modelo por defecto
    python app_sonrisa.py --source 1 --model mi_modelo.keras
    python app_sonrisa.py --source video.mp4 --threshold 0.4

Teclas durante la ejecución
----------------------------
    q          Salir
    + / -      Aumentar/reducir margen del recorte
    [ / ]      Bajar/subir umbral de sonrisa
    d          Activar/desactivar vista de recortes (debug)
    r          Iniciar/detener grabación de video de respaldo
    s          Captura de pantalla

Dependencias
------------
    numpy, opencv-python, tensorflow (opcional; sin él solo se detecta)

Archivos que necesita
---------------------
    models/face_detection_yunet_2023mar.onnx  – detector (se descarga si falta)
    models/model_A_cnn.keras                  – clasificador (opcional)
    models/A_metrics.json                     – umbral recomendado (opcional)
"""
import argparse, json, os, time, urllib.request
from collections import deque

import cv2
import numpy as np

def find_file(filename, search_dirs=("models", ".")):
    """Busca un archivo en las carpetas comunes y devuelve su ruta si existe.

    Args:
        filename (str): Nombre o ruta del archivo a buscar.
        search_dirs (tuple[str, ...]): Carpetas donde buscar, en orden.

    Returns:
        str: Ruta al archivo encontrado, o ruta sugerida dentro de 'models'.
    """
    if os.path.exists(filename):
        return filename
    for d in search_dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    pref = "models" if os.path.isdir("models") else "."
    return os.path.join(pref, filename) if pref != "." else filename


YUNET_FILE = find_file("face_detection_yunet_2023mar.onnx")
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
IMG_SIZE = 64            # debe coincidir con el entrenamiento
GREEN, RED, WHITE = (60, 200, 60), (50, 50, 230), (255, 255, 255)


# ----------------------------------------------------------------- detector
def load_detector(path, score_thr):
    """Carga el detector YuNet, descargándolo si no existe.

    Args:
        path (str): Ruta al archivo .onnx de YuNet.
        score_thr (float): Umbral de confianza mínimo para aceptar detecciones.

    Returns:
        cv2.FaceDetectorYN: Detector listo para usar.

    Raises:
        SystemExit: Si no se puede descargar el modelo.
    """
    path = find_file(path)
    if not os.path.exists(path):
        print(f"Descargando YuNet a {path} ...")
        try:
            urllib.request.urlretrieve(YUNET_URL, path)
        except Exception as e:
            raise SystemExit(f"No se pudo descargar YuNet ({e}).\n"
                             f"Descárgalo manualmente de:\n{YUNET_URL}\ny ponlo en {path}.")
    return cv2.FaceDetectorYN.create(path, "", (320, 320), score_thr, 0.3, 10)


def detect_faces(detector, frame):
    """Detecta rostros en el frame con YuNet y devuelve cajas (x, y, w, h) enteras.

    Args:
        detector (cv2.FaceDetectorYN): Detector YuNet ya cargado.
        frame (np.ndarray): Frame BGR de entrada.

    Returns:
        list[tuple[int, int, int, int]]: Lista de cajas (x, y, w, h).
    """
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    return [tuple(int(v) for v in f[:4]) for f in faces]


# --------------------------------------------------------------- clasificador
def load_classifier(path):
    """Carga el modelo clasificador de sonrisas, o devuelve None en modo solo-detección.

    Modo solo-detección sin modelo
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Si el archivo .keras no existe (ruta no encontrada), la función imprime un
    aviso claro y devuelve None.  Esto permite que la app arranque sin el modelo
    y muestre los rostros detectados con recuadro blanco (sin clasificación).
    El import de TensorFlow se hace aquí (tardío) para que la app abra rápido
    incluso si TensorFlow no está instalado.

    Args:
        path (str): Ruta al archivo .keras del modelo.

    Returns:
        callable | None: Función ``batch(N,64,64,1 float32 0-255) -> probabilidades (N,)``,
            o None si el modelo no existe.
    """
    resolved = find_file(path) if not os.path.exists(path) else path
    if not os.path.exists(resolved):
        print(f"[aviso] No se encontró {path}: modo solo-detección (sin clasificar).")
        print("        Para obtener los modelos entrenados, ejecute el notebook 01_modelo_A_cnn_desde_cero.ipynb.")
        return None
    from tensorflow import keras           # import tardío: la app arranca rápido sin modelo
    model = keras.models.load_model(resolved)
    model(np.zeros((1, IMG_SIZE, IMG_SIZE, 1), np.float32), training=False)   # calentamiento
    return lambda batch: model(batch, training=False).numpy().ravel()


def crop_face(frame, box, margin):
    """Recorte cuadrado alrededor de la caja, ampliado por ``margin`` (0.1 = +10% por lado).

    Args:
        frame (np.ndarray): Frame BGR de entrada.
        box (tuple[int, int, int, int]): Caja de la cara (x, y, w, h).
        margin (float): Fracción del lado a añadir alrededor del recorte.

    Returns:
        np.ndarray | None: Recorte BGR, o None si es demasiado pequeño.
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
    """Preprocesa un recorte según el contrato del modelo: gris, 64x64, float32 0-255, forma (64,64,1).

    Args:
        crop (np.ndarray): Recorte BGR de la cara.

    Returns:
        np.ndarray: Array (64, 64, 1) float32, valores 0–255.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32)[..., None]


# ------------------------------------------------------- seguimiento + suavizado
def iou(a, b):
    """Calcula el IoU entre dos cajas (x, y, w, h).

    Args:
        a (tuple): Primera caja (x, y, w, h).
        b (tuple): Segunda caja (x, y, w, h).

    Returns:
        float: IoU en [0, 1].
    """
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class Track:
    """Representa el seguimiento de una cara individual.

    Args:
        tid (int): ID único del track.
        box (tuple): Caja inicial (x, y, w, h).
        n (int): Tamaño de la ventana de suavizado de probabilidades.
    """
    def __init__(self, tid, box, n):
        """Inicializa un track de rostro.

        Args:
            tid (int): Identificador numérico del track.
            box (tuple[int, int, int, int]): Coordenadas de la caja (x, y, w, h).
            n (int): Longitud del búfer circular para el promedio móvil de probabilidades.
        """
        self.id, self.box, self.missed = tid, box, 0
        self.probs = deque(maxlen=n)   # historial de probabilidades (ventana deslizante)

    @property
    def smooth(self):
        """Probabilidad promedio suavizada.

        Returns:
            float: Media de las últimas ``n`` probabilidades (0.0 si no hay historial).
        """
        return float(np.mean(self.probs)) if self.probs else 0.0


class Tracker:
    """Asocia las caras de cada frame con las del anterior (IoU) para suavizar por persona."""
    def __init__(self, smooth_n=5, max_missed=10, min_iou=0.25):
        """Inicializa el tracker.

        Args:
            smooth_n (int): Frames para promediar la probabilidad.
            max_missed (int): Frames sin detección antes de eliminar un track.
            min_iou (float): IoU mínimo para asociar una caja a un track.
        """
        self.tracks, self.next_id = {}, 0
        self.smooth_n, self.max_missed, self.min_iou = smooth_n, max_missed, min_iou

    def update(self, boxes):
        """Asocia nuevas cajas con tracks existentes y devuelve los tracks activos.

        Args:
            boxes (list[tuple]): Cajas detectadas en el frame actual.

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
            used_t.add(tid); used_i.add(i); assigned[i] = self.tracks[tid]
        for i, b in enumerate(boxes):
            if i not in assigned:
                assigned[i] = self.tracks[self.next_id] = Track(self.next_id, b, self.smooth_n)
                self.next_id += 1
        for i, t in assigned.items():
            t.box, t.missed = boxes[i], 0
        self.tracks = {k: t for k, t in self.tracks.items() if t.missed <= self.max_missed}
        return [assigned[i] for i in range(len(boxes))]


# ------------------------------------------------------------------- dibujo
def draw_face(frame, track, thr, has_model):
    """Dibuja el recuadro y la etiqueta de sonrisa sobre el frame.

    Si no hay modelo cargado, dibuja recuadro blanco sin etiqueta.

    Args:
        frame (np.ndarray): Frame BGR a anotar (in-place).
        track (Track): Track de la cara a dibujar.
        thr (float): Umbral de probabilidad para decidir si sonríe.
        has_model (bool): True si hay clasificador activo.
    """
    x, y, w, h = track.box
    if not has_model:
        cv2.rectangle(frame, (x, y), (x + w, y + h), WHITE, 2)
        return
    p = track.smooth
    smiling = p >= thr
    color = GREEN if smiling else RED
    label = f"{'Sonrie' if smiling else 'No sonrie'} {p*100:.0f}%"
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), color, -1)
    cv2.putText(frame, label, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)


def debug_strip(crops):
    """Muestra los recortes tal como los ve el modelo (para calibrar el margen).

    Args:
        crops (list[np.ndarray]): Lista de recortes preprocesados (64, 64, 1) float32.

    Returns:
        np.ndarray: Tira horizontal de hasta 6 recortes ampliados a 128×128 px.
    """
    tiles = [cv2.resize(c[..., 0].astype(np.uint8), (128, 128), interpolation=cv2.INTER_NEAREST)
             for c in crops[:6]]
    return np.hstack(tiles) if tiles else np.zeros((128, 128), np.uint8)


# --------------------------------------------------------------------- main
def main():
    """Punto de entrada del detector de sonrisas en consola.

    Pipeline detect+classify en hilo principal (bucle de captura)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    El procesamiento es síncrono: en cada iteración del while se captura un
    frame, se ejecuta YuNet (detección), se recortan y preprocesan las caras
    y se hace un único ``model(batch)`` por frame.  No usa hilos separados
    (a diferencia de app_gui.py) para mantener el código simple.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="índice de cámara (0,1..) o ruta de un video")
    ap.add_argument("--model", default=find_file("model_A_cnn.keras"))
    ap.add_argument("--metrics", default=find_file("A_metrics.json"), help="para leer el umbral recomendado")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--margin", type=float, default=-0.10,
                    help="Fracción del lado a añadir alrededor del recorte YuNet. "
                         "Negativo = recorta el padding extra que ya incluye YuNet. "
                         "Valor calibrado con modo debug para parecerse al dataset SMILES.")
    ap.add_argument("--smooth", type=int, default=5, help="frames para promediar (1 = sin suavizado)")
    ap.add_argument("--det-every", type=int, default=2,
                    help="Ejecutar YuNet cada N frames (1=siempre, 2=cada 2, 3=cada 3). "
                         "Los frames intermedios reutilizan las cajas del tracker.")
    ap.add_argument("--det-score", type=float, default=0.7)
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    thr = args.threshold
    if thr is None:
        thr = 0.5
        metrics_file = find_file(args.metrics) if not os.path.exists(args.metrics) else args.metrics
        if os.path.exists(metrics_file):
            thr = json.load(open(metrics_file)).get("umbral_recomendado", 0.5)
    margin = args.margin

    detector = load_detector(YUNET_FILE, args.det_score)
    # load_classifier devuelve None si el modelo no existe: modo solo-detección
    classify = load_classifier(args.model)
    tracker = Tracker(args.smooth)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente: {args.source}")

    writer, show_debug, fps, t_prev = None, False, 0.0, time.time()
    frame_idx = 0   # contador para saber cuándo toca detectar con YuNet
    print("Listo. Teclas: q salir | +/- margen | [ ] umbral | d debug | r grabar | s captura")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not args.no_mirror and isinstance(src, int):
            frame = cv2.flip(frame, 1)

        # --- Detección: YuNet solo cada `det_every` frames; el resto usa el tracker ---
        frame_idx += 1
        if frame_idx % args.det_every == 0:
            boxes = detect_faces(detector, frame)
            tracks = tracker.update(boxes)
        else:
            # Reutiliza las cajas actuales del tracker (sin llamar a YuNet)
            tracks = list(tracker.tracks.values())
            boxes  = [t.box for t in tracks]

        crops = []
        if classify is not None and boxes:
            valid = []
            for t in tracks:
                c = crop_face(frame, t.box, margin)
                if c is not None:
                    crops.append(preprocess(c)); valid.append(t)
            if crops:
                probs = classify(np.stack(crops))          # un solo predict por frame
                for t, p in zip(valid, probs):
                    t.probs.append(float(p))
        for t in tracks:
            draw_face(frame, t, thr, classify is not None and len(t.probs) > 0)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(now - t_prev, 1e-6)); t_prev = now
        hud = f"FPS {fps:4.1f} | caras {len(boxes)} | margen {margin:.2f} | umbral {thr:.2f} | det/{args.det_every}"
        # Fondo semitransparente para que el HUD sea legible sobre cualquier imagen
        (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (6, 6), (tw + 14, th + 16), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, hud, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        if writer is not None:
            writer.write(frame)
            cv2.circle(frame, (frame.shape[1] - 20, 20), 8, (0, 0, 255), -1)

        cv2.imshow("Detector de sonrisas", frame)
        if show_debug:
            cv2.imshow("Lo que ve el modelo (64x64)", debug_strip(crops))

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k in (ord("+"), ord("=")):
            margin = round(min(0.6, margin + 0.02), 2)
        elif k in (ord("-"), ord("_")):
            margin = round(max(-0.1, margin - 0.02), 2)
        elif k == ord("]"):
            thr = round(min(0.95, thr + 0.02), 2)
        elif k == ord("["):
            thr = round(max(0.05, thr - 0.02), 2)
        elif k == ord("d"):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow("Lo que ve el modelo (64x64)")
        elif k == ord("s"):
            name = f"captura_{int(time.time())}.png"; cv2.imwrite(name, frame); print("guardado", name)
        elif k == ord("r"):
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(f"respaldo_{int(time.time())}.mp4",
                                         cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                print("grabando...")
            else:
                writer.release(); writer = None; print("grabación guardada")

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

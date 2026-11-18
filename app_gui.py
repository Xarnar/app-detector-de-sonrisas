"""
Interfaz gráfica del detector de sonrisas · Versión 2.
=======================================================

Propósito del módulo
--------------------
Aplicación de escritorio (CustomTkinter) que muestra video en tiempo real con
detección de rostros y clasificación de sonrisas.  Permite cambiar detectores
(YuNet, SSD, Haar, Práctica 6), modelos CNN (.keras), umbral, margen y otros
parámetros en vivo sin cortar la captura.

Arquitectura de hilos
---------------------
  - **Hilo de trabajo** (``_worker``): captura frames, ejecuta detección +
    clasificación y deposita el frame anotado en ``self.latest``.
  - **Hilo principal Tk** (``_refresh``): lee ``self.latest`` cada 15 ms y lo
    muestra.  NUNCA se llama a Tk desde el hilo de trabajo; toda comunicación
    pasa por ``self.post()`` (cola ``ui_q``).

Cómo ejecutarlo
---------------
    python app_gui.py                        # cámara 0, modelo por defecto
    python app_gui.py --source video.mp4
    python app_gui.py --model mi_modelo.keras --theme light

    Si no hay modelos .keras disponibles, la app arranca en modo solo-detección
    (encuadra rostros en blanco).  Para obtener los modelos, ejecuta el
    notebook 01_modelo_A_cnn_desde_cero.ipynb en Kaggle o localmente.

Dependencias
------------
    opencv-python>=4.9,<5  numpy  pillow  tensorflow>=2.16  customtkinter>=5.2
    (tkinter viene con Python en Windows/macOS; en Linux: sudo apt install python3-tk)

Archivos que necesita
---------------------
    smile_core.py                          núcleo de detección y clasificación
    practica_6_deteccion_rostros.py        algoritmo Práctica 6 (opcional)
    models/face_detection_yunet_2023mar.onnx   detector YuNet (se descarga solo)
    models/deploy.prototxt                 arquitectura SSD (se descarga solo)
    models/res10_300x300_ssd_iter_140000.caffemodel  pesos SSD (se descarga solo)
    models/model_A_cnn.keras              clasificador de sonrisas (opcional)
    models/A_metrics.json                  umbral recomendado del modelo A (opcional)

Atajos de teclado
-----------------
    q / Esc   salir          s   captura         r   grabar
    d         ver recortes   F11 pantalla completa  H   panel lateral
"""

import argparse
import glob
import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import customtkinter as ctk

import smile_core as sc

# ---------------------------------------------------------------------------
# Constantes de colores y configuración
# ---------------------------------------------------------------------------

# Paleta modo oscuro
DARK = {
    "bg":     "#1b1d23",
    "panel":  "#23262e",
    "fg":     "#e8eaed",
    "muted":  "#9aa0aa",
    "accent": "#4cc38a",
    "warn":   "#f0b429",
    "border": "#343843",
    "field":  "#2d303a",
    "btn":    "#343843",
    "btn_act":"#444a58",
}

# Paleta modo claro
LIGHT = {
    "bg":     "#f0f2f5",
    "panel":  "#ffffff",
    "fg":     "#1a1c20",
    "muted":  "#6b7280",
    "accent": "#16a34a",
    "warn":   "#d97706",
    "border": "#d1d5db",
    "field":  "#f9fafb",
    "btn":    "#e5e7eb",
    "btn_act":"#d1d5db",
}

# Colores del video (siempre oscuros para legibilidad)
VID_GREEN = (60, 200, 60)
VID_RED   = (50,  50, 230)
VID_WHITE = (255, 255, 255)

NO_MODEL = "(sin clasificador)"
MAX_W    = 960          # ancho máximo del frame de trabajo
CONFIG_FILE = "config.json"
SALIDAS_DIR = "salidas"

# Detectores disponibles
DETECTORS = {
    "yunet":  ("YuNet · rápido",
               "Detector ligero moderno: analiza la imagen completa de una sola vez."),
    "ssd":    ("SSD ResNet-10 · red de la Práctica 6",
               "La red de la Práctica 6 aplicada al frame completo (1 pasada)."),
    "haar":   ("Haar · clásico (pirámide + ventana)",
               "Clásico: pirámide + ventana deslizante con clasificadores muy livianos."),
    "p6roi":  ("Práctica 6 + seguimiento · experimental",
               "Ventana deslizante solo cerca de la última cara. Pocos FPS."),
}
DET_KEY = {v[0]: k for k, v in DETECTORS.items()}

P6_PRESETS = {
    "Demo · paso 48, escala 1.40 (~10 s)": (48, 1.40),
    "Rápido · paso 24, escala 1.40":       (24, 1.40),
    "Estándar · paso 16, escala 1.25":     (16, 1.25),
}
P6_BTN_TEXT = "Analizar fotograma (algoritmo Práctica 6)"

# Longitud del historial de sparklines
SPARK_LEN = 60

# Alfa para el suavizado exponencial de las cajas (animación)
BOX_ALPHA = 0.35    # cuánto del valor nuevo se toma (1 = sin suavizado)
PROB_ALPHA = 0.20   # suavizado de la barra de probabilidad animada


# ---------------------------------------------------------------------------
# Búsqueda de fuente TrueType con tilde
# ---------------------------------------------------------------------------

def _encontrar_fuente(tamanio=14):
    """Busca una fuente TrueType del sistema que soporte tildes.
    Si no encuentra ninguna, devuelve None (se usará la fuente PIL por defecto)."""
    candidatos = [
        # Windows
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for ruta in candidatos:
        if os.path.exists(ruta):
            try:
                return ImageFont.truetype(ruta, tamanio)
            except Exception:
                continue
    return None


# Fuente global (se inicializa una vez al arrancar)
_FUENTE_ETIQUETA = _encontrar_fuente(13)
_FUENTE_GRANDE   = _encontrar_fuente(15)


# ---------------------------------------------------------------------------
# Dibujo de etiquetas PIL sobre el frame (píldora con bordes redondeados)
# ---------------------------------------------------------------------------

def _dibujar_etiqueta_pil(frame_bgr, x, y, texto, color_bgr, encima=True):
    """Dibuja una etiqueta tipo píldora con PIL sobre `frame_bgr` (in-place).
    Se convierte SOLO el parche pequeño donde va la etiqueta, no el frame completo.
    Acepta caracteres con tilde ('Sonríe')."""
    fuente = _FUENTE_ETIQUETA or ImageFont.load_default()

    # Medir el texto con PIL
    img_tmp = Image.new("RGB", (1, 1))
    draw_tmp = ImageDraw.Draw(img_tmp)
    bbox = draw_tmp.textbbox((0, 0), texto, font=fuente)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad_x, pad_y = 7, 4
    bw = tw + pad_x * 2
    bh = th + pad_y * 2
    radio = bh // 2          # bordes completamente redondeados (cápsula)

    # Coordenadas del parche en el frame
    px0 = max(0, x)
    py0 = max(0, y - bh - 2) if encima else max(0, y + 2)
    px1 = min(frame_bgr.shape[1], px0 + bw)
    py1 = min(frame_bgr.shape[0], py0 + bh)
    if px1 <= px0 or py1 <= py0:
        return

    # Extraer el parche BGR y convertirlo a PIL RGB
    parche_bgr = frame_bgr[py0:py1, px0:px1].copy()
    parche_pil = Image.fromarray(cv2.cvtColor(parche_bgr, cv2.COLOR_BGR2RGB))

    # Crear la píldora (fondo con color de la etiqueta)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # BGR→RGB
    pildora = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(pildora)
    d.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=radio, fill=color_rgb + (230,))
    d.text((pad_x - bbox[0], pad_y - bbox[1]), texto,
           fill=(255, 255, 255, 255), font=fuente)

    # Pegar la píldora sobre el parche (solo la región que cabe)
    pw = px1 - px0
    ph = py1 - py0
    pildora_crop = pildora.crop((0, 0, pw, ph))
    parche_pil.paste(pildora_crop, (0, 0), pildora_crop)

    # Escribir de vuelta al frame
    frame_bgr[py0:py1, px0:px1] = cv2.cvtColor(
        np.array(parche_pil), cv2.COLOR_RGB2BGR)


def _dibujar_barra_prob(frame_bgr, x, y, w, prob, umbral):
    """Barra de probabilidad (animada) bajo el recuadro de la cara."""
    if prob is None:
        return
    bh = 5
    # Fondo gris
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + bh), (60, 60, 60), -1)
    # Relleno proporcional a la prob
    ancho_lleno = int(w * prob)
    color = VID_GREEN if prob >= umbral else VID_RED
    if ancho_lleno > 0:
        cv2.rectangle(frame_bgr, (x, y), (x + ancho_lleno, y + bh), color, -1)


def _dibujar_esquinas(frame_bgr, x, y, w, h, color, largo=12, grosor=2):
    """Dibuja las esquinas estilo 'bracket' en lugar de un rectángulo completo."""
    pts = [
        # Esquina superior-izquierda
        ((x, y + largo), (x, y), (x + largo, y)),
        # Esquina superior-derecha
        ((x + w - largo, y), (x + w, y), (x + w, y + largo)),
        # Esquina inferior-derecha
        ((x + w, y + h - largo), (x + w, y + h), (x + w - largo, y + h)),
        # Esquina inferior-izquierda
        ((x + largo, y + h), (x, y + h), (x, y + h - largo)),
    ]
    for p1, p2, p3 in pts:
        cv2.line(frame_bgr, p1, p2, color, grosor, cv2.LINE_AA)
        cv2.line(frame_bgr, p2, p3, color, grosor, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Suavizador exponencial de cajas
# ---------------------------------------------------------------------------

class SuavizadorCajas:
    """Suaviza el movimiento de las cajas con un promedio exponencial por persona."""

    def __init__(self):
        """Inicializa el suavizador de cajas."""
        self._cajas = {}   # tid -> (x, y, w, h) suavizado como floats

    def actualizar(self, faces, anim_on):
        """Devuelve la lista de cajas suavizadas para cada cara."""
        cajas_suavizadas = {}
        tids_vistos = set()
        resultado = []
        for f in faces:
            tid = f.tid
            tids_vistos.add(tid)
            bx, by, bw, bh = f.box
            if anim_on and tid in self._cajas:
                ox, oy, ow, oh = self._cajas[tid]
                # Promedio exponencial: nuevo = alpha * actual + (1-alpha) * anterior
                sx = BOX_ALPHA * bx + (1 - BOX_ALPHA) * ox
                sy = BOX_ALPHA * by + (1 - BOX_ALPHA) * oy
                sw = BOX_ALPHA * bw + (1 - BOX_ALPHA) * ow
                sh = BOX_ALPHA * bh + (1 - BOX_ALPHA) * oh
            else:
                sx, sy, sw, sh = float(bx), float(by), float(bw), float(bh)
            cajas_suavizadas[tid] = (sx, sy, sw, sh)
            resultado.append((int(sx), int(sy), int(sw), int(sh)))
        self._cajas = cajas_suavizadas
        return resultado

    def limpiar(self):
        """Borra el historial de cajas suavizadas."""
        self._cajas.clear()


# ---------------------------------------------------------------------------
# Suavizador de probabilidades (para la barra animada en el panel)
# ---------------------------------------------------------------------------

class SuavizadorProb:
    """Interpola suavemente la probabilidad visible hacia el valor real."""

    def __init__(self):
        """Inicializa el suavizador de probabilidades."""
        self._vals = {}    # tid -> float suavizado

    def actualizar(self, faces, anim_on):
        """Devuelve dict tid -> prob suavizada."""
        nuevo = {}
        for f in faces:
            if f.prob is None:
                continue
            tid = f.tid
            if anim_on and tid in self._vals:
                nuevo[tid] = PROB_ALPHA * f.prob + (1 - PROB_ALPHA) * self._vals[tid]
            else:
                nuevo[tid] = f.prob
        self._vals = nuevo
        return nuevo

    def limpiar(self):
        """Borra el historial de probabilidades suavizadas."""
        self._vals.clear()


# ---------------------------------------------------------------------------
# Detector de pulso (transición No sonríe → Sonríe)
# ---------------------------------------------------------------------------

class DetectorPulso:
    """Detecta cuándo una persona pasa de No Sonríe a Sonríe."""

    DURACION = 0.30   # segundos que dura el pulso

    def __init__(self):
        """Inicializa el detector de pulso."""
        self._estados = {}   # tid -> bool (última decisión)
        self._pulsos  = {}   # tid -> float (tiempo de inicio del pulso)

    def actualizar(self, faces, umbral):
        """Llama con la lista de caras; registra transiciones."""
        for f in faces:
            if f.prob is None:
                continue
            tid = f.tid
            sonriendo = f.prob >= umbral
            if tid in self._estados and not self._estados[tid] and sonriendo:
                self._pulsos[tid] = time.time()   # nueva transición
            self._estados[tid] = sonriendo

    def en_pulso(self, tid):
        """¿Está este tid en el período de pulso activo?"""
        t = self._pulsos.get(tid)
        return t is not None and (time.time() - t) < self.DURACION

    def limpiar(self):
        """Borra el historial de pulsos y estados."""
        self._estados.clear()
        self._pulsos.clear()


# ---------------------------------------------------------------------------
# Grad-CAM: mapa de atención del modelo (Tarea 5 - opcional)
# ---------------------------------------------------------------------------

class GradCAM:
    """Calcula mapas de atención Grad-CAM para el clasificador de sonrisas.

    Usa la última capa Activation anterior al GlobalAveragePooling2D,
    igual que en el notebook 04_gradcam_y_errores_modelo_A.ipynb.

    El cálculo se hace con tf.GradientTape compilado con @tf.function
    para que sea rápido (batch dinámico).
    """

    def __init__(self, clf):
        """Inicializa Grad-CAM a partir del SmileClassifier ya cargado.

        Args:
            clf: instancia de sc.SmileClassifier (ya tiene el modelo cargado).
        Raises:
            RuntimeError: si no encuentra la capa objetivo en el modelo.
            ImportError: si TensorFlow no está disponible.
        """
        import tensorflow as tf  # importar aquí para no romper si no hay TF
        self._tf = tf
        self.lock = clf.lock    # mismo lock que el clasificador

        modelo = clf.model

        # Buscar la última capa Activation antes del GlobalAveragePooling2D
        capa_objetivo = None
        for capa in modelo.layers:
            tipo = type(capa).__name__
            if tipo == "GlobalAveragePooling2D":
                break
            if tipo == "Activation":
                capa_objetivo = capa

        if capa_objetivo is None:
            raise RuntimeError(
                "No se encontró capa Activation antes de GlobalAveragePooling2D"
            )

        # Modelo intermedio: entrada → (activaciones de la capa, salida final)
        self._modelo_grad = tf.keras.Model(
            inputs=modelo.inputs,
            outputs=[capa_objetivo.output, modelo.output]
        )

        # Compilar con tf.function para batch dinámico (más rápido)
        @tf.function(input_signature=[
            tf.TensorSpec([None, 64, 64, 1], tf.float32)
        ])
        def _calcular(x):
            with tf.GradientTape() as tape:
                activaciones, predicciones = self._modelo_grad(x, training=False)
                tape.watch(activaciones)
                # Clase 0 = probabilidad de sonrisa (salida del modelo)
                clase = predicciones[:, 0]
            gradientes = tape.gradient(clase, activaciones)
            # Pooled grads por imagen del batch: (N, canales)
            grad_pool = tf.reduce_mean(gradientes, axis=[1, 2])
            # Mapa de calor por imagen: (N, H, W)
            mapas = tf.einsum("bhwc,bc->bhw", activaciones, grad_pool)
            mapas = tf.nn.relu(mapas)   # solo valores positivos
            return mapas, predicciones

        self._calcular = _calcular

        # Calentamiento: primera llamada compila el grafo
        dummy = tf.zeros([1, 64, 64, 1], tf.float32)
        self._calcular(dummy)

    def calcular(self, batch_np):
        """Calcula los mapas Grad-CAM para un batch de recortes.

        Args:
            batch_np: array numpy (N, 64, 64, 1) float32, igual que el
                      clasificador (valores 0-255).
        Returns:
            Lista de N imágenes BGR de tamaño 64×64 con el mapa de calor.
        """
        import tensorflow as tf
        with self.lock:
            mapas_tf, _ = self._calcular(tf.constant(batch_np))
        mapas = mapas_tf.numpy()   # (N, H_act, W_act)

        resultado = []
        for m in mapas:
            # Normalizar al rango 0-1
            mx = m.max()
            if mx > 1e-8:
                m = m / mx
            # Convertir a imagen de color (colormap JET: azul→rojo)
            m_uint8 = (m * 255).astype(np.uint8)
            m_color = cv2.applyColorMap(
                cv2.resize(m_uint8, (64, 64)), cv2.COLORMAP_JET
            )
            resultado.append(m_color)
        return resultado


def _superponer_gradcam(frame_bgr, x, y, w, h, mapa_color):
    """Superpone el mapa Grad-CAM dentro del recuadro de la cara.

    Solo modifica los píxeles dentro de la caja de la cara;
    nunca procesa el frame completo.

    Args:
        frame_bgr: frame de video (modificado in-place).
        x, y, w, h: posición y tamaño de la caja de la cara.
        mapa_color: imagen BGR 64×64 con el mapa de calor.
    """
    # Recortar la ROI al interior de la cara (clampear al borde del frame)
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_bgr.shape[1], x + w)
    y1 = min(frame_bgr.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    ancho = x1 - x0
    alto  = y1 - y0
    # Redimensionar el mapa al tamaño real de la cara en el frame
    mapa_resized = cv2.resize(mapa_color, (ancho, alto))
    roi = frame_bgr[y0:y1, x0:x1]
    # Mezcla: 50% original + 50% mapa de calor
    frame_bgr[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.5, mapa_resized, 0.5, 0)


# ---------------------------------------------------------------------------
# Sparklines (mini-gráficos de rendimiento)
# ---------------------------------------------------------------------------

class Sparkline:
    """Dibuja un mini-gráfico de línea en un tk.Canvas."""

    def __init__(self, canvas, color, maximo=None):
        """Inicializa un gráfico sparkline.

        Args:
            canvas (tk.Canvas): Lienzo donde dibujar la gráfica.
            color (str): Color de la línea (hex o nombre).
            maximo (float | None): Valor máximo de escala; si None, escala auto.
        """
        self.canvas = canvas
        self.color  = color
        self.maximo = maximo   # si None, se escala automáticamente
        self.datos  = deque(maxlen=SPARK_LEN)

    def agregar(self, valor):
        """Agrega una nueva muestra al historial del sparkline.

        Args:
            valor (float): Nuevo valor a registrar.
        """
        self.datos.append(valor)

    def redibujar(self):
        """Redibuja la curva en el canvas según los datos acumulados."""
        c = self.canvas
        c.delete("all")
        if len(self.datos) < 2:
            return
        W = int(c.cget("width"))
        H = int(c.cget("height"))
        datos = list(self.datos)
        maximo = self.maximo or max(datos) or 1
        minimo = 0
        rango = maximo - minimo or 1

        # Calcular puntos
        puntos = []
        for i, v in enumerate(datos):
            px = int(i * W / (len(datos) - 1))
            py = H - int((v - minimo) / rango * (H - 2)) - 1
            puntos.append((px, py))

        # Dibujar línea
        for i in range(len(puntos) - 1):
            c.create_line(puntos[i][0], puntos[i][1],
                          puntos[i+1][0], puntos[i+1][1],
                          fill=self.color, width=1)


# ---------------------------------------------------------------------------
# Clase principal de la aplicación
# ---------------------------------------------------------------------------

class App:
    """Aplicación principal de detección de sonrisas con interfaz CustomTkinter.

    Maneja la arquitectura desacoplada de hilos: captura y procesamiento
    en segundo plano (``_worker``), e interfaz visual en el hilo de Tkinter.
    """

    def __init__(self, root, args):
        """Inicializa la aplicación y construye la interfaz.

        Args:
            root (ctk.CTk): Ventana principal de CustomTkinter.
            args (argparse.Namespace): Argumentos de línea de comandos parseados.
        """
        self.root = root
        self.args = args

        # Pipeline de detección y clasificación
        self.pipe = sc.Pipeline()

        # Sincronización entre hilos
        self.lock      = threading.Lock()
        self.latest    = None          # último frame procesado (escribe el hilo de trabajo)
        self.ui_q      = queue.Queue() # mensajes del hilo de trabajo al hilo principal
        self.stop_evt  = threading.Event()
        self.worker    = None

        # Estado de grabación y Práctica 6
        self.rec_flag  = False
        self.p6_busy   = False
        self.p6_state, self.p6_frame, self.p6_last = None, None, 0.0
        self.p6_cancel = threading.Event()
        self.p6_upsample_var = ctk.BooleanVar(value=False)  # Rostros pequeños (ampliación de pirámide)
        self.freeze_img, self.freeze_dirty = None, False

        # Estado de la visualización
        self.shown_id   = -1
        self.photo      = None
        self.thumb_photo = None

        # Métricas de la interfaz
        self.ui_ms    = 0.0
        self.ui_fps   = 0.0
        self.ui_count = 0
        self.ui_t0    = time.time()
        self.last_stats = 0.0

        # Tamaño de video escalado
        self._video_w = 640
        self._video_h = 480
        self._last_win_size = (0, 0)

        # Modo pantalla completa
        self._fullscreen = False
        self._panel_visible = True

        # Animaciones
        self._suav_cajas = SuavizadorCajas()
        self._suav_prob  = SuavizadorProb()
        self._pulso      = DetectorPulso()

        # Grad-CAM (Tarea 5 - opcional, apagado por defecto)
        self._gradcam         = None    # instancia GradCAM (None hasta que se inicialice)
        self._gradcam_on      = False   # interruptor del usuario
        self._gradcam_frame   = 0       # contador de frames para "cada 3 frames"
        self._gradcam_mapas   = {}      # tid → mapa de calor BGR más reciente
        self._gradcam_auto_off = False  # True si se apagó automáticamente (FPS < 10)

        # Sparklines (historial de métricas)
        self._hist_fps  = deque(maxlen=SPARK_LEN)
        self._hist_det  = deque(maxlen=SPARK_LEN)
        self._hist_cls  = deque(maxlen=SPARK_LEN)
        self._hist_draw = deque(maxlen=SPARK_LEN)

        # Verificar si está disponible la Práctica 6
        try:
            import practica_6_deteccion_rostros  # noqa: F401
            self.p6_ok = True
        except ImportError:
            self.p6_ok = False

        # Cargar configuración guardada
        self._config = self._cargar_config()

        # Determinar tema inicial
        tema_inicial = args.theme or self._config.get("theme", "dark")
        self._tema_actual = tema_inicial

        # Configurar CustomTkinter
        ctk.set_appearance_mode(
            "dark" if tema_inicial == "dark"
            else "light" if tema_inicial == "light"
            else "system"
        )

        # Crear la interfaz
        self._construir_ui()
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.bind_all("<Key>", self._on_key)
        root.bind("<Configure>", self._on_resize)

        # Cargar detector y modelo
        self._load_detector("yunet")
        self._load_model(self._initial_model())

        # Crear carpeta de salidas si no existe
        os.makedirs(SALIDAS_DIR, exist_ok=True)

        # Abrir fuente de video
        src = args.source
        self.open_source(int(src) if str(src).isdigit() else src)

        # Iniciar el bucle de refresco de la ventana
        self.root.after(15, self._refresh)

    # -----------------------------------------------------------------------
    # Gestión de configuración (config.json)
    # -----------------------------------------------------------------------

    def _cargar_config(self):
        """Carga la configuración guardada (tema, etc.) desde config.json."""
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _guardar_config(self):
        """Guarda la configuración actual en config.json."""
        try:
            self._config["theme"] = self._tema_actual
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[config] No se pudo guardar: {e}")

    # -----------------------------------------------------------------------
    # Construcción de la interfaz
    # -----------------------------------------------------------------------

    def _construir_ui(self):
        """Construye todos los widgets de la ventana principal."""
        self.root.title("Detector de Sonrisas · Visión por Computadora")
        self.root.minsize(900, 580)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # ---- Barra superior ------------------------------------------------
        self._barra_top = ctk.CTkFrame(self.root, corner_radius=0, height=46)
        self._barra_top.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._barra_top.columnconfigure(1, weight=1)

        # Título
        ctk.CTkLabel(
            self._barra_top,
            text="🎯  Detector de Sonrisas",
            font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, padx=14, pady=8)

        # Selector de tema (claro / oscuro / sistema)
        self._tema_var = ctk.StringVar(value=self._tema_actual)
        ctk.CTkLabel(self._barra_top, text="Tema:").grid(row=0, column=2, padx=(0, 4))
        ctk.CTkSegmentedButton(
            self._barra_top,
            values=["dark", "light", "system"],
            variable=self._tema_var,
            command=self._cambiar_tema,
            width=200
        ).grid(row=0, column=3, padx=(0, 12))

        # Modo rendimiento (apaga animaciones)
        self._perf_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            self._barra_top,
            text="Modo rendimiento",
            variable=self._perf_var,
            onvalue=True, offvalue=False,
            command=self._on_perf_mode
        ).grid(row=0, column=4, padx=(0, 12))

        # ---- Contenedor principal ------------------------------------------
        self._main = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        self._main.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self._main.columnconfigure(0, weight=1)
        self._main.rowconfigure(0, weight=1)

        # ---- Columna izquierda: video + barra de stats ----------------------
        izq = ctk.CTkFrame(self._main, corner_radius=0, fg_color="transparent")
        izq.grid(row=0, column=0, sticky="nsew", padx=(10, 6), pady=10)
        izq.columnconfigure(0, weight=1)
        izq.rowconfigure(0, weight=1)

        # Etiqueta de video (fondo negro fijo, como en v1)
        self.video = tk.Label(izq, bg="black", fg="#9aa0aa",
                              text="Iniciando fuente de video...",
                              width=80, height=24)
        self.video.grid(row=0, column=0, sticky="nsew")

        # Barra de rendimiento
        self._construir_barra_stats(izq)

        # Tira de recortes 64x64
        self.thumbs = tk.Label(izq, bg="#1b1d23")
        # (se muestra/oculta con _toggle_crops)

        # ---- Panel lateral (pestañas) --------------------------------------
        self._panel = ctk.CTkFrame(self._main, width=310, corner_radius=8)
        self._panel.grid(row=0, column=1, sticky="ns", padx=(6, 10), pady=10)
        self._panel.columnconfigure(0, weight=1)

        self._tabs = ctk.CTkTabview(self._panel, width=295)
        self._tabs.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._panel.rowconfigure(0, weight=1)

        # Pestaña Fuente
        tab_fuente = self._tabs.add("Fuente")
        self._construir_tab_fuente(tab_fuente)

        # Pestaña Detector
        tab_det = self._tabs.add("Detector")
        self._construir_tab_detector(tab_det)

        # Pestaña Clasificador
        tab_clf = self._tabs.add("Clasif.")
        self._construir_tab_clasificador(tab_clf)

        # Pestaña Práctica 6 + Acciones
        tab_p6 = self._tabs.add("P6 / Acc.")
        self._construir_tab_p6(tab_p6)

        # Pestaña Rendimiento
        tab_perf = self._tabs.add("Render.")
        self._construir_tab_rendimiento(tab_perf)

        # ---- Barra de estado inferior ----------------------------------------
        self.status = tk.StringVar(value="Listo")
        tk.Label(self.root, textvariable=self.status,
                 bg="#14161a", fg="#9aa0aa",
                 font=("Segoe UI", 9),
                 anchor="w", padx=8, pady=4
                 ).grid(row=2, column=0, columnspan=2, sticky="ew")

    def _construir_barra_stats(self, padre):
        """Franja compacta de estadísticas bajo el video."""
        bar = ctk.CTkFrame(padre, corner_radius=6, height=50)
        bar.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        bar.columnconfigure(list(range(8)), weight=1)

        self.stat = {}
        campos = [
            ("fps",   "FPS proc."),
            ("ui",    "FPS UI"),
            ("faces", "Caras"),
            ("smile", "Sonriendo"),
            ("det",   "Detección"),
            ("cls",   "Clasif."),
            ("proc",  "Total"),
            ("draw",  "Pantalla"),
        ]
        for col, (k, nombre) in enumerate(campos):
            celda = ctk.CTkFrame(bar, corner_radius=0, fg_color="transparent")
            celda.grid(row=0, column=col, padx=2, pady=4, sticky="nsew")
            ctk.CTkLabel(celda, text=nombre,
                         font=ctk.CTkFont(size=8), text_color=("gray50", "gray60")
                         ).pack()
            self.stat[k] = tk.StringVar(value="-")
            ctk.CTkLabel(celda, textvariable=self.stat[k],
                         font=ctk.CTkFont(family="Consolas", size=10, weight="bold")
                         ).pack()

    def _construir_tab_fuente(self, padre):
        """Pestaña: Fuente de video."""
        padre.columnconfigure(0, weight=1)

        ctk.CTkLabel(padre, text="Índice de cámara:").pack(anchor="w", pady=(6, 2))
        fila = ctk.CTkFrame(padre, fg_color="transparent")
        fila.pack(fill="x")

        self.cam_var = tk.StringVar(value="0")
        ctk.CTkEntry(fila, textvariable=self.cam_var, width=50).pack(side="left")
        ctk.CTkButton(fila, text="Usar cámara", width=120,
                      command=self._use_camera).pack(side="left", padx=6)
        ctk.CTkButton(padre, text="Abrir video...",
                      command=self._open_video).pack(fill="x", pady=(8, 0))

        # Modo espejo
        self.mirror_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(padre, text="Modo espejo (solo cámara)",
                        variable=self.mirror_var, onvalue=True, offvalue=False,
                        command=lambda: setattr(self.pipe, "mirror", self.mirror_var.get())
                        ).pack(anchor="w", pady=(10, 0))

    def _construir_tab_detector(self, padre):
        """Pestaña: Detector de rostros."""
        padre.columnconfigure(0, weight=1)

        ctk.CTkLabel(padre, text="Detector:").pack(anchor="w", pady=(6, 2))
        self.det_var = ctk.StringVar(value=DETECTORS["yunet"][0])
        self.det_cb = ctk.CTkOptionMenu(
            padre,
            variable=self.det_var,
            values=[v[0] for v in DETECTORS.values()],
            command=lambda _: self._on_detector()
        )
        self.det_cb.pack(fill="x")

        # Descripción del detector
        self.det_desc = ctk.CTkLabel(
            padre, text=DETECTORS["yunet"][1],
            wraplength=270, justify="left",
            font=ctk.CTkFont(size=9),
            text_color=("gray50", "gray60")
        )
        self.det_desc.pack(anchor="w", pady=(4, 6))

        # Control deslizante: confianza mínima
        self.set_score = self._slider_ctk(
            padre, "Confianza mínima", 0.3, 0.95,
            self.pipe.det_score, "{:.2f}",
            lambda x: setattr(self.pipe, "det_score", x)
        )

    def _construir_tab_clasificador(self, padre):
        """Pestaña: Clasificador de sonrisa."""
        padre.columnconfigure(0, weight=1)

        ctk.CTkLabel(padre, text="Modelo:").pack(anchor="w", pady=(6, 2))
        self.model_var = ctk.StringVar(value=NO_MODEL)
        self.model_cb = ctk.CTkOptionMenu(
            padre,
            variable=self.model_var,
            values=[NO_MODEL],
            command=lambda _: self._on_model()
        )
        self.model_cb.pack(fill="x")
        self._refresh_models()

        # Control deslizante: margen del recorte
        self.set_margin = self._slider_ctk(
            padre, "Margen del recorte", -0.10, 0.50, 0.10, "{:+.2f}",
            lambda x: self.pipe.margins.__setitem__(self.pipe.det_name, x)
        )
        # Control deslizante: umbral de sonrisa
        self.set_thr = self._slider_ctk(
            padre, "Umbral de sonrisa", 0.05, 0.95, 0.5, "{:.2f}",
            lambda x: setattr(self.pipe, "threshold", x)
        )
        # Control deslizante: suavizado temporal
        self.set_smooth = self._slider_ctk(
            padre, "Suavizado (frames)", 1, 15, 5, "{:.0f}",
            lambda x: self.pipe.tracker.set_smooth(int(round(x)))
        )

        # Ver recortes 64x64
        self.crops_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            padre, text="Ver recortes 64×64",
            variable=self.crops_var, onvalue=True, offvalue=False,
            command=self._toggle_crops
        ).pack(anchor="w", pady=(10, 0))

        # Barra de probabilidad de la cara principal (en el panel)
        ctk.CTkLabel(padre, text="Prob. cara principal:").pack(anchor="w", pady=(10, 2))
        self._prob_panel_var = tk.DoubleVar(value=0.0)
        self._prob_barra = ctk.CTkProgressBar(padre)
        self._prob_barra.set(0)
        self._prob_barra.pack(fill="x")
        self._prob_label = ctk.CTkLabel(padre, text="–")
        self._prob_label.pack(anchor="e")

        # Grad-CAM: interruptor (Tarea 5 - opcional)
        ctk.CTkLabel(padre, text="Atención del modelo:",
                     font=ctk.CTkFont(size=10)).pack(anchor="w", pady=(12, 2))
        self._gradcam_var = ctk.BooleanVar(value=False)
        self._gradcam_chk = ctk.CTkCheckBox(
            padre,
            text="Ver atención Grad-CAM",
            variable=self._gradcam_var,
            onvalue=True,
            offvalue=False,
            command=self._on_gradcam_toggle,
            state="disabled",        # se habilita cuando hay modelo cargado
        )
        self._gradcam_chk.pack(anchor="w")
        # Etiqueta de estado pequeña debajo del checkbox
        self._gradcam_status = ctk.CTkLabel(
            padre, text="Carga un modelo para activarlo",
            font=ctk.CTkFont(size=9),
            text_color=("gray50", "gray60"),
        )
        self._gradcam_status.pack(anchor="w")

    def _construir_tab_p6(self, padre):
        """Pestaña: Práctica 6 y Acciones."""
        padre.columnconfigure(0, weight=1)

        # Acciones: Grabar y Captura
        ctk.CTkLabel(padre, text="Acciones:").pack(anchor="w", pady=(6, 4))
        fila = ctk.CTkFrame(padre, fg_color="transparent")
        fila.pack(fill="x")
        self.rec_btn = ctk.CTkButton(fila, text="● Grabar",
                                     command=self._toggle_rec)
        self.rec_btn.pack(side="left", expand=True, fill="x", padx=(0, 3))
        ctk.CTkButton(fila, text="📷 Captura",
                      command=self._screenshot).pack(side="left", expand=True,
                                                     fill="x", padx=(3, 0))

        # Separador
        ctk.CTkLabel(padre, text="Práctica 6 (ventana deslizante):",
                     font=ctk.CTkFont(size=10)
                     ).pack(anchor="w", pady=(12, 2))
        ctk.CTkLabel(
            padre,
            text="Muestra el proceso paso a paso sobre 1 fotograma.",
            font=ctk.CTkFont(size=9),
            text_color=("gray50", "gray60"),
            wraplength=270, justify="left"
        ).pack(anchor="w")

        # Selector de preset
        self.p6_var = ctk.StringVar(value=list(P6_PRESETS)[0])
        ctk.CTkOptionMenu(
            padre,
            variable=self.p6_var,
            values=list(P6_PRESETS),
            command=lambda _: self._actualizar_tam_min_p6()
        ).pack(fill="x", pady=(4, 6))

        # Casilla: Rostros pequeños (ampliación de pirámide)
        self.p6_upsample_chk = ctk.CTkCheckBox(
            padre,
            text="Rostros pequeños (más lento)",
            variable=self.p6_upsample_var,
            command=self._on_p6_upsample_toggle,
            font=ctk.CTkFont(size=11),
        )
        self.p6_upsample_chk.pack(anchor="w", pady=(2, 2))

        # Etiqueta: tamaño mínimo de rostro detectable
        self.p6_min_size_lbl = ctk.CTkLabel(
            padre,
            text="",
            font=ctk.CTkFont(size=9),
            text_color=("gray50", "gray60"),
            justify="left"
        )
        self.p6_min_size_lbl.pack(anchor="w", pady=(0, 6))
        self._actualizar_tam_min_p6()

        # Botón de Práctica 6
        self.p6_btn = ctk.CTkButton(
            padre, text=P6_BTN_TEXT,
            command=self._toggle_p6,
            fg_color=("#16a34a", "#4cc38a"),
            hover_color=("#15803d", "#63d69f"),
            text_color=("white", "#0e1a14")
        )
        self.p6_btn.pack(fill="x")

        if not self.p6_ok:
            self.p6_btn.configure(state="disabled")
            self.p6_upsample_chk.configure(state="disabled")
            ctk.CTkLabel(
                padre,
                text="⚠ Falta practica_6_deteccion_rostros.py",
                font=ctk.CTkFont(size=9),
                text_color=("#d97706", "#f0b429")
            ).pack(anchor="w", pady=(4, 0))

    def _construir_tab_rendimiento(self, padre):
        """Pestaña: Sparklines de rendimiento."""
        padre.columnconfigure(0, weight=1)

        ctk.CTkLabel(padre, text="Métricas en tiempo real:",
                     font=ctk.CTkFont(size=10)).pack(anchor="w", pady=(6, 4))

        # Colores según el modo
        c_fps  = "#4cc38a"
        c_det  = "#60a5fa"
        c_cls  = "#f59e0b"
        c_draw = "#a78bfa"

        def _fila_spark(label, color, maximo=None):
            """Crea un renglón con nombre y canvas del sparkline."""
            fila = ctk.CTkFrame(padre, fg_color="transparent")
            fila.pack(fill="x", pady=2)
            ctk.CTkLabel(fila, text=label, width=85,
                         font=ctk.CTkFont(size=9)).pack(side="left")
            canvas = tk.Canvas(fila, width=170, height=28,
                               bg="#23262e", highlightthickness=0)
            canvas.pack(side="left")
            spark = Sparkline(canvas, color, maximo)
            return spark

        self._spark_fps  = _fila_spark("FPS proc.",  c_fps,  maximo=60)
        self._spark_det  = _fila_spark("Det. (ms)",  c_det,  maximo=100)
        self._spark_cls  = _fila_spark("Cls. (ms)",  c_cls,  maximo=200)
        self._spark_draw = _fila_spark("Pantalla ms",c_draw, maximo=15)

        # Botón para limpiar el historial
        ctk.CTkButton(padre, text="Limpiar historial",
                      command=self._limpiar_hist
                      ).pack(fill="x", pady=(10, 0))

    # -----------------------------------------------------------------------
    # Helpers para sliders con CTkSlider
    # -----------------------------------------------------------------------

    def _slider_ctk(self, padre, etiqueta, lo, hi, inicio, fmt, cb):
        """Crea un slider con etiqueta y valor en CustomTkinter; devuelve función setter."""
        fila_top = ctk.CTkFrame(padre, fg_color="transparent")
        fila_top.pack(fill="x", pady=(6, 0))
        ctk.CTkLabel(fila_top, text=etiqueta, font=ctk.CTkFont(size=10)).pack(side="left")
        val_lbl = ctk.CTkLabel(fila_top, text=fmt.format(inicio),
                               font=ctk.CTkFont(size=10))
        val_lbl.pack(side="right")

        var = tk.DoubleVar(value=inicio)

        def on_move(v):
            x = float(v)
            var.set(x)
            val_lbl.configure(text=fmt.format(x))
            cb(x)

        slider = ctk.CTkSlider(padre, from_=lo, to=hi, variable=var, command=on_move)
        slider.pack(fill="x")

        def setter(x):
            var.set(x)
            val_lbl.configure(text=fmt.format(x))

        return setter

    # -----------------------------------------------------------------------
    # Cambio de tema
    # -----------------------------------------------------------------------

    def _cambiar_tema(self, tema):
        """Cambia el tema en vivo sin reiniciar ni cortar el video."""
        self._tema_actual = tema
        ctk.set_appearance_mode(
            "dark" if tema == "dark"
            else "light" if tema == "light"
            else "system"
        )
        # Actualizar fondo negro del video (siempre oscuro)
        self.video.configure(bg="black")
        self._guardar_config()
        self.status.set(f"Tema cambiado: {tema}")

    def _on_perf_mode(self):
        """Activa/desactiva el modo rendimiento (apaga animaciones)."""
        activo = self._perf_var.get()
        if activo:
            # Reiniciar suavizadores para que no haya efecto visual residual
            self._suav_cajas.limpiar()
            self._suav_prob.limpiar()
            self._pulso.limpiar()
            self._gradcam_mapas.clear()
            self.status.set("Modo rendimiento ON: animaciones desactivadas")
        else:
            self.status.set("Modo rendimiento OFF: animaciones activadas")

    def _on_gradcam_toggle(self):
        """Activa o desactiva la visualización de atención Grad-CAM (Tarea 5)."""
        activo = self._gradcam_var.get()
        if activo:
            if self.pipe.classifier is None:
                self._gradcam_var.set(False)
                self._gradcam_status.configure(text="Sin clasificador cargado")
                return
            if self._gradcam is None:
                try:
                    self._gradcam = GradCAM(self.pipe.classifier)
                except Exception as e:
                    msg = str(e)
                    self._gradcam_var.set(False)
                    self._gradcam_status.configure(text=f"No disponible: {msg[:45]}")
                    return
            self._gradcam_on = True
            self._gradcam_auto_off = False
            self._gradcam_status.configure(text="Grad-CAM activo (máx 3 caras, c/3 frames)")
        else:
            self._gradcam_on = False
            self._gradcam_mapas.clear()
            if not self._gradcam_auto_off:
                self._gradcam_status.configure(text="")

    # -----------------------------------------------------------------------
    # Comunicación entre hilos
    # -----------------------------------------------------------------------

    def post(self, fn):
        """Encola una función para que el hilo principal la ejecute. Seguro desde cualquier hilo."""
        self.ui_q.put(fn)

    def say(self, text):
        """Muestra un mensaje en la barra de estado (seguro desde cualquier hilo)."""
        self.post(lambda: self.status.set(text))

    # -----------------------------------------------------------------------
    # Carga del detector
    # -----------------------------------------------------------------------

    def _load_detector(self, name):
        """Carga el detector indicado en un hilo secundario."""
        self.say(f"Cargando detector {name}...")

        def job():
            try:
                if name == "yunet":
                    det = sc.YuNetDetector()
                elif name == "ssd":
                    det = sc.SSDDetector(sc.load_ssd_net())
                elif name == "haar":
                    det = sc.HaarDetector()
                else:
                    if not self.p6_ok:
                        raise RuntimeError("Falta practica_6_deteccion_rostros.py")
                    upsample = 1 if (hasattr(self, "p6_upsample_var") and self.p6_upsample_var.get()) else 0
                    det = sc.P6RoiDetector(sc.load_ssd_net(), upsample_levels=upsample)
            except Exception as e:                       # noqa: BLE001
                msg = str(e)   # copiar el mensaje ANTES de salir del except
                print(f"[detector {name}] {msg}")
                self.post(lambda: self._detector_failed(name, msg))
                return
            self.post(lambda: self._detector_ready(name, det))

        threading.Thread(target=job, daemon=True).start()

    def _detector_ready(self, name, det):
        """Se llama desde el hilo principal cuando el detector quedó listo."""
        old = self.pipe.detector
        if old is not None and hasattr(old, "close"):
            old.close()
        self.pipe.detector, self.pipe.det_name = det, name
        self.det_var.set(DETECTORS[name][0])
        self.det_desc.configure(text=DETECTORS[name][1])
        self.set_margin(self.pipe.margins[name])
        self.set_score(self.pipe.det_score)
        self.pipe.tracker.tracks.clear()
        self._suav_cajas.limpiar()
        self._pulso.limpiar()
        self.status.set(f"Detector activo: {DETECTORS[name][0]}")

    def _detector_failed(self, name, msg):
        """Se llama desde el hilo principal si el detector no pudo cargarse."""
        cur = self.pipe.det_name if self.pipe.detector else "yunet"
        self.det_var.set(DETECTORS[cur][0] if self.pipe.detector else "")
        self.status.set(f"No se pudo cargar {name}: {msg[:150]}")
        if self.pipe.detector is None and name == "yunet":
            self._load_detector("ssd")   # plan B si YuNet falla

    def _on_detector(self):
        """Responde al cambio de selector de detector."""
        name = DET_KEY.get(self.det_var.get())
        if name and (name != self.pipe.det_name or self.pipe.detector is None):
            self._load_detector(name)

    # -----------------------------------------------------------------------
    # Carga del modelo clasificador
    # -----------------------------------------------------------------------

    def _refresh_models(self):
        """Busca archivos .keras en las carpetas de modelos."""
        dirs = [self.args.models_dir]
        for extra in ("models", "."):
            if extra not in dirs and os.path.isdir(extra):
                dirs.append(extra)
        self.model_files = {}
        for d in dirs:
            for f in sorted(glob.glob(os.path.join(d, "*.keras"))):
                name = os.path.basename(f)
                if name not in self.model_files:
                    self.model_files[name] = f
        opciones = [NO_MODEL] + list(self.model_files)
        self.model_cb.configure(values=opciones)

    def _initial_model(self):
        """Decide el modelo inicial (argumento CLI, o model_A_cnn.keras si existe)."""
        if self.args.model:
            return self.args.model
        pref = sc.find_file("model_A_cnn.keras", (self.args.models_dir, "models", "."))
        if pref and os.path.exists(pref):
            return pref
        return next(iter(self.model_files.values()), None)

    def _load_model(self, path):
        """Carga el modelo clasificador en un hilo secundario.

        Modo solo-detección sin modelo
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Si ``path`` es None o vacío (el usuario seleccionó "(sin clasificador)"),
        se pone ``self.pipe.classifier = None`` y la app sigue funcionando en modo
        solo-detección: los rostros se encuadran en blanco sin etiqueta de sonrisa.
        Así la app arranca y funciona correctamente aunque no haya ningún .keras.

        Args:
            path (str | None): Ruta al archivo .keras, o None para modo solo-detección.
        """
        if not path:
            self.pipe.classifier = None
            self.model_var.set(NO_MODEL)
            self._gradcam = None
            self._gradcam_mapas.clear()
            self._gradcam_on = False
            self._gradcam_var.set(False)
            self._gradcam_chk.configure(state="disabled")
            self._gradcam_status.configure(text="Carga un modelo para activarlo")
            self.status.set("Modo solo-detección (sin modelo .keras). Para obtener los modelos, ejecute el notebook 01_modelo_A_cnn_desde_cero.ipynb.")
            return
        self.say(f"Cargando modelo {os.path.basename(path)}...")

        def job():
            try:
                clf = sc.SmileClassifier(path)
            except Exception as e:                       # noqa: BLE001
                msg = str(e)
                print(f"[modelo] {msg}")
                self.post(lambda: self.status.set(f"No se pudo cargar el modelo: {msg[:150]}"))
                return
            self.post(lambda: self._model_ready(clf))

        threading.Thread(target=job, daemon=True).start()

    def _model_ready(self, clf):
        """Se llama desde el hilo principal cuando el modelo quedó listo."""
        self.pipe.classifier = clf
        self.pipe.threshold = min(0.95, max(0.05, clf.threshold))
        self.set_thr(self.pipe.threshold)
        self.pipe.tracker.clear_probs()
        self.model_var.set(clf.name)
        # Resetear Grad-CAM para el nuevo modelo
        self._gradcam = None
        self._gradcam_mapas.clear()
        self._gradcam_on = False
        self._gradcam_var.set(False)
        self._gradcam_auto_off = False
        self._gradcam_chk.configure(state="normal")
        self._gradcam_status.configure(text="Listo para activar")
        src = (f"de {clf.metrics_file}" if clf.metrics_file
               else "por defecto (no hay JSON de métricas)")
        self.status.set(f"Modelo {clf.name} · umbral {self.pipe.threshold:.2f} {src}")

    def _on_model(self):
        """Responde al cambio de selector de modelo."""
        name = self.model_var.get()
        self._load_model(None if name == NO_MODEL else self.model_files.get(name))

    # -----------------------------------------------------------------------
    # Fuente de video
    # -----------------------------------------------------------------------

    def _use_camera(self):
        """Conecta a la cámara según el índice indicado en el selector."""
        try:
            self.open_source(int(self.cam_var.get()))
        except ValueError:
            self.status.set("Índice de cámara no válido")

    def _open_video(self):
        """Abre un diálogo de archivo para seleccionar y reproducir un video."""
        path = filedialog.askopenfilename(
            title="Elegir video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")]
        )
        if path:
            self.open_source(path)

    def open_source(self, src):
        """Cierra el hilo de captura actual y abre una nueva fuente."""
        self.stop_evt.set()
        if self.worker is not None:
            self.worker.join(timeout=2)
        self.stop_evt = threading.Event()
        self.worker = threading.Thread(
            target=self._worker, args=(src, self.stop_evt), daemon=True
        )
        self.worker.start()

    # -----------------------------------------------------------------------
    # Hilo de procesamiento (captura + detección + clasificación)
    # -----------------------------------------------------------------------

    def _worker(self, src, evt):
        """Hilo de trabajo: captura frames y ejecuta el pipeline de visión.
        NUNCA modifica widgets de Tk; envía todo mediante self.post()."""
        cap = cv2.VideoCapture(src)
        is_file = not isinstance(src, int)
        if not is_file:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.say(f"No se pudo abrir la fuente: {src}")
            return
        self.say(f"Fuente abierta: {src}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer, fps, t_prev, fid = None, 0.0, time.perf_counter(), 0

        while not evt.is_set():
            t_loop = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # el video se repite
                    continue
                self.say("Se perdió la señal de la cámara")
                break

            h, w = frame.shape[:2]
            if w > MAX_W:
                frame = cv2.resize(frame, (MAX_W, int(h * MAX_W / w)))
            if self.pipe.mirror and not is_file:
                frame = cv2.flip(frame, 1)

            # Mientras corre la Práctica 6 dejamos la CPU libre
            if self.p6_busy:
                time.sleep(0.03)
                continue

            # Pipeline: detectar + clasificar (tolerante a excepciones transitorias)
            try:
                faces, crops, t_det, t_cls = self.pipe.process(frame)
            except Exception as e:
                print(f"[worker warning] Error en procesamiento de fotograma: {e}")
                faces, crops, t_det, t_cls = [], [], 0.0, 0.0

            # Grad-CAM: mapa de atención (máx 3 caras, cada 3 frames, Tarea 5)
            anim_on = not self._perf_var.get()
            self._gradcam_frame += 1
            if (self._gradcam_on and anim_on and self._gradcam is not None
                    and crops and self.pipe.classifier is not None):
                if self._gradcam_frame % 3 == 0:
                    try:
                        valid_faces = [f for f in faces if f.prob is not None][:3]
                        n_c = min(len(crops), len(valid_faces))
                        if n_c > 0:
                            batch = np.stack(crops[:n_c])
                            mapas = self._gradcam.calcular(batch)
                            for f, m in zip(valid_faces[:n_c], mapas):
                                self._gradcam_mapas[f.tid] = m
                    except Exception as e:
                        print(f"[gradcam] {e}")
            elif not self._gradcam_on or not anim_on:
                self._gradcam_mapas.clear()

            # Limpiar caras que ya no están visibles
            if self._gradcam_mapas:
                vistos = {f.tid for f in faces}
                self._gradcam_mapas = {
                    tid: m for tid, m in self._gradcam_mapas.items() if tid in vistos
                }

            # Dibujar el frame anotado
            out = frame.copy()
            self._dibujar_frame(out, faces)

            # Información extra del detector P6ROI
            info = getattr(self.pipe.detector, "status", "")
            if info:
                cv2.putText(out, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(out, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 255), 1, cv2.LINE_AA)

            # Calcular FPS de procesamiento
            now = time.perf_counter()
            inst = 1.0 / max(now - t_prev, 1e-6)
            fps = inst if fid == 0 else 0.9 * fps + 0.1 * inst
            t_prev = now

            # Grabación de video de respaldo en salidas/
            if self.rec_flag and writer is None:
                hh, ww = out.shape[:2]
                nombre = os.path.join(SALIDAS_DIR, f"respaldo_{int(time.time())}.mp4")
                writer = cv2.VideoWriter(
                    nombre, cv2.VideoWriter_fourcc(*"mp4v"), 20, (ww, hh)
                )
            elif not self.rec_flag and writer is not None:
                writer.release()
                writer = None
            if writer is not None:
                writer.write(out)

            fid += 1
            thr = self.pipe.threshold
            with self.lock:
                self.latest = dict(
                    id=fid, raw=frame, img=out, crops=crops, fps=fps,
                    faces=faces,
                    n=len(faces),
                    smiling=sum(1 for f in faces
                                if f.prob is not None and f.prob >= thr),
                    t_det=t_det, t_cls=t_cls,
                    t_proc=(time.perf_counter() - t_loop) * 1000
                )

            if is_file:   # a la velocidad original del video
                time.sleep(max(0.0, 1.0 / src_fps - (time.perf_counter() - t_loop)))

        cap.release()
        if writer is not None:
            writer.release()

    # -----------------------------------------------------------------------
    # Dibujo sobre el frame (anotaciones mejoradas)
    # -----------------------------------------------------------------------

    def _dibujar_frame(self, frame, faces):
        """Dibuja las anotaciones sobre el frame BGR.
        Se llama desde el hilo de trabajo; no debe tocar Tk."""
        anim_on = not self._perf_var.get()
        umbral  = self.pipe.threshold

        # Actualizar suavizadores
        cajas_suav = self._suav_cajas.actualizar(faces, anim_on)
        probs_suav = self._suav_prob.actualizar(faces, anim_on)
        self._pulso.actualizar(faces, umbral)

        for i, f in enumerate(faces):
            if i >= len(cajas_suav):
                break
            x, y, w, h = cajas_suav[i]
            prob = probs_suav.get(f.tid)
            en_pulso = anim_on and self._pulso.en_pulso(f.tid)

            if f.prob is None:
                # Sin clasificador: recuadro blanco
                _dibujar_esquinas(frame, x, y, w, h, VID_WHITE)
                continue

            # Superponer mapa Grad-CAM dentro del recuadro de la cara (Tarea 5)
            if anim_on and self._gradcam_on and f.tid in self._gradcam_mapas:
                _superponer_gradcam(frame, x, y, w, h, self._gradcam_mapas[f.tid])

            sonriendo = f.prob >= umbral
            color = VID_GREEN if sonriendo else VID_RED

            # Pulso: añadir brillo al borde
            grosor = 3 if en_pulso else 2

            if anim_on:
                # Estilo moderno: solo esquinas
                _dibujar_esquinas(frame, x, y, w, h, color, largo=14, grosor=grosor)
            else:
                # Modo rendimiento: rectángulo clásico (igual que v1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Etiqueta PIL tipo píldora (soporta tildes)
            if anim_on:
                texto = f"{'Sonríe' if sonriendo else 'No sonríe'}  {f.prob * 100:.0f}%"
                _dibujar_etiqueta_pil(frame, x, y, texto, color, encima=(y > 30))
            else:
                # Modo rendimiento: texto Hershey clásico
                label = f"{'Sonrie' if sonriendo else 'No sonrie'} {f.prob * 100:.0f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ytxt = y - 6 if y - th - 10 >= 0 else y + h + th + 6
                cv2.rectangle(frame, (x, ytxt - th - 6), (x + tw + 8, ytxt + 4), color, -1)
                cv2.putText(frame, label, (x + 4, ytxt - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, VID_WHITE, 2)

            # Barra de probabilidad animada bajo el recuadro
            if anim_on and prob is not None:
                _dibujar_barra_prob(frame, x, y + h + 2, w, prob, umbral)

    # -----------------------------------------------------------------------
    # Refresco de la ventana (hilo principal, 15 ms)
    # -----------------------------------------------------------------------

    def _refresh(self):
        """Bucle principal de la interfaz: drena la cola, actualiza el video y las estadísticas.
        Se llama cada 15 ms desde el hilo principal (Tk after)."""
        # 1) Ejecutar todas las funciones que el hilo de trabajo mandó
        while True:
            try:
                self.ui_q.get_nowait()()
            except queue.Empty:
                break
            except Exception as e:                               # noqa: BLE001
                print("[ui]", e)

        # 2) Mostrar el frame correcto
        with self.lock:
            lat = self.latest

        if self.p6_busy and self.p6_state is not None:
            # Mostrar la animación de la Práctica 6 (~14 fps)
            if time.time() - self.p6_last > 0.07:
                self.p6_last = time.time()
                self._show(sc.render_p6(self.p6_frame, self.p6_state))
        elif self.freeze_img is not None:
            if self.freeze_dirty:
                self._show(self.freeze_img)
                self.freeze_dirty = False
        elif lat is not None and lat["id"] != self.shown_id:
            self.shown_id = lat["id"]
            self._show(lat["img"])
            self.ui_count += 1
            if self.crops_var.get():
                self._show_crops(lat["crops"])

        # 3) Actualizar estadísticas (máx. 4 veces por segundo)
        now = time.time()
        if lat is not None and now - self.last_stats > 0.25:
            self.last_stats = now
            dt = now - self.ui_t0
            if dt >= 1.0:
                self.ui_fps   = self.ui_count / dt
                self.ui_count = 0
                self.ui_t0    = now

            s = self.stat
            s["fps"].set(f"{lat['fps']:.1f}")
            s["ui"].set(f"{self.ui_fps:.1f}")
            s["faces"].set(str(lat["n"]))
            s["smile"].set(
                f"{lat['smiling']} / {lat['n']}" if self.pipe.classifier else "-"
            )
            s["det"].set(f"{lat['t_det']:.1f} ms")
            s["cls"].set(f"{lat['t_cls']:.1f} ms" if self.pipe.classifier else "-")
            s["proc"].set(f"{lat['t_proc']:.1f} ms")
            s["draw"].set(f"{self.ui_ms:.1f} ms")

            # Auto-desactivar Grad-CAM si los FPS bajan de 10 (presupuesto de rendimiento)
            if self._gradcam_on and lat["fps"] < 10.0 and not self._gradcam_auto_off:
                self._gradcam_auto_off = True
                self._gradcam_on = False
                self._gradcam_var.set(False)
                self._gradcam_mapas.clear()
                self._gradcam_status.configure(
                    text="⚠ Auto-desactivado (FPS < 10)"
                )

            # Actualizar barra de probabilidad de la cara principal en el panel
            faces = lat.get("faces", [])
            if faces and self.pipe.classifier:
                f0 = faces[0]
                if f0.prob is not None:
                    self._prob_barra.set(f0.prob)
                    self._prob_label.configure(
                        text=f"{f0.prob * 100:.0f}%  {'😊 Sonríe' if f0.prob >= self.pipe.threshold else '😐 No sonríe'}"
                    )
            else:
                self._prob_barra.set(0)
                self._prob_label.configure(text="–")

            # Actualizar sparklines
            self._hist_fps.append(lat["fps"])
            self._hist_det.append(lat["t_det"])
            self._hist_cls.append(lat["t_cls"] if self.pipe.classifier else 0)
            self._hist_draw.append(self.ui_ms)

            if not self._perf_var.get():
                for spark, hist in [
                    (self._spark_fps,  self._hist_fps),
                    (self._spark_det,  self._hist_det),
                    (self._spark_cls,  self._hist_cls),
                    (self._spark_draw, self._hist_draw),
                ]:
                    for v in hist:
                        spark.datos.append(v) if len(spark.datos) == 0 else None
                    spark.datos = deque(list(hist)[-SPARK_LEN:], maxlen=SPARK_LEN)
                    spark.redibujar()

        self.root.after(15, self._refresh)

    def _show(self, bgr):
        """Convierte un frame BGR a PhotoImage y lo muestra en self.video.
        Escala el frame al tamaño disponible del contenedor (manteniendo proporción).
        Permite ampliar Y reducir para que el video llene el espacio disponible."""
        t0 = time.perf_counter()

        # Obtener el tamaño disponible del widget de video
        w_disp = self.video.winfo_width()
        h_disp = self.video.winfo_height()
        if w_disp < 10:
            w_disp = 640    # antes del primer Configure
        if h_disp < 10:
            h_disp = 480

        h_orig, w_orig = bgr.shape[:2]
        # Escalar ajustando tanto alto como ancho (sin límite de 1.0):
        # se puede ampliar cuando la ventana es más grande que el frame
        escala = min(w_disp / w_orig, h_disp / h_orig)
        if abs(escala - 1.0) > 0.01:
            new_w = int(w_orig * escala)
            new_h = int(h_orig * escala)
            # INTER_AREA para reducir (mejor calidad), INTER_LINEAR para ampliar
            interp = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_LINEAR
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=interp)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video.config(image=self.photo, text="")
        self.ui_ms = 0.8 * self.ui_ms + 0.2 * (time.perf_counter() - t0) * 1000

    def _show_crops(self, crops):
        """Muestra los recortes 64x64 en la tira de miniaturas."""
        if not crops:
            return
        strip = np.hstack([c[..., 0].astype(np.uint8) for c in crops[:8]])
        self.thumb_photo = ImageTk.PhotoImage(Image.fromarray(strip))
        self.thumbs.config(image=self.thumb_photo)

    def _toggle_crops(self):
        """Muestra u oculta la tira de recortes 64x64."""
        if self.crops_var.get():
            self.thumbs.grid(row=2, column=0, sticky="w", pady=(4, 0))
        else:
            self.thumbs.grid_forget()

    def _on_resize(self, event):
        """Responde al cambio de tamaño de la ventana."""
        # Solo actualizar si el frame de video cambió de tamaño
        pass   # el escalado se hace en _show() en cada cuadro

    # -----------------------------------------------------------------------
    # Pantalla completa y panel
    # -----------------------------------------------------------------------

    def _toggle_fullscreen(self):
        """Alterna el modo pantalla completa (F11)."""
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)
        if self._fullscreen:
            # Ocultar panel lateral en pantalla completa
            self._panel.grid_remove()
            self.status.set("Pantalla completa · H: panel · F11: salir")
        else:
            self._panel.grid()
            self.status.set("Modo ventana")

    def _toggle_panel(self):
        """Muestra u oculta el panel lateral (tecla H)."""
        self._panel_visible = not self._panel_visible
        if self._panel_visible:
            self._panel.grid()
        else:
            self._panel.grid_remove()

    # -----------------------------------------------------------------------
    # Acciones: grabar, captura, Práctica 6
    # -----------------------------------------------------------------------

    def _toggle_rec(self):
        """Inicia o detiene la grabación del video de respaldo."""
        self.rec_flag = not self.rec_flag
        self.rec_btn.configure(
            text="■ Detener" if self.rec_flag else "● Grabar"
        )
        self.status.set(
            "Grabando video de respaldo en salidas/..." if self.rec_flag
            else f"Grabación guardada en {SALIDAS_DIR}/"
        )

    def _screenshot(self):
        """Captura el frame actual y lo guarda en salidas/."""
        with self.lock:
            lat = self.latest
        if lat is None:
            return
        nombre = os.path.join(SALIDAS_DIR, f"captura_{int(time.time())}.png")
        cv2.imwrite(nombre, self.freeze_img if self.freeze_img is not None else lat["img"])
        self.status.set(f"Captura guardada: {nombre}")

    def _actualizar_tam_min_p6(self):
        """Actualiza la etiqueta con el tamaño mínimo de rostro detectable."""
        if not hasattr(self, "p6_min_size_lbl") or not hasattr(self, "p6_upsample_var"):
            return
        preset = self.p6_var.get()
        _, scale = P6_PRESETS.get(preset, (24, 1.40))
        upsample = 1 if self.p6_upsample_var.get() else 0
        min_px = sc.p6_min_face_size(upsample_levels=upsample, scale=scale)
        txt = f"Tamaño mín. detectable: ≈ {min_px:.0f} px"
        if upsample > 0:
            txt += f" (ampliación x{scale:.2f})"
        self.p6_min_size_lbl.configure(text=txt)

    def _on_p6_upsample_toggle(self):
        """Gestiona el cambio de la casilla de rostros pequeños (upsample).

        Si cambia mientras hay un análisis en curso, el nuevo valor se aplicará
        al siguiente análisis. Si el detector actual es P6RoiDetector, actualiza
        su atributo upsample_levels para el próximo escaneo o re-escaneo.
        """
        self._actualizar_tam_min_p6()
        upsample = 1 if self.p6_upsample_var.get() else 0
        if hasattr(self, "pipe") and isinstance(self.pipe.detector, sc.P6RoiDetector):
            self.pipe.detector.upsample_levels = upsample

    def _toggle_p6(self):
        """Inicia, cancela o cierra el análisis de la Práctica 6."""
        if self.p6_busy:
            self.p6_cancel.set()
            self.p6_btn.configure(state="disabled", text="Cancelando...")
            return
        if self.freeze_img is not None:
            # Volver al video en vivo
            self.freeze_img = None
            self.shown_id   = -1
            self.p6_btn.configure(text=P6_BTN_TEXT)
            return
        with self.lock:
            lat = self.latest
        if lat is None:
            self.status.set("Aún no hay imagen para analizar")
            return
        step, scale = P6_PRESETS[self.p6_var.get()]
        upsample = 1 if self.p6_upsample_var.get() else 0
        frame = lat["raw"].copy()
        self.p6_frame  = frame
        self.p6_state  = None
        self.p6_cancel = threading.Event()
        self.freeze_img  = sc.draw_banner(frame.copy(), ["Practica 6: preparando la piramide..."])
        self.freeze_dirty = True
        self.p6_busy     = True
        self.p6_btn.configure(text="Cancelar análisis")
        threading.Thread(target=self._p6_job, args=(frame, step, scale, upsample), daemon=True).start()

    def _p6_job(self, frame, step, scale, upsample=0):
        """Hilo del análisis de la Práctica 6."""
        try:
            st = dict(evaluated=[], accepted=[], scanning=[], final=[], done=0, total=0,
                      elapsed=0.0, phase="start", li=0, win_lvl=[], step=step, batch=8,
                      levels=[], level_imgs=[], n_raw=0)
            result = None
            for ev in sc.p6_scan(frame, sc.load_ssd_net(), step, scale,
                                  batch=8, cancel=self.p6_cancel,
                                  upsample_levels=upsample):
                ph = ev["phase"]
                if ph == "cancelled":
                    self.post(self._p6_cancelled)
                    return
                if ph == "start":
                    st.update(levels=ev["levels"], total=ev["total"],
                              level_imgs=ev["level_imgs"], step=ev["step"], batch=ev["batch"])
                elif ph == "scan":
                    st.update(phase="scan", li=ev["li"], scanning=ev["boxes"],
                              win_lvl=ev["win_lvl"], done=ev["done"], elapsed=ev["elapsed"])
                elif ph == "scored":
                    st.update(phase="scored", scanning=[], done=ev["done"],
                              elapsed=ev["elapsed"],
                              evaluated=st["evaluated"] + ev["boxes"],
                              accepted=st["accepted"] + ev["accepted"])
                elif ph == "nms":
                    st.update(phase="nms", final=ev["kept"], n_raw=ev["n_raw"],
                              done=ev["done"], elapsed=ev["elapsed"])
                    time.sleep(1.2)   # pausa para ver el paso NMS
                elif ph == "done":
                    result = ev
                self.p6_state = dict(st)

            boxes = [(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                     for x1, y1, x2, y2 in result["boxes"]]
            secs, nwin = result["elapsed"], result["total"]
            det, name = self.pipe.detector, self.pipe.det_name
            up_txt = f" (upsample {upsample})" if upsample > 0 else ""
            lines = [f"Practica 6 (ventana deslizante): {nwin} ventanas{up_txt}, "
                     f"{secs:.1f} s, {len(boxes)} rostro(s)"]
            if det is not None and name != "p6roi":
                t0 = time.perf_counter()
                live = det.detect(frame, self.pipe.det_score)
                live_ms = (time.perf_counter() - t0) * 1000
                lines.append(f"Detector en vivo ({name}): {live_ms:.0f} ms, "
                              f"{len(live)} rostro(s)  ->  "
                              f"{secs * 1000 / max(live_ms, 0.1):.0f}x mas rapido")
            faces = self.pipe.classify_boxes(frame, boxes)
            out = frame.copy()
            for b in result["raw_boxes"]:
                cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              (0, 140, 255), 1)
            sc.draw_faces(out, faces, self.pipe.threshold)
            sc.draw_banner(out, lines)
            self.post(lambda: self._p6_done(out, lines[0]))
        except Exception as e:                                # noqa: BLE001
            msg = str(e)
            print(f"[practica6] {msg}")
            self.post(lambda: self._p6_failed(msg))

    def _p6_done(self, img, summary):
        """Muestra el resultado final del análisis de la Práctica 6 en pantalla.

        Args:
            img (np.ndarray): Imagen congelada con las detecciones finales dibujadas.
            summary (str): Texto resumen para la barra de estado.
        """
        self.freeze_img   = img
        self.freeze_dirty = True
        self.p6_busy      = False
        self.p6_btn.configure(state="normal", text="Volver al video en vivo")
        self.status.set(summary)

    def _p6_cancelled(self):
        """Restaura el video en vivo tras cancelar el análisis de la Práctica 6."""
        self.freeze_img  = None
        self.p6_busy     = False
        self.shown_id    = -1
        self.p6_btn.configure(state="normal", text=P6_BTN_TEXT)
        self.status.set("Análisis de la Práctica 6 cancelado")

    def _p6_failed(self, msg):
        """Maneja errores ocurridos durante el análisis de la Práctica 6.

        Args:
            msg (str): Mensaje de error capturado.
        """
        self.freeze_img  = None
        self.p6_busy     = False
        self.shown_id    = -1
        self.p6_btn.configure(state="normal", text=P6_BTN_TEXT)
        self.status.set(f"Práctica 6 falló: {msg[:150]}")

    # -----------------------------------------------------------------------
    # Sparklines: limpiar historial
    # -----------------------------------------------------------------------

    def _limpiar_hist(self):
        """Borra el historial de sparklines."""
        for h in (self._hist_fps, self._hist_det, self._hist_cls, self._hist_draw):
            h.clear()
        for s in (self._spark_fps, self._spark_det, self._spark_cls, self._spark_draw):
            s.datos.clear()
            s.redibujar()
        self.status.set("Historial de rendimiento limpiado")

    # -----------------------------------------------------------------------
    # Teclado
    # -----------------------------------------------------------------------

    def _on_key(self, e):
        """Manejador de atajos de teclado."""
        # No interceptar si el foco está en un campo de texto
        if e.widget.winfo_class() in ("TCombobox", "TSpinbox", "Entry", "CTkEntry"):
            return
        k = e.keysym.lower()
        if k in ("q", "escape") and not self._fullscreen:
            self.quit()
        elif k == "escape" and self._fullscreen:
            self._toggle_fullscreen()
        elif k == "f11":
            self._toggle_fullscreen()
        elif k == "h":
            self._toggle_panel()
        elif k == "s":
            self._screenshot()
        elif k == "r":
            self._toggle_rec()
        elif k == "d":
            self.crops_var.set(not self.crops_var.get())
            self._toggle_crops()

    # -----------------------------------------------------------------------
    # Cierre de la aplicación
    # -----------------------------------------------------------------------

    def quit(self):
        """Detiene el hilo de trabajo y cierra la ventana."""
        self.stop_evt.set()
        self.rec_flag = False
        if self.worker is not None:
            self.worker.join(timeout=2)
        self._guardar_config()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Argumentos de línea de comandos
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    """Parsea los argumentos de línea de comandos para la GUI.

    Args:
        argv (list[str] | None): Lista de argumentos opcionales.

    Returns:
        argparse.Namespace: Objeto con los parámetros parseados.
    """
    ap = argparse.ArgumentParser(description="Detector de Sonrisas v2")
    ap.add_argument("--source", default="0",
                    help="índice de cámara o ruta de un video")
    default_models_dir = "models" if os.path.isdir("models") else "."
    ap.add_argument("--models-dir", default=default_models_dir,
                    help="carpeta con los .keras y sus *_metrics.json")
    ap.add_argument("--model", default=None,
                    help="modelo inicial (ruta a un .keras)")
    ap.add_argument("--theme", default=None, choices=["light", "dark", "system"],
                    help="tema inicial: light, dark o system")
    ap.add_argument("--no-anim", action="store_true",
                    help="arranca en modo rendimiento (sin animaciones)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main():
    """Punto de entrada principal para la aplicación gráfica."""
    args = parse_args()
    # CustomTkinter usa su propia ventana raíz
    root = ctk.CTk()
    app = App(root, args)

    # Si se pidió modo sin animaciones desde la línea de comandos
    if args.no_anim:
        app._perf_var.set(True)
        app._on_perf_mode()

    root.mainloop()


if __name__ == "__main__":
    main()

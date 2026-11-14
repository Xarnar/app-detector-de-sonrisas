"""
Interfaz gráfica del detector de sonrisas (tkinter, sin dependencias extra).

Arquitectura: un HILO de trabajo lee la cámara y ejecuta detección + clasificación;
la ventana solo muestra el último frame ya procesado. Así la interfaz no frena el video.

Atajos: q/Esc salir | s captura | r grabar | d ver recortes
"""
import argparse
import glob
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import smile_core as sc

BG, PANEL, FG = "#1b1d23", "#23262e", "#e8eaed"
MUTED, ACCENT, WARN = "#9aa0aa", "#4cc38a", "#f0b429"
NO_MODEL = "(sin clasificador)"
P6_PRESETS = {"Demo · paso 48, escala 1.40 (~10 s)": (48, 1.40),
              "Rápido · paso 24, escala 1.40": (24, 1.40),
              "Estándar · paso 16, escala 1.25": (16, 1.25)}
DETECTORS = {   # clave: (texto del selector, descripción)
    "yunet": ("YuNet · rápido", "Detector ligero moderno: analiza la imagen completa de una sola vez."),
    "ssd": ("SSD ResNet-10 · red de la Práctica 6",
            "La red de la Práctica 6 aplicada al frame completo (1 pasada)."),
    "haar": ("Haar · clásico (pirámide + ventana)",
             "Clásico: pirámide + ventana deslizante con clasificadores muy livianos."),
    "p6roi": ("Práctica 6 + seguimiento · experimental",
              "Ventana deslizante de la Práctica 6 solo cerca de la última cara. Pocos FPS; "
              "al inicio hace un escaneo completo (lento)."),
}
DET_KEY = {v[0]: k for k, v in DETECTORS.items()}
MAX_W = 960
P6_BTN_TEXT = "Analizar fotograma (algoritmo Práctica 6)"


class App:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.pipe = sc.Pipeline()
        self.lock = threading.Lock()
        self.latest = None                 # último frame procesado (lo escribe el hilo de trabajo)
        self.ui_q = queue.Queue()          # el hilo de trabajo pide cosas a la ventana por aquí
        self.stop_evt = threading.Event()
        self.worker = None
        self.rec_flag = False
        self.p6_busy = False
        self.p6_state, self.p6_frame, self.p6_last = None, None, 0.0
        self.p6_cancel = threading.Event()
        self.freeze_img, self.freeze_dirty = None, False
        self.shown_id, self.photo, self.thumb_photo = -1, None, None
        self.ui_ms, self.ui_fps, self.ui_count, self.ui_t0 = 0.0, 0.0, 0, time.time()
        self.last_stats = 0.0
        try:
            import practica_6_deteccion_rostros  # noqa: F401
            self.p6_ok = True
        except ImportError:
            self.p6_ok = False

        self._style()
        self._build()
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.bind_all("<Key>", self._on_key)

        self._load_detector("yunet")
        self._load_model(self._initial_model())
        self.open_source(int(args.source) if str(args.source).isdigit() else args.source)
        self.root.after(15, self._refresh)

    # ------------------------------------------------------------ estilo y UI
    def _style(self):
        s = ttk.Style()
        s.theme_use("clam")
        self.root.configure(bg=BG)
        s.configure(".", background=BG, foreground=FG, fieldbackground=PANEL, font=("Segoe UI", 10))
        s.configure("TFrame", background=BG)
        s.configure("Card.TLabelframe", background=PANEL, bordercolor="#343843", relief="solid")
        s.configure("Card.TLabelframe.Label", background=PANEL, foreground=ACCENT,
                    font=("Segoe UI", 10, "bold"))
        s.configure("TLabel", background=PANEL, foreground=FG)
        s.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        s.configure("Status.TLabel", background="#14161a", foreground=MUTED, padding=(8, 4))
        s.configure("Big.TLabel", background=PANEL, foreground=FG, font=("Consolas", 10))
        s.configure("TButton", background="#343843", foreground=FG, borderwidth=0, padding=6)
        s.map("TButton", background=[("active", "#444a58"), ("disabled", "#2a2d35")],
              foreground=[("disabled", "#666b75")])
        s.configure("Accent.TButton", background=ACCENT, foreground="#0e1a14",
                    font=("Segoe UI", 10, "bold"))
        s.map("Accent.TButton", background=[("active", "#63d69f"), ("disabled", "#2f5f49")])
        s.configure("TRadiobutton", background=PANEL, foreground=FG)
        s.configure("TCheckbutton", background=PANEL, foreground=FG)
        s.map("TRadiobutton", background=[("active", PANEL)])
        s.map("TCheckbutton", background=[("active", PANEL)])
        s.configure("TCombobox", fieldbackground="#2d303a", background="#343843", foreground=FG,
                    arrowcolor=FG)
        s.map("TCombobox", fieldbackground=[("readonly", "#2d303a")], foreground=[("readonly", FG)])
        s.configure("TSpinbox", fieldbackground="#2d303a", foreground=FG, arrowcolor=FG)
        s.configure("Horizontal.TScale", background=PANEL, troughcolor="#343843")
        self.root.option_add("*TCombobox*Listbox.background", "#2d303a")
        self.root.option_add("*TCombobox*Listbox.foreground", FG)

    def _card(self, title):
        f = ttk.LabelFrame(self.side, text=title, style="Card.TLabelframe", padding=6)
        f.pack(fill="x", pady=(0, 5))
        return f

    def _slider(self, parent, label, lo, hi, init, fmt, cb):
        var = tk.DoubleVar(value=init)
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=2)
        top = tk.Frame(row, bg=PANEL)
        top.pack(fill="x")
        tk.Label(top, text=label, bg=PANEL, fg=FG).pack(side="left")
        val = tk.Label(top, text=fmt.format(init), bg=PANEL, fg=ACCENT)
        val.pack(side="right")

        def on(v):
            x = float(v)
            val.config(text=fmt.format(x))
            cb(x)

        ttk.Scale(row, from_=lo, to=hi, variable=var, command=on).pack(fill="x")

        def setter(x):
            var.set(x)
            val.config(text=fmt.format(x))

        return setter

    def _build(self):
        self.root.title("Detector de sonrisas · Visión por Computadora")
        self.root.minsize(1000, 600)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = tk.Frame(self.root, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 6), pady=10)
        self.video = tk.Label(left, bg="black", fg=MUTED, text="Iniciando fuente de video...",
                              width=80, height=24)
        self.video.pack(anchor="n")
        # --- franja de rendimiento (compacta, bajo el video)
        bar = tk.Frame(left, bg=PANEL, highlightbackground="#343843", highlightthickness=1)
        bar.pack(fill="x", pady=(8, 0))
        self.stat = {}
        cells = (("fps", "FPS proc."), ("ui", "FPS interfaz"), ("faces", "Caras"),
                 ("smile", "Sonriendo"), ("det", "Detección"), ("cls", "Clasif."),
                 ("proc", "Total"), ("draw", "Pantalla"))
        for k, name in cells:
            cell = tk.Frame(bar, bg=PANEL)
            cell.pack(side="left", expand=True, fill="x", pady=4)
            tk.Label(cell, text=name, bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack()
            self.stat[k] = tk.StringVar(value="-")
            tk.Label(cell, textvariable=self.stat[k], bg=PANEL, fg=FG,
                     font=("Consolas", 10, "bold")).pack()
        self.thumbs = tk.Label(left, bg=BG)

        self.side = tk.Frame(self.root, bg=BG, width=320)
        self.side.grid(row=0, column=1, sticky="ns", padx=(6, 10), pady=10)

        # --- fuente
        c = self._card("Fuente de video")
        row = tk.Frame(c, bg=PANEL)
        row.pack(fill="x")
        tk.Label(row, text="Cámara", bg=PANEL, fg=FG).pack(side="left")
        self.cam_var = tk.StringVar(value="0")
        ttk.Spinbox(row, from_=0, to=5, width=3, textvariable=self.cam_var).pack(side="left", padx=6)
        ttk.Button(row, text="Usar cámara", command=self._use_camera).pack(side="left")
        ttk.Button(c, text="Abrir video...", command=self._open_video).pack(fill="x", pady=(6, 0))

        # --- detector
        c = self._card("Detector de rostros")
        self.det_var = tk.StringVar(value=DETECTORS["yunet"][0])
        self.det_cb = ttk.Combobox(c, textvariable=self.det_var, state="readonly",
                                   values=[v[0] for v in DETECTORS.values()])
        self.det_cb.pack(fill="x")
        self.det_cb.bind("<<ComboboxSelected>>", lambda e: self._on_detector())
        self.det_desc = tk.Label(c, text=DETECTORS["yunet"][1], bg=PANEL, fg=MUTED, justify="left",
                                 font=("Segoe UI", 8), wraplength=290)
        self.det_desc.pack(anchor="w", pady=(3, 2))
        self.set_score = self._slider(c, "Confianza mínima", 0.3, 0.95, self.pipe.det_score,
                                      "{:.2f}", lambda x: setattr(self.pipe, "det_score", x))

        # --- clasificador
        c = self._card("Clasificador de sonrisa")
        self.model_var = tk.StringVar(value=NO_MODEL)
        self.model_cb = ttk.Combobox(c, textvariable=self.model_var, state="readonly",
                                     postcommand=self._refresh_models)
        self.model_cb.pack(fill="x")
        self.model_cb.bind("<<ComboboxSelected>>", lambda e: self._on_model())
        self._refresh_models()
        self.set_margin = self._slider(c, "Margen del recorte", -0.10, 0.50, 0.10, "{:+.2f}",
                                       lambda x: self.pipe.margins.__setitem__(self.pipe.det_name, x))
        self.set_thr = self._slider(c, "Umbral de sonrisa", 0.05, 0.95, 0.5, "{:.2f}",
                                    lambda x: setattr(self.pipe, "threshold", x))
        self.set_smooth = self._slider(c, "Suavizado (frames)", 1, 15, 5, "{:.0f}",
                                       lambda x: self.pipe.tracker.set_smooth(int(round(x))))
        self.mirror_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(c, text="Modo espejo (solo cámara)", variable=self.mirror_var,
                        command=lambda: setattr(self.pipe, "mirror", self.mirror_var.get())
                        ).pack(anchor="w", pady=(4, 0))
        self.crops_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(c, text="Ver recortes 64x64 (lo que ve el modelo)", variable=self.crops_var,
                        command=self._toggle_crops).pack(anchor="w")

        # --- acciones
        c = self._card("Acciones")
        row = tk.Frame(c, bg=PANEL)
        row.pack(fill="x")
        self.rec_btn = ttk.Button(row, text="● Grabar", command=self._toggle_rec)
        self.rec_btn.pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(row, text="Captura", command=self._screenshot).pack(side="left", expand=True,
                                                                       fill="x", padx=(3, 0))
        tk.Label(c, text="Muestra el proceso paso a paso sobre 1 fotograma:",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w", pady=(6, 0))
        self.p6_var = tk.StringVar(value=list(P6_PRESETS)[0])
        ttk.Combobox(c, textvariable=self.p6_var, values=list(P6_PRESETS), state="readonly"
                     ).pack(fill="x", pady=(2, 3))
        self.p6_btn = ttk.Button(c, text=P6_BTN_TEXT,
                                 style="Accent.TButton", command=self._toggle_p6)
        self.p6_btn.pack(fill="x")
        if not self.p6_ok:
            self.p6_btn.config(state="disabled")
            tk.Label(c, text="Falta practica_6_deteccion_rostros.py en esta carpeta", bg=PANEL,
                     fg=WARN, font=("Segoe UI", 8)).pack(anchor="w")

        self.status = tk.StringVar(value="Listo")
        ttk.Label(self.root, textvariable=self.status, style="Status.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="ew")

    # ------------------------------------------------------ utilidades de hilos
    def post(self, fn):
        """Pide a la ventana (hilo principal) ejecutar fn; seguro desde cualquier hilo."""
        self.ui_q.put(fn)

    def say(self, text):
        self.post(lambda: self.status.set(text))

    # ---------------------------------------------------- detector y clasificador
    def _load_detector(self, name):
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
                        raise RuntimeError("Falta practica_6_deteccion_rostros.py en esta carpeta")
                    det = sc.P6RoiDetector(sc.load_ssd_net())
            except Exception as e:                       # noqa: BLE001
                msg = str(e)            # copiar: Python borra `e` al salir del except
                print(f"[detector {name}] {msg}")
                self.post(lambda: self._detector_failed(name, msg))
                return
            self.post(lambda: self._detector_ready(name, det))

        threading.Thread(target=job, daemon=True).start()

    def _detector_ready(self, name, det):
        old = self.pipe.detector
        if old is not None and hasattr(old, "close"):
            old.close()                                  # detiene el escaneo en segundo plano
        self.pipe.detector, self.pipe.det_name = det, name
        self.det_var.set(DETECTORS[name][0])
        self.det_desc.config(text=DETECTORS[name][1])
        self.set_margin(self.pipe.margins[name])
        self.set_score(self.pipe.det_score)
        self.pipe.tracker.tracks.clear()
        self.status.set(f"Detector activo: {DETECTORS[name][0]}")

    def _detector_failed(self, name, msg):
        cur = self.pipe.det_name if self.pipe.detector else "yunet"
        self.det_var.set(DETECTORS[cur][0] if self.pipe.detector else "")
        self.status.set(f"No se pudo cargar {name}: {msg[:150]}")
        if self.pipe.detector is None and name == "yunet":
            self._load_detector("ssd")                   # plan B para no quedarse sin detector

    def _on_detector(self):
        name = DET_KEY.get(self.det_var.get())
        if name and (name != self.pipe.det_name or self.pipe.detector is None):
            self._load_detector(name)

    def _refresh_models(self):
        files = sorted(glob.glob(os.path.join(self.args.models_dir, "*.keras")))
        self.model_files = {os.path.basename(f): f for f in files}
        self.model_cb["values"] = [NO_MODEL] + list(self.model_files)

    def _initial_model(self):
        if self.args.model:
            return self.args.model
        pref = os.path.join(self.args.models_dir, "model_A_cnn.keras")
        if os.path.exists(pref):
            return pref
        return next(iter(self.model_files.values()), None)

    def _load_model(self, path):
        if not path:
            self.pipe.classifier = None
            self.model_var.set(NO_MODEL)
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
        self.pipe.classifier = clf
        self.pipe.threshold = min(0.95, max(0.05, clf.threshold))
        self.set_thr(self.pipe.threshold)
        self.pipe.tracker.clear_probs()
        self.model_var.set(clf.name)
        src = f"de {clf.metrics_file}" if clf.metrics_file else "por defecto (no hay JSON de métricas)"
        self.status.set(f"Modelo {clf.name} · umbral {self.pipe.threshold:.2f} {src}")

    def _on_model(self):
        name = self.model_var.get()
        self._load_model(None if name == NO_MODEL else self.model_files.get(name))

    # --------------------------------------------------------------- fuente
    def _use_camera(self):
        try:
            self.open_source(int(self.cam_var.get()))
        except ValueError:
            self.status.set("Índice de cámara no válido")

    def _open_video(self):
        path = filedialog.askopenfilename(title="Elegir video",
                                          filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")])
        if path:
            self.open_source(path)

    def open_source(self, src):
        self.stop_evt.set()
        if self.worker is not None:
            self.worker.join(timeout=2)
        self.stop_evt = threading.Event()
        self.worker = threading.Thread(target=self._worker, args=(src, self.stop_evt), daemon=True)
        self.worker.start()

    # ---------------------------------------------------- hilo de procesamiento
    def _worker(self, src, evt):
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
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)          # el video se repite
                    continue
                self.say("Se perdió la señal de la cámara")
                break
            h, w = frame.shape[:2]
            if w > MAX_W:
                frame = cv2.resize(frame, (MAX_W, int(h * MAX_W / w)))
            if self.pipe.mirror and not is_file:
                frame = cv2.flip(frame, 1)
            if self.p6_busy:                       # deja la CPU libre mientras corre la Práctica 6
                time.sleep(0.03)
                continue

            faces, crops, t_det, t_cls = self.pipe.process(frame)
            out = frame.copy()
            sc.draw_faces(out, faces, self.pipe.threshold)
            info = getattr(self.pipe.detector, "status", "")
            if info:
                cv2.putText(out, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(out, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

            now = time.perf_counter()
            inst = 1.0 / max(now - t_prev, 1e-6)
            fps, t_prev = (inst if fid == 0 else 0.9 * fps + 0.1 * inst), now

            if self.rec_flag and writer is None:
                hh, ww = out.shape[:2]
                writer = cv2.VideoWriter(f"respaldo_{int(time.time())}.mp4",
                                         cv2.VideoWriter_fourcc(*"mp4v"), 20, (ww, hh))
            elif not self.rec_flag and writer is not None:
                writer.release()
                writer = None
            if writer is not None:
                writer.write(out)

            fid += 1
            thr = self.pipe.threshold
            with self.lock:
                self.latest = dict(id=fid, raw=frame, img=out, crops=crops, fps=fps,
                                   n=len(faces), smiling=sum(1 for f in faces
                                                             if f.prob is not None and f.prob >= thr),
                                   t_det=t_det, t_cls=t_cls, t_proc=(time.perf_counter() - t_loop) * 1000)
            if is_file:                                          # a la velocidad original del video
                time.sleep(max(0.0, 1.0 / src_fps - (time.perf_counter() - t_loop)))
        cap.release()
        if writer is not None:
            writer.release()

    # ----------------------------------------------------- refresco de ventana
    def _refresh(self):
        while True:
            try:
                self.ui_q.get_nowait()()
            except queue.Empty:
                break
            except Exception as e:                               # noqa: BLE001
                print("[ui]", e)

        with self.lock:
            lat = self.latest
        if self.p6_busy and self.p6_state is not None:
            if time.time() - self.p6_last > 0.07:          # ~14 cuadros/s del proceso
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

        now = time.time()
        if lat is not None and now - self.last_stats > 0.25:
            self.last_stats = now
            dt = now - self.ui_t0
            if dt >= 1.0:
                self.ui_fps, self.ui_count, self.ui_t0 = self.ui_count / dt, 0, now
            s = self.stat
            s["fps"].set(f"{lat['fps']:.1f}")
            s["ui"].set(f"{self.ui_fps:.1f}")
            s["faces"].set(str(lat["n"]))
            s["smile"].set(f"{lat['smiling']} / {lat['n']}" if self.pipe.classifier else "-")
            s["det"].set(f"{lat['t_det']:.1f} ms")
            s["cls"].set(f"{lat['t_cls']:.1f} ms" if self.pipe.classifier else "-")
            s["proc"].set(f"{lat['t_proc']:.1f} ms")
            s["draw"].set(f"{self.ui_ms:.1f} ms")
        self.root.after(15, self._refresh)

    def _show(self, bgr):
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video.config(image=self.photo, text="", width=rgb.shape[1], height=rgb.shape[0])
        self.ui_ms = 0.8 * self.ui_ms + 0.2 * (time.perf_counter() - t0) * 1000

    def _show_crops(self, crops):
        if not crops:
            return
        strip = np.hstack([c[..., 0].astype(np.uint8) for c in crops[:8]])   # 64x64 reales
        self.thumb_photo = ImageTk.PhotoImage(Image.fromarray(strip))
        self.thumbs.config(image=self.thumb_photo)

    def _toggle_crops(self):
        if self.crops_var.get():
            self.thumbs.pack(anchor="w", pady=(6, 0))
        else:
            self.thumbs.pack_forget()

    # ------------------------------------------------------------------ acciones
    def _toggle_rec(self):
        self.rec_flag = not self.rec_flag
        self.rec_btn.config(text="■ Detener grabación" if self.rec_flag else "● Grabar")
        self.status.set("Grabando video de respaldo..." if self.rec_flag
                        else "Grabación guardada en la carpeta de la app (respaldo_*.mp4)")

    def _screenshot(self):
        with self.lock:
            lat = self.latest
        if lat is None:
            return
        name = f"captura_{int(time.time())}.png"
        cv2.imwrite(name, self.freeze_img if self.freeze_img is not None else lat["img"])
        self.status.set(f"Captura guardada: {name}")

    def _toggle_p6(self):
        if self.p6_busy:                                       # cancelar el análisis en curso
            self.p6_cancel.set()
            self.p6_btn.config(state="disabled", text="Cancelando...")
            return
        if self.freeze_img is not None:                       # volver al video en vivo
            self.freeze_img = None
            self.shown_id = -1
            self.p6_btn.config(text=P6_BTN_TEXT)
            return
        with self.lock:
            lat = self.latest
        if lat is None:
            self.status.set("Aún no hay imagen para analizar")
            return
        step, scale = P6_PRESETS[self.p6_var.get()]
        frame = lat["raw"].copy()
        self.p6_frame, self.p6_state, self.p6_cancel = frame, None, threading.Event()
        self.freeze_img = sc.draw_banner(frame.copy(), ["Practica 6: preparando la piramide..."])
        self.freeze_dirty, self.p6_busy = True, True
        self.p6_btn.config(text="Cancelar análisis")
        threading.Thread(target=self._p6_job, args=(frame, step, scale), daemon=True).start()

    def _p6_job(self, frame, step, scale):
        try:
            st = dict(evaluated=[], accepted=[], scanning=[], final=[], done=0, total=0, elapsed=0.0,
                      phase="start", li=0, win_lvl=[], step=step, batch=8, levels=[], level_imgs=[],
                      n_raw=0)
            result = None
            for ev in sc.p6_scan(frame, sc.load_ssd_net(), step, scale, batch=8, cancel=self.p6_cancel):
                ph = ev["phase"]
                if ph == "cancelled":
                    self.post(self._p6_cancelled)
                    return
                if ph == "start":
                    st.update(levels=ev["levels"], total=ev["total"], level_imgs=ev["level_imgs"],
                              step=ev["step"], batch=ev["batch"])
                elif ph == "scan":
                    st.update(phase="scan", li=ev["li"], scanning=ev["boxes"], win_lvl=ev["win_lvl"],
                              done=ev["done"], elapsed=ev["elapsed"])
                elif ph == "scored":
                    st.update(phase="scored", scanning=[], done=ev["done"], elapsed=ev["elapsed"],
                              evaluated=st["evaluated"] + ev["boxes"], accepted=st["accepted"] + ev["accepted"])
                elif ph == "nms":
                    st.update(phase="nms", final=ev["kept"], n_raw=ev["n_raw"], done=ev["done"],
                              elapsed=ev["elapsed"])
                    time.sleep(1.2)                       # pausa para que se vea el paso del NMS
                elif ph == "done":
                    result = ev
                self.p6_state = dict(st)

            boxes = [(int(x1), int(y1), int(x2 - x1), int(y2 - y1)) for x1, y1, x2, y2 in result["boxes"]]
            secs, nwin = result["elapsed"], result["total"]
            det, name = self.pipe.detector, self.pipe.det_name
            lines = [f"Practica 6 (ventana deslizante): {nwin} ventanas, {secs:.1f} s, {len(boxes)} rostro(s)"]
            if det is not None and name != "p6roi":
                t0 = time.perf_counter()
                live = det.detect(frame, self.pipe.det_score)
                live_ms = (time.perf_counter() - t0) * 1000
                lines.append(f"Detector en vivo ({name}): {live_ms:.0f} ms, {len(live)} rostro(s)  ->  "
                             f"{secs * 1000 / max(live_ms, 0.1):.0f}x mas rapido")
            faces = self.pipe.classify_boxes(frame, boxes)
            out = frame.copy()
            for b in result["raw_boxes"]:                  # ventanas aceptadas antes del NMS (naranja)
                cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 140, 255), 1)
            sc.draw_faces(out, faces, self.pipe.threshold)
            sc.draw_banner(out, lines)
            self.post(lambda: self._p6_done(out, lines[0]))
        except Exception as e:                                # noqa: BLE001
            msg = str(e)
            print(f"[practica6] {msg}")
            self.post(lambda: self._p6_failed(msg))

    def _p6_done(self, img, summary):
        self.freeze_img, self.freeze_dirty, self.p6_busy = img, True, False
        self.p6_btn.config(state="normal", text="Volver al video en vivo")
        self.status.set(summary)

    def _p6_cancelled(self):
        self.freeze_img, self.p6_busy = None, False
        self.shown_id = -1
        self.p6_btn.config(state="normal", text=P6_BTN_TEXT)
        self.status.set("Análisis de la Práctica 6 cancelado")

    def _p6_failed(self, msg):
        self.freeze_img, self.p6_busy = None, False
        self.shown_id = -1
        self.p6_btn.config(state="normal", text=P6_BTN_TEXT)
        self.status.set(f"Práctica 6 falló: {msg[:150]}")

    # --------------------------------------------------------------- teclado/cierre
    def _on_key(self, e):
        if e.widget.winfo_class() in ("TCombobox", "TSpinbox", "Entry"):
            return
        k = e.keysym.lower()
        if k in ("q", "escape"):
            self.quit()
        elif k == "s":
            self._screenshot()
        elif k == "r":
            self._toggle_rec()
        elif k == "d":
            self.crops_var.set(not self.crops_var.get())
            self._toggle_crops()

    def quit(self):
        self.stop_evt.set()
        self.rec_flag = False
        if self.worker is not None:
            self.worker.join(timeout=2)
        self.root.destroy()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="índice de cámara o ruta de un video")
    ap.add_argument("--models-dir", default=".", help="carpeta con los .keras y sus *_metrics.json")
    ap.add_argument("--model", default=None, help="modelo inicial (ruta a un .keras)")
    return ap.parse_args(argv)


def main():
    root = tk.Tk()
    App(root, parse_args())
    root.mainloop()


if __name__ == "__main__":
    main()
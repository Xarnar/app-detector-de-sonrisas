# Detector de Sonrisas en Tiempo Real

Proyecto universitario de Visión por Computadora para la detección facial y clasificación de sonrisas en tiempo real, integrando múltiples detectores de rostros y modelos convolucionales en TensorFlow/Keras.

---

## Características principales

- **Interfaz Gráfica interactiva (`app_gui.py`)**: Construida con Tkinter (multi-hilo para no ralentizar el video). Permite alternar detectores y modelos en caliente, ajustar márgenes, umbrales y suavizado en vivo, ver los recortes 64×64 preprocesados y medir métricas de rendimiento (FPS de procesamiento e interfaz, tiempos de detección, clasificación y dibujo).
- **Inferencia en tiempo real por OpenCV (`app_sonrisa.py`)**: Versión ligera y directa para ejecutar desde consola.
- **Múltiples detectores faciales**:
  - **YuNet** (`face_detection_yunet_2023mar.onnx`): Detector ligero moderno en una sola pasada (tiempo real a altos FPS).
  - **SSD ResNet-10** (`modelo/`): Red neuronal profunda de detección facial en una pasada.
  - **Práctica 6** (`practica_6_deteccion_rostros.py`): Análisis detallado usando pirámide de escalas + ventana deslizante + NMS, permitiendo comparar visualmente su tiempo frente a detectores en una pasada.
- **Múltiples clasificadores de sonrisa**:
  - **Modelo A (CNN desde cero)**: `model_A_cnn.keras` (ligero y rápido, ~25 ms/cara).
  - **Modelo B (MobileNetV2 Transfer Learning)**: `model_B_mobilenetv2.keras`.
- **Seguimiento y suavizado temporal**: Algoritmo de tracking por IoU para asociar rostros entre frames y suavizar las probabilidades por individuo.

---

## Pipeline

```text
Cámara / Video
  └─> Detector Facial (YuNet / SSD ResNet-10)
        └─> Recorte cuadrado + preprocesado (escala de grises 64×64, float32 0–255)
              └─> Clasificador CNN (Modelo A / Modelo B)
                    └─> Seguimiento por IoU + Suavizado temporal
                          └─> Visualización y métricas en pantalla
```

---

## Estructura del proyecto

```text
├── app_gui.py                      # Interfaz gráfica principal (Tkinter)
├── smile_core.py                   # Núcleo modular: detectores, modelos, tracking y pipeline
├── app_sonrisa.py                  # Versión de consola con OpenCV
├── practica_6_deteccion_rostros.py # Módulo de ventana deslizante + pirámide
├── model_A_cnn.keras               # Pesos del Modelo A (CNN desde cero)
├── A_metrics.json                  # Métricas y umbral óptimo del Modelo A
├── model_B_mobilenetv2.keras       # Pesos del Modelo B (MobileNetV2)
├── B_metrics.json                  # Métricas y umbral óptimo del Modelo B
├── face_detection_yunet_2023mar.onnx # Pesos del detector YuNet
├── modelo/                         # Pesos y arquitectura del detector Caffe SSD
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Documentación
```

---

## Requisitos e instalación

Se recomienda **Python 3.10 – 3.12**.

```powershell
# 1. Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt
```

> **Nota sobre OpenCV**: `requirements.txt` especifica `opencv-python>=4.9,<5` para asegurar compatibilidad con el importador de modelos Caffe.

---

## Uso

### 1. Interfaz Gráfica (Recomendada)

```powershell
# Usando cámara por defecto
python app_gui.py

# O usando un archivo de video
python app_gui.py --source video.mp4
```

**Controles en la interfaz:**
- **Selector de detector**: Cambia entre YuNet, SSD ResNet-10, Haar o Práctica 6 ROI.
- **Selector de modelo**: Detecta automáticamente los archivos `.keras` en la carpeta y carga el umbral de su respectivo `*_metrics.json`.
- **Deslizadores en vivo**:
  - *Margen del recorte*: Ajusta cuánto contexto extra alrededor del rostro recibe el modelo (se recuerda de forma independiente por detector).
  - *Umbral de sonrisa*: Probabilidad de corte para catalogar `Sonríe` vs `No sonríe`.
  - *Suavizado*: Ventana de fotogramas para promediar la predicción en cada rostro.
- **Botón "Analizar fotograma con la Práctica 6"**: Congela el fotograma y corre la pirámide multiescala con ventana deslizante, mostrando la comparativa de velocidad frente al detector en vivo.
- **Atajos de teclado:** `q` / `Esc` (salir), `s` (captura de pantalla), `r` (iniciar/detener grabación de video), `d` (mostrar/ocultar tira de recortes 64×64).

---

### 2. Versión de consola en tiempo real (OpenCV)

```powershell
python app_sonrisa.py
```

**Parámetros útiles:**
- `--source 0`: Índice de cámara o ruta a video.
- `--model model_A_cnn.keras`: Modelo a utilizar.
- `--margin -0.10`: Margen de recorte calibrado para YuNet.
- `--smooth 5`: Número de frames para el suavizado de probabilidades.
- `--det-every 2`: Frecuencia de inferencia de YuNet (los frames intermedios usan el tracker).

---

## Comparativa de Modelos

| Métrica | Modelo A (CNN desde cero) | Modelo B (MobileNetV2 TL) |
| :--- | :---: | :---: |
| **Arquitectura** | CNN personalizada | MobileNetV2 preentrenado |
| **Test Accuracy** | **93.22%** | 91.19% |
| **Test F1 (Sonríe)** | **0.8820** | 0.8412 |
| **Test AUC** | **0.9731** | 0.9622 |
| **Umbral recomendado** | 0.34 | 0.51 |
| **Latencia estimada** | **~25.25 ms / cara** | ~157.88 ms / cara |

# Detector de Sonrisas en Tiempo Real

Proyecto universitario de Visión por Computadora para la detección facial y clasificación de sonrisas en tiempo real, integrando múltiples detectores de rostros y modelos convolucionales en TensorFlow/Keras.

---

## Características principales

- **Interfaz Gráfica interactiva (`app_gui.py`)**: Construida con **CustomTkinter** v2 (multi-hilo para no ralentizar el video). Permite alternar detectores y modelos en caliente, cambiar temas (claro/oscuro/sistema con persistencia), activar modo rendimiento, ajustar márgenes, umbrales y suavizado en vivo, ver recortes 64×64 y monitorear métricas con sparklines de rendimiento en tiempo real.
- **Inferencia en tiempo real por OpenCV (`app_sonrisa.py`)**: Versión ligera y directa para ejecutar desde consola.
- **Modo solo-detección**: Si no se dispone de un modelo `.keras`, tanto la GUI como la consola operan en modo solo-detección (rostros encuadrados en blanco sin clasificar).
- **Múltiples detectores faciales**:
  - **YuNet** (`models/face_detection_yunet_2023mar.onnx`): Detector ligero moderno en una sola pasada (tiempo real a altos FPS).
  - **SSD ResNet-10** (`models/`): Red neuronal profunda de detección facial en una pasada.
  - **Haar Cascades** (Viola-Jones): Detector clásico integrado en OpenCV que emplea pirámide de imágenes y ventana deslizante con clasificadores en cascada e imágenes integrales.
  - **Práctica 6** (`practica_6_deteccion_rostros.py`): Análisis detallado usando pirámide multiescala + ventana deslizante + NMS, con soporte de sobremuestreo (`upsample_levels`) para detectar rostros pequeños que miden menos que la ventana base (Desafío 2b).
  - **Práctica 6 ROI** (experimental): Ventana deslizante acotada a la región de interés alrededor del rostro rastreado por el tracker.
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
├── app_gui.py                      # Interfaz gráfica v2 (CustomTkinter, temas, animaciones)
├── smile_core.py                   # Núcleo modular: detectores, modelos, tracking y pipeline
├── app_sonrisa.py                  # Versión de consola con OpenCV
├── practica_6_deteccion_rostros.py # Módulo de ventana deslizante + pirámide
├── models/                         # Modelos entrenados, pesos y métricas (no mover)
│   ├── model_A_cnn.keras           # Pesos del Modelo A (CNN desde cero)
│   ├── A_metrics.json              # Métricas y umbral óptimo del Modelo A
│   ├── model_B_mobilenetv2.keras   # Pesos del Modelo B (MobileNetV2 Transfer Learning)
│   ├── B_metrics.json              # Métricas y umbral óptimo del Modelo B
│   ├── face_detection_yunet_2023mar.onnx # Detector facial YuNet
│   ├── deploy.prototxt             # Arquitectura Caffe SSD ResNet-10
│   └── res10_300x300_ssd_iter_140000.caffemodel # Pesos Caffe SSD ResNet-10
├── notebooks/                      # Notebooks Jupyter de entrenamiento y experimentos
│   ├── 01_modelo_A_cnn_desde_cero.ipynb
│   ├── 02_modelo_B_mobilenetv2_transfer_learning.ipynb
│   ├── 03_practica_6_deteccion_rostros_ventana_deslizante.ipynb
│   └── 04_gradcam_y_errores_modelo_A.ipynb
├── resultados/                     # Resultados de experimentos (análisis de errores Grad-CAM y benchmark P6)
├── requirements.txt                # Dependencias del proyecto (incluye customtkinter)
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

### 1. Interfaz Gráfica v2 (Recomendada)

```powershell
# Usando cámara por defecto (tema guardado en config.json)
python app_gui.py

# Usando un archivo de video con tema claro
python app_gui.py --source ruta/al/video.mp4 --theme light

# Modo máximo rendimiento (sin animaciones)
python app_gui.py --no-anim

# Tema oscuro explícito
python app_gui.py --theme dark
```

**Controles en la interfaz:**
- **Barra superior**: Selector de tema (claro/oscuro/sistema) e interruptor "Modo rendimiento".
- **Panel en pestañas**: Fuente · Detector · Clasif. · P6/Acc. · Render.
- **Selector de detector**: Cambia entre YuNet, SSD ResNet-10, Haar o Práctica 6 ROI.
- **Selector de modelo**: Detecta automáticamente los archivos `.keras` en la carpeta y carga el umbral de su respectivo `*_metrics.json`.
- **Deslizadores en vivo**:
  - *Margen del recorte*: Ajusta cuánto contexto extra alrededor del rostro recibe el modelo (se recuerda de forma independiente por detector).
  - *Umbral de sonrisa*: Probabilidad de corte para catalogar `Sonríe` vs `No sonríe`.
  - *Suavizado*: Ventana de fotogramas para promediar la predicción en cada rostro.
- **Botón "Analizar fotograma con la Práctica 6"**: Congela el fotograma y corre la pirámide multiescala con ventana deslizante (con presets configurables y soporte de sobremuestreo para rostros pequeños), mostrando la comparativa de velocidad frente al detector en vivo.
- **Atajos de teclado:**

| Tecla | Acción |
|-------|--------|
| `q` / `Esc` | Salir |
| `s` | Captura → `salidas/` |
| `r` | Grabar / detener → `salidas/` |
| `d` | Mostrar / ocultar recortes 64×64 |
| `F11` | Pantalla completa |
| `H` | Mostrar / ocultar panel lateral |

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

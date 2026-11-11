# Detector de Sonrisas en Tiempo Real

Proyecto universitario de Visión por Computadora para la detección y clasificación de sonrisas en tiempo real usando **YuNet** (detección facial) y una **CNN en Keras/TensorFlow** (clasificación `sonríe` / `no sonríe`).

## Pipeline

```text
Cámara / Video
  └─> YuNet (face_detection_yunet_2023mar.onnx)
        └─> Recorte cuadrado + escala de grises 64×64 (float32 0–255)
              └─> CNN Keras (model_A_cnn.keras)
                    └─> Seguimiento por IoU + suavizado temporal -> Overlay en pantalla
```

- Si `face_detection_yunet_2023mar.onnx` no está presente, [`app_sonrisa.py`](app_sonrisa.py) lo descarga automáticamente desde OpenCV Zoo.
- Si `model_A_cnn.keras` no está presente, la aplicación funciona en modo solo-detección de rostros.

## Estructura del proyecto

- `app_sonrisa.py`: Aplicación principal de inferencia en tiempo real con seguimiento por IoU y suavizado de predicciones.
- `capture_dataset.py`: Herramienta para capturar recortes propios (`64×64` en escala de grises) con el mismo preprocesado que `app_sonrisa.py`, pensada para fine-tuning.
- `model_A_cnn.keras`: Modelo CNN entrenado (entrada `(N, 64, 64, 1)` en `float32` rango `0–255`; el reescalado `1/255` está integrado en el modelo).
- `A_metrics.json`: Métricas de evaluación del modelo y umbral recomendado (`0.34`).
- `face_detection_yunet_2023mar.onnx`: Modelo YuNet para detección de rostros.
- `results/`: Gráficas de entrenamiento/evaluación (`A_curvas.png`, `A_confusion.png`) y respaldo de métricas/modelo.
- `requirements.txt`: Dependencias del proyecto.

## Requisitos e instalación

Se recomienda **Python 3.10 – 3.12** (por compatibilidad con TensorFlow).

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uso

### 1. Detector en tiempo real

```powershell
python app_sonrisa.py
```

Argumentos útiles:
- `--source 0`: Índice de cámara (`0`, `1`, ...) o ruta a un video.
- `--margin -0.10`: Fracción del recorte alrededor del rostro detectado por YuNet.
- `--threshold`: Umbral de probabilidad para clasificar sonrisa (por defecto lee `umbral_recomendado` de `A_metrics.json`).
- `--smooth 5`: Número de frames para promediar la probabilidad por persona (`1` desactiva el suavizado).
- `--det-every 2`: Ejecuta YuNet cada `N` frames y reutiliza las cajas del tracker en los frames intermedios.
- `--no-mirror`: Desactiva el efecto espejo en cámara web.

Controles de teclado:
- `q`: Salir
- `+` / `-`: Ajustar el margen del recorte
- `[` / `]`: Ajustar el umbral de clasificación
- `d`: Mostrar/ocultar la vista debug de los recortes `64×64` que recibe el modelo
- `r`: Iniciar/detener grabación de video (`respaldo_<timestamp>.mp4`)
- `s`: Guardar captura de pantalla (`captura_<timestamp>.png`)

### 2. Captura de dataset propio

```powershell
python capture_dataset.py
```

Controles de teclado:
- `s`: Guardar recorte en `mis_fotos/sonrie/`
- `n`: Guardar recorte en `mis_fotos/no_sonrie/`
- `+` / `-`: Ajustar margen del recorte
- `q`: Salir

## Métricas del modelo (`A_metrics.json`)

- **Modelo:** `A_cnn_desde_cero` (22 épocas)
- **Umbral recomendado:** `0.34`
- **Test Accuracy:** `93.22%`
- **Test F1 (sonríe):** `0.8820`
- **Test AUC:** `0.9731`
- **Latencia:** `25.25 ms` por cara

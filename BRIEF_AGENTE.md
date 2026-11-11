# Brief para el agente: app de detección de sonrisas en tiempo real

## Contexto
Proyecto universitario de Visión por Computadora. `app_sonrisa.py` ya existe y su lógica base está probada.
Pipeline: cámara -> YuNet (rostros) -> recorte -> CNN Keras (sonríe / no sonríe) -> overlay.
Los estudiantes deben poder explicar cada parte, así que prioriza código simple y comentado.

## Restricciones (NO cambiar sin avisar)
- Contrato del modelo: entrada gris, 64x64, float32 con valores 0-255, forma (N,64,64,1).
  El reescalado a 0-1 ya está dentro del modelo.
- El modelo se carga de `model_A_cnn.keras`; el umbral viene de `A_metrics.json` (`umbral_recomendado`).
- Sin el modelo, la app debe seguir funcionando en modo solo-detección.

## Tareas (en orden)
1. Crear un entorno virtual, instalar `requirements.txt` y ejecutar `python app_sonrisa.py`.
   Confirmar que YuNet se descarga, que abre la cámara y dibuja recuadros.
2. Medir FPS con 1 y con 3+ caras. Si bajan de ~15 FPS, proponer optimizaciones
   (detectar cada N frames y seguir con el tracker en los intermedios, o convertir el modelo a ONNX).
3. Con la tecla `d` (debug) revisar los recortes 64x64 y proponer un valor de `--margin`
   que se parezca a los recortes ajustados del dataset SMILES.
4. Escribir `capture_dataset.py`: usa el MISMO detector y el MISMO `crop_face`/`preprocess`
   para guardar recortes propios en `mis_fotos/sonrie/` y `mis_fotos/no_sonrie/`
   (teclas: `s` guarda como sonríe, `n` como no sonríe). Servirán para el fine-tuning.
5. Reportar qué cambió y qué mediste. No tocar el preprocesado sin explicarlo.

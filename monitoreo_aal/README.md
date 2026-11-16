# Monitoreo no invasivo AAL con VLC + IRS

Abrir `monitoreo_aal.ipynb` desde esta carpeta (los módulos se importan desde aquí).

| Archivo | Contenido |
|---|---|
| `ris_vlc.py` | Modelo de canal original, sin cambios |
| `sim_aal.py` | Rutina del ocupante, sensado (x, y) ruidoso, terminal VLC, eventos anómalos |
| `modelo_aal.py` | CNN–GRU (predicción a 1 s), residuo, baseline de velocidad constante y regla de inmovilidad |
| `evaluacion_aal.py` | Calibración del umbral, detección, falsas alarmas/h, evaluación del enlace |
| `cnn_gru_aal.pt`, `entrenamiento.json` | Pesos entrenados; si se borran, el notebook reentrena (~8 min en CPU) |
| `resultados_aal.json` | Números del informe |
| `figuras/` | Figuras en PDF para el paper |

Requisitos: `numpy`, `matplotlib`, `torch`, `jupyter`. Ejecución completa con pesos: ~7 min en CPU.

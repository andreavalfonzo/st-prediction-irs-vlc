# Simulador de canal VLC asistido por IRS de espejos

Implementación numérica del modelo de canal para el Hito 2, con las pruebas de
verificación y los barridos de sensibilidad que generan las figuras del informe.

## Uso

```bash
pip install numpy matplotlib

python3 verificacion.py                        # 6 pruebas; deben pasar todas
python3 sensibilidad.py                        # 7 barridos -> figuras/*.pdf
python3 tabla_parametros.py > tabla_parametros.tex
```

Tiempo total de ejecución: unos 15 segundos. Semilla fija (`SEMILLA = 20260926`),
de modo que dos corridas producen resultados idénticos.

## Archivos

| Archivo | Contenido |
|---|---|
| `ris_vlc.py` | Modelo: geometría, canal LoS, reflexión en el IRS, bloqueo, métricas |
| `verificacion.py` | Pruebas contra resultados conocidos sin simular |
| `sensibilidad.py` | Barridos de parámetros y generación de figuras |
| `tabla_parametros.py` | Emite la tabla de parámetros en LaTeX desde el código |
| `resultados.tex` | Sección de resultados redactada, con los números de esta corrida |
| `figuras/` | Figuras en PDF vectorial, dimensionadas para columna IEEE |

## Convenciones

Origen en una esquina del piso; `x`, `y` sobre el piso y `z` hacia arriba. El
LED va en el techo apuntando hacia abajo, el IRS sobre la pared `x = 0` con
normal nominal hacia el interior, y el receptor a la altura del dispositivo.

Los ángulos de control de cada espejo, `alpha` (elevación) y `beta` (azimut),
se miden respecto de la normal nominal de la pared y valen cero cuando el
espejo mira perpendicularmente hacia el interior.

## Dos decisiones de modelado que conviene conocer

**Criterio de acoplamiento.** Un espejo plano forma una imagen virtual puntual
del emisor. Hay acoplamiento si la recta que une esa imagen con el detector
atraviesa la apertura del espejo. La tolerancia angular queda fijada por el
tamaño del espejo (1,92° en el escenario de referencia) y no por el del
fotodiodo (0,14°), que produciría degradaciones discontinuas.

**Cota de conservación.** La contribución de cada espejo se limita a la
potencia que recolecta, repartida sobre la huella del haz reflejado, de radio
`a (1 + d2/d1) + r_s d2/d1`. Sin esta cota la ganancia crecería sin límite al
subdividir el panel.

## Qué falta

- Validación contra trazado de rayos, para acotar el error del modelo analítico.
- Reflexiones difusas de pared, hoy omitidas.
- Sombreado parcial del arreglo, que exige una prueba de visibilidad por espejo.
- Modelo de movilidad y orientación estadístico en lugar de trayectorias fijas.

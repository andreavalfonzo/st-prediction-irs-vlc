# Control predictivo del IRS: alcance y reproduccion

Esta implementacion agrega la solucion propuesta en la Seccion III-A del informe:
CNN sobre mapas espaciales, GRU sobre el historial, y salida de 2N angulos para
el instante siguiente. El simulador y sus figuras anteriores siguen disponibles.
El entrenamiento implementado es **imitacion supervisada**, una de las dos vias
que propone el informe. No implementa la otra via exploratoria de optimizacion
diferenciable de J. No hay mediciones reales ni validacion por trazado de rayos.

## Ejecutar

Desde la raiz del repositorio, con Python 3.10 o superior:

```bash
python -m pip install -r requirements.txt
python verificacion.py
python -m unittest discover -s tests -v
python entrenar.py --quick --output resultados/integracion_nueva
python entrenar.py --output resultados/experimento_nuevo --noise-sweep
```

`--quick` comprueba la integracion con solo dos epocas: no es un resultado
cientifico. La corrida completa entrena los tres modelos en CPU. La cantidad
de hilos se controla con `threads`; no se requiere GPU. Cada 10 epocas muestra
el error de entrenamiento y validacion. No sobrescribe checkpoints existentes.

Para repetir la evaluacion de un entrenamiento terminado:

```bash
python entrenar.py --evaluate-only --output resultados/experimento_nuevo --noise-sweep
```

Si se uso `--quick`, se debe volver a incluir `--quick` para cargar su configuracion.
Para parametros distintos, copiar `config_prediccion.json`, editarlo y pasarlo
con `--config archivo.json` a entrenamiento y evaluacion. La configuracion debe
coincidir exactamente con los checkpoints. No se descargan pesos ni datasets.

## Parametros provisionales autorizados

Los valores fisicos de `ris_vlc.Escenario` reproducen la Tabla II del informe;
`scenario` permite cambiar sus argumentos sin editar el codigo. Los siguientes
valores son decisiones de simulacion, **no especificaciones de un actuador real**:

| Parametro | Valor por defecto | Interpretacion |
|---|---:|---|
| dt | 50 ms | Un paso entre sensado y aplicacion, incluye retardo de actuacion |
| history | 8 muestras | Historial causal de 350 ms entre primera y ultima observacion |
| steps | 120 | 6 s por trayectoria; ultima muestra a 5.95 s |
| Episodios train/val/test | 64/12/16 | Semillas consecutivas, sin compartir trayectorias |
| grid | 20 x 20 | Proyeccion espacial sobre el piso |
| Ruido de posicion | 2 cm | Desviacion estandar por coordenada horizontal, gaussiano independiente |
| Ruido de orientacion | 1 grado | Ruido de componentes de la normal, seguido de normalizacion |
| Velocidad de usuario | 0.1 a 1.5 m/s | Caminata correlacionada con giros y reflexion en bordes |
| Rango angular | -80 a 80 grados | Para cada uno de los dos angulos de cada espejo |
| Velocidad angular | 120 grados/s | Maximo 6 grados por paso, por componente |
| Rmin | 5 bits/s/Hz | Umbral de indisponibilidad, configurable |
| lambda_s | 0.001 | Penalizacion por suma de desplazamientos en radianes |
| lambda_q | 1 | Penalizacion del deficit de tasa |
| Bloqueador | radio 0.20 m, altura 1.8 m | Cilindro vertical, posicion actual perfectamente observable |

Las orientaciones cambian suavemente con dos componentes sinusoidales. El ruido
de orientacion es una aproximacion en componentes, no una distribucion exacta
de rotaciones de desviacion angular dada. La altura del receptor se conoce y
permanece fija en 0.85 m. En la mitad de los episodios una persona sigue al
usuario a 0.32 m, con desplazamiento rotatorio, para incluir obstrucciones;
en la otra mitad camina independientemente. No se modela el propio cuerpo del
usuario. El estado verdadero y la observacion ruidosa se mantienen separados.

## Datos, arquitectura y aprendizaje

Cada mapa tiene ocho canales: LED, panel con altura normalizada, usuario como
gaussiana subpixel, tres componentes de la orientacion estimada ponderadas por
la presencia del usuario, bloqueador y altura de LED ponderada por su presencia.
La proyeccion cenital aproxima la distribucion vertical del panel con la altura
maxima de los espejos en cada celda. Esta representacion se evalua en una sola
geometria fija; no demuestra generalizacion a otras salas o alturas.

La CNN tiene dos convoluciones (12 y 16 filtros), pooling y embedding de 64
componentes. La GRU tiene estado de dimension 64. El cabezal devuelve 2N
correcciones angulares acotadas por tanh a +/-0.15 rad, que se suman a la
solucion geometrica de la **posicion observada actual**. Esta conexion residual
es una decision explicita de implementacion: integra geometria conocida y
facilita aprender el pequeno desplazamiento de un paso. La red produce los
angulos futuros de cada espejo, no una posicion futura intermedia. No es una
red que aprenda la geometria completa desde cero. Las dos ablaciones comparten
el mismo residual para comparar la CNN y la memoria en condiciones semejantes.

Las etiquetas son los angulos geometricos ideales de la posicion verdadera en
t+1. La informacion futura se usa exclusivamente como etiqueta de entrenamiento
y para evaluar; nunca se pasa a un controlador. La perdida es MSE angular en
radianes, reescalada por 0.01 para acondicionamiento numerico. El mejor checkpoint
se selecciona por validacion. Se proyecta la salida a A(theta_t) al aplicar cada
decision, no durante la imitacion: las etiquetas son geometricas y no incorporan
la historia de saturacion de cada controlador. Esa diferencia limita la imitacion
cuando los limites mecanicos son activos. No se afirma que las etiquetas maximicen J.

## Comparacion causal

Todos los metodos reciben observaciones hasta t, deciden para t+1 y usan el mismo
canal, configuracion mecanica, ruido y trayectorias de test. Se inicializan en
cero y se orientan durante el calentamiento solo con observaciones disponibles.

| Metodo | Decision |
|---|---|
| reactive | Geometria de la ultima posicion observada |
| constant_velocity | Velocidad por regresion lineal del historial y extrapolacion de un paso |
| geometric_j | Elige por J entre mantener angulos, geometria reactiva y extrapolacion; estado futuro estimado causalmente |
| cnn_gru | Mapas historicos, CNN y GRU, cabezal angular residual |
| coord_gru | Coordenadas normalizadas de posicion/orientacion/bloqueador y GRU, sin CNN |
| cnn | Solo mapa actual, sin historial |

`geometric_j` es una busqueda finita de candidatos, no una solucion global del
problema no convexo. Mantiene la ultima orientacion y bloqueador observados al
predecir la tasa. El aprendizaje usa imitacion y no maximiza J; J realizado se
reporta por separado junto a las metricas fisicas. No se incluye un oraculo con
acceso al futuro como competidor causal. Un RMSE angular alto puede ser racional
para `geometric_j` cuando decide mantener espejos porque el enlace directo basta.

El canal final utiliza las funciones originales de ganancia, FOV, acoplamiento
y cota de flujo, sin suavizarlas. Se agrega interseccion exacta segmento-cilindro
para LED-PD y **cada** LED-espejo y espejo-PD; el bloqueador puede anular tambien
el IRS. No hay reflexion difusa, sombreado mutuo ni modelo de settling continuo:
el estado de aplicacion representa la orientacion al final de cada intervalo.
La causalidad se comprueba con una prueba que altera todos los estados futuros
y verifica que ninguna decision actual cambie.

## Resultados guardados

En la carpeta `--output` quedan:

- `config.json`, `environment.json`, `split_seeds.json`: parametros, versiones y particion.
- `cnn_gru.pt`, `coord_gru.pt`, `cnn.pt`: pesos del mejor checkpoint de validacion.
- `*_training.csv`: evolucion por epoca de entrenamiento y validacion.
- `metrics.csv`: tasa media, P5, outage, tasa bajo bloqueo, ganancia IRS, movimiento,
  error angular, J y latencia media/P95; IC95 de tasa media por bootstrap de episodios.
- `traces.csv`: medidas por metodo, episodio e intervalo.
- `applied_angles.npz`: angulos efectivamente aplicados, en orden episodio/tiempo.
- `test_episodes.npz`: estados verdaderos y observaciones para auditar la evaluacion.
- `report.md`, `comparison.png`: tabla legible y figura comparativa.
- `trajectory.png`, `time_series.png`: geometria, observaciones y desempeno temporal del primer episodio de test.
- `noise_00cm/`, `noise_02cm/`, `noise_05cm/`, `noise_10cm/` al usar `--noise-sweep`:
  mismo test y mismos pesos, variando solo el ruido de localizacion, sin reentrenar.
- `noise_metrics.csv` y `noise_comparison.png`: resumen del barrido de ruido.

Los tiempos medidos son de inferencia/control en CPU con mapas preparados:
no incluyen sensado, rasterizacion ni bus. Las metricas fisicas son reproducibles
en el mismo entorno; la latencia depende de la carga del equipo. El bootstrap
considera episodios como unidades independientes, no ventanas correlacionadas.
Una sola semilla de entrenamiento no permite concluir significancia de una
mejora entre redes; conviene repetir semillas antes de publicar esa conclusion.

## Discrepancia del informe que debe corregirse

La ecuacion (6) del PDF asigna beta a la componente vertical y usa arcsin para
alpha. El repositorio define alpha como elevacion y beta como azimut, con
normal `(cos(beta)cos(alpha), sin(beta)cos(alpha), sin(alpha))`.
Se mantiene la convencion del repositorio: `alpha=asin(nz)` y
`beta=atan2(ny,nx)`. Permite reconstruir inequívocamente la normal, y la prueba
de ida y vuelta lo verifica. Los nombres roll/yaw del texto no deben confundirse
con una parametrizacion de Euler fisica de un actuador no especificado.

Los parametros de configuracion, las simplificaciones de movilidad y la conexion
residual deben describirse como decisiones nuevas en el informe. Los resultados
del entrenamiento deben incorporarse a la Seccion VI solo despues de revisar
`metrics.csv`; el codigo no presupone superioridad del modelo hibrido.

"""Informe reproducible de la prueba de movimientos inesperados, sin entrenar."""
from pathlib import Path
import math


LABELS = {
    'normal': 'Movimiento normal',
    'sudden_stop': 'Detención repentina',
    'stationary': 'Posición fija',
    'erratic': 'Movimiento errático',
    'fall_proxy': 'Caída simulada',
}
METHODS = ('cnn_gru', 'reactive', 'constant_velocity')
METHOD_LABELS = {
    'cnn_gru': 'CNN-GRU',
    'reactive': 'Reactivo',
    'constant_velocity': 'Velocidad constante',
}


def _number(value, digits=3, signed=False):
    if value is None or not math.isfinite(float(value)):
        return '—'
    return format(float(value), f'{"+" if signed else ""}.{digits}f')


def _percent(value):
    return _number(None if value is None else 100 * float(value), 2)


def _rate_interval(row):
    return (f"{_number(row['mean_rate'])} "
            f"[{_number(row['rate_ci_low'])}; {_number(row['rate_ci_high'])}]")


def write_stress_report(out, cfg, metadata, summaries, paired,
                        reference_summaries, provenance):
    """Escribe informe.md a partir de resultados, sin realizar inferencia ni ajuste.

    ``after`` incluye el primer paso afectado y todo el resto del episodio;
    ``transition`` cubre aproximadamente los primeros 0.5 s y ``settled``
    comienza tras esa ventana y al menos un historial completo posterior.
    Los IC de diferencias se reciben ya calculados por episodio emparejado.
    """
    out = Path(out)
    by_key = {(r['scenario'], r['method'], r['window']): r for r in summaries}
    diffs = {(r['scenario'], r['method'], r['window'], r['contrast']): r
             for r in paired}
    refs = {(r['scenario'], r['window']): r for r in reference_summaries}
    scenarios = list(dict.fromkeys(r['scenario'] for r in summaries))
    event_index = metadata.get('event_index', cfg['steps'] // 2)
    event_time = metadata.get('event_time_s', event_index * cfg['dt'])
    duration = (cfg['steps'] - 1) * cfg['dt']
    transition_steps = max(1, int(round(0.5 / cfg['dt'])))
    transition_duration = transition_steps * cfg['dt']
    transition_end = min(event_time + transition_duration, duration + cfg['dt'])
    settled_delay = max(transition_steps, cfg['history']) * cfg['dt']
    unchanged = provenance.get('checkpoints_unchanged')

    lines = [
        '# Prueba del modelo ante movimientos inesperados', '',
        '**Objetivo:** evaluar cómo responde el modelo ya entrenado cuando la '
        'persona se detiene, permanece en una posición, cambia de dirección '
        'bruscamente o su dispositivo baja y se inclina como aproximación a una caída.', '',
        'Se reutilizaron los pesos existentes, sin reentrenamiento ni ajuste con '
        'estas trayectorias. La red entrega los ángulos de los espejos para el '
        'paso siguiente; no entrega una posición futura de la persona.', '',
        '## Resultados principales', '',
    ]
    if unchanged is True:
        lines.append('La comprobación de las huellas SHA256 confirmó que los archivos '
                     'de pesos son idénticos antes y después de la evaluación.')
    elif unchanged is False:
        lines.append('**La verificación detectó cambios en los archivos de pesos. '
                     'Estos resultados requieren revisar su procedencia antes de compartirlos.**')
    else:
        lines.append('La procedencia de los pesos se conserva junto con los resultados; '
                     'esta entrada del informe no incluye una confirmación de igualdad de sus huellas.')
    lines.append('')

    # Use only measured quantities and paired comparisons in the narrative.
    stop = by_key.get(('sudden_stop', 'cnn_gru', 'onset'))
    if stop is not None:
        text = (f"- **Detención repentina:** en el primer paso afectado, CNN-GRU obtuvo "
                f"{_number(stop['mean_rate'])} bits/s/Hz y un error angular de "
                f"{_number(stop['angle_rmse_deg'])}°. El evento aún no era visible "
                'cuando se emitió esa decisión.')
        delta = diffs.get(('sudden_stop', 'cnn_gru', 'onset', 'versus_reactive'))
        if delta is not None:
            text += (f" La diferencia de tasa respecto del reactivo fue "
                     f"{_number(delta['delta_mean_rate'], signed=True)} bits/s/Hz "
                     f"(IC95: [{_number(delta['delta_rate_ci_low'])}; "
                     f"{_number(delta['delta_rate_ci_high'])}]).")
        lines.append(text)
    for scenario, subject in [('stationary', 'Posición fija'),
                              ('erratic', 'Movimiento errático')]:
        row = by_key.get((scenario, 'cnn_gru', 'after'))
        if row is not None:
            lines.append(
                f"- **{subject}:** en la ventana posterior, CNN-GRU obtuvo "
                f"{_number(row['mean_rate'])} bits/s/Hz, con "
                f"{_percent(row['outage_fraction'])}% de intervalos por debajo del "
                f"umbral y {_number(row['angle_rmse_deg'])}° de error angular.")
    fall = by_key.get(('fall_proxy', 'cnn_gru', 'after'))
    if fall is not None:
        lines.append(
            f"- **Caída simulada:** CNN-GRU obtuvo {_number(fall['mean_rate'])} "
            f"bits/s/Hz y {_percent(fall['outage_fraction'])}% de intervalos bajo "
            'el umbral en la ventana posterior. Este escenario modifica tanto '
            'la altura como la orientación del receptor; su degradación incluye '
            'cambios físicos del canal.')

    lines += [
        '', '## CNN-GRU después del evento', '',
        f'Ventana común: desde {_number(event_time, 2)} s hasta '
        f'{_number(duration, 2)} s, incluyendo la primera muestra afectada. '
        'Para posición fija, el dispositivo ya estaba detenido desde el inicio; '
        'se usa el mismo intervalo temporal para facilitar la lectura. '
        'El caso normal no tiene perturbación.', '',
        '| Escenario | Tasa media e IC95 [bits/s/Hz] | P5 de tasa | Bajo umbral [%] '
        '| Error angular RMSE [°] | Máximo RMSE por intervalo [°] |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for scenario in scenarios:
        row = by_key.get((scenario, 'cnn_gru', 'after'))
        if row is not None:
            lines.append(
                f"| {LABELS.get(scenario, scenario)} | {_rate_interval(row)} "
                f"| {_number(row['p5_rate'])} | {_percent(row['outage_fraction'])} "
                f"| {_number(row['angle_rmse_deg'])} "
                f"| {_number(row['peak_angle_rmse_deg'])} |")
    lines += [
        '', 'P5 es el percentil 5 de las tasas por intervalo. El RMSE angular '
        'compara los ángulos aplicados con los ángulos geométricos de la posición '
        'verdadera al aplicar el control. Incluye ruido, error de anticipación '
        'y restricciones de movimiento de los espejos.', '',
        '![Comparación posterior al evento](resumen_postevento.png)', '',
        '## Primer paso y respuesta inicial', '',
        f'El primer paso corresponde a la aplicación en {_number(event_time, 2)} s. '
        f'La ventana inicial abarca [{_number(event_time, 2)}, '
        f'{_number(transition_end, 2)}) s; incluye ese primer paso. '
        'Las ventanas se solapan deliberadamente para distinguir la sorpresa '
        'inicial de la respuesta durante los siguientes pasos.', '',
        '| Escenario | Tasa primer paso | Error primer paso [°] | Bajo umbral primer paso [%] '
        '| Tasa primeros 0.5 s | Error primeros 0.5 s [°] | Bajo umbral primeros 0.5 s [%] |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for scenario in scenarios:
        first = by_key.get((scenario, 'cnn_gru', 'onset'))
        transient = by_key.get((scenario, 'cnn_gru', 'transition'))
        if first is not None and transient is not None:
            lines.append(
                f"| {LABELS.get(scenario, scenario)} | {_number(first['mean_rate'])} "
                f"| {_number(first['angle_rmse_deg'])} | {_percent(first['outage_fraction'])} "
                f"| {_number(transient['mean_rate'])} | {_number(transient['angle_rmse_deg'])} "
                f"| {_percent(transient['outage_fraction'])} |")
    lines += [
        '', 'Las tasas están en bits/s/Hz. En el primer paso, el porcentaje bajo '
        'umbral corresponde a la proporción de episodios. Esta evaluación no '
        'estima un tiempo de recuperación ni presupone que el modelo se recupere.', '',
        '![Respuesta alrededor del evento](respuesta_evento.png)', '',
        '## Comparación con controles de referencia', '',
        'Los tres métodos de esta tabla reciben la misma trayectoria y las mismas '
        'observaciones dentro de cada escenario. El reactivo apunta usando la '
        'última posición medida; velocidad constante extrapola el movimiento '
        'estimado del historial. Todos actúan con un paso de retraso. '
        'Se muestra la ventana posterior completa.', '',
        '| Escenario | Método | Tasa media [bits/s/Hz] | Bajo umbral [%] | RMSE angular [°] |',
        '|---|---|---:|---:|---:|',
    ]
    for scenario in scenarios:
        for method in METHODS:
            row = by_key.get((scenario, method, 'after'))
            if row is not None:
                lines.append(
                    f"| {LABELS.get(scenario, scenario)} | {METHOD_LABELS[method]} "
                    f"| {_number(row['mean_rate'])} | {_percent(row['outage_fraction'])} "
                    f"| {_number(row['angle_rmse_deg'])} |")
    lines += [
        '', '### Diferencias emparejadas de CNN-GRU', '',
        'Cada diferencia se calcula con los mismos episodios. Un valor positivo '
        'de tasa significa mayor tasa para CNN-GRU en el escenario indicado; '
        'un valor positivo de indisponibilidad significa más intervalos bajo el umbral. '
        'La comparación con normal incluye el cambio de trayectoria y de canal, '
        'además de la respuesta del controlador.', '',
        '| Escenario | Comparación posterior | Diferencia de tasa e IC95 [bits/s/Hz] '
        '| Diferencia bajo umbral [puntos porcentuales] | Diferencia RMSE [°] |',
        '|---|---|---:|---:|---:|',
    ]
    for scenario in scenarios:
        for contrast, label in [('versus_reactive', 'CNN-GRU − reactivo'),
                                ('versus_normal', 'CNN-GRU escenario − CNN-GRU normal')]:
            if scenario == 'normal' and contrast == 'versus_normal':
                continue
            row = diffs.get((scenario, 'cnn_gru', 'after', contrast))
            if row is not None:
                lines.append(
                    f"| {LABELS.get(scenario, scenario)} | {label} "
                    f"| {_number(row['delta_mean_rate'], signed=True)} "
                    f"[{_number(row['delta_rate_ci_low'])}; {_number(row['delta_rate_ci_high'])}] "
                    f"| {_number(row['delta_outage_pp'], 2, signed=True)} "
                    f"| {_number(row['delta_angle_rmse_deg'], signed=True)} |")
    lines += [
        '', 'Los intervalos de confianza se obtienen remuestreando episodios '
        'completos. Describen la variación entre estas trayectorias sintéticas; '
        'una sola corrida de entrenamiento y una sola sala no permiten afirmar '
        'superioridad general de un método.', '',
        '## Cómo interpretar la caída simulada', '',
        'La caída es una perturbación cinemática del receptor: disminuye su altura '
        'y cambia su orientación. No representa una simulación biomecánica ni '
        'valida detección de caídas. La orientación puede sacar los enlaces del '
        'campo de visión del fotodiodo. Por eso una caída de tasa no implica, '
        'por sí sola, un fallo de anticipación del modelo.', '',
        'Para separar parcialmente ambos efectos se calculan dos diagnósticos: '
        'la tasa con ángulos geométricos instantáneos obtenidos de la posición '
        'verdadera y la tasa del enlace directo sin IRS. El primer diagnóstico '
        'usa información futura y omite límites mecánicos; no es un competidor '
        'causal ni una cota máxima garantizada de la tasa.', '',
        '| Escenario, ventana posterior | Tasa CNN-GRU | Tasa geometría instantánea '
        '| Bajo umbral geometría [%] | Tasa solo enlace directo |',
        '|---|---:|---:|---:|---:|',
    ]
    for scenario in ('normal', 'fall_proxy'):
        ref = refs.get((scenario, 'after'))
        row = by_key.get((scenario, 'cnn_gru', 'after'))
        if ref is not None and row is not None:
            lines.append(
                f"| {LABELS[scenario]} | {_number(row['mean_rate'])} "
                f"| {_number(ref['aligned_mean_rate'])} "
                f"| {_percent(ref['aligned_outage_fraction'])} "
                f"| {_number(ref['los_only_mean_rate'])} |")
    lines += [
        '', 'Tasas en bits/s/Hz. Los diagnósticos conservan la orientación real '
        'del receptor y los bloqueos de cada escenario.', '',
        '**Limitaciones particulares de altura:** la CNN no tiene un canal de '
        'altura del receptor en sus mapas; el cálculo geométrico que se suma a '
        'su salida sí recibe la altura actual observada. La GRU de coordenadas '
        'recibe esa altura entre sus entradas. Los controles originales de '
        'velocidad constante y `geometric_j` fuerzan la altura futura al valor '
        'fijo del entrenamiento. Se conservaron esos controles: su desempeño '
        'en la caída incluye esta limitación y no debe interpretarse como una '
        'comparación aislada de capacidad predictiva.', '',
        '## Protocolo y alcance', '',
        f'- Se evaluaron {cfg["test_episodes"]} episodios por escenario, separados '
        'del entrenamiento y la validación, con las mismas semillas de test del experimento original.',
        f'- Cada trayectoria contiene {cfg["steps"]} muestras, separadas por '
        f'{_number(1000 * cfg["dt"], 0)} ms. El historial contiene {cfg["history"]} muestras. '
        'En cada decisión solo se utilizan observaciones disponibles hasta ese instante.',
        f'- Los eventos comienzan en la muestra {event_index}, a los '
        f'{_number(event_time, 2)} s. Detención, movimiento errático y caída '
        'comparten el prefijo original hasta la muestra anterior. Posición fija '
        'se aplica desde el inicio del episodio.',
        '- Se conserva la realización de ruido y la trayectoria original del '
        'bloqueador al construir las variantes. Su bloqueo efectivo puede cambiar '
        'porque cambia la posición del receptor. Posición fija significa ausencia '
        'de traslación; la orientación conserva su evolución original.',
        f'- El umbral de disponibilidad es Rmin = {_number(cfg["r_min"], 2)} bits/s/Hz. '
        'La indisponibilidad es la fracción de intervalos con tasa menor que ese umbral.',
        '- Los archivos incluyen las ventanas `all` (toda la evaluación), `before` '
        '(antes del evento), `onset` (primera muestra afectada), `transition` '
        f'(primeros {_number(transition_duration, 2)} s), `after` '
        '(desde el evento hasta el final) y `settled` '
        f'(desde {_number(settled_delay, 2)} s después del evento hasta el final). '
        'Esta última ventana comienza después de la transición y de un historial '
        'completo de muestras posteriores al evento. El nombre `settled` '
        'identifica una ventana temporal; no demuestra estabilización del modelo.',
        '- Se usa el mismo modelo físico de canal, con bloqueo, campo de visión '
        'y restricciones angulares del experimento original. Son resultados '
        'sintéticos en una geometría fija, sin mediciones reales.', '',
        '![Trayectorias de los escenarios](trayectorias_estres.png)', '',
        '## Archivos para revisar o compartir', '',
        '- [Métricas por escenario, método y ventana](metrics_by_window.csv).',
        '- [Diferencias emparejadas e intervalos de confianza](paired_differences.csv).',
        '- [Diagnósticos de geometría instantánea y enlace directo](alignment_by_window.csv).',
        '- [Procedencia, configuración y verificación de los pesos](provenance.json).',
        '- [Figura resumen posterior al evento](resumen_postevento.png).',
        '- [Figura de respuesta temporal](respuesta_evento.png).',
        '- [Figura de trayectorias](trayectorias_estres.png).', '',
        'Las carpetas de cada escenario conservan las métricas completas, las '
        'trazas por intervalo, los ángulos aplicados y las trayectorias verdaderas '
        'y observadas. Las cifras de este informe se generan directamente de '
        'los resultados de esta evaluación.', '',
    ]
    (out / 'informe.md').write_text('\n'.join(lines), encoding='utf-8')

"""Figuras de las pruebas de movilidad con controladores congelados.

Se usa el primer episodio en las figuras de trayectoria, sin seleccionar un
ejemplo por su resultado. Las curvas de desempeño agregan todos los episodios.
"""
from pathlib import Path

import numpy as np


SCENARIO_ORDER = ('normal', 'sudden_stop', 'stationary', 'erratic', 'fall_proxy')
SCENARIO_LABELS = {
    'normal': 'Normal', 'sudden_stop': 'Detención\nrepentina',
    'stationary': 'Posición\nfija', 'erratic': 'Giros\nerráticos',
    'fall_proxy': 'Caída\nsimplificada',
}
SCENARIO_COLORS = {
    'normal': '#64748b', 'sudden_stop': '#dc5a40', 'stationary': '#b28515',
    'erratic': '#7c4bb3', 'fall_proxy': '#be3c82',
}
METHODS = ('cnn_gru', 'reactive', 'constant_velocity')
METHOD_LABELS = {
    'cnn_gru': 'CNN–GRU', 'reactive': 'Reactivo',
    'constant_velocity': 'Velocidad constante',
}
METHOD_COLORS = {
    'cnn_gru': '#007c91', 'reactive': '#d47b21', 'constant_velocity': '#6d55a3',
}


def _save(fig, out, stem):
    fig.savefig(out / (stem + '.png'), dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(out / (stem + '.pdf'), bbox_inches='tight', facecolor='white')


def _event_time(cfg, metadata):
    return float(metadata.get('event_time_s',
                              metadata.get('event_index', cfg['steps'] // 2) * cfg['dt']))


def _summary_figure(plt, summaries, cfg, metadata, out):
    rows = {(r['scenario'], r['method']): r for r in summaries if r['window'] == 'after'}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.24
    for ax, key, label, scale in zip(
            axes, ('mean_rate', 'outage_fraction', 'angle_rmse_deg'),
            ('Tasa media [bits/s/Hz]', 'Indisponibilidad [%]', 'RMSE angular [°]'),
            (1, 100, 1)):
        for i, method in enumerate(METHODS):
            values = [scale * float(rows[(scenario, method)][key])
                      for scenario in SCENARIO_ORDER]
            ax.bar(x + (i - 1) * width, values, width=width,
                   label=METHOD_LABELS[method], color=METHOD_COLORS[method])
        ax.set_xticks(x, [SCENARIO_LABELS[s] for s in SCENARIO_ORDER], fontsize=9)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(bottom=0)
    axes[0].axhline(cfg['r_min'], color='#374151', linestyle=':', linewidth=1)
    axes[0].text(0.01, cfg['r_min'], ' Rmín', transform=axes[0].get_yaxis_transform(),
                 va='bottom', fontsize=8, color='#374151')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.92))
    fig.suptitle('Respuesta posterior al cambio · pesos originales, sin reentrenar',
                 fontsize=14, y=0.99)
    fig.text(0.5, 0.02,
             f"Aplicación desde {_event_time(cfg, metadata):.2f} s. "
             'Posición fija desde t = 0. Caída: extrapolación conserva la altura nominal.',
             ha='center', fontsize=9, color='#475569')
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.06, right=0.99, wspace=0.28)
    _save(fig, out, 'resumen_postevento')
    plt.close(fig)


def _time_curve(traces, scenario, method, key, cfg, event_time):
    grouped = {}
    for row in traces:
        if row['scenario'] == scenario and row['method'] == method:
            grouped.setdefault(int(row['t']), []).append(float(row[key]))
    times = sorted(grouped)
    return (np.array([(t + 1) * cfg['dt'] - event_time for t in times]),
            np.array([np.mean(grouped[t]) for t in times]))


def _response_figure(plt, traces, cfg, metadata, out):
    scenarios = ('sudden_stop', 'erratic', 'fall_proxy', 'stationary')
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True)
    event_time = _event_time(cfg, metadata)
    for i, scenario in enumerate(scenarios):
        title = SCENARIO_LABELS[scenario].replace('\n', ' ')
        if scenario == 'stationary':
            title += ' (desde t = 0)'
        for j, key in enumerate(('angle_rmse_deg', 'rate')):
            ax = axes[i, j]
            for method in METHODS:
                x, y = _time_curve(traces, scenario, method, key, cfg, event_time)
                ax.plot(x, y, color=METHOD_COLORS[method], label=METHOD_LABELS[method],
                        linewidth=1.7, alpha=0.95)
            x, y = _time_curve(traces, 'normal', 'cnn_gru', key, cfg, event_time)
            ax.plot(x, y, color='#64748b', linestyle='--', linewidth=1.4,
                    label='CNN–GRU, trayectoria normal', alpha=0.85)
            ax.axvline(0, color='#334155', linestyle=':', linewidth=1)
            if scenario != 'stationary':
                ax.axvspan(0, 0.5, color='#64748b', alpha=0.08)
            ax.set_title(title, fontsize=10, loc='left', fontweight='bold')
            ax.set_ylabel('RMSE angular medio [°]' if j == 0 else 'Tasa media [bits/s/Hz]')
            ax.grid(alpha=0.22)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=max(-event_time, -1), right=(cfg['steps'] - 1) * cfg['dt'] - event_time)
    for ax in axes[-1]:
        ax.set_xlabel('Tiempo relativo al evento de referencia [s]')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.962), fontsize=9)
    fig.suptitle('Dinámica del error y del enlace ante movimientos no habituales',
                 y=0.99, fontsize=14)
    n = len({r['episode'] for r in traces if r['scenario'] == 'normal'})
    fig.text(0.5, 0.017,
             f'Media sobre {n} episodios por instante; sombreado: primeros 0,5 s. '
             'El error angular no es un error de posición.\n'
             'Caída: altura e inclinación cambian el canal; velocidad constante sigue suponiendo altura nominal.',
             ha='center', fontsize=9, color='#475569')
    fig.subplots_adjust(top=0.91, bottom=0.09, left=0.075, right=0.99,
                        hspace=0.38, wspace=0.22)
    _save(fig, out, 'respuesta_evento')
    plt.close(fig)


def _trajectory_figure(plt, episodes, cfg, metadata, out):
    import ris_vlc as rv

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    esc = rv.Escenario(**cfg.get('scenario', {}))
    event_index = int(metadata.get('event_index', cfg['steps'] // 2))
    event_time = _event_time(cfg, metadata)
    for scenario in SCENARIO_ORDER:
        ep = episodes[scenario][0]
        position, normal = np.asarray(ep['position']), np.asarray(ep['normal'])
        times = np.arange(len(position)) * cfg['dt']
        label = SCENARIO_LABELS[scenario].replace('\n', ' ')
        color = SCENARIO_COLORS[scenario]
        style = '--' if scenario == 'normal' else '-'
        axes[0, 0].plot(position[:, 0], position[:, 1], label=label, color=color,
                        linestyle=style, linewidth=1.8, alpha=0.9)
        if scenario == 'stationary':
            axes[0, 0].scatter(*position[0, :2], s=75, marker='x', color=color, zorder=10)
        elif scenario != 'normal':
            axes[0, 0].scatter(*position[event_index, :2], s=32, color=color, zorder=8)
        axes[0, 1].plot(times, position[:, 2], color=color, linestyle=style,
                        linewidth=1.6, alpha=0.85)
        inclination = np.degrees(np.arccos(np.clip(normal[:, 2] / np.linalg.norm(normal, axis=1), -1, 1)))
        axes[1, 0].plot(times, inclination, color=color, linestyle=style,
                        linewidth=1.6, alpha=0.85)
        speed = np.linalg.norm(np.diff(position[:, :2], axis=0), axis=1) / cfg['dt']
        axes[1, 1].plot(times[1:], speed, color=color, linestyle=style,
                        linewidth=1.6, alpha=0.85)
    axes[0, 0].set(xlabel='x [m]', ylabel='y [m]', title='Recorrido horizontal verdadero',
                   xlim=(0, esc.sala[0]), ylim=(0, esc.sala[1]))
    axes[0, 0].set_aspect('equal', adjustable='box')
    axes[0, 1].set(xlabel='Tiempo [s]', ylabel='Altura [m]', title='Altura del receptor')
    axes[1, 0].set(xlabel='Tiempo [s]', ylabel='Inclinación respecto a vertical [°]',
                   title='Orientación del receptor')
    axes[1, 1].set(xlabel='Tiempo [s]', ylabel='Velocidad horizontal [m/s]',
                   title='Movimiento entre muestras')
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.axvline(event_time, color='#334155', linestyle=':', linewidth=1)
        ax.set_xlim(0, (cfg['steps'] - 1) * cfg['dt'])
        ax.set_ylim(bottom=0)
    for ax in axes.flat:
        ax.grid(alpha=0.22)
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.947), fontsize=9)
    seed = episodes['normal'][0].get('seed', '—')
    fig.suptitle(f'Trayectorias de prueba · primer episodio, semilla {seed}',
                 fontsize=14, y=0.99)
    fig.text(0.5, 0.02,
             'Mismo prefijo antes del evento, excepto posición fija desde el inicio. '
             'La caída es una perturbación cinemática simplificada.\n'
             'Las curvas coincidentes se superponen; posición fija conserva la orientación original.',
             ha='center', fontsize=9, color='#475569')
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.08, right=0.99,
                        hspace=0.38, wspace=0.24)
    _save(fig, out, 'trayectorias_estres')
    plt.close(fig)


def plot_stress(episodes, traces, summaries, cfg, metadata, out):
    """Escribe tres figuras en PNG y PDF; no modifica datos ni controladores.

    ``summaries`` requiere la ventana ``after`` para cada escenario/método.
    ``traces.t`` es el índice de decisión: el resultado se aplica en ``t+1``.
    En respuesta_evento cada ordenada es la media de las métricas por episodio
    en ese instante; el RMSE agregado del resumen lo suministra el evaluador.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        _summary_figure(plt, summaries, cfg, metadata, out)
        _response_figure(plt, traces, cfg, metadata, out)
        _trajectory_figure(plt, episodes, cfg, metadata, out)

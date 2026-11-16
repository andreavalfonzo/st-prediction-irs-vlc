"""Evalua trayectorias inesperadas con checkpoints existentes, sin entrenamiento.

python evaluar_estres.py --checkpoint-dir resultados/experimento_sintetico \
    --output resultados/estres_movilidad
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

import ris_vlc as rv
from entrenar import validate
from prediccion.data import prepare
from prediccion.experiment import KINDS, evaluate, load_models, setup, write_csv
from prediccion.physics import angles, link
from prediccion.stress import SCENARIOS, make_scenario, scenario_metadata


WINDOWS = ('all', 'before', 'onset', 'transition', 'after', 'settled')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def in_window(application_index, window, cfg, metadata):
    """Ventanas sobre el instante de APLICACION, nunca sobre el de decision."""
    onset = metadata['event_index']
    transition = max(1, int(round(0.5 / cfg['dt'])))
    return {
        'all': True,
        'before': application_index < onset,
        'onset': application_index == onset,
        'transition': onset <= application_index < onset + transition,
        'after': application_index >= onset,
        'settled': application_index >= onset + max(transition, cfg['history']),
    }[window]


def read_traces(path, scenario):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    integer_fields = ('episode', 'seed', 't', 'blocked')
    return [dict(scenario=scenario, **{
        key: value if key == 'method' else int(value) if key in integer_fields else float(value)
        for key, value in row.items()}) for row in rows]


def bootstrap_mean(values, seed):
    """Remuestrea episodios independientes, no intervalos correlacionados."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
    return tuple(float(v) for v in np.percentile(boot, [2.5, 97.5]))


def summarize(traces, cfg, metadata):
    summaries, episode_rows = [], []
    groups = {}
    for row in traces:
        groups.setdefault((row['scenario'], row['method']), []).append(row)
    for (scenario, method), rows in groups.items():
        for window in WINDOWS:
            samples = [r for r in rows if in_window(r['t'] + 1, window, cfg, metadata)]
            if not samples:
                continue
            per_episode = {}
            for row in samples:
                per_episode.setdefault(row['episode'], []).append(row)
            ep_metrics = []
            for episode_id, subset in per_episode.items():
                rate = np.array([r['rate'] for r in subset])
                metric = dict(scenario=scenario, method=method, window=window,
                              episode=episode_id, seed=subset[0]['seed'], samples=len(subset),
                              mean_rate=float(rate.mean()),
                              outage_fraction=float(np.mean(rate < cfg['r_min'])),
                              angle_mse_deg2=float(np.mean([r['angle_rmse_deg']**2 for r in subset])))
                episode_rows.append(metric)
                ep_metrics.append(metric)
            rate = np.array([r['rate'] for r in samples])
            low, high = bootstrap_mean([r['mean_rate'] for r in ep_metrics], cfg['seed'])
            summaries.append(dict(
                scenario=scenario, method=method, window=window,
                episodes=len(ep_metrics), samples=len(samples),
                mean_rate=float(rate.mean()), rate_ci_low=low, rate_ci_high=high,
                p5_rate=float(np.percentile(rate, 5)),
                outage_fraction=float(np.mean(rate < cfg['r_min'])),
                blocked_fraction=float(np.mean([r['blocked'] for r in samples])),
                mean_h_irs=float(np.mean([r['h_irs'] for r in samples])),
                angle_rmse_deg=float(np.sqrt(np.mean([r['angle_rmse_deg']**2 for r in samples]))),
                peak_angle_rmse_deg=float(max(r['angle_rmse_deg'] for r in samples)),
            ))
    return summaries, episode_rows


def paired_differences(episode_rows, cfg):
    groups = {}
    for row in episode_rows:
        groups.setdefault((row['scenario'], row['method'], row['window']), {})[row['seed']] = row
    result = []
    for (scenario, method, window), current in groups.items():
        comparisons = [('versus_reactive', (scenario, 'reactive', window))]
        if scenario != 'normal':
            comparisons.append(('versus_normal', ('normal', method, window)))
        for contrast, reference_key in comparisons:
            if contrast == 'versus_reactive' and method == 'reactive':
                continue
            reference = groups[reference_key]
            if set(reference) != set(current):
                raise ValueError('Las comparaciones requieren las mismas semillas de test')
            seeds = sorted(current)
            delta_rate = [current[s]['mean_rate'] - reference[s]['mean_rate'] for s in seeds]
            low, high = bootstrap_mean(delta_rate, cfg['seed'])
            result.append(dict(
                scenario=scenario, method=method, window=window, contrast=contrast,
                episodes=len(seeds), delta_mean_rate=float(np.mean(delta_rate)),
                delta_rate_ci_low=low, delta_rate_ci_high=high,
                delta_outage_pp=float(100*np.mean([
                    current[s]['outage_fraction'] - reference[s]['outage_fraction'] for s in seeds])),
                delta_angle_rmse_deg=float(
                    np.sqrt(np.mean([current[s]['angle_mse_deg2'] for s in seeds]))
                    - np.sqrt(np.mean([reference[s]['angle_mse_deg2'] for s in seeds]))),
            ))
    return result


def alignment_reference(episodes, esc, cfg):
    """Referencia diagnostica no causal, sin restricciones mecanicas.

    Geometria hacia posicion real de aplicacion: no es una cota de tasa ni un
    optimizador global. Distingue desalineacion de perdida fisica por FOV/bloqueo.
    """
    rows = []
    for scenario, eps in episodes.items():
        for episode_id, ep in enumerate(eps):
            for index in range(cfg['history'], cfg['steps']):
                position, normal = ep['position'][index], ep['normal'][index]
                rate, h_irs, blocked = link(esc, position, normal, ep['blocker'][index],
                                          angles(esc, position), cfg)
                h_los = 0.0 if blocked else float(rv.ganancia_los(esc, position, normal))
                rows.append(dict(scenario=scenario, episode=episode_id, seed=ep['seed'],
                                 t=index-1, aligned_rate=rate, aligned_h_irs=h_irs,
                                 los_only_rate=float(rv.tasa(esc, h_los))))
    return rows


def summarize_reference(rows, cfg, metadata):
    result = []
    for scenario in SCENARIOS:
        for window in WINDOWS:
            samples = [r for r in rows if r['scenario'] == scenario
                       and in_window(r['t']+1, window, cfg, metadata)]
            if samples:
                rate = np.array([r['aligned_rate'] for r in samples])
                result.append(dict(scenario=scenario, window=window,
                                   aligned_mean_rate=float(rate.mean()),
                                   aligned_outage_fraction=float(np.mean(rate < cfg['r_min'])),
                                   los_only_mean_rate=float(np.mean([r['los_only_rate'] for r in samples]))))
    return result


def verify_causal_prefix(output, cfg, metadata):
    """La orden previa al evento debe coincidir incluso en el primer paso afectado."""
    length = cfg['steps'] - cfg['history']
    # Column zero is application at history; include application at event_index.
    prefix_length = metadata['event_index'] - cfg['history'] + 1
    with np.load(output/'normal'/'applied_angles.npz') as normal:
        for scenario in ('sudden_stop', 'erratic', 'fall_proxy'):
            with np.load(output/scenario/'applied_angles.npz') as changed:
                for method in normal.files:
                    shape = (cfg['test_episodes'], length, -1)
                    reference = normal[method].reshape(shape)[:, :prefix_length]
                    candidate = changed[method].reshape(shape)[:, :prefix_length]
                    if not np.array_equal(reference, candidate):
                        raise AssertionError(f'Prefijo causal distinto: {scenario}/{method}')
    return True


def verify_original_normal(checkpoint_dir, output):
    """Auditoria opcional contra la evaluacion original guardada, excluye latencia."""
    path = checkpoint_dir/'traces.csv'
    if not path.exists():
        return {'available': False}
    old, new = read_traces(path, 'normal'), read_traces(output/'normal'/'traces.csv', 'normal')
    keys = ('method', 'episode', 'seed', 't')
    old_lookup = {tuple(row[k] for k in keys): row for row in old}
    if len(old) != len(new) or any(tuple(r[k] for k in keys) not in old_lookup for r in new):
        raise AssertionError('La evaluacion normal no reproduce el conjunto original')
    fields = ('rate', 'h_irs', 'blocked', 'movement_deg', 'angle_rmse_deg', 'j')
    differences = {}
    for field in fields:
        original = np.array([old_lookup[tuple(r[k] for k in keys)][field] for r in new])
        measured = np.array([r[field] for r in new])
        differences[field] = float(np.max(np.abs(original-measured)))
        if not np.allclose(original, measured, rtol=1e-5, atol=1e-8):
            raise AssertionError(f'La evaluacion normal difiere de la original: {field}')
    return {'available': True, 'reproduced': True, 'max_absolute_differences': differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('resultados/experimento_sintetico'))
    parser.add_argument('--output', type=Path, default=Path('resultados/estres_movilidad'))
    parser.add_argument('--event-index', type=int, help='Primera muestra alterada; defecto steps//2')
    args = parser.parse_args()
    checkpoint_dir, output = args.checkpoint_dir.resolve(), args.output.resolve()
    if output == checkpoint_dir or checkpoint_dir.is_relative_to(output):
        raise ValueError('El output debe ser independiente de la carpeta de checkpoints')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output no vacio: usa una carpeta nueva para preservar los resultados')
    cfg = json.loads((checkpoint_dir/'config.json').read_text(encoding='utf-8'))
    validate(cfg)
    setup(cfg)
    esc = rv.Escenario(**cfg['scenario'])
    metadata = scenario_metadata(cfg, esc, args.event_index)
    if metadata['fall_end_index'] >= cfg['steps']:
        raise ValueError('El evento debe dejar al menos 0.5 s para completar la caida')
    # Keep a stable key for reports regardless of the generator's extra metadata.
    metadata['event_time_s'] = metadata['event_index']*cfg['dt']
    manifest = json.loads((checkpoint_dir/'split_seeds.json').read_text(encoding='utf-8'))
    seeds = manifest['test']
    if len(seeds) != cfg['test_episodes'] or len(set(seeds)) != len(seeds):
        raise ValueError('Particion test incompatible con configuracion')
    if set(seeds) & (set(manifest['train']) | set(manifest['val'])):
        raise ValueError('Test debe ser disjunto de entrenamiento y validacion')
    files = {kind: checkpoint_dir/(kind+'.pt') for kind in KINDS}
    hashes = {kind: sha256(path) for kind, path in files.items()}
    models = load_models(checkpoint_dir, cfg)
    checkpoints = {}
    for kind, path in files.items():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        checkpoints[kind] = dict(path=str(path), sha256_before=hashes[kind],
                                 selected_epoch=saved['epoch'],
                                 validation_mse_scaled=saved['validation_mse_scaled'])
    output.mkdir(parents=True, exist_ok=True)
    (output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8')
    provenance = dict(
        checkpoint_dir=str(checkpoint_dir), training_performed=False,
        checkpoints=checkpoints, test_seeds=seeds, scenarios=list(SCENARIOS),
        intervention=metadata, python=sys.version, platform=platform.platform(),
        numpy=np.__version__, torch=torch.__version__, device='cpu',
        source_sha256={name: sha256(Path(__file__).parent/name) for name in (
            'evaluar_estres.py', 'prediccion/stress.py', 'prediccion/stress_plots.py',
            'prediccion/stress_report.py', 'prediccion/data.py', 'prediccion/models.py',
            'prediccion/experiment.py', 'prediccion/physics.py', 'ris_vlc.py')},
    )
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    episodes, traces, all_metrics = {}, [], []
    for scenario in SCENARIOS:
        print(f'\nEscenario {scenario}: {len(seeds)} trayectorias; pesos congelados', flush=True)
        eps = [prepare(make_scenario(cfg, esc, seed, scenario, metadata['event_index']), esc, cfg)
               for seed in seeds]
        episodes[scenario] = eps
        scenario_out = output/scenario
        scenario_out.mkdir()
        metrics = evaluate({'test': eps}, models, cfg, esc, scenario_out)
        all_metrics.extend(dict(scenario=scenario, **row) for row in metrics)
        traces.extend(read_traces(scenario_out/'traces.csv', scenario))
    print('\nAgregando ventanas de evento y referencias de alineacion...', flush=True)
    summaries, episode_rows = summarize(traces, cfg, metadata)
    paired = paired_differences(episode_rows, cfg)
    reference = alignment_reference(episodes, esc, cfg)
    reference_summaries = summarize_reference(reference, cfg, metadata)
    write_csv(output/'metrics.csv', all_metrics)
    write_csv(output/'metrics_by_window.csv', summaries)
    write_csv(output/'episode_metrics.csv', episode_rows)
    write_csv(output/'paired_differences.csv', paired)
    write_csv(output/'alignment_reference.csv', reference)
    write_csv(output/'alignment_by_window.csv', reference_summaries)
    provenance['causal_prefix_verified'] = verify_causal_prefix(output, cfg, metadata)
    provenance['original_normal_verification'] = verify_original_normal(checkpoint_dir, output)
    after = {kind: sha256(path) for kind, path in files.items()}
    provenance['checkpoints_unchanged'] = after == hashes
    for kind in KINDS:
        provenance['checkpoints'][kind]['sha256_after'] = after[kind]
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    if after != hashes:
        raise AssertionError('Los checkpoints cambiaron durante la evaluacion')
    from prediccion.stress_plots import plot_stress
    from prediccion.stress_report import write_stress_report
    plot_stress(episodes, traces, summaries, cfg, metadata, output)
    write_stress_report(output, cfg, metadata, summaries, paired, reference_summaries, provenance)
    print('Pesos intactos y causalidad verificada. Resultados: ' + str(output), flush=True)


if __name__ == '__main__':
    main()

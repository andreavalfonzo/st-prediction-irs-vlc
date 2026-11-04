"""CLI reproducible: python entrenar.py --config config_prediccion.json."""
import argparse
import json
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import ris_vlc as rv
from prediccion.data import splits, episode, prepare
from prediccion.experiment import KINDS, setup, train, load_models, evaluate, write_csv, plot_noise


def validate(cfg):
    if not 2 <= cfg['history'] < cfg['steps']:
        raise ValueError('Se requiere 2 <= history < steps')
    for name in ('train_episodes', 'val_episodes', 'test_episodes', 'epochs', 'batch_size', 'threads'):
        if not isinstance(cfg[name], int) or cfg[name] < 1:
            raise ValueError(name + ' debe ser entero positivo')
    for name in ('dt', 'slew_deg_s', 'max_speed_m_s', 'learning_rate',
                 'blocker_radius_m', 'blocker_height_m'):
        if not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(name + ' debe ser positivo y finito')
    for name in ('position_noise_m', 'orientation_noise_deg', 'lambda_s', 'lambda_q', 'r_min'):
        if not np.isfinite(cfg[name]) or cfg[name] < 0:
            raise ValueError(name + ' debe ser no negativo y finito')
    lo, hi = np.array(cfg['angle_min_deg']), np.array(cfg['angle_max_deg'])
    if lo.shape != (2,) or hi.shape != (2,) or not np.all(np.isfinite([lo, hi])):
        raise ValueError('Limites angulares deben contener dos valores finitos')
    if np.any(lo >= hi) or np.any(lo > 0) or np.any(hi < 0):
        raise ValueError('Limites deben contener cero y tener min < max')
    if not isinstance(cfg['grid'], int) or cfg['grid'] < 8:
        raise ValueError('grid debe ser entero >=8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config_prediccion.json'))
    parser.add_argument('--output', type=Path, default=Path('resultados/prediccion'))
    parser.add_argument('--evaluate-only', action='store_true', help='Reutilizar checkpoints, sin entrenar')
    parser.add_argument('--quick', action='store_true', help='Solo prueba de integracion, no resultado cientifico')
    parser.add_argument('--noise-sweep', action='store_true', help='Evaluar ruido de posicion 0/2/5/10 cm sin reentrenar')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if args.quick:
        cfg.update(train_episodes=6, val_episodes=2, test_episodes=2, steps=40, epochs=2)
    validate(cfg)
    setup(cfg)
    esc = rv.Escenario(**cfg['scenario'])
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.evaluate_only and any(args.output.glob('*.pt')):
        raise FileExistsError('Ya existen checkpoints. Usa --evaluate-only o una carpeta --output nueva.')
    if args.evaluate_only:
        saved_cfg = json.loads((args.output/'config.json').read_text(encoding='utf-8'))
        if saved_cfg != cfg:
            raise ValueError('La configuracion no coincide con el experimento guardado')
    (args.output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8')
    (args.output/'environment.json').write_text(json.dumps(dict(python=sys.version,
        platform=platform.platform(), numpy=np.__version__, torch=torch.__version__, device='cpu'), indent=2), encoding='utf-8')
    print('Generando trayectorias y mapas separados por episodio...', flush=True)
    data = splits(cfg, esc)
    manifest = {name: [ep['seed'] for ep in eps] for name, eps in data.items()}
    (args.output/'split_seeds.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if not args.evaluate_only:
        for kind in KINDS:
            train(kind, data, cfg, args.output)
    models = load_models(args.output, cfg)
    evaluate(data, models, cfg, esc, args.output)
    if args.noise_sweep:
        noise_rows = []
        for noise in (0.0, 0.02, 0.05, 0.10):
            noisy_cfg = dict(cfg, position_noise_m=noise)
            noisy_test = [prepare(episode(noisy_cfg, esc, seed), esc, noisy_cfg)
                          for seed in manifest['test']]
            noise_out = args.output / f'noise_{round(noise*100):02d}cm'
            noise_out.mkdir(exist_ok=True)
            print(f'Evaluacion sin reentrenar: ruido={noise*100:.0f} cm', flush=True)
            metrics = evaluate({'test': noisy_test}, models, noisy_cfg, esc, noise_out)
            noise_rows.extend(dict(position_noise_m=noise, **row) for row in metrics)
        write_csv(args.output/'noise_metrics.csv', noise_rows)
        plot_noise(noise_rows, args.output)
    print('Resultados: ' + str(args.output.resolve()), flush=True)


if __name__ == '__main__':
    main()

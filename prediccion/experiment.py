"""Entrenamiento por imitacion y evaluacion cerrada, con protocolo comun."""
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import ris_vlc as rv
from .data import Windows, splits
from .models import Controller
from .physics import angles, project, link, utility

KINDS = ('cnn_gru', 'coord_gru', 'cnn')


def setup(cfg):
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)


def train(kind, data, cfg, out):
    # Re-seed separately so adding/removing an ablation does not change another.
    setup(cfg)
    model = Controller(data['train'][0]['base'].shape[1], kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    generator = torch.Generator().manual_seed(cfg['seed'])
    loaders = {name: DataLoader(Windows(data[name], cfg), batch_size=cfg['batch_size'],
                               shuffle=name == 'train', generator=generator)
               for name in ('train', 'val')}
    best, log = float('inf'), []
    for epoch in range(cfg['epochs']):
        row = {'epoch': epoch+1}
        for name, loader in loaders.items():
            model.train(name == 'train')
            total, count = 0.0, 0
            with torch.set_grad_enabled(name == 'train'):
                for maps, coords, base, target in loader:
                    predicted = model(maps, coords, base)
                    # Scale radians for numerically useful loss magnitudes.
                    loss = ((predicted-target)/0.01).square().mean()
                    if name == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()
                    total += loss.item()*len(base)
                    count += len(base)
            row[name + '_mse_scaled'] = total/count
        log.append(row)
        if row['val_mse_scaled'] < best:
            best = row['val_mse_scaled']
            torch.save({'state_dict': model.state_dict(), 'kind': kind, 'config': cfg,
                        'epoch': epoch+1, 'validation_mse_scaled': best,
                        'n_angles': data['train'][0]['base'].shape[1]}, out / (kind+'.pt'))
        if epoch == 0 or (epoch+1) % 10 == 0:
            print(f"{kind}: epoch {epoch+1}/{cfg['epochs']}, "
                  f"train={row['train_mse_scaled']:.5f}, val={row['val_mse_scaled']:.5f}", flush=True)
    write_csv(out / (kind+'_training.csv'), log)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_models(out, cfg):
    result = {}
    for kind in KINDS:
        saved = torch.load(out/(kind+'.pt'), map_location='cpu', weights_only=True)
        # Reject accidental evaluation with a different experiment definition.
        if saved['config'] != cfg:
            raise ValueError('La configuracion no coincide con el checkpoint: ' + kind)
        model = Controller(saved['n_angles'], kind)
        model.load_state_dict(saved['state_dict'])
        result[kind] = model.eval()
    return result


def decide(method, ep, t, previous, esc, cfg, models):
    window = slice(t-cfg['history']+1, t+1)
    if method in models:
        inputs = [torch.from_numpy(a).unsqueeze(0) for a in
                  (ep['maps'][window], ep['coords'][window], ep['base'][t])]
        with torch.inference_mode():
            desired = models[method](*inputs).numpy()[0]
    elif method == 'reactive':
        desired = angles(esc, ep['observed'][t])
    else:
        # Linear least-squares velocity over the history reduces noise versus
        # a two-sample finite difference. Uses only observations <= t.
        history = ep['observed'][window]
        times = np.arange(len(history))*cfg['dt']
        centered = times-times.mean()
        velocity = (centered[:, None]*history).sum(0)/(centered**2).sum()
        predicted = np.clip(history[-1] + cfg['dt']*velocity,
                            [0.05, 0.05, esc.altura_dispositivo],
                            [esc.sala[0]-0.05, esc.sala[1]-0.05, esc.altura_dispositivo])
        desired = angles(esc, predicted)
        if method == 'geometric_j':
            # Finite candidate search, NOT a claim of a global optimum of J.
            # Unknown future orientation/blocker use zero-order hold.
            candidates = [previous, project(ep['base'][t], previous, cfg),
                          project(desired, previous, cfg)]
            scores = [utility(link(esc, predicted, ep['observed_normal'][t],
                                   ep['blocker'][t], theta, cfg)[0], theta, previous, cfg)
                      for theta in candidates]
            desired = candidates[int(np.argmax(scores))]
    return project(desired, previous, cfg)


def evaluate(data, models, cfg, esc, out):
    methods = ['reactive', 'constant_velocity', 'geometric_j', *KINDS]
    rows, traces, angular = [], [], {}
    # Warm up inference outside the recorded CPU latency samples.
    ep = data['test'][0]
    for method in KINDS:
        decide(method, ep, cfg['history']-1, project(ep['base'][0],
               np.zeros(2*esc.n_total), cfg), esc, cfg, models)
    for method in methods:
        theta_records = []
        for episode_id, ep in enumerate(data['test']):
            # Identical feasible initialization for every method; warm-up uses
            # only observed positions, respecting slew from a zero-angle panel.
            previous = np.zeros(2*esc.n_total)
            for t in range(cfg['history']-1):
                previous = project(ep['base'][t], previous, cfg)
            for t in range(cfg['history']-1, len(ep['position'])-1):
                start = time.perf_counter()
                theta = decide(method, ep, t, previous, esc, cfg, models)
                elapsed = (time.perf_counter()-start)*1000
                rate, h_irs, blocked = link(esc, ep['position'][t+1], ep['normal'][t+1],
                                           ep['blocker'][t+1], theta, cfg)
                angular_error = np.degrees(theta-ep['target'][t+1])
                trace = dict(method=method, episode=episode_id, seed=ep['seed'], t=t,
                             rate=rate, h_irs=h_irs, blocked=int(blocked),
                             movement_deg=float(np.degrees(np.abs(theta-previous)).sum()),
                             angle_rmse_deg=float(np.sqrt(np.mean(angular_error**2))),
                             decision_ms=elapsed, j=utility(rate, theta, previous, cfg))
                traces.append(trace)
                theta_records.append(theta.copy())
                previous = theta
        angular[method] = np.array(theta_records, dtype='float32')
        samples = [r for r in traces if r['method'] == method]
        rate = np.array([r['rate'] for r in samples])
        blocked = np.array([r['blocked'] for r in samples], bool)
        means = [np.mean([r['rate'] for r in samples if r['episode'] == i])
                 for i in range(len(data['test']))]
        rng = np.random.default_rng(cfg['seed'])
        boot = rng.choice(means, (2000, len(means)), replace=True).mean(axis=1)
        row = dict(method=method, mean_rate=float(rate.mean()),
                   rate_ci_low=float(np.percentile(boot, 2.5)),
                   rate_ci_high=float(np.percentile(boot, 97.5)),
                   p5_rate=float(np.percentile(rate, 5)),
                   outage_fraction=float(np.mean(rate < cfg['r_min'])),
                   blocked_fraction=float(blocked.mean()),
                   blocked_mean_rate=float(rate[blocked].mean()) if blocked.any() else None,
                   mean_h_irs=float(np.mean([r['h_irs'] for r in samples])),
                   mean_movement_deg=float(np.mean([r['movement_deg'] for r in samples])),
                   angle_rmse_deg=float(np.sqrt(np.mean([r['angle_rmse_deg']**2 for r in samples]))),
                   mean_decision_ms=float(np.mean([r['decision_ms'] for r in samples])),
                   p95_decision_ms=float(np.percentile([r['decision_ms'] for r in samples], 95)),
                   mean_j=float(np.mean([r['j'] for r in samples])))
        rows.append(row)
        print(f"{method}: R={row['mean_rate']:.4f}, P5={row['p5_rate']:.4f}, "
              f"outage={row['outage_fraction']:.3f}, RMSE={row['angle_rmse_deg']:.3f} deg", flush=True)
    write_csv(out/'metrics.csv', rows)
    write_csv(out/'traces.csv', traces)
    np.savez_compressed(out/'applied_angles.npz', **angular)
    # Save latent truth and measurements to make the generated experiment auditable.
    np.savez_compressed(out/'test_episodes.npz', **{
        f"episode_{i}_{key}": ep[key] for i, ep in enumerate(data['test'])
        for key in ('position', 'observed', 'normal', 'observed_normal', 'blocker')})
    report(rows, cfg, out)
    plot(rows, out)
    plot_episode(data['test'][0], traces, cfg, esc, out)
    return rows


def report(rows, cfg, out):
    text = ['# Evaluacion sintetica de control predictivo IRS-VLC', '',
            'Datos sinteticos; una sola geometria. No constituye validacion experimental.', '',
            'Entrenamiento: imitacion de angulos geometricos futuros, sin optimizacion diferenciable de J.',
            'Test separado por trayectorias; todas las decisiones usan observaciones hasta t y se aplican en t+1.',
            'CNN-GRU, GRU de coordenadas y CNN sin historial comparten un cabezal angular residual respecto de la geometria observada.',
            'Se evalua el canal original sin suavizar, agregando bloqueo exacto de los tres segmentos opticos.',
            'geometric_j selecciona entre mantener, geometria reactiva y extrapolacion; no es un optimizador global.', '',
            '| Metodo | Tasa media | P5 | Fraccion R<Rmin | RMSE angular (deg) | Decision media (ms) |',
            '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['method']} | {row['mean_rate']:.4f} | {row['p5_rate']:.4f} | "
                    f"{row['outage_fraction']:.4f} | {row['angle_rmse_deg']:.4f} | {row['mean_decision_ms']:.3f} |")
    text += ['', 'Tasas en bits/s/Hz. Movimiento = suma de cambios absolutos de los 200 angulos por intervalo.',
             'metrics.csv incluye intervalos de confianza bootstrap por episodio y latencia P95.',
             'La latencia mide inferencia/control/proyeccion en CPU con mapas ya preparados; excluye sensado, rasterizacion y bus.',
             'No se presupone que CNN-GRU supere los baselines. La tasa total puede ocultar diferencias del IRS cuando LoS domina.',
             'Los hiperparametros se seleccionan por error angular de validacion, nunca por test.', '',
             '## Parametros utilizados', '', '```json', json.dumps(cfg, indent=2), '```']
    (out/'report.md').write_text('\n'.join(text)+'\n', encoding='utf-8')


def plot(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, key, label in zip(axes, ['mean_rate', 'outage_fraction', 'angle_rmse_deg'],
                              ['Mean rate [bits/s/Hz]', 'Fraction below Rmin', 'Angle RMSE [deg]']):
        ax.bar([r['method'] for r in rows], [r[key] for r in rows])
        ax.set_ylabel(label)
        ax.tick_params(axis='x', rotation=55)
        ax.grid(axis='y', alpha=0.25)
    fig.savefig(out/'comparison.png', dpi=160)
    plt.close(fig)


def plot_noise(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for method in dict.fromkeys(r['method'] for r in rows):
        samples = [r for r in rows if r['method'] == method]
        x = [r['position_noise_m']*100 for r in samples]
        for ax, key in zip(axes, ['mean_rate', 'angle_rmse_deg']):
            ax.plot(x, [r[key] for r in samples], marker='o', label=method)
    for ax, label in zip(axes, ['Mean rate [bits/s/Hz]', 'Angle RMSE [deg]']):
        ax.set_xlabel('Position noise standard deviation [cm]')
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(out/'noise_comparison.png', dpi=160)
    plt.close(fig)


def plot_episode(ep, traces, cfg, esc, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    ax.plot(ep['position'][:, 0], ep['position'][:, 1], label='True user')
    ax.scatter(ep['observed'][:, 0], ep['observed'][:, 1], s=5, alpha=0.4, label='Observed user')
    ax.plot(ep['blocker'][:, 0], ep['blocker'][:, 1], '--', label='Blocker')
    ax.plot([0, 0], esc.irs_y, linewidth=4, label='IRS')
    ax.scatter(*esc.led_pos[:2], marker='*', s=100, label='LED')
    ax.set(xlim=(0, esc.sala[0]), ylim=(0, esc.sala[1]), xlabel='x [m]', ylabel='y [m]',
           title=f'Synthetic test episode 0, seed {ep["seed"]}')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    fig.savefig(out/'trajectory.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout='constrained')
    for method in dict.fromkeys(r['method'] for r in traces):
        rows = [r for r in traces if r['method'] == method and r['episode'] == 0]
        times = [(r['t']+1)*cfg['dt'] for r in rows]
        axes[0].plot(times, [r['rate'] for r in rows], label=method, alpha=0.8)
        axes[1].plot(times, [r['h_irs'] for r in rows], label=method, alpha=0.8)
    axes[0].axhline(cfg['r_min'], color='black', linestyle=':', label='Rmin')
    axes[0].set_ylabel('Rate [bits/s/Hz]')
    axes[0].legend(ncol=4, fontsize=8)
    axes[1].set(xlabel='Application time [s]', ylabel='IRS DC gain')
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(out/'time_series.png', dpi=160)
    plt.close(fig)

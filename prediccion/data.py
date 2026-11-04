"""Trayectorias independientes y observaciones causales, separadas por episodio."""
import numpy as np
import torch
from torch.utils.data import Dataset
import ris_vlc as rv
from .physics import angles


def episode(cfg, esc, seed):
    rng = np.random.default_rng(seed)
    count, dt = cfg['steps'], cfg['dt']
    position = np.empty((count, 3))
    position[0] = [*rng.uniform(0.6, esc.sala[:2] - 0.6), esc.altura_dispositivo]
    heading = rng.uniform(-np.pi, np.pi)
    speed = rng.uniform(0.3, cfg['max_speed_m_s'])
    blocker = np.empty((count, 3))
    blocker[0] = [*rng.uniform(0.5, esc.sala[:2] - 0.5), 0]
    bv = rng.normal(0, 0.25, 2)
    turn = 0.0
    for t in range(1, count):
        turn = 0.9 * turn + rng.normal(0, 0.16)
        heading += turn * dt
        if t % 25 == 0:
            speed = np.clip(speed + rng.normal(0, 0.15), 0.1, cfg['max_speed_m_s'])
        velocity = speed * np.array([np.cos(heading), np.sin(heading)])
        xy = position[t-1, :2] + dt * velocity
        for axis in range(2):
            if xy[axis] < 0.35 or xy[axis] > esc.sala[axis] - 0.35:
                velocity[axis] *= -1
                xy[axis] = position[t-1, axis] + dt * velocity[axis]
        heading = np.arctan2(velocity[1], velocity[0])
        position[t] = [*xy, esc.altura_dispositivo]
        bxy = blocker[t-1, :2] + dt * bv
        for axis in range(2):
            if bxy[axis] < 0.25 or bxy[axis] > esc.sala[axis] - 0.25:
                bv[axis] *= -1
        blocker[t] = [*np.clip(blocker[t-1, :2] + dt * bv, 0.25, esc.sala[:2] - 0.25), 0]
    # In half the episodes a nearby person follows the user, causing meaningful
    # blockage as well as unblocked intervals. Still only current sensing is used.
    if seed % 2 == 0:
        phase = rng.uniform(0, 2*np.pi)
        phi = phase + np.arange(count) * dt * 0.6
        blocker[:, :2] = np.clip(position[:, :2] + 0.32 * np.column_stack(
            [np.cos(phi), np.sin(phi)]), 0.2, esc.sala[:2] - 0.2)
    time = np.arange(count) * dt
    phase = rng.uniform(0, 2*np.pi, 2)
    normal = rv.unit(np.column_stack([0.15*np.sin(time*0.7 + phase[0]),
                                     0.15*np.sin(time*0.5 + phase[1]), np.ones(count)]))
    observed = position + rng.normal(0, cfg['position_noise_m'], position.shape)
    # Height is fixed and known in this initial scenario.
    observed[:, 2] = esc.altura_dispositivo
    observed[:, :2] = np.clip(observed[:, :2], 0.05, esc.sala[:2] - 0.05)
    observed_normal = rv.unit(normal + rng.normal(
        0, np.radians(cfg['orientation_noise_deg']), normal.shape))
    return dict(position=position, normal=normal, blocker=blocker,
                observed=observed, observed_normal=observed_normal, seed=seed)


def coordinates(ep, esc):
    return np.column_stack([ep['observed'] / esc.sala, ep['observed_normal'],
                            ep['blocker'][:, :2] / esc.sala[:2]]).astype('float32')


def maps(ep, esc, cfg):
    """Top-down 8 canales: LED, IRS/altura, usuario, nx,ny,nz, bloqueo, altura LED.

    Las alturas de usuario y panel son fijas para un experimento. El kernel
    gaussiano subpixel evita perder desplazamientos menores que una celda.
    """
    g = cfg['grid']
    xx, yy = np.meshgrid((np.arange(g)+0.5)*esc.sala[0]/g,
                         (np.arange(g)+0.5)*esc.sala[1]/g, indexing='ij')
    width = max(esc.sala[:2]) / g
    def blob(points, sigma):
        return np.exp(-((xx[None]-points[:, 0, None, None])**2
                        +(yy[None]-points[:, 1, None, None])**2)/(2*sigma**2))
    n = len(ep['observed'])
    out = np.zeros((n, 8, g, g), dtype='float32')
    out[:, 0] = blob(esc.led_pos[None], width)
    for mirror in esc.pos_espejos:
        # Preserve the fixed panel footprint and its maximum height per cell.
        i, j = np.minimum((mirror[:2] / esc.sala[:2] * g).astype(int), g-1)
        out[:, 1, i, j] = max(out[0, 1, i, j], mirror[2]/esc.sala[2])
    out[:, 2] = blob(ep['observed'], width)
    out[:, 3:6] = out[:, 2:3] * ep['observed_normal'][:, :, None, None]
    out[:, 6] = blob(ep['blocker'], cfg['blocker_radius_m'])
    out[:, 7] = out[:, 0] * esc.led_pos[2]/esc.sala[2]
    return out


def prepare(ep, esc, cfg):
    return dict(**ep, maps=maps(ep, esc, cfg), coords=coordinates(ep, esc),
                base=np.array([angles(esc, p) for p in ep['observed']], dtype='float32'),
                target=np.array([angles(esc, p) for p in ep['position']], dtype='float32'))


def splits(cfg, esc):
    groups, cursor = {}, cfg['seed']
    for name in ['train', 'val', 'test']:
        count = cfg[name + '_episodes']
        groups[name] = [prepare(episode(cfg, esc, s), esc, cfg)
                        for s in range(cursor, cursor+count)]
        cursor += count
    return groups


class Windows(Dataset):
    def __init__(self, episodes, cfg):
        self.episodes, self.history = episodes, cfg['history']
        self.index = [(i, t) for i, ep in enumerate(episodes)
                      for t in range(self.history-1, len(ep['position'])-1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, t = self.index[index]
        ep = self.episodes[i]
        window = slice(t-self.history+1, t+1)
        return tuple(torch.from_numpy(a) for a in
                     (ep['maps'][window], ep['coords'][window], ep['base'][t], ep['target'][t+1]))

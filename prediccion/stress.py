"""Perturbaciones cinematicas emparejadas para evaluar sin reentrenar.

Cada variante conserva la trayectoria del bloqueador original (aunque en el
generador normal este siga al usuario), el ruido XY y el prefijo antes del
evento. Esto permite aislar la perturbacion del receptor. ``stationary`` es un
control aparte: su posicion es constante desde el inicio del episodio.

``fall_proxy`` es un descenso e inclinacion sinteticos, no una simulacion
biomecanica ni un detector de caidas. La altura se supone conocida al observar
cada muestra: el residual geometrico y coord_gru la reciben; los mapas CNN no
codifican la altura del receptor. Los controladores existentes no se modifican:
constant_velocity y la prediccion interna de geometric_j mantienen la altura
nominal, una limitacion especialmente relevante para este caso fuera de dominio.
"""

import numpy as np

import ris_vlc as rv
from .data import episode


SCENARIOS = ('normal', 'sudden_stop', 'stationary', 'erratic', 'fall_proxy')
FALL_DURATION_S = 0.5
FALL_HEIGHT_M = 0.25
ERRATIC_TURN_INTERVAL_S = (0.2, 0.35)


def scenario_metadata(cfg, esc, event_index=None):
    """Parametros comunes; el evento es la primera muestra perturbada.

    La decision para esa muestra se toma en event_index-1 y, por tanto, no
    puede conocer la perturbacion. El descenso termina tras fall_steps pasos
    desde la ultima muestra intacta; fall_end_index es inclusivo.
    """
    event = cfg['steps'] // 2 if event_index is None else event_index
    if not isinstance(event, (int, np.integer)) or isinstance(event, bool):
        raise ValueError('event_index debe ser entero')
    if not cfg['history'] <= event < cfg['steps']:
        raise ValueError('Se requiere history <= event_index < steps')
    fall_steps = max(1, int(round(FALL_DURATION_S / cfg['dt'])))
    return {
        'event_index': int(event),
        'event_time_s': float(event * cfg['dt']),
        'last_unperturbed_time_s': float((event-1) * cfg['dt']),
        'fall_steps': fall_steps,
        'fall_duration_s': float(fall_steps * cfg['dt']),
        'fall_end_index': int(event + fall_steps - 1),
        'fall_initial_height_m': float(esc.altura_dispositivo),
        'fall_final_height_m': FALL_HEIGHT_M,
        'erratic_turn_interval_s': list(ERRATIC_TURN_INTERVAL_S),
        'blocker_policy': 'original trajectory unchanged in every scenario',
        'position_noise_policy': 'reuse original XY observation residual, then clip',
        'orientation_noise_policy': 'rotate original observed and true normals together',
        'height_observation_policy': 'current true height is known without noise',
    }


def _reflect_step(point, velocity, dt, low, high):
    """Reflexion especular en bordes, incluso para mas de un cruce por paso."""
    width = high - low
    phase = np.mod(point + dt*velocity - low, 2*width)
    reflected = low + np.where(phase <= width, phase, 2*width-phase)
    velocity = velocity * np.where(phase <= width, 1.0, -1.0)
    return reflected, velocity


def _rotate_observation(observed, old_normal, new_normal):
    """Rotacion minima que conserva exactamente el error angular observado."""
    cross = np.cross(old_normal, new_normal)
    dot = float(np.clip(np.dot(old_normal, new_normal), -1, 1))
    if dot < -1 + 1e-10:
        # Complete the definition for antipodal normals, although the downward
        # height proxy rotates an upward normal to a horizontal one instead.
        axis = np.cross(old_normal, np.eye(3)[np.argmin(abs(old_normal))])
        axis = rv.unit(axis)
        return rv.unit(2*axis*np.dot(axis, observed) - observed)
    return rv.unit(observed + np.cross(cross, observed)
                   + np.cross(cross, np.cross(cross, observed))/(1+dot))


def make_scenario(cfg, esc, seed, scenario, event_index=None):
    """Devuelve un episodio bruto compatible con ``data.prepare``.

    ``normal`` es exactamente ``data.episode``. Para las perturbaciones desde
    el evento, todas las entradas antes de event_index son identicas al normal.
    Ningun peso, configuracion del modelo o controlador se altera aqui.
    """
    if scenario not in SCENARIOS:
        raise ValueError('Escenario desconocido: ' + str(scenario))
    meta = scenario_metadata(cfg, esc, event_index)
    original = episode(cfg, esc, seed)
    if scenario == 'normal':
        return original
    ep = {key: value.copy() if isinstance(value, np.ndarray) else value
          for key, value in original.items()}
    event, dt = meta['event_index'], cfg['dt']
    position = ep['position']
    first_changed = event

    if scenario == 'stationary':
        position[:] = position[0].copy()
        first_changed = 0
    elif scenario == 'sudden_stop':
        position[event:] = position[event-1].copy()
    elif scenario == 'erratic':
        # Separate RNG: selecting a scenario cannot change the original world
        # or sensor noise. Direction changes by 90-180 degrees at each turn.
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 43671]))
        velocity = (position[event-1, :2] - position[event-2, :2])/dt
        speed = min(float(np.linalg.norm(velocity)), cfg['max_speed_m_s'])
        if speed < 1e-12:
            speed = min(0.5, cfg['max_speed_m_s'])
            velocity = np.array([speed, 0.0])
        min_interval = max(1, int(np.ceil(ERRATIC_TURN_INTERVAL_S[0]/dt)))
        max_interval = max(min_interval, int(np.floor(ERRATIC_TURN_INTERVAL_S[1]/dt + 1e-10)))
        next_turn = event
        for t in range(event, len(position)):
            if t == next_turn:
                heading = np.arctan2(velocity[1], velocity[0])
                heading += rng.choice([-1, 1])*rng.uniform(np.pi/2, np.pi)
                velocity = speed*np.array([np.cos(heading), np.sin(heading)])
                next_turn = t + int(rng.integers(min_interval, max_interval+1))
            position[t, :2], velocity = _reflect_step(
                position[t-1, :2], velocity, dt, np.full(2, 0.35), esc.sala[:2]-0.35)
    else:  # fall_proxy
        if esc.altura_dispositivo <= FALL_HEIGHT_M:
            raise ValueError('fall_proxy requiere altura_dispositivo > 0.25 m')
        steps = meta['fall_steps']
        if meta['fall_end_index'] >= len(position):
            raise ValueError('No quedan muestras suficientes para completar fall_proxy')
        origin = position[event-1].copy()
        direction = position[event-1, :2] - position[event-2, :2]
        if np.linalg.norm(direction) < 1e-12:
            direction = np.array([1.0, 0.0])
        final_normal = np.r_[rv.unit(direction), 0.0]
        start_normal = ep['normal'][event-1].copy()
        angle = np.arccos(np.clip(np.dot(start_normal, final_normal), -1, 1))
        for t in range(event, len(position)):
            u = min((t-event+1)/steps, 1.0)
            weight = u*u*(3-2*u)  # Continuous position/tilt with zero endpoint slope.
            position[t] = origin
            position[t, 2] = origin[2] + weight*(FALL_HEIGHT_M-origin[2])
            if weight == 1:
                ep['normal'][t] = final_normal
            else:
                ep['normal'][t] = rv.unit(
                    np.sin((1-weight)*angle)*start_normal + np.sin(weight*angle)*final_normal)
            ep['observed_normal'][t] = _rotate_observation(
                original['observed_normal'][t], original['normal'][t], ep['normal'][t])

    # Original XY locations stay away from sensor clipping boundaries. Reuse
    # their sensor residual and clip only after applying it to the new path.
    residual = original['observed'][first_changed:, :2] - original['position'][first_changed:, :2]
    ep['observed'][first_changed:, :2] = np.clip(
        position[first_changed:, :2] + residual, 0.05, esc.sala[:2]-0.05)
    ep['observed'][first_changed:, 2] = position[first_changed:, 2]
    return ep

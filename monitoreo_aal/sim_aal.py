"""
Simulación de rutina, sensado no invasivo y eventos anómalos para el
escenario AAL. El canal óptico se toma sin cambios de ris_vlc.py.
"""
import numpy as np
import ris_vlc as rv

CFG = dict(
    dt=0.1,                 # s, intervalo de sensado y decisión
    steps=900,              # 90 s por episodio
    history=10,             # 1 s de historial
    horizon=10,             # 1 s de predicción
    grid=20,
    pos_noise=0.05,         # m, error del estimador de posición por sombra
    device_offset=0.20,     # m, terminal sostenido delante del cuerpo
    body_radius=0.15,
    body_height=1.60,
    device_height=0.85,
    slew_deg_s=120.0,
    angle_lim_deg=80.0,
    r_min=5.0,
    t_ev_range=(20.0, 60.0),  # s, ventana donde puede ocurrir el evento
)

# Lugares habituales: posición y duración de la estadía [s]
LUGARES = {
    'cama':   dict(xy=(4.00, 4.00), estadia=(6.0, 14.0)),
    'bano':   dict(xy=(0.80, 4.30), estadia=(4.0, 9.0)),
    'sillon': dict(xy=(0.90, 1.20), estadia=(6.0, 14.0)),
    'mesa':   dict(xy=(2.00, 2.80), estadia=(5.0, 12.0)),
    'cocina': dict(xy=(3.90, 0.90), estadia=(4.0, 10.0)),
}
NOMBRES = list(LUGARES)
POI = np.array([LUGARES[n]['xy'] for n in NOMBRES])

# Rutina: probabilidad de ir de un lugar a otro
TRANSICION = {
    'cama':   {'bano': 0.6, 'cocina': 0.2, 'sillon': 0.2},
    'bano':   {'cama': 0.4, 'sillon': 0.3, 'mesa': 0.3},
    'sillon': {'cocina': 0.35, 'mesa': 0.35, 'bano': 0.3},
    'mesa':   {'cocina': 0.4, 'sillon': 0.4, 'bano': 0.2},
    'cocina': {'mesa': 0.6, 'sillon': 0.3, 'cama': 0.1},
}

ANOMALIAS = ['caida', 'detencion', 'deambulacion', 'marcha_lenta']
ETIQUETAS = {'normal': 'Normal', 'caida': 'Fall', 'detencion': 'Prolonged stop',
             'deambulacion': 'Wandering', 'marcha_lenta': 'Slow gait'}
LUGARES_EN = {'cama': 'bed', 'bano': 'bathroom', 'sillon': 'armchair', 'mesa': 'table', 'cocina': 'kitchen'}

SALA = np.array([5.0, 5.0])
CENTRO = SALA / 2


def _desvio_ruta(a, b):
    """Desvío lateral habitual de cada ruta: fijo para el par (a, b)."""
    i, j = sorted((NOMBRES.index(a), NOMBRES.index(b)))
    return 0.35 * np.sin(1.7 * i + 2.9 * j + 0.5)


def _ruta(rng, a, b, p0, p1):
    """Curva de Bézier cuadrática que sigue la ruta habitual entre dos lugares."""
    mid = (p0 + p1) / 2
    d = p1 - p0
    perp = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    w = mid + 0.3 * (CENTRO - mid) + _desvio_ruta(a, b) * perp + rng.normal(0, 0.06, 2)
    u = np.linspace(0, 1, 300)[:, None]
    pts = (1 - u) ** 2 * p0 + 2 * (1 - u) * u * w + u ** 2 * p1
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return pts, s


def _sobre_ruta(pts, s, x):
    k = int(np.clip(np.searchsorted(s, x) - 1, 0, len(s) - 2))
    f = (x - s[k]) / (s[k + 1] - s[k] + 1e-12)
    p = pts[k] + f * (pts[k + 1] - pts[k])
    tang = pts[k + 1] - pts[k]
    return p, tang / (np.linalg.norm(tang) + 1e-12)


def _suave(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def generar_episodio(rng, anomalia=None, cfg=CFG):
    """
    Devuelve la trayectoria verdadera del cuerpo, la del terminal VLC y la
    observación ruidosa que obtiene la infraestructura. Si se pide una
    anomalía, ocurre una sola vez a mitad de un trayecto habitual.
    """
    T, dt = cfg['steps'], cfg['dt']
    acc, dec = 0.6, 1.0
    v0 = rng.uniform(0.55, 0.75)                  # velocidad habitual del ocupante
    lugar = rng.choice(NOMBRES)
    anchor = POI[NOMBRES.index(lugar)] + rng.normal(0, 0.08, 2)
    modo, timer = 'estadia', rng.uniform(0.5, LUGARES[lugar]['estadia'][1])
    ou = np.zeros(2)
    heading = rng.uniform(-np.pi, np.pi)
    factor = 1.0
    t_trigger = rng.uniform(*cfg['t_ev_range']) if anomalia else np.inf
    t_ev = None

    body = np.empty((T, 2)); head = np.empty(T)
    dev_z = np.full(T, cfg['device_height']); tilt = np.zeros(T); tilt_az = np.zeros(T)
    body_h = np.full(T, cfg['body_height'])
    caida = None

    for t in range(T):
        tiempo = t * dt
        if modo == 'estadia':
            ou = 0.9 * ou + rng.normal(0, 0.006, 2)
            pos = anchor + ou
            timer -= dt
            if timer <= 0:
                opciones = TRANSICION[lugar]
                destino = rng.choice(list(opciones), p=list(opciones.values()))
                p1 = POI[NOMBRES.index(destino)] + rng.normal(0, 0.08, 2)
                pts, s_acum = _ruta(rng, lugar, destino, anchor, p1)
                L, s, v = s_acum[-1], 0.0, 0.0
                vt = v0 * factor * rng.normal(1.0, 0.07)
                modo = 'camino'
        elif modo == 'camino':
            v_obj = min(vt, np.sqrt(2 * acc * max(L - s, 0.0)))
            v = max(v + np.clip(v_obj - v, -dec * dt, acc * dt), 0.05)
            s = min(s + v * dt, L)
            p, tang = _sobre_ruta(pts, s_acum, s)
            heading = np.arctan2(tang[1], tang[0])
            lateral = 0.015 * np.sin(2 * np.pi * 1.8 * tiempo) * np.array([-tang[1], tang[0]])
            pos = p + lateral
            if (anomalia and t_ev is None and tiempo >= t_trigger
                    and 0.3 <= s / L <= 0.7):
                t_ev = t
                if anomalia == 'caida':
                    ang = heading + rng.uniform(-0.7, 0.7)
                    caida = dict(p0=pos.copy(), d=rng.uniform(0.5, 0.8),
                                 u=np.array([np.cos(ang), np.sin(ang)]), k=0,
                                 tilt=np.radians(rng.uniform(60, 90)), az=ang)
                    modo, tilt_now, z_now = 'cayendo', 0.0, cfg['device_height']
                elif anomalia == 'detencion':
                    modo = 'frenando'
                elif anomalia == 'deambulacion':
                    modo, w, vw = 'deambula', 0.0, rng.uniform(0.3, 0.45)
                elif anomalia == 'marcha_lenta':
                    factor = rng.uniform(0.40, 0.55)
                    vt *= factor
            if modo == 'camino' and s >= L - 1e-6:
                modo, lugar, anchor = 'estadia', destino, pos.copy()
                timer = rng.uniform(*LUGARES[lugar]['estadia'])
                ou = np.zeros(2)
        elif modo == 'cayendo':
            caida['k'] += 1
            f = _suave(caida['k'] / 8)
            pos = caida['p0'] + f * caida['d'] * caida['u']
            tilt_now, z_now = f * caida['tilt'], cfg['device_height'] - f * 0.65
            if caida['k'] >= 8:
                modo, quieto = 'caido', pos.copy()
        elif modo == 'caido':
            pos = quieto + rng.normal(0, 0.003, 2)
        elif modo == 'frenando':
            v = max(v - dec * dt, 0.0)
            s = min(s + v * dt, L)
            pos, _ = _sobre_ruta(pts, s_acum, s)
            if v == 0.0:
                modo, quieto, ou = 'quieto', pos.copy(), np.zeros(2)
        elif modo == 'quieto':
            ou = 0.9 * ou + rng.normal(0, 0.004, 2)
            pos = quieto + ou
        elif modo == 'deambula':
            w = 0.85 * w + rng.normal(0, 0.9)
            heading += w * dt
            vel = vw * np.array([np.cos(heading), np.sin(heading)])
            nueva = pos + vel * dt
            for ax in range(2):
                if nueva[ax] < 0.4 or nueva[ax] > SALA[ax] - 0.4:
                    vel[ax] *= -1
                    heading = np.arctan2(vel[1], vel[0])
            pos = np.clip(pos + vel * dt, 0.3, SALA - 0.3)

        body[t] = pos
        head[t] = heading
        if modo in ('cayendo', 'caido'):
            if modo == 'cayendo':
                tilt[t], dev_z[t] = tilt_now, z_now
            else:
                tilt[t], dev_z[t] = caida['tilt'], cfg['device_height'] - 0.65
            tilt_az[t] = caida['az']
            body_h[t] = cfg['body_height'] - _suave(caida['k'] / 8) * 1.25

    if anomalia and t_ev is None:          # no hubo trayecto apto: se repite
        return generar_episodio(rng, anomalia, cfg)

    # Terminal VLC sostenido delante del cuerpo; inclinación natural pequeña
    tiempo = np.arange(T) * dt
    fase = rng.uniform(0, 2 * np.pi, 2)
    nat = np.column_stack([0.12 * np.sin(0.7 * tiempo + fase[0]),
                           0.12 * np.sin(0.5 * tiempo + fase[1]), np.ones(T)])
    nat = rv.unit(nat)
    caida_n = np.column_stack([np.sin(tilt) * np.cos(tilt_az),
                               np.sin(tilt) * np.sin(tilt_az), np.cos(tilt)])
    dev_normal = rv.unit(np.where(tilt[:, None] > 0, caida_n, nat))
    dev_xy = np.clip(body + cfg['device_offset'] * np.column_stack(
        [np.cos(head), np.sin(head)]), 0.05, SALA - 0.05)
    dev = np.column_stack([dev_xy, dev_z])

    # Lo único que observa la infraestructura: posición (x, y) estimada por sombra
    obs = body + rng.normal(0, cfg['pos_noise'], body.shape)

    return dict(body=body, heading=head, dev=dev, dev_normal=dev_normal,
                body_h=body_h, obs=obs, t_ev=t_ev, tipo=anomalia or 'normal')


def generar(seed, n, anomalia=None, cfg=CFG):
    rng = np.random.default_rng(seed)
    return [generar_episodio(rng, anomalia, cfg) for _ in range(n)]

"""Evaluación de la alerta al cuidador y de la capa de comunicación."""
import numpy as np
import ris_vlc as rv
from sim_aal import CFG
import modelo_aal as m

METODOS = ['cnn_gru', 'cv', 'regla']
NOMBRE_METODO = {'cnn_gru': 'CNN–GRU residual', 'cv': 'Constant-velocity residual',
                 'regla': 'Immobility rule'}


def puntajes(e, modelo, cfg=CFG):
    pm = m.predicciones_modelo(modelo, e, cfg)
    pc = m.predicciones_cv(e, cfg)
    return dict(pm=pm, pc=pc,
                cnn_gru=m.suavizar(m.residuo(pm, e['obs'], cfg)),
                cv=m.suavizar(m.residuo(pc, e['obs'], cfg)),
                regla=m.puntaje_inmovilidad(e, cfg))


def alarmas(score, tau, refractario):
    """Índices donde el puntaje cruza el umbral, con período refractario."""
    idx, ultimo = [], -np.inf
    arriba = np.nan_to_num(score, nan=-np.inf) >= tau
    for t in np.flatnonzero(arriba & ~np.r_[False, arriba[:-1]]):
        if t - ultimo >= refractario:
            idx.append(t); ultimo = t
    return idx


def falsas_por_hora(eps, sc, metodo, tau, cfg=CFG, refractario_s=30.0):
    """Alarmas en episodios normales y en el tramo previo al evento de los anómalos."""
    n, horas = 0, 0.0
    for e, s in zip(eps, sc):
        fin = e['t_ev'] if e['t_ev'] is not None else len(e['obs'])
        n += len(alarmas(s[metodo][:fin], tau, int(refractario_s / cfg['dt'])))
        horas += fin * cfg['dt'] / 3600
    return n / horas


def deteccion(eps, sc, metodo, tau, cfg=CFG, ventana_s=10.0):
    """Fracción de eventos detectados dentro de la ventana y retardos [s]."""
    w = int(ventana_s / cfg['dt'])
    det, ret = 0, []
    for e, s in zip(eps, sc):
        seg = np.nan_to_num(s[metodo][e['t_ev']:e['t_ev'] + w], nan=-np.inf)
        k = np.flatnonzero(seg >= tau)
        if len(k):
            det += 1; ret.append(k[0] * cfg['dt'])
    return det / len(eps), ret


def umbral_calibrado(eps_cal, sc_cal, metodo, margen=1.0):
    """Máximo puntaje observado en el historial normal del ocupante."""
    return margen * max(np.nanmax(s[metodo]) for s in sc_cal)


# ---------------------------------------------------------------------
# Capa de comunicación
# ---------------------------------------------------------------------

def _rumbo(d, previo, minimo=0.15):
    n = np.linalg.norm(d)
    return d / n if n > minimo else previo


def evaluar_enlace(esc, e, s, cfg=CFG, desde=None):
    """
    Tasa por intervalo con los espejos apuntados por cada estrategia. Todas
    usan solo la posición (x, y) estimada por la infraestructura; la altura y
    la orientación del terminal se desconocen (se asume 0.85 m). El canal se
    evalúa con la posición, orientación y bloqueo verdaderos.
    """
    obs, dev, nrm, body, bh = e['obs'], e['dev'], e['dev_normal'], e['body'], e['body_h']
    pm, pc = s['pm'], s['pc']
    T, dt = len(obs), cfg['dt']
    paso = np.radians(cfg['slew_deg_s']) * dt
    lim = np.radians(cfg['angle_lim_deg'])
    off, z = cfg['device_offset'], cfg['device_height']
    t0 = desde if desde is not None else cfg['history'] + cfg['horizon']
    ctrls = ['ideal', 'reactivo', 'cv', 'cnn_gru']
    ang = {c: np.array(rv.angulos_optimos(esc, [*obs[t0], z])) for c in ctrls}
    u_obs = u_pm = np.array([1.0, 0.0])
    tasas = {c: [] for c in ctrls}
    for t in range(t0, T - 1):
        u_obs = _rumbo(obs[t - 2:t + 1].mean(0) - obs[t - 9:t - 6].mean(0), u_obs)
        u_pm = _rumbo(pm[t, 2] - obs[t - 6:t - 3].mean(0), u_pm)
        objetivo = {
            'ideal': dev[t + 1],
            'reactivo': [*(obs[t] + off * u_obs), z],
            'cv': [*(pc[t, 0] + off * u_obs), z],
            'cnn_gru': [*(pm[t, 0] + off * u_pm), z],
        }
        bloq = rv.bloquea_los(dev[t + 1], esc.led_pos, [*body[t + 1], 0.0],
                              radio=cfg['body_radius'], altura=bh[t + 1])
        for c in ctrls:
            obj = np.array(rv.angulos_optimos(esc, objetivo[c]))
            if c != 'ideal':
                obj = np.clip(ang[c] + np.clip(obj - ang[c], -paso, paso), -lim, lim)
            ang[c] = obj
            normales = rv.normal_desde_angulos(obj[0], obj[1])
            h = rv.ganancia_total(esc, dev[t + 1], nrm[t + 1], normales, bloq)
            tasas[c].append(rv.tasa(esc, h))
    return {c: np.array(v) for c, v in tasas.items()}

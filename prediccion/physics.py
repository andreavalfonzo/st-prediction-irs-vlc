"""Control factible y visibilidad por trayecto, reutilizando el canal original."""
import numpy as np
import ris_vlc as rv


def angles(esc, position):
    """Orden intercalado [alpha_1,beta_1,...], convencion de ris_vlc."""
    return np.stack(rv.angulos_optimos(esc, position), axis=-1).reshape(-1)


def project(desired, previous, cfg):
    lo = np.tile(np.radians(cfg['angle_min_deg']), len(previous) // 2)
    hi = np.tile(np.radians(cfg['angle_max_deg']), len(previous) // 2)
    step = np.radians(cfg['slew_deg_s']) * cfg['dt']
    return np.clip(desired, np.maximum(lo, previous - step),
                   np.minimum(hi, previous + step))


def segment_blocked(start, end, center, radius, height):
    """Interseccion EXACTA segmento-cilindro cerrado (base z=0). Vectorizable.

    Primero se intersecta el intervalo del segmento con el slab 0<=z<=h;
    luego se minimiza la distancia horizontal en ese intervalo. Usar solo el
    punto horizontal mas cercano, como el simulador inicial, omite casos en
    los que el segmento entra al cilindro por debajo de su tapa.
    """
    start, end = np.broadcast_arrays(np.asarray(start, float), np.asarray(end, float))
    direction = end - start
    dz = direction[..., 2]
    safe = np.where(abs(dz) < 1e-15, 1.0, dz)
    za, zb = -start[..., 2] / safe, (height - start[..., 2]) / safe
    lo, hi = np.maximum(0, np.minimum(za, zb)), np.minimum(1, np.maximum(za, zb))
    flat = abs(dz) < 1e-15
    inside = (start[..., 2] >= 0) & (start[..., 2] <= height)
    lo, hi = np.where(flat, 0, lo), np.where(flat, 1, hi)
    valid = (lo <= hi) & (~flat | inside)
    offset = start[..., :2] - np.asarray(center)[..., :2]
    delta = direction[..., :2]
    norm2 = np.sum(delta * delta, axis=-1)
    t = -np.sum(offset * delta, axis=-1) / np.where(norm2 > 0, norm2, 1)
    t = np.minimum(np.maximum(t, lo), hi)
    dist2 = np.sum((offset + t[..., None] * delta) ** 2, axis=-1)
    return valid & (dist2 <= radius ** 2)


def link(esc, position, normal, blocker, theta, cfg):
    """Canal sin suavizar: bloqueo LED-PD, LED-espejo y espejo-PD."""
    radius, height = cfg['blocker_radius_m'], cfg['blocker_height_m']
    blocked = bool(segment_blocked(esc.led_pos, position, blocker, radius, height))
    mirror_angles = np.asarray(theta).reshape(-1, 2)
    normals = rv.normal_desde_angulos(mirror_angles[:, 0], mirror_angles[:, 1])
    hk = rv.ganancia_irs(esc, position, normal, normals, devolver_por_espejo=True)
    hidden = segment_blocked(esc.led_pos, esc.pos_espejos, blocker, radius, height)
    hidden |= segment_blocked(esc.pos_espejos, position, blocker, radius, height)
    h_irs = float(np.sum(np.where(hidden, 0, hk)))
    h_los = 0.0 if blocked else float(rv.ganancia_los(esc, position, normal))
    return float(rv.tasa(esc, h_los + h_irs)), h_irs, blocked


def utility(rate, theta, previous, cfg):
    return (rate - cfg['lambda_s'] * np.abs(theta - previous).sum()
            - cfg['lambda_q'] * max(0.0, cfg['r_min'] - rate))

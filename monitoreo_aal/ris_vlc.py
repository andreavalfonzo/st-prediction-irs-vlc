"""
ris_vlc.py — Modelo de canal VLC asistido por IRS de espejos.

Sistema de coordenadas
----------------------
Origen en una esquina del piso. Ejes x, y sobre el piso; z hacia arriba.
    - LED en el techo, apuntando hacia abajo: normal (0, 0, -1).
    - IRS sobre la pared x = 0, normal nominal (1, 0, 0) hacia el interior.
    - Receptor a la altura del dispositivo, normal nominal (0, 0, 1).

Convención de ángulos del espejo
--------------------------------
La normal de cada espejo se parametriza en esféricas respecto de la normal
nominal de la pared:
    n = (cos(beta) cos(alpha), sin(beta) cos(alpha), sin(alpha))
con alpha la elevación (positivo = inclina hacia arriba) y beta el azimut
(positivo = gira hacia +y). Ambos son cero cuando el espejo mira
perpendicularmente hacia el interior de la sala.

Criterio de acoplamiento
------------------------
Un espejo plano forma una imagen virtual puntual de la fuente detrás de su
plano. El detector recibe potencia si la recta que une esa imagen virtual con
el detector atraviesa la apertura física del espejo. La tolerancia angular
queda entonces fijada por el tamaño del espejo, no por el del fotodiodo.
El espejo se aproxima por un disco de área equivalente.

Unidades: SI (metros, watts, radianes salvo donde se indique en grados).
"""

import numpy as np

# ---------------------------------------------------------------------
# Utilidades geométricas
# ---------------------------------------------------------------------

def unit(v):
    """Vector unitario. Acepta (3,) o (..., 3)."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norm == 0, 1.0, norm)


def lambertian_order(half_angle_deg):
    """Orden lambertiano m a partir del semiángulo de media potencia."""
    phi_half = np.radians(half_angle_deg)
    return -np.log(2.0) / np.log(np.cos(phi_half))


def cos_between(a, b):
    """Coseno del ángulo entre dos vectores. Numéricamente estable."""
    return np.clip(np.sum(unit(a) * unit(b), axis=-1), -1.0, 1.0)


def angle_between(a, b):
    """Ángulo en radianes entre dos vectores."""
    return np.arccos(cos_between(a, b))


# ---------------------------------------------------------------------
# Parámetros del escenario
# ---------------------------------------------------------------------

class Escenario:
    """Configuración física del sistema. Todo lo que no varía en una corrida."""

    def __init__(self,
                 sala=(5.0, 5.0, 3.0),
                 led_pos=(2.5, 2.5, 3.0),
                 semiangulo_deg=60.0,
                 potencia=20.0,          # W ópticos
                 area_pd=1e-4,           # m^2 (1 cm^2)
                 fov_deg=60.0,
                 n_refr=1.5,             # índice del concentrador
                 t_filtro=1.0,           # transmitancia del filtro óptico
                 responsividad=0.5,      # A/W
                 ruido_var=1e-14,        # A^2
                 # --- IRS ---
                 irs_y=(1.5, 3.5),       # extensión en y del panel
                 irs_z=(1.5, 2.5),       # extensión en z del panel
                 n_espejos=10,           # por lado; total N = n_espejos^2
                 reflectividad=0.95,
                 radio_fuente=5e-3,      # m; extensión física del emisor
                 altura_dispositivo=0.85):

        self.sala = np.array(sala, dtype=float)
        self.led_pos = np.array(led_pos, dtype=float)
        self.led_normal = np.array([0.0, 0.0, -1.0])
        self.semiangulo_deg = semiangulo_deg
        self.m = lambertian_order(semiangulo_deg)
        self.potencia = potencia
        self.area_pd = area_pd
        self.fov = np.radians(fov_deg)
        self.n_refr = n_refr
        self.t_filtro = t_filtro
        self.responsividad = responsividad
        self.ruido_var = ruido_var

        self.irs_y = irs_y
        self.irs_z = irs_z
        self.n_espejos = n_espejos
        self.reflectividad = reflectividad
        self.radio_fuente = radio_fuente
        self.altura_dispositivo = altura_dispositivo

        self.pos_espejos = self._construir_grilla()

    # -- geometría del arreglo -----------------------------------------

    def _construir_grilla(self):
        """Centros de los espejos sobre la pared x = 0."""
        n = self.n_espejos
        y0, y1 = self.irs_y
        z0, z1 = self.irs_z
        ys = y0 + (np.arange(n) + 0.5) * (y1 - y0) / n
        zs = z0 + (np.arange(n) + 0.5) * (z1 - z0) / n
        YY, ZZ = np.meshgrid(ys, zs, indexing="ij")
        pos = np.stack([np.zeros_like(YY), YY, ZZ], axis=-1)
        return pos.reshape(-1, 3)                      # (N, 3)

    @property
    def n_total(self):
        return self.n_espejos ** 2

    @property
    def area_panel(self):
        return (self.irs_y[1] - self.irs_y[0]) * (self.irs_z[1] - self.irs_z[0])

    @property
    def area_espejo(self):
        """Área de un espejo. El área total del panel se mantiene constante."""
        return self.area_panel / self.n_total

    @property
    def radio_espejo(self):
        """Radio del disco de área equivalente al espejo."""
        return np.sqrt(self.area_espejo / np.pi)

    def ganancia_concentrador(self, psi):
        """g(psi) = n^2 / sin^2(Psi_c), nula fuera del campo de visión."""
        g = self.n_refr ** 2 / np.sin(self.fov) ** 2
        return np.where(psi <= self.fov, g, 0.0)


# ---------------------------------------------------------------------
# Enlace directo
# ---------------------------------------------------------------------

def ganancia_los(esc, rx_pos, rx_normal):
    """Ganancia DC del enlace LED -> fotodiodo."""
    rx_pos = np.asarray(rx_pos, dtype=float)
    d_vec = rx_pos - esc.led_pos
    d = np.linalg.norm(d_vec, axis=-1)

    cos_phi = cos_between(esc.led_normal, d_vec)       # irradiancia en el LED
    psi = angle_between(rx_normal, -d_vec)             # incidencia en el PD

    h = ((esc.m + 1) * esc.area_pd / (2 * np.pi * d ** 2)
         * np.maximum(cos_phi, 0.0) ** esc.m
         * np.cos(psi)
         * esc.t_filtro
         * esc.ganancia_concentrador(psi))

    return np.where((psi <= esc.fov) & (cos_phi > 0), h, 0.0)


# ---------------------------------------------------------------------
# Orientación de los espejos
# ---------------------------------------------------------------------

def normales_optimas(esc, rx_pos):
    """
    Normal de cada espejo que refleja el rayo del LED hacia el receptor:
    vector mitad entre las direcciones al emisor y al detector.
    """
    rx_pos = np.asarray(rx_pos, dtype=float)
    hacia_led = unit(esc.led_pos - esc.pos_espejos)
    hacia_rx = unit(rx_pos - esc.pos_espejos)
    return unit(hacia_led + hacia_rx)


def angulos_desde_normal(n):
    """(alpha, beta) a partir de la normal. Inversa de normal_desde_angulos."""
    n = unit(n)
    alpha = np.arcsin(np.clip(n[..., 2], -1.0, 1.0))
    beta = np.arctan2(n[..., 1], n[..., 0])
    return alpha, beta


def normal_desde_angulos(alpha, beta):
    """Normal del espejo a partir de sus ángulos de control."""
    alpha = np.atleast_1d(alpha)
    beta = np.atleast_1d(beta)
    return unit(np.stack([np.cos(beta) * np.cos(alpha),
                          np.sin(beta) * np.cos(alpha),
                          np.sin(alpha)], axis=-1))


def angulos_optimos(esc, rx_pos):
    """Atajo: ángulos de control óptimos para una posición del receptor."""
    return angulos_desde_normal(normales_optimas(esc, rx_pos))


def reflejar(direccion, normal):
    """Ley de reflexión especular: r = d - 2 (d . n) n."""
    d = unit(direccion)
    n = unit(normal)
    return unit(d - 2 * np.sum(d * n, axis=-1, keepdims=True) * n)


def aplicar_error_apuntamiento(alpha, beta, sigma_deg, rng):
    """Ruido gaussiano independiente sobre cada ángulo de control."""
    if sigma_deg <= 0:
        return alpha, beta
    s = np.radians(sigma_deg)
    return (alpha + rng.normal(0.0, s, size=np.shape(alpha)),
            beta + rng.normal(0.0, s, size=np.shape(beta)))


# ---------------------------------------------------------------------
# Camino vía IRS
# ---------------------------------------------------------------------

def _acopla(esc, rx_pos, normales):
    """
    Test de apertura. El espejo forma una imagen virtual puntual del LED;
    hay acoplamiento si la recta imagen-detector cruza la apertura del espejo.

    Devuelve además la fracción de solape, que vale 1 en el centro y decae
    linealmente hasta 0 en el borde del disco equivalente. Esto representa el
    recorte progresivo del haz cuando el punto de cruce se acerca al borde.
    """
    R = esc.pos_espejos                                  # (N,3)
    n = unit(normales)                                   # (N,3)
    S = esc.led_pos
    D = np.asarray(rx_pos, dtype=float)

    # Imagen virtual del LED tras el plano de cada espejo
    d_plano = np.sum((S - R) * n, axis=-1, keepdims=True)
    S_img = S - 2.0 * d_plano * n                        # (N,3)

    # Cruce de la recta S_img -> D con el plano del espejo
    direc = D - S_img
    denom = np.sum(direc * n, axis=-1)
    num = np.sum((R - S_img) * n, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = num / denom
    P = S_img + t[:, None] * direc
    offset = np.linalg.norm(P - R, axis=-1)

    dentro = np.isfinite(offset) & (t > 0.0) & (t < 1.0)
    frac = np.clip(1.0 - offset / esc.radio_espejo, 0.0, 1.0)
    return np.where(dentro, frac, 0.0)


def _cota_flujo(esc, d1, d2, cos_phi, psi):
    """
    Cota de conservación de energía. Un espejo no puede entregar al detector
    más potencia de la que recolecta, repartida sobre la huella del haz
    reflejado. Con una fuente de radio finito r_s y un espejo de radio a, la
    huella sobre el plano del receptor tiene radio

        R = a (1 + d2/d1) + r_s d2/d1

    El primer término es la proyección de la apertura y el segundo el
    desenfoque por el tamaño del emisor, que domina cuando el espejo es
    pequeño. Esta cota hace que la ganancia sature al subdividir el panel, en
    lugar de crecer de forma indefinida.
    """
    a = esc.radio_espejo
    r_huella = a * (1.0 + d2 / d1) + esc.radio_fuente * d2 / d1

    irr_espejo = ((esc.m + 1) * np.maximum(cos_phi, 0.0) ** esc.m
                  / (2 * np.pi * d1 ** 2))               # por unidad de potencia
    captado = irr_espejo * esc.area_espejo * esc.reflectividad
    fraccion = esc.area_pd * np.cos(psi) / (np.pi * r_huella ** 2)

    return captado * np.minimum(fraccion, 1.0) * esc.t_filtro * esc.ganancia_concentrador(psi)


def ganancia_irs(esc, rx_pos, rx_normal, normales, devolver_por_espejo=False):
    """Ganancia DC aportada por el arreglo de espejos."""
    rx_pos = np.asarray(rx_pos, dtype=float)
    normales = unit(normales)

    v_led = esc.pos_espejos - esc.led_pos                # LED -> espejo
    d1 = np.linalg.norm(v_led, axis=-1)
    v_rx = rx_pos - esc.pos_espejos                      # espejo -> receptor
    d2 = np.linalg.norm(v_rx, axis=-1)

    cos_phi = cos_between(esc.led_normal, v_led)         # irradiancia del LED
    psi = angle_between(rx_normal, -v_rx)                # incidencia en el PD

    h = (esc.reflectividad * (esc.m + 1) * esc.area_pd
         / (2 * np.pi * (d1 + d2) ** 2)
         * np.maximum(cos_phi, 0.0) ** esc.m
         * np.cos(psi)
         * esc.t_filtro
         * esc.ganancia_concentrador(psi))

    # Ninguna trayectoria puede superar la potencia que el espejo recolecta
    h = np.minimum(h, _cota_flujo(esc, d1, d2, cos_phi, psi))

    # El LED debe iluminar la cara frontal del espejo
    ilumina = np.sum(unit(-v_led) * normales, axis=-1) > 0

    h_k = h * _acopla(esc, rx_pos, normales)
    h_k = np.where(ilumina & (psi <= esc.fov) & (cos_phi > 0), h_k, 0.0)

    return h_k if devolver_por_espejo else float(np.sum(h_k))


# ---------------------------------------------------------------------
# Bloqueo
# ---------------------------------------------------------------------

def bloquea_los(rx_pos, led_pos, centro_bloqueador, radio=0.15, altura=1.8):
    """
    Cilindro vertical que representa a una persona. Devuelve True si el
    segmento LED-receptor lo atraviesa.
    """
    p0 = np.asarray(led_pos, dtype=float)
    p1 = np.asarray(rx_pos, dtype=float)
    c = np.asarray(centro_bloqueador, dtype=float)

    # Distancia mínima en planta entre el segmento y el eje del cilindro
    a, b = p0[:2], p1[:2]
    ab = b - a
    denom = np.dot(ab, ab)
    t = 0.0 if denom == 0 else np.clip(np.dot(c[:2] - a, ab) / denom, 0.0, 1.0)
    dist = np.linalg.norm(a + t * ab - c[:2])
    if dist > radio:
        return False
    z_cruce = p0[2] + t * (p1[2] - p0[2])
    return z_cruce <= altura


# ---------------------------------------------------------------------
# Canal total y métricas
# ---------------------------------------------------------------------

def ganancia_total(esc, rx_pos, rx_normal, normales, bloqueado=False):
    """H_total = (1 - B) H_LoS + H_IRS."""
    h_los = 0.0 if bloqueado else float(ganancia_los(esc, rx_pos, rx_normal))
    return h_los + ganancia_irs(esc, rx_pos, rx_normal, normales)


def snr(esc, h):
    """SNR eléctrica para modulación de intensidad y detección directa."""
    return (esc.responsividad * np.asarray(h) * esc.potencia) ** 2 / esc.ruido_var


def tasa(esc, h):
    """
    Cota inferior de capacidad para IM/DD, en bits/s/Hz.
    No se usa Shannon porque la señal es real y no negativa.
    """
    return 0.5 * np.log2(1.0 + (np.e / (2 * np.pi)) * snr(esc, h))

"""
verificacion.py — Pruebas del modelo de canal RIS-VLC.

Cada prueba compara el simulador contra un resultado conocido sin simular.
Si alguna falla, no confíes en ningún barrido posterior.

Ejecutar:  python3 verificacion.py
"""

import numpy as np
import ris_vlc as rv

# arccos pierde precisión cerca de cero: el error de redondeo en el coseno
# (~1e-16) se traduce en ~1e-6 grados de error angular. La tolerancia debe
# estar por encima de ese piso.
TOL_ANG_DEG = 1e-4


def ok(nombre, condicion, detalle=""):
    print(f"  [{'PASA' if condicion else 'FALLA'}] {nombre}"
          + (f"  -> {detalle}" if detalle else ""))
    return bool(condicion)


def test_1_caso_analitico():
    """Receptor bajo el LED, ambos apuntándose: phi = psi = 0."""
    print("\n1. Enlace LoS en configuración axial")
    esc = rv.Escenario()
    rx = np.array([esc.led_pos[0], esc.led_pos[1], esc.altura_dispositivo])
    h_sim = float(rv.ganancia_los(esc, rx, [0.0, 0.0, 1.0]))

    d = esc.led_pos[2] - esc.altura_dispositivo
    g = esc.n_refr ** 2 / np.sin(esc.fov) ** 2
    h_ana = (esc.m + 1) * esc.area_pd / (2 * np.pi * d ** 2) * esc.t_filtro * g

    err = abs(h_sim - h_ana) / h_ana
    return ok("coincide con la fórmula cerrada", err < 1e-12,
              f"error relativo = {err:.2e}")


def test_2_corte_fov():
    """Fuera del campo de visión la ganancia debe ser exactamente cero."""
    print("\n2. Corte por campo de visión")
    esc = rv.Escenario(fov_deg=30.0)
    rx = np.array([esc.led_pos[0], esc.led_pos[1], esc.altura_dispositivo])

    t = np.radians(80.0)                       # dispositivo muy inclinado
    h_fuera = float(rv.ganancia_los(esc, rx, [np.sin(t), 0.0, np.cos(t)]))
    h_dentro = float(rv.ganancia_los(esc, rx, [0.0, 0.0, 1.0]))

    return (ok("nula fuera del FoV", h_fuera == 0.0)
            & ok("no nula dentro del FoV", h_dentro > 0.0))


def test_3_ley_cuadrado():
    """Al duplicar la distancia la ganancia cae a un cuarto."""
    print("\n3. Ley del inverso del cuadrado")
    esc = rv.Escenario()
    x, y, zt = esc.led_pos
    n_rx = [0.0, 0.0, 1.0]
    h1 = float(rv.ganancia_los(esc, [x, y, zt - 1.0], n_rx))
    h2 = float(rv.ganancia_los(esc, [x, y, zt - 2.0], n_rx))

    razon = h1 / h2
    return ok("razón de ganancias igual a 4", abs(razon - 4.0) < 1e-12,
              f"razón = {razon:.9f}")


def test_4_alineamiento_espejo():
    """La normal óptima refleja el rayo del LED hacia el receptor."""
    print("\n4. Orientación óptima de los espejos")
    esc = rv.Escenario()
    rx = np.array([2.0, 2.5, esc.altura_dispositivo])

    n_opt = rv.normales_optimas(esc, rx)
    reflejado = rv.reflejar(esc.pos_espejos - esc.led_pos, n_opt)
    desvio = np.degrees(rv.angle_between(reflejado, rx - esc.pos_espejos))
    r1 = ok("todos los espejos apuntan al receptor",
            np.max(desvio) < TOL_ANG_DEG, f"desvío máximo = {np.max(desvio):.2e}°")

    alpha, beta = rv.angulos_desde_normal(n_opt)
    err = np.max(np.abs(rv.normal_desde_angulos(alpha, beta) - n_opt))
    r2 = ok("conversión normal <-> ángulos es consistente", err < 1e-12,
            f"error máximo = {err:.2e}")

    return r1 & r2


def test_5_degradacion_suave():
    """
    El acoplamiento debe decaer de forma gradual con el desalineamiento, con
    una tolerancia del orden del radio angular del espejo, no del fotodiodo.
    """
    print("\n5. Degradación gradual ante desalineamiento")
    esc = rv.Escenario()
    rx = np.array([2.0, 2.5, esc.altura_dispositivo])
    n_rx = np.array([0.0, 0.0, 1.0])
    alpha, beta = rv.angulos_optimos(esc, rx)

    desv = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    h = np.array([rv.ganancia_irs(esc, rx, n_rx,
                                  rv.normal_desde_angulos(alpha + np.radians(d), beta))
                  for d in desv])

    monotona = np.all(np.diff(h) <= 1e-18)
    r1 = ok("la ganancia no crece al desalinear", monotona)
    r2 = ok("un error de 0.1° conserva parte del enlace", h[1] > 0.5 * h[0],
            f"{100 * h[1] / h[0]:.1f}% de la ganancia óptima")
    r3 = ok("un error de 5° extingue el enlace", h[-1] == 0.0)

    print("       desvío [°]:", "  ".join(f"{d:6.2f}" for d in desv))
    print("       H_IRS/H_opt:", "  ".join(f"{v / h[0]:6.3f}" for v in h))
    return r1 & r2 & r3


def test_6_conservacion_bloqueo():
    """Bajo bloqueo el término directo desaparece y el del IRS sobrevive."""
    print("\n6. Bloqueo de la línea de vista")
    esc = rv.Escenario()
    rx = np.array([2.0, 2.5, esc.altura_dispositivo])
    n_rx = np.array([0.0, 0.0, 1.0])
    n_opt = rv.normales_optimas(esc, rx)

    h_libre = rv.ganancia_total(esc, rx, n_rx, n_opt, bloqueado=False)
    h_bloq = rv.ganancia_total(esc, rx, n_rx, n_opt, bloqueado=True)
    h_irs = rv.ganancia_irs(esc, rx, n_rx, n_opt)

    r1 = ok("bloqueado equivale a la sola contribución del IRS",
            abs(h_bloq - h_irs) < 1e-18)
    r2 = ok("el IRS sostiene el enlace tras el bloqueo", h_bloq > 0.0,
            f"conserva el {100 * h_bloq / h_libre:.1f}% de la ganancia")

    # Enlace oblicuo: el rayo baja lo suficiente como para ser obstruido
    rx_lejos = np.array([1.0, 2.5, esc.altura_dispositivo])
    r3 = ok("detecta un bloqueador interpuesto en un enlace oblicuo",
            rv.bloquea_los(rx_lejos, esc.led_pos, [1.3, 2.5, 0.0]))
    r4 = ok("ignora un bloqueador fuera de la trayectoria",
            not rv.bloquea_los(rx_lejos, esc.led_pos, [1.3, 4.0, 0.0]))

    # Un enlace casi vertical pasa por encima de la cabeza del bloqueador
    r5 = ok("un enlace cercano al nadir no se obstruye",
            not rv.bloquea_los(rx, esc.led_pos, [2.25, 2.5, 0.0]))
    return r1 & r2 & r3 & r4 & r5


if __name__ == "__main__":
    print("=" * 64)
    print("Verificación del modelo de canal RIS-VLC")
    print("=" * 64)

    resultados = [test_1_caso_analitico(),
                  test_2_corte_fov(),
                  test_3_ley_cuadrado(),
                  test_4_alineamiento_espejo(),
                  test_5_degradacion_suave(),
                  test_6_conservacion_bloqueo()]

    print("\n" + "=" * 64)
    fallas = resultados.count(False)
    print(f"Las {len(resultados)} pruebas pasaron." if fallas == 0
          else f"ATENCIÓN: {fallas} de {len(resultados)} fallaron.")
    print("=" * 64)

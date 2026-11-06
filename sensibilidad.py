"""
sensibilidad.py — Análisis de sensibilidad del modelo de canal RIS-VLC.

Genera las figuras en PDF vectorial dentro de figuras/, y un
resumen cuantitativo por consola.

Ejecutar:  python3 sensibilidad.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ris_vlc as rv

SALIDA = "figuras"
SEMILLA = 20260926

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "figure.constrained_layout.use": True,
})

UNA_COL = (3.5, 2.6)
UNA_COL_ALTA = (3.5, 4.4)
DOS_COL = (7.16, 2.5)

RX_REF = np.array([1.5, 2.5, 0.85])       # receptor de referencia
N_RX_REF = np.array([0.0, 0.0, 1.0])      # dispositivo horizontal
BLOQUEADOR = np.array([2.0, 2.5, 0.0])    # persona entre el LED y el receptor
RADIO_BLOQ = 0.20                         # m; medio ancho del torso


def guardar(fig, nombre):
    os.makedirs(SALIDA, exist_ok=True)
    ruta = os.path.join(SALIDA, nombre)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {ruta}")


def rejilla_receptores(esc, n=21):
    """Posiciones de muestreo sobre el plano del dispositivo."""
    xs = np.linspace(0.4, esc.sala[0] - 0.4, n)
    ys = np.linspace(0.4, esc.sala[1] - 0.4, n)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([XX.ravel(), YY.ravel(),
                     np.full(XX.size, esc.altura_dispositivo)], axis=-1)


def tasa_media_sala(esc, usar_irs):
    """Tasa promediada sobre el plano del dispositivo."""
    pts = rejilla_receptores(esc)
    if usar_irs:
        vals = [rv.tasa(esc, rv.ganancia_irs(esc, p, N_RX_REF,
                                             rv.normales_optimas(esc, p)))
                for p in pts]
    else:
        vals = [rv.tasa(esc, float(rv.ganancia_los(esc, p, N_RX_REF)))
                for p in pts]
    return float(np.mean(vals))


def sigma_media_ganancia(esc, rx=RX_REF, n_real=120, rng=None):
    """
    Error de apuntamiento, en grados, que reduce la ganancia del IRS a la
    mitad. Búsqueda por bisección sobre el valor medio de N realizaciones.
    """
    rng = rng or np.random.default_rng(SEMILLA)
    alpha, beta = rv.angulos_optimos(esc, rx)
    h0 = rv.ganancia_irs(esc, rx, N_RX_REF, rv.normal_desde_angulos(alpha, beta))
    if h0 <= 0:
        return np.nan

    def media(sigma):
        acc = 0.0
        for _ in range(n_real):
            a, b = rv.aplicar_error_apuntamiento(alpha, beta, sigma, rng)
            acc += rv.ganancia_irs(esc, rx, N_RX_REF, rv.normal_desde_angulos(a, b))
        return acc / n_real / h0

    lo, hi = 0.0, 8.0
    if media(hi) > 0.5:
        return hi
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if media(mid) > 0.5:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------
# 1. Semiángulo de media potencia
# ---------------------------------------------------------------------

def barrido_semiangulo():
    print("  [1/7] Half-power semiangle")
    semis = np.linspace(15.0, 75.0, 25)
    punto, media = [], []
    for s in semis:
        esc = rv.Escenario(semiangulo_deg=s)
        punto.append(rv.tasa(esc, float(rv.ganancia_los(esc, RX_REF, N_RX_REF))))
        media.append(tasa_media_sala(esc, usar_irs=False))
    punto, media = np.array(punto), np.array(media)

    fig, ax = plt.subplots(figsize=UNA_COL)
    ax.plot(semis, punto, label="Reference Receiver")
    ax.plot(semis, media, "--", label="Room Average")
    ax.set_xlabel(r"Semiángulo de media potencia $\phi_{1/2}$ [degrees]")
    ax.set_ylabel("Direct link rate [bits/s/Hz]")
    ax.legend()
    guardar(fig, "sens_semiangulo.pdf")
    return semis, punto, media


# ---------------------------------------------------------------------
# 2. Campo de visión del receptor
# ---------------------------------------------------------------------

def barrido_fov():
    print("  [2/7] Field of View")
    fovs = np.linspace(20.0, 85.0, 27)
    punto, media, via_irs = [], [], []
    for f in fovs:
        esc = rv.Escenario(fov_deg=f)
        punto.append(rv.tasa(esc, float(rv.ganancia_los(esc, RX_REF, N_RX_REF))))
        media.append(tasa_media_sala(esc, usar_irs=False))
        via_irs.append(rv.tasa(esc, rv.ganancia_irs(
            esc, RX_REF, N_RX_REF, rv.normales_optimas(esc, RX_REF))))
    punto, media, via_irs = map(np.array, (punto, media, via_irs))

    fig, ax = plt.subplots(figsize=UNA_COL)
    ax.plot(fovs, punto, label="Direct, reference receiver")
    ax.plot(fovs, media, "--", label="Direct, room average")
    ax.plot(fovs, via_irs, ":", label="With IRS, reference receiver")
    ax.set_xlabel(r"Campo de visión $\Psi_c$ [degrees]")
    ax.set_ylabel("achievable rate [bits/s/Hz]")
    ax.legend()
    guardar(fig, "sens_fov.pdf")
    return fovs, punto, media, via_irs


# ---------------------------------------------------------------------
# 3. Número de espejos: ganancia frente a tolerancia
# ---------------------------------------------------------------------

def barrido_n_espejos():
    print("  [3/7] number of mirrors")
    lados = np.array([2, 3, 5, 7, 10, 14, 20, 28, 40, 56, 80])
    rng = np.random.default_rng(SEMILLA)
    n_tot, tasas, sigmas, radios = [], [], [], []
    for n in lados:
        esc = rv.Escenario(n_espejos=int(n))
        h = rv.ganancia_irs(esc, RX_REF, N_RX_REF, rv.normales_optimas(esc, RX_REF))
        n_tot.append(esc.n_total)
        tasas.append(rv.tasa(esc, h))
        sigmas.append(sigma_media_ganancia(esc, n_real=60, rng=rng))
        radios.append(esc.radio_espejo)
    n_tot = np.array(n_tot)
    tasas, sigmas, radios = map(np.array, (tasas, sigmas, radios))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=UNA_COL_ALTA, sharex=True)
    ax1.plot(n_tot, tasas, marker="o", markersize=3)
    ax1.set_xscale("log")
    ax1.set_ylabel("IRS rate [bits/s/Hz]")
    ax1.set_title("(a) Gain")

    ax2.plot(n_tot, sigmas, marker="s", markersize=3, color="C1")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Número de espejos $N$ (área del panel constante)")
    ax2.set_ylabel(r"$\sigma$ que reduce $H_{\mathrm{IRS}}$ a la mitad [$^\circ$]")
    ax2.set_title("(b) Tolerance to pointing error")
    guardar(fig, "sens_n_espejos.pdf")
    return n_tot, tasas, sigmas, radios


# ---------------------------------------------------------------------
# 4. Reflectividad
# ---------------------------------------------------------------------

def barrido_reflectividad():
    print("  [4/7] reflectivity")
    rhos = np.linspace(0.5, 1.0, 51)
    tasas = np.array([
        rv.tasa(esc := rv.Escenario(reflectividad=r),
                rv.ganancia_irs(esc, RX_REF, N_RX_REF,
                                rv.normales_optimas(esc, RX_REF)))
        for r in rhos])

    fig, ax = plt.subplots(figsize=UNA_COL)
    ax.plot(rhos, tasas)
    ax.set_xlabel(r"Reflectividad del espejo $\rho$")
    ax.set_ylabel("IRS rate [bits/s/Hz]")
    guardar(fig, "sens_reflectividad.pdf")
    return rhos, tasas


# ---------------------------------------------------------------------
# 5. Error de apuntamiento
# ---------------------------------------------------------------------

def barrido_error_apuntamiento(n_real=400):
    print("  [5/7] pointing error")
    rng = np.random.default_rng(SEMILLA)
    esc = rv.Escenario()
    alpha, beta = rv.angulos_optimos(esc, RX_REF)
    h_ref = rv.ganancia_irs(esc, RX_REF, N_RX_REF,
                            rv.normal_desde_angulos(alpha, beta))

    sigmas = np.linspace(0.0, 2.5, 26)
    media, p10, p90 = [], [], []
    for s in sigmas:
        m = np.array([rv.ganancia_irs(
            esc, RX_REF, N_RX_REF,
            rv.normal_desde_angulos(*rv.aplicar_error_apuntamiento(alpha, beta, s, rng)))
            for _ in range(n_real)]) / h_ref
        media.append(m.mean())
        p10.append(np.percentile(m, 10))
        p90.append(np.percentile(m, 90))
    media, p10, p90 = map(np.array, (media, p10, p90))

    fig, ax = plt.subplots(figsize=UNA_COL)
    ax.fill_between(sigmas, p10, p90, alpha=0.2, linewidth=0,
                    label="Percentiles 10--90")
    ax.plot(sigmas, media, label="Media")
    ax.set_xlabel(r"Desviación del error de apuntamiento $\sigma$ [degrees]")
    ax.set_ylabel(r"$H_{\mathrm{IRS}}$ normalizada")
    ax.set_ylim(0, 1.05)
    ax.legend()
    guardar(fig, "sens_error_apuntamiento.pdf")
    return sigmas, media


# ---------------------------------------------------------------------
# 6. Cobertura con un bloqueador presente
# ---------------------------------------------------------------------

def mapa_cobertura(paso=0.10):
    print("  [6/7] coverage map")
    esc = rv.Escenario()
    xs = np.arange(paso / 2, esc.sala[0], paso)
    ys = np.arange(paso / 2, esc.sala[1], paso)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    t_libre = np.zeros_like(XX)
    t_bloq = np.zeros_like(XX)
    t_bloq_irs = np.zeros_like(XX)
    mascara_bloq = np.zeros(XX.shape, dtype=bool)

    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            rx = np.array([XX[i, j], YY[i, j], esc.altura_dispositivo])
            h_los = float(rv.ganancia_los(esc, rx, N_RX_REF))
            b = rv.bloquea_los(rx, esc.led_pos, BLOQUEADOR, radio=RADIO_BLOQ)
            mascara_bloq[i, j] = b
            h_irs = rv.ganancia_irs(esc, rx, N_RX_REF, rv.normales_optimas(esc, rx))
            t_libre[i, j] = rv.tasa(esc, h_los)
            t_bloq[i, j] = rv.tasa(esc, 0.0 if b else h_los)
            t_bloq_irs[i, j] = rv.tasa(esc, (0.0 if b else h_los) + h_irs)

    vmax = max(t_libre.max(), t_bloq_irs.max())
    fig, axes = plt.subplots(1, 3, figsize=DOS_COL)
    paneles = [(t_libre, "(a) No blocking, just direct"),
               (t_bloq, "(b) Blocking, just direct"),
               (t_bloq_irs, "(c) Blocking, direct + IRS")]
    for ax, (Z, titulo) in zip(axes, paneles):
        im = ax.imshow(Z.T, origin="lower", aspect="equal", vmin=0, vmax=vmax,
                       extent=[0, esc.sala[0], 0, esc.sala[1]], cmap="viridis")
        ax.plot([0, 0], esc.irs_y, color="white", linewidth=3, solid_capstyle="butt")
        ax.plot(esc.led_pos[0], esc.led_pos[1], "w*", markersize=7)
        ax.plot(BLOQUEADOR[0], BLOQUEADOR[1], "wx", markersize=6, markeredgewidth=1.4)
        ax.set_title(titulo)
        ax.set_xlabel("x [m]")
        ax.grid(False)
    axes[0].set_ylabel("y [m]")
    fig.colorbar(im, ax=axes, label="Rate [bits/s/Hz]", shrink=0.9)
    guardar(fig, "mapa_cobertura.pdf")
    return t_libre, t_bloq, t_bloq_irs, mascara_bloq


# ---------------------------------------------------------------------
# 7. Retardo de actuación
# ---------------------------------------------------------------------

def barrido_retardo(velocidades=(0.5, 1.0, 1.5), t_max=0.30, n_pasos=40):
    print("  [7/7] attenuation delay")
    esc = rv.Escenario()
    retardos = np.linspace(0.0, t_max, 31)
    y0, z0 = 2.5, esc.altura_dispositivo
    x_ini, x_fin = 1.0, 3.5

    ganancia, tasa_med = {}, {}
    for v in velocidades:
        g_v, t_v = [], []
        for tau in retardos:
            hs, ts = [], []
            for x in np.linspace(x_ini, x_fin, n_pasos):
                rx_real = np.array([x, y0, z0])
                # El controlador solo conoce dónde estaba el usuario hace tau.
                # La trayectoria se supone iniciada antes de la ventana medida.
                rx_visto = np.array([x - v * tau, y0, z0])
                h = rv.ganancia_irs(esc, rx_real, N_RX_REF,
                                    rv.normales_optimas(esc, rx_visto))
                hs.append(h)
                ts.append(rv.tasa(esc, h))
            g_v.append(np.mean(hs))
            t_v.append(np.mean(ts))
        ganancia[v] = np.array(g_v) / g_v[0]
        tasa_med[v] = np.array(t_v)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=UNA_COL_ALTA, sharex=True)
    for v in velocidades:
        ax1.plot(retardos * 1e3, ganancia[v], label=f"$v$ = {v:.1f} m/s")
        ax2.plot(retardos * 1e3, tasa_med[v], label=f"$v$ = {v:.1f} m/s")
    ax1.set_ylabel(r"$H_{\mathrm{IRS}}$ normalizada")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("(a) Channel gain")
    ax1.legend()
    ax2.set_xlabel(r"Retardo de actuación $\tau$ [ms]")
    ax2.set_ylabel("Media rate [bits/s/Hz]")
    ax2.set_title("(b) achievable rate")
    guardar(fig, "sens_retardo.pdf")
    return retardos, ganancia, tasa_med


# ---------------------------------------------------------------------

def main():
    print("Sensitivity Analysis of the RIS-VLC Channel")
    print("-" * 56)

    semis, p_s, m_s = barrido_semiangulo()
    fovs, p_f, m_f, i_f = barrido_fov()
    n_tot, t_n, sig_n, rad_n = barrido_n_espejos()
    rhos, t_rho = barrido_reflectividad()
    sig, med_sig = barrido_error_apuntamiento()
    t_libre, t_bloq, t_bloq_irs, masc = mapa_cobertura()
    retardos, g_tau, t_tau = barrido_retardo()

    esc = rv.Escenario()
    print("\n" + "-" * 56)
    print("Quantitative Summary")
    print("-" * 56)
    print(f"Room {esc.sala[0]:.0f}x{esc.sala[1]:.0f}x{esc.sala[2]:.0f} m; "
          f"panel {esc.area_panel:.2f} m^2; N = {esc.n_total} mirrors "
          f"(equivalent radio {esc.radio_espejo * 1e3:.0f} mm)")
    print(f"Reference receiver {RX_REF.tolist()}, "
          f"blockage in {BLOQUEADOR[:2].tolist()}")

    print(f"\n[1] Optimal semi-angle at the reference point: "
          f"{semis[np.argmax(p_s)]:.0f}°")
    print(f"    Optimal average semi-angle for the room:  "
          f"{semis[np.argmax(m_s)]:.0f}°")

    print(f"\n[2] Optimal field of view, right on target: {fovs[np.argmax(p_f)]:.0f}°; "
          f"room average: {fovs[np.argmax(m_f)]:.0f}°; "
          f"via IRS: {fovs[np.argmax(i_f)]:.0f}°")

    print(f"\n[3] Gain and tolerance relative to N:")
    for n, t, s, r in zip(n_tot, t_n, sig_n, rad_n):
        print(f"      N = {n:5d}  radio = {r * 1e3:6.1f} mm  "
              f"rate = {t:5.2f} b/s/Hz  sigma_50 = {s:5.2f}°")

    print(f"\n[4] Reflectivity: The Rate Falls "
          f"{t_rho[-1] - t_rho[0]:.2f} b/s/Hz as rho changes from 1.0 to 0.5")

    i50 = int(np.argmin(np.abs(med_sig - 0.5)))
    print(f"\n[5] Pointing error that cuts the gain in half: "
          f"sigma = {sig[i50]:.2f}°")

    n_bloq = int(np.count_nonzero(masc))
    if n_bloq:
        rec = 100.0 * np.mean(t_bloq_irs[masc] > 0.5 * t_libre[masc])
        print(f"\n[6] The blocker cuts off the line of sight to the "
              f"{100.0 * n_bloq / masc.size:.1f}% of the area.")
        print(f"    In that area, the IRS collects at least half of the rate "
              f"original in {rec:.0f}% of the points.")
        print(f"    Average rate in the blocked area: "
              f"{t_bloq[masc].mean():.2f} without IRS  ->  "
              f"{t_bloq_irs[masc].mean():.2f} with IRS  [b/s/Hz]")

    print("\n[7] Actuation delay:")
    for v in g_tau:
        g, t = g_tau[v], t_tau[v]
        bajo = np.where(g < 0.5)[0]
        tau50 = retardos[bajo[0]] * 1e3 if bajo.size else float("nan")
        print(f"      v = {v:.1f} m/s: H drops to "
              f"{100 * np.interp(0.050, retardos, g):.0f}% with tau = 50 ms; "
              f"to 50% with tau = {tau50:.0f} ms; "
              f"rate {t[0]:.2f} -> {np.interp(0.050, retardos, t):.2f} b/s/Hz")


if __name__ == "__main__":
    main()

"""
tabla_parametros.py — Emite la tabla de parámetros del escenario en LaTeX.

La tabla se genera a partir de los valores por defecto de ris_vlc.Escenario,
de modo que el informe no pueda desincronizarse del simulador.

Ejecutar:  python3 tabla_parametros.py > tabla_parametros.tex
"""

import numpy as np
import ris_vlc as rv

PLANTILLA = r"""% Generado por tabla_parametros.py -- no editar a mano.
% Requiere: \usepackage{{tabularx}}
\begin{{table}}[t]
\caption{{Parámetros del escenario de referencia}}
\label{{tab:parametros}}
\centering
\scriptsize
\renewcommand{{\arraystretch}}{{1.25}}
\begin{{tabularx}}{{\columnwidth}}{{|X|p{{1.5cm}}|p{{2.0cm}}|}}
\hline
\textbf{{Parámetro}} & \textbf{{Símbolo}} & \textbf{{Valor}} \\
\hline
\multicolumn{{3}}{{|l|}}{{\textit{{Recinto y emisor}}}} \\
\hline
Dimensiones de la sala & --- & ${sx:.0f} \times {sy:.0f} \times {sz:.0f}$ m \\
\hline
Posición del LED & $S$ & $({lx:.1f}, {ly:.1f}, {lz:.1f})$ m \\
\hline
Semiángulo de media potencia & $\phi_{{1/2}}$ & ${semi:.0f}^\circ$ \\
\hline
Orden lambertiano & $m$ & ${m:.2f}$ \\
\hline
Potencia óptica transmitida & $P$ & ${pot:.0f}$ W \\
\hline
Radio efectivo del emisor & $r_s$ & ${rs:.0f}$ mm \\
\hline
\multicolumn{{3}}{{|l|}}{{\textit{{Receptor}}}} \\
\hline
Área del fotodiodo & $A$ & ${apd:.0f}$ cm$^2$ \\
\hline
Campo de visión & $\Psi_c$ & ${fov:.0f}^\circ$ \\
\hline
Índice del concentrador & $n_r$ & ${nr:.1f}$ \\
\hline
Transmitancia del filtro & $T_s$ & ${ts:.1f}$ \\
\hline
Responsividad & $R$ & ${resp:.1f}$ A/W \\
\hline
Altura del dispositivo & --- & ${hdev:.2f}$ m \\
\hline
Varianza de ruido & $\sigma^2$ & ${ruido:.0e} A$^2$ \\
\hline
\multicolumn{{3}}{{|l|}}{{\textit{{Superficie reflectante}}}} \\
\hline
Extensión del panel & --- & ${py:.1f} \times {pz:.1f}$ m \\
\hline
Número de espejos & $N$ & ${ntot:d}$ \\
\hline
Radio equivalente por espejo & $a$ & ${rad:.0f}$ mm \\
\hline
Reflectividad & $\rho$ & ${rho:.2f}$ \\
\hline
\end{{tabularx}}
\end{{table}}
"""


def main():
    e = rv.Escenario()
    print(PLANTILLA.format(
        sx=e.sala[0], sy=e.sala[1], sz=e.sala[2],
        lx=e.led_pos[0], ly=e.led_pos[1], lz=e.led_pos[2],
        semi=e.semiangulo_deg, m=e.m, pot=e.potencia,
        rs=e.radio_fuente * 1e3,
        apd=e.area_pd * 1e4, fov=np.degrees(e.fov), nr=e.n_refr,
        ts=e.t_filtro, resp=e.responsividad, hdev=e.altura_dispositivo,
        ruido=e.ruido_var,
        py=e.irs_y[1] - e.irs_y[0], pz=e.irs_z[1] - e.irs_z[0],
        ntot=e.n_total, rad=e.radio_espejo * 1e3, rho=e.reflectividad))


if __name__ == "__main__":
    main()

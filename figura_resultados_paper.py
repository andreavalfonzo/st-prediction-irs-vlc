"""Export one English IEEE-width figure from the saved mobility stress results.

Run: python figura_resultados_paper.py
No training or simulation is performed. Confidence intervals are read from
the original episode-bootstrap summaries, not computed from time samples.
"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "resultados" / "estres_movilidad"
OUTPUT = ROOT / "output" / "pdf"
STEM = "mobility_stress_results_en"
SCENARIOS = ("normal", "sudden_stop", "stationary", "erratic", "fall_proxy")
LABELS = ("Normal", "Sudden\nstop", "Stationary", "Erratic", "Simulated\nfall")
METHODS = ("cnn_gru", "reactive", "constant_velocity")
LEGEND = ("CNN-GRU", "Reactive", "Constant velocity*")
COLORS = ("#0072B2", "#E69F00", "#A6A6A6")
HATCHES = (None, "///", "...")


def main():
    cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    provenance = json.loads((SOURCE / "provenance.json").read_text(encoding="utf-8"))
    csv_path = SOURCE / "metrics_by_window.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        selected = [r for r in csv.DictReader(stream)
                    if r["window"] == "after" and r["method"] in METHODS]
    rows = {(r["scenario"], r["method"]): r for r in selected}
    assert len(rows) == len(selected) == len(SCENARIOS) * len(METHODS)
    assert all(int(r["episodes"]) == 16 and int(r["samples"]) == 960
               for r in selected)
    assert cfg["dt"] == 0.05 and cfg["r_min"] == 5.0
    assert provenance["training_performed"] is False
    assert provenance["checkpoints_unchanged"] is True
    assert provenance["intervention"]["event_time_s"] == 3.0
    assert np.isclose((cfg["steps"] - 1) * cfg["dt"], 5.95)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "hatch.linewidth": 0.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.85))
        x = np.arange(len(SCENARIOS))
        width = 0.24
        for i, method in enumerate(METHODS):
            group = [rows[(scenario, method)] for scenario in SCENARIOS]
            means = np.array([float(r["mean_rate"]) for r in group])
            low = np.array([float(r["rate_ci_low"]) for r in group])
            high = np.array([float(r["rate_ci_high"]) for r in group])
            assert np.all(low <= means) and np.all(means <= high)
            bars = dict(width=width, color=COLORS[i], edgecolor="#222222",
                        linewidth=0.4, hatch=HATCHES[i], label=LEGEND[i], zorder=3)
            axes[0].bar(x + (i - 1) * width, means,
                        yerr=np.array([means - low, high - means]),
                        error_kw={"elinewidth": 0.65, "capsize": 1.6,
                                  "capthick": 0.65, "ecolor": "#222222"}, **bars)
            axes[1].bar(x + (i - 1) * width,
                        [100 * float(r["outage_fraction"]) for r in group], **bars)

        for ax in axes:
            ax.set_xticks(x, LABELS)
            ax.set_xlim(-0.55, len(SCENARIOS) - 0.45)
            ax.tick_params(axis="x", length=0, pad=5)
            ax.tick_params(axis="y", length=3)
            ax.grid(axis="y", color="#D8D8D8", linewidth=0.45, zorder=0)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set(title="(a) Mean link rate", ylabel="Rate [bits/s/Hz]",
                    ylim=(0, 12), yticks=np.arange(0, 13, 2))
        axes[1].set(title="(b) Link outage", ylabel="Outage [%]",
                    ylim=(0, 100), yticks=np.arange(0, 101, 20))
        axes[0].axhline(cfg["r_min"], color="#555555", linestyle=(0, (3, 2)),
                        linewidth=0.7, zorder=2)
        axes[0].text(-0.43, cfg["r_min"] + 0.2, r"$R_{\min}=5$", fontsize=8,
                     color="#333333", va="bottom", zorder=4,
                     bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False,
                   bbox_to_anchor=(0.52, 0.995), handlelength=1.8,
                   columnspacing=2.0, handletextpad=0.6)
        fig.text(0.5, 0.025,
                 "*Constant velocity retains nominal receiver height during the simulated fall.",
                 ha="center", va="bottom", fontsize=8)
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.235, top=0.785, wspace=0.26)
        fig.savefig(OUTPUT / f"{STEM}.pdf", metadata={
            "Title": "Mobility stress: link rate and outage",
            "Subject": "16 test trajectories per scenario; 50 ms delay; window 3.00-5.95 s",
            "Keywords": "VLC, IRS, CNN-GRU, mobility stress, simulation",
        })
        fig.savefig(OUTPUT / f"{STEM}.png", dpi=600)
        plt.close(fig)

    caption = (
        "Link performance over 3.00--5.95~s across 16 simulated test trajectories "
        "per scenario, without retraining (50~ms control delay). Error bars show "
        "95\\% bootstrap confidence intervals across trajectories; outage is the "
        "fraction of intervals below 5~bits/s/Hz. The stationary case starts at "
        "$t=0$. The simulated fall changes receiver height and orientation; "
        "constant-velocity control retains the nominal height."
    )
    tex = (
        "% Place the PDF beside the main .tex file, then insert this block\n"
        "% in Results and analysis. figure* spans both IEEE columns.\n"
        "\\begin{figure*}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\textwidth]{{{STEM}.pdf}}\n"
        f"  \\caption{{{caption}}}\n"
        "  \\label{fig:mobility_stress}\n"
        "\\end{figure*}\n"
    )
    (OUTPUT / f"{STEM}.tex").write_text(tex, encoding="utf-8")
    # Preserve the exact plotted rows alongside the figure for audit/reuse.
    with (OUTPUT / f"{STEM}_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(rows[(scenario, method)]
                         for scenario in SCENARIOS for method in METHODS)
    manifest = {
        "source": csv_path.relative_to(ROOT).as_posix(),
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "window": "after", "window_seconds": [3.0, 5.95],
        "test_trajectories_per_scenario": 16, "control_delay_ms": 50,
        "rate_ci": "95% percentile bootstrap of whole test trajectories, 2000 resamples",
        "outage_threshold_bits_s_hz": 5,
        "paper_width_inches": 7.16, "png_dpi": 600,
    }
    (OUTPUT / f"{STEM}_source.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created {OUTPUT / STEM} (.pdf, .png, .tex, _data.csv, _source.json)")


if __name__ == "__main__":
    main()

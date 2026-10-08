#!/usr/bin/env python3
"""Figura F3 del paper #1: el sintético con ground truth — el phase bias
acumulado imita subsidencia y la corrección recupera la velocidad exacta.

Insumo: validation/phase_bias_export/synthetic.json
        (cargo run --release --example phase_bias_synthetic_figs)

Salida: docs/phase_bias/figs/fig_synthetic.pdf (+ preview PNG 300 dpi).
Columna simple (88 mm).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "phase_bias" / "figs"
d = json.loads((ROOT / "validation" / "phase_bias_export" / "synthetic.json").read_text())

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
})
INK, C_BIAS, C_CORR = "#24303a", "#D55E00", "#0072B2"

days = np.array(d["years"]) * 365.25
t_true = np.array(d["series_true_mm"])
t_bias = np.array(d["series_biased_mm"])
t_corr = np.array(d["series_corrected_mm"])

fig, ax = plt.subplots(figsize=(88 / 25.4, 66 / 25.4), layout="constrained")

ax.plot(days, t_true, "-", color=INK, lw=1.3, zorder=2,
        label=f"truth ({d['v_true_mmyr']:.0f} mm yr$^{{-1}}$)")
ax.plot(days, t_bias, "s--", color=C_BIAS, lw=1.0, ms=4.5, mec="white",
        mew=0.5, zorder=3,
        label=f"biased ({d['v_biased_mmyr']:.1f} mm yr$^{{-1}}$)")
ax.plot(days, t_corr, "o", color=C_CORR, ms=4.5, mec="white", mew=0.5,
        zorder=4, label=f"corrected ({d['v_corrected_mmyr']:.2f} mm yr$^{{-1}}$)")

# El sesgo acumulado: brecha entre sesgada y verdad en la última época.
ax.annotate("", xy=(days[-1], t_bias[-1]), xytext=(days[-1], t_true[-1]),
            arrowprops=dict(arrowstyle="<->", color=C_BIAS, lw=0.9))
ax.annotate(f"accumulated bias\n{t_bias[-1] - t_true[-1]:+.1f} mm in {days[-1]:.0f} d",
            xy=(days[-1] - 4, (t_bias[-1] + t_true[-1]) / 2), ha="right",
            va="center", fontsize=7, color=C_BIAS, zorder=5,
            bbox=dict(facecolor="white", alpha=0.85, lw=0, pad=1.5))

ax.text(0.03, 0.03,
        f"closure phase RMS: {d['closure_rms_before']:.3f} → "
        f"{d['closure_rms_after']:.1e} rad",
        transform=ax.transAxes, fontsize=7, color=INK, va="bottom")

ax.set_xlabel("days since first epoch")
ax.set_ylabel("LOS displacement (mm)")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=2.5, width=0.6)
ax.legend(frameon=False, fontsize=7, loc="upper right")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_synthetic.pdf")
fig.savefig(OUT / "fig_synthetic_preview.png", dpi=300)
print("escrito:", OUT / "fig_synthetic.pdf")

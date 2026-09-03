#!/usr/bin/env python3
"""Figura headline post-validación: cierre de fase SISTEMÁTICO (medio) antes y
después de la corrección hold-out, con estratificación por ESA WorldCover.

Reemplaza la métrica RMS por píxel (dominada por ruido de decorrelación y por
absorción de grados de libertad — ver phase_bias_validation.rs) por la media
del cierre, donde el ruido se cancela y el sesgo sobrevive; la corrección se
evalúa out-of-sample (δ̂ del bloque donut, nunca del propio píxel).

Insumos: validation/phase_bias_export/{mean_closure_before,mean_closure_after,
vel_bias_removed_mmyr}.f32 + wc_grid.npy + meta.json
(cargo run --release --example phase_bias_validation -- data/licsar_083D_12636/GEOC)

Salida: docs/phase_bias/figs/fig_systematic.pdf (+ preview PNG 300 dpi).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "validation" / "phase_bias_export"
OUT = ROOT / "docs" / "phase_bias" / "figs"

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
C_BEFORE, C_AFTER = "#D55E00", "#0072B2"

meta = json.loads((EXP / "meta.json").read_text())
nr, nc, g = meta["rows"], meta["cols"], meta["geo"]
rd = lambda f: np.fromfile(EXP / f, np.float32).reshape(nr, nc)
before, after = rd("mean_closure_before.f32"), rd("mean_closure_after.f32")

def smooth(m, k=7):
    """Media móvil k×k ignorando NaN — SOLO para display de los mapas (la
    estadística del panel d usa los valores por píxel sin suavizar)."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = k // 2
    mp = np.pad(m, pad, constant_values=np.nan)
    w = sliding_window_view(mp, (k, k))
    return np.nanmean(w, axis=(2, 3))

before_s, after_s = smooth(before), smooth(after)
wc = np.load(EXP / "wc_grid.npy")
strata = json.loads((EXP / "worldcover_strata.json").read_text())
extent = [g["lon0"], g["lon0"] + nc * g["dlon"], g["lat0"] + nr * g["dlat"], g["lat0"]]

fig = plt.figure(figsize=(180 / 25.4, 150 / 25.4), layout="constrained")
axs = fig.subplot_mosaic([["a", "b"], ["c", "d"]])

# (a,b) cierre medio antes / después (mismo rango, divergente centrado en 0)
vmax = 0.12
im_kw = dict(extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
im_a = axs["a"].imshow(before_s, **im_kw)
axs["b"].imshow(after_s, **im_kw)
cb = fig.colorbar(im_a, ax=[axs["a"], axs["b"]], location="right", shrink=0.9, pad=0.015)
cb.set_label("mean closure phase (rad)", fontsize=8)
cb.ax.tick_params(labelsize=7)
cb.outline.set_linewidth(0.6)

# (c) ESA WorldCover con su paleta oficial
wc_codes = [10, 20, 30, 40, 50, 60, 80]
wc_colors = ["#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000", "#b4b4b4", "#0064c8"]
wc_names = ["Tree", "Shrub", "Grass", "Crop", "Built", "Bare", "Water"]
lut = np.full(101, np.nan)
for i, code in enumerate(wc_codes):
    lut[code] = i
wc_idx = lut[np.clip(wc, 0, 100).astype(int)]
cmap_wc = ListedColormap(wc_colors)
norm_wc = BoundaryNorm(np.arange(-0.5, len(wc_codes)), cmap_wc.N)
im_c = axs["c"].imshow(wc_idx, extent=extent, cmap=cmap_wc, norm=norm_wc,
                       interpolation="nearest")
cbc = fig.colorbar(im_c, ax=axs["c"], location="right", shrink=0.9, pad=0.015,
                   ticks=range(len(wc_codes)))
cbc.ax.set_yticklabels(wc_names, fontsize=6.5)
cbc.outline.set_linewidth(0.6)

for k in ("a", "b", "c"):
    ax = axs[k]
    ax.set_xticks([-72.2, -72.0, -71.8])
    ax.set_yticks([-36.8, -36.6, -36.4])
    ax.tick_params(length=2.5, width=0.6)
    ax.set_xticklabels([]) if k in ("a", "b") else ax.set_xticklabels(
        ["72.2°W", "72.0°W", "71.8°W"])
    ax.set_yticklabels([]) if k == "b" else ax.set_yticklabels(
        ["36.8°S", "36.6°S", "36.4°S"])

# (d) cierre medio por clase, antes → después (hold-out), con equivalente en
# velocidad del sesgo removido anotado.
ax = axs["d"]
names = ["Cropland", "Grassland", "Tree cover", "Built-up"]
x = np.arange(len(names))
w = 0.36
vb = [strata[n]["before"] for n in names]
va = [strata[n]["after"] for n in names]
ax.bar(x - w / 2, vb, w, color=C_BEFORE, label="before")
ax.bar(x + w / 2, va, w, color=C_AFTER, label="after (held-out)")
ax.axhline(0, color="#24303a", lw=0.6)
for xi, n in zip(x, names):
    b, a, v = strata[n]["before"], strata[n]["after"], strata[n]["vel"]
    ax.text(xi - w / 2, b - 0.004, f"{b:+.3f}", ha="center", va="top", fontsize=6.5)
    ax.text(xi + w / 2, a - 0.004, f"{a:+.3f}", ha="center", va="top", fontsize=6.5)
    ax.text(xi, 0.004, f"{v:+.1f}", ha="center", va="bottom", fontsize=7,
            fontweight="bold", color="#24303a")
ax.text(0.02, 0.97, "bold: spurious LOS velocity removed (mm yr$^{-1}$)",
        transform=ax.transAxes, fontsize=6.5, va="top", color="#24303a")
ax.set_xticks(x, [n.replace(" ", "\n") for n in names], fontsize=7)
ax.set_ylabel("mean closure phase (rad)")
ax.set_ylim(min(vb) * 1.45, 0.052)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=2.5, width=0.6)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")

for k, ax in axs.items():
    dx = -0.17 if k != "d" else -0.18
    ax.text(dx, 1.10, k, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_systematic.pdf")
fig.savefig(OUT / "fig_systematic_preview.png", dpi=300)
print("clases:", {n: (round(strata[n]['before'], 4), round(strata[n]['after'], 4),
                      round(strata[n]['vel'], 2)) for n in names})
print("escrito:", OUT / "fig_systematic.pdf")

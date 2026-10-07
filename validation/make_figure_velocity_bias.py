#!/usr/bin/env python3
"""Figura de la revisión post blind b2: sesgo espurio de VELOCIDAD por phase
bias en Ñuble, medido como diferencia de inversiones SBAS (no Σδ̂).

(a) Mapa de la velocidad SBAS de la pantalla de corrección (sin corregir −
    corregida, sin referenciar): el sesgo que la corrección remueve.
(b) ESA WorldCover.
(c) Sesgo por clase (mediana, IC95 por bootstrap de bloques de 32 px), toda la
    escena y píxeles coherentes, junto a la cifra Σδ̂ de la versión anterior.
(d) Cambio de velocidad al aplicar la corrección antes vs después del
    desenrollado (SNAPHU), relativo a la referencia urbana.

aₙ = Maghsoudi et al. (2025) Fig. 11, base 12 d (≈0.55, 0.30); enmascarado
robusto de cierres (outlier_sigma = 2).

Insumos (validation/phase_bias_export/unw_snaphu_fig11/):
  vel_screen_unref_mmyr.f32, vel_{biased,corrected,postcorr}_mpyr.f32,
  summary.json + ../wc_grid.npy, ../mean_coh.f32
  (examples/phase_bias_velocity_unw + phase_bias_screen_velocity con
   PB_COEFS=0.55,0.30, OUT_TAG=_fig11).
Salida: docs/phase_bias/figs/fig_velocity_bias.pdf (+ preview PNG 300 dpi).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "validation" / "phase_bias_export"
RUN = EXP / "unw_snaphu_fig11"
OUT = ROOT / "docs" / "phase_bias" / "figs"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.linewidth": 0.6,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
})
INK = "#24303a"
C_SCENE, C_COH, C_OLD = "#0072B2", "#56B4E9", "#999999"
C_PRE, C_POST = "#D55E00", "#009E73"
CLASSES = [(40, "Crop"), (30, "Grass"), (10, "Tree"), (50, "Built")]
OLD_SUM_DELTA = {"Crop": -4.8, "Grass": -4.9, "Tree": -3.8, "Built": -1.8}
BLOCK, N_BOOT, MIN_COH = 32, 1000, 0.5

s = json.loads((RUN / "summary.json").read_text())
nr, nc, g = s["rows"], s["cols"], s["geo"]
rd = lambda p: np.fromfile(p, np.float32).reshape(nr, nc)
screen = rd(RUN / "vel_screen_unref_mmyr.f32")
vb = rd(RUN / "vel_biased_mpyr.f32") * 1000.0
vc = rd(RUN / "vel_corrected_mpyr.f32") * 1000.0
vp = rd(RUN / "vel_postcorr_mpyr.f32") * 1000.0
coh = rd(EXP / "mean_coh.f32")
wc = np.load(EXP / "wc_grid.npy")
valid = np.isfinite(screen) & (coh < 0.99)  # 6 px con coherencia saturada (artefacto)
coherent = valid & (coh >= MIN_COH) & np.isfinite(vb) & np.isfinite(vc) & np.isfinite(vp)

# Referencia regional para los caminos desenrollados: mediana urbana coherente.
urban = coherent & (wc == 50)
for a in (vb, vc, vp):
    a -= np.median(a[urban])

rng = np.random.default_rng(0)
bid = (np.arange(nr)[:, None] // BLOCK) * (nc // BLOCK + 1) + np.arange(nc)[None, :] // BLOCK


def med_ci(v, m):
    vals, blocks = v[m], bid[m]
    groups = [vals[blocks == b] for b in np.unique(blocks)]
    meds = [np.median(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
            for _ in range(N_BOOT)]
    return float(np.median(vals)), np.percentile(meds, [2.5, 97.5])


def smooth(m, k=7):
    """Media móvil k×k ignorando NaN: solo para el mapa."""
    from numpy.lib.stride_tricks import sliding_window_view
    mp = np.pad(m, k // 2, constant_values=np.nan)
    return np.nanmean(sliding_window_view(mp, (k, k)), axis=(2, 3))


extent = [g["lon0"], g["lon0"] + nc * g["dlon"], g["lat0"] + nr * g["dlat"], g["lat0"]]
fig = plt.figure(figsize=(180 / 25.4, 165 / 25.4), layout="constrained")
axs = fig.subplot_mosaic([["a", "b"], ["c", "d"]], height_ratios=[1.25, 1])
MAP_ASPECT = 1 / np.cos(np.deg2rad(36.6))  # grados lon/lat a escala real

# (a) velocidad de la pantalla (sesgo removido), suavizada solo para display.
vmax = 4.0
im_a = axs["a"].imshow(np.where(valid, smooth(screen), np.nan), extent=extent, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest", aspect=MAP_ASPECT)
cba = fig.colorbar(im_a, ax=axs["a"], location="right", shrink=0.8, pad=0.015, extend="both")
cba.set_label("spurious LOS velocity (mm yr$^{-1}$)", fontsize=8)
cba.ax.tick_params(labelsize=7)
cba.outline.set_linewidth(0.6)

# (b) ESA WorldCover, paleta oficial.
wc_codes = [10, 20, 30, 40, 50, 60, 80]
wc_colors = ["#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000", "#b4b4b4", "#0064c8"]
wc_names = ["Tree", "Shrub", "Grass", "Crop", "Built", "Bare", "Water"]
lut = np.full(101, np.nan)
for i, code in enumerate(wc_codes):
    lut[code] = i
im_b = axs["b"].imshow(lut[np.clip(wc, 0, 100).astype(int)], extent=extent,
                       cmap=ListedColormap(wc_colors),
                       norm=BoundaryNorm(np.arange(-0.5, len(wc_codes)), len(wc_codes)),
                       interpolation="nearest", aspect=MAP_ASPECT)
cbb = fig.colorbar(im_b, ax=axs["b"], location="right", shrink=0.8, pad=0.015,
                   ticks=range(len(wc_codes)))
cbb.ax.set_yticklabels(wc_names, fontsize=6.5)
cbb.outline.set_linewidth(0.6)

for k in ("a", "b"):
    ax = axs[k]
    ax.set_xticks([-72.2, -72.0, -71.8], ["72.2°W", "72.0°W", "71.8°W"])
    ax.set_yticks([-36.8, -36.6, -36.4], ["36.8°S", "36.6°S", "36.4°S"] if k == "a" else [])
    ax.tick_params(length=2.5, width=0.6)
for st, (lon, lat) in {"CLL1": (-72.0800, -36.5951), "BN16": (-72.0950, -36.6085)}.items():
    axs["a"].plot(lon, lat, marker="^", ms=4, mfc="white", mec=INK, mew=0.7)
axs["a"].annotate("GNSS CLL1, BN16", (-72.085, -36.60), (-72.02, -36.72), fontsize=6.5, color=INK,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                  arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))

# (c) sesgo por clase: escena y coherentes con IC95, y la cifra anterior (Σδ̂).
ax = axs["c"]
names = [n for _, n in CLASSES]
x = np.arange(len(names))
w = 0.34
stats = {}
for code, name in CLASSES:
    stats[name] = (med_ci(screen, valid & (wc == code)), med_ci(screen, coherent & (wc == code)))
for off, idx, col, lab in ((-w / 2, 0, C_SCENE, "whole scene"), (w / 2, 1, C_COH, "coherent pixels")):
    meds = np.array([stats[n][idx][0] for n in names])
    ci = np.array([stats[n][idx][1] for n in names])
    ax.bar(x + off, meds, w, color=col, label=lab)
    ax.errorbar(x + off, meds, yerr=[meds - ci[:, 0], ci[:, 1] - meds], fmt="none",
                ecolor=INK, elinewidth=0.6, capsize=1.5)
ax.scatter(x, [OLD_SUM_DELTA[n] for n in names], marker="_", s=180, color=C_OLD, lw=1.4,
           zorder=3, label=r"previous estimate ($\Sigma\hat\delta$)")
ax.axhline(0, color=INK, lw=0.6)
ax.set_xticks(x, names, fontsize=7)
ax.set_ylabel("spurious LOS velocity (mm yr$^{-1}$)")
ax.set_ylim(-5.4, 0.3)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=2.5, width=0.6)
ax.legend(frameon=False, fontsize=6.5, loc="lower right", handlelength=1.2)

# (d) corregir antes vs después del desenrollado (coherentes, ref. urbana).
ax = axs["d"]
pre, post = {}, {}
for code, name in CLASSES:
    m = coherent & (wc == code)
    pre[name], post[name] = med_ci(vc - vb, m), med_ci(vp - vb, m)
for off, d, col, lab in ((-w / 2, post, C_POST, "corrected after unwrapping"),
                         (w / 2, pre, C_PRE, "corrected before unwrapping")):
    meds = np.array([d[n][0] for n in names])
    ci = np.array([d[n][1] for n in names])
    ax.bar(x + off, meds, w, color=col, label=lab)
    ax.errorbar(x + off, meds, yerr=[meds - ci[:, 0], ci[:, 1] - meds], fmt="none",
                ecolor=INK, elinewidth=0.6, capsize=1.5)
ax.axhline(0, color=INK, lw=0.6)
cyc = rd(RUN / "cycle_changes_layers.f32")[coherent]
ax.text(0.02, 0.97,
        f"correction before unwrapping changes the integer\ncycle in {np.median(cyc):.0f} of {s['n_pairs']} "
        f"interferograms per pixel (median; P90 {np.percentile(cyc, 90):.0f})",
        transform=ax.transAxes, fontsize=6.3, va="top", color=INK, linespacing=1.3)
ax.set_xticks(x, names, fontsize=7)
ax.set_ylabel("velocity change, corrected − uncorrected\n(mm yr$^{-1}$, built-up reference)")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=2.5, width=0.6)
ax.legend(frameon=False, fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.86), handlelength=1.2)
lo = min(min(v[1][0] for v in pre.values()), min(v[1][0] for v in post.values()))
hi = max(max(v[1][1] for v in pre.values()), max(v[1][1] for v in post.values()))
ax.set_ylim(min(lo * 1.3, -1.0), hi * 1.5)

for k, ax in axs.items():
    ax.text(-0.02, 1.02, k, transform=ax.transAxes, fontsize=11, ha="right",
            fontweight="bold", va="bottom")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_velocity_bias.pdf")
fig.savefig(OUT / "fig_velocity_bias_preview.png", dpi=300)
print({n: (round(stats[n][0][0], 2), round(stats[n][1][0], 2), round(pre[n][0], 2), round(post[n][0], 2))
       for n in names})
print("escrito:", OUT / "fig_velocity_bias.pdf")

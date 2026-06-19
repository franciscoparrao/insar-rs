#!/usr/bin/env python3
"""Genera las figuras del showcase desde los productos reales de insar-rs
(CLI `insar isce` sobre Fernandina) + velocity.h5 de MintPy."""
import glob
import json
import os

import numpy as np
import tifffile
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = "/tmp/insar_isce_out"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)

INK = "#24303a"
PAPER = "#f5f1ea"
WARM = "#b5451f"
COOL = "#2f6f86"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "figure.dpi": 150,
    "savefig.transparent": True, "axes.facecolor": "none",
})

def read_tif(p):
    a = tifffile.imread(p).astype(np.float64)
    return np.where(np.isfinite(a), a, np.nan)

vel = read_tif(f"{OUT}/velocity.tif")          # m/año, ref (282,91)
coh = read_tif(f"{OUT}/temporal_coherence.tif")
nr, nc = vel.shape

# Referenciar insar al mismo píxel que MintPy (147,579) para comparar/leer.
RY, RX = 147, 579
with h5py.File("data/FernandinaSenDT128/mintpy/velocity.h5") as f:
    mvel = f["velocity"][:]
vel_ref = vel - vel[RY, RX]

# Píxel de máxima deformación entre píxeles coherentes.
mask = coh > 0.85
cand = np.where(mask, np.abs(vel_ref), -np.inf)
py, px = np.unravel_index(np.nanargmax(cand), cand.shape)
print(f"peak pixel=({py},{px}) vel={vel_ref[py,px]*1000:.1f} mm/año coh={coh[py,px]:.3f}")

# ---- Fig 1: mapa de velocidad LOS (mm/año), enmascarado por coherencia ----
vshow = np.where(coh > 0.5, vel_ref * 1000, np.nan)
vmax = np.nanpercentile(np.abs(vshow), 99)
fig, ax = plt.subplots(figsize=(5.4, 4.2))
im = ax.imshow(vshow, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.plot(px, py, "o", mfc="none", mec=INK, mew=1.6, ms=11)
ax.annotate("máx. deformación", (px, py), (px-150, py-28), color=INK, fontsize=8,
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
ax.plot(RX, RY, "^", color=INK, ms=7); ax.annotate("ref", (RX, RY), (RX-30, RY+34), fontsize=8)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cb.set_label("velocidad LOS (mm/año)", fontsize=9); cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=8)
fig.tight_layout(pad=0.4)
fig.savefig(f"{FIG}/velocity.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 2: coherencia temporal ----
fig, ax = plt.subplots(figsize=(5.4, 4.2))
im = ax.imshow(coh, cmap="cividis", vmin=0, vmax=1)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cb.set_label("coherencia temporal γ", fontsize=9); cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=8)
fig.tight_layout(pad=0.4)
fig.savefig(f"{FIG}/coherence.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 3: serie temporal del píxel de máxima deformación ----
files = sorted(glob.glob(f"{OUT}/series/disp_*.tif"))
dates = [os.path.basename(f)[5:13] for f in files]
import datetime as dt
days = np.array([(dt.datetime.strptime(d, "%Y%m%d") - dt.datetime.strptime(dates[0], "%Y%m%d")).days for d in dates])
years = days / 365.25
series_peak = np.array([read_tif(f)[py, px] for f in files])
series_ref = np.array([read_tif(f)[RY, RX] for f in files])
disp = (series_peak - series_ref) * 100.0  # cm, relativo al píxel de referencia
# ajuste lineal
A = np.vstack([years, np.ones_like(years)]).T
v, b = np.linalg.lstsq(A, disp, rcond=None)[0]

fig, ax = plt.subplots(figsize=(7.6, 3.5))
ax.axhline(0, color=INK, lw=0.5, alpha=0.3)
ax.plot(days, disp, "o", ms=4.5, color=WARM, mec="white", mew=0.5, alpha=0.9, zorder=3)
ax.plot(days, v*years + b, "-", color=INK, lw=1.4, alpha=0.8,
        label=f"v = {v*10:.1f} mm/año")
ax.set_xlabel("días desde 2014-12-13", fontsize=9)
ax.set_ylabel("desplazamiento LOS (cm)", fontsize=9)
ax.tick_params(labelsize=8)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="best")
fig.tight_layout(pad=0.5)
fig.savefig(f"{FIG}/timeseries.svg", bbox_inches="tight")
plt.close(fig)

# ---- Fig 4: paridad insar-rs vs MintPy ----
m = np.isfinite(vel_ref) & np.isfinite(mvel)
r = np.corrcoef(vel_ref[m], mvel[m])[0, 1]
fig, ax = plt.subplots(figsize=(4.6, 4.4))
lim = np.nanpercentile(np.abs(mvel[m])*1000, 99)
ax.plot([-lim, lim], [-lim, lim], "-", color=WARM, lw=1.2, zorder=1)
ax.hexbin(mvel[m]*1000, vel_ref[m]*1000, gridsize=70, cmap="Greys", bins="log",
          mincnt=1, zorder=2)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("MintPy (mm/año)", fontsize=9)
ax.set_ylabel("insar-rs (mm/año)", fontsize=9)
ax.tick_params(labelsize=8)
ax.set_aspect("equal")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.text(0.05, 0.92, f"r = {r:.6f}", transform=ax.transAxes, fontsize=11, color=INK)
fig.tight_layout(pad=0.5)
fig.savefig(f"{FIG}/parity.svg", bbox_inches="tight")
plt.close(fig)

stats = {
    "peak": {"row": int(py), "col": int(px),
             "vel_mm_yr": round(float(vel_ref[py, px]*1000), 1),
             "coh": round(float(coh[py, px]), 3),
             "total_disp_cm": round(float(disp[-1]), 1)},
    "fit_v_mm_yr": round(float(v*10), 1),
    "coh_median": round(float(np.nanmedian(coh)), 3),
    "coh_gt09_frac": round(float(np.nanmean(coh > 0.9)), 3),
    "parity_r": round(float(r), 6),
    "n_epochs": len(files), "grid": [nr, nc],
}
json.dump(stats, open(f"{FIG}/stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))

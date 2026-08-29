#!/usr/bin/env python3
"""Figura F2 del paper #1: paridad numérica insar-rs vs MintPy (Fernandina).

Requiere la cadena regenerada (ver docs/phase_bias/closure_figs.md):
  1. MintPy: load_data + reference_point + ifgram_inversion (timeseries.h5)
  2. timeseries2velocity.py → velocity con la MISMA referencia (se pasa como
     argv[1]; default el scratchpad de la sesión)
  3. export_ifgstack.py + cargo run --example validate_fernandina

Salida: docs/phase_bias/figs/fig_parity.pdf (+ preview PNG 300 dpi).
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "validation" / "export"
MINTPY = ROOT / "data" / "FernandinaSenDT128" / "mintpy"
VEL_H5 = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP / "velocity_ref.h5"
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
INK = "#24303a"

meta = json.loads((EXP / "meta.json").read_text())
ne, nr, nc = meta["n_epochs"], meta["rows"], meta["cols"]
its = np.fromfile(EXP / "insar_timeseries.f32", np.float32).reshape(ne, nr, nc)
ivel = np.fromfile(EXP / "insar_velocity.f32", np.float32).reshape(nr, nc)

with h5py.File(MINTPY / "timeseries.h5") as f:
    mts = f["timeseries"][:]
    ry, rx = int(f.attrs["REF_Y"]), int(f.attrs["REF_X"])
with h5py.File(VEL_H5) as f:
    mvel = f["velocity"][:]

# Ambos lados referenciados al mismo píxel (el de MintPy).
its = its - its[:, ry, rx][:, None, None]
ivel = ivel - ivel[ry, rx]

# ── stats ──────────────────────────────────────────────────────────────────
k = np.isfinite(its) & np.isfinite(mts)
a, b = its[k] * 100, mts[k] * 100  # cm
d_um = (a - b) * 1e4               # µm
rmse_mm = float(np.sqrt(np.mean((a - b) ** 2))) * 10
r = float(np.corrcoef(a, b)[0, 1])

kv = np.isfinite(ivel) & np.isfinite(mvel)
dv_um = np.where(kv, (ivel - mvel) * 1e6, np.nan)  # µm/año
rmse_v = float(np.sqrt(np.nanmean(dv_um ** 2)))
print(f"serie: n={k.sum():,}  RMSE={rmse_mm:.4f} mm  r={r:.6f}")
print(f"velocidad: RMSE={rmse_v:.2f} µm/año")

# ── figura: 3 paneles, doble columna ───────────────────────────────────────
fig = plt.figure(figsize=(180 / 25.4, 60 / 25.4), layout="constrained")
axs = fig.subplot_mosaic([["a", "b", "c"]])

# (a) densidad 2D de la serie: insar-rs vs MintPy
ax = axs["a"]
lim = np.percentile(np.abs(b), 99.9)
H, xe, ye = np.histogram2d(b, a, bins=240, range=[[-lim, lim], [-lim, lim]])
pm = ax.pcolormesh(xe, ye, H.T, norm=LogNorm(vmin=1), cmap="magma",
                   rasterized=True)
ax.plot([-lim, lim], [-lim, lim], "--", color="#5a6a75", lw=0.7, zorder=3)
ax.set_xlabel("MintPy LOS displacement (cm)")
ax.set_ylabel("insar-rs LOS displacement (cm)")
ax.set_aspect("equal")
ax.text(0.04, 0.96, f"r = {r:.6f}\nRMSE = {rmse_mm:.4f} mm\nn = {k.sum() / 1e6:.1f} M",
        transform=ax.transAxes, va="top", fontsize=7)
cb = fig.colorbar(pm, ax=ax, location="right", shrink=0.85, pad=0.02)
cb.set_label("count", fontsize=7)
cb.ax.tick_params(labelsize=6)
cb.outline.set_linewidth(0.5)

# (b) diferencia de velocidad (µm/año)
ax = axs["b"]
vmax = 20.0
im = ax.imshow(dv_um, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
ax.set_xticks([])
ax.set_yticks([])
ax.plot(rx, ry, "o", mfc="none", mec=INK, mew=1.0, ms=7)
ax.annotate("ref.", xy=(rx, ry), xytext=(rx - 90, ry + 60), fontsize=7,
            color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
cb = fig.colorbar(im, ax=ax, location="right", shrink=0.85, pad=0.02)
cb.set_label("Δ velocity (µm yr$^{-1}$)", fontsize=7)
cb.ax.tick_params(labelsize=6)
cb.outline.set_linewidth(0.5)
ax.text(0.03, 0.04, f"RMSE = {rmse_v:.1f} µm yr$^{{-1}}$", transform=ax.transAxes,
        va="bottom", fontsize=7)

# (c) histograma de diferencias de la serie (µm)
ax = axs["c"]
sub = d_um[:: max(1, d_um.size // 2_000_000)]
ax.hist(sub, bins=161, range=(-4, 4), color="#0072B2", lw=0)
ax.set_xlabel("series difference (µm)")
ax.set_ylabel("pixels × epochs")
ax.set_yscale("log")
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.04, 0.96, f"P99.9 |Δ| = {np.percentile(np.abs(d_um), 99.9):.2f} µm",
        transform=ax.transAxes, va="top", fontsize=7)

for kk, ax in axs.items():
    ax.tick_params(length=2.5, width=0.6)
    ax.text(-0.16, 1.06, kk, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_parity.pdf", dpi=300)
fig.savefig(OUT / "fig_parity_preview.png", dpi=300)
print("escrito:", OUT / "fig_parity.pdf")

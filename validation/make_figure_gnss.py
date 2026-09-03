#!/usr/bin/env python3
"""Figura suplementaria S1: diagnóstico GNSS que justifica NO reportar
velocidades InSAR para Ñuble (Issue 3 del review C&G).

Compara la velocidad LOS del pipeline con desenrollado flood-fill en el píxel
de la estación CLL1 (Chillán, NGL, serie fija a placa Sudamericana) contra la
velocidad GNSS proyectada a LOS. La discrepancia (decenas de mm/año en un
sitio estable) muestra que el error de desenrollado domina el campo de
velocidad — la razón del pivote a validación en dominio de cierre.

Insumos: scratchpad/cll1_sa.tenv3 (NGL IGS20/SA) +
         docs/phase_bias/figs/vel_{biased,corrected}_mpyr.tif
Proyección LOS: descendente, incidencia 39°, heading −169° → u=(0.62,−0.12,0.78).

Salida: docs/phase_bias/figs/fig_gnss_supp.pdf (+ preview).
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path("/tmp/claude-1000/-home-franciscoparrao-proyectos-insar-rs/"
               "937ff52c-ac48-42c4-b662-e8729779435a/scratchpad")
OUT = ROOT / "docs" / "phase_bias" / "figs"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.linewidth": 0.6,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
})
INK, C_G, C_I = "#24303a", "#0072B2", "#D55E00"

# ── GNSS: velocidades E/N/U sobre la ventana InSAR ─────────────────────────
rows = []
for ln in open(SCRATCH / "cll1_sa.tenv3"):
    p = ln.split()
    if not p or p[0] != "CLL1":
        continue
    rows.append((float(p[2]), float(p[8]), float(p[10]), float(p[12])))
t, e, n, u = map(np.array, zip(*rows))
win = (t >= 2021.97) & (t <= 2023.0)
fit = lambda y: np.polyfit(t[win], y[win], 1)[0] * 1000.0  # mm/año
ve, vn, vu = fit(e), fit(n), fit(u)

# LOS descendente (θ=39°, heading −169°, right-looking): u=(E,N,U)
theta, head = np.radians(39.0), np.radians(-169.0)
look_az = head + np.pi / 2
uE = np.sin(theta) * np.sin(look_az - np.pi)
uN = np.sin(theta) * np.cos(look_az - np.pi)
uU = np.cos(theta)
v_gnss_los = ve * uE + vn * uN + vu * uU
print(f"GNSS CLL1 (SA-fixed, 2021.97–2023.0): vE={ve:+.1f} vN={vn:+.1f} "
      f"vU={vu:+.1f} mm/a → LOS {v_gnss_los:+.1f} mm/a  "
      f"(u=({uE:.2f},{uN:.2f},{uU:.2f}))")

# ── InSAR flood-fill en el píxel de CLL1 ───────────────────────────────────
from osgeo import gdal
gdal.UseExceptions()
LON, LAT = -72.079958, -36.595071
def sample(path):
    ds = gdal.Open(str(path))
    gt = ds.GetGeoTransform()
    a = ds.ReadAsArray()
    c = int((LON - gt[0]) / gt[1]); r = int((LAT - gt[3]) / gt[5])
    w = a[max(0, r - 1):r + 2, max(0, c - 1):c + 2] * 1000.0  # m/a → mm/a
    return float(np.nanmedian(w))
v_ins_corr = sample(ROOT / "docs/phase_bias/figs/vel_corrected_mpyr.tif")
v_ins_bias = sample(ROOT / "docs/phase_bias/figs/vel_biased_mpyr.tif")
print(f"InSAR flood-fill en CLL1 (mediana 3×3): sin corregir {v_ins_bias:+.1f}, "
      f"corregida {v_ins_corr:+.1f} mm/a → discrepancia "
      f"{v_ins_corr - v_gnss_los:+.1f} mm/a")

# ── figura ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(120 / 25.4, 70 / 25.4), layout="constrained")
m = (t >= 2021.0) & (t <= 2023.4)
los_mm = (e * uE + n * uN + u * uU) * 1000.0
los_mm -= np.nanmean(los_mm[m & (t < 2021.2)])
ax.plot(t[m], los_mm[m], ".", ms=2.5, color=C_G, alpha=0.7,
        label=f"GNSS CLL1 → LOS  ({v_gnss_los:+.1f} mm yr$^{{-1}}$)")
t0 = 2021.97
tt = np.array([t0, 2023.0])
base = np.interp(t0, t[m], los_mm[m])
ax.plot(tt, base + v_ins_corr * (tt - t0), "-", color=C_I, lw=1.6,
        label=f"InSAR flood-fill at CLL1  ({v_ins_corr:+.1f} mm yr$^{{-1}}$)")
ax.axvspan(t0, 2023.0, color="#009E73", alpha=0.07, lw=0)
ax.annotate("InSAR analysis window", xy=(2022.48, ax.get_ylim()[0]),
            fontsize=7, color="#009E73", ha="center", va="bottom")
ax.set_xlabel("year")
ax.set_ylabel("LOS displacement (mm)")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=2.5, width=0.6)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_gnss_supp.pdf")
fig.savefig(OUT / "fig_gnss_supp_preview.png", dpi=300)
print("escrito:", OUT / "fig_gnss_supp.pdf")

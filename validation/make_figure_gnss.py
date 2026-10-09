#!/usr/bin/env python3
"""Figura A.1 del paper #1: chequeo de consistencia GNSS diferencial CLL1 − BN16.

Las velocidades InSAR son relativas a una referencia y las soluciones NGL están
fijas a la placa Sudamericana, así que se compara la DIFERENCIA entre las dos
estaciones (1.6 km, ambas en Chillán), que no depende de la referencia InSAR.
La tasa GNSS se ajusta sobre la serie ya diferenciada (días comunes), lo que
cancela el modo común.

  (a) serie LOS diferencial diaria con tendencia + términos anual y semianual
      sobre el período común, y la ventana InSAR sombreada;
  (b) tasas diferenciales: GNSS secular (±2σ, bootstrap de bloques de 1 año
      sobre los residuos: el ruido es coloreado y el σ formal lo subestima
      ~6×), la distribución de tasas GNSS en ventanas deslizantes de la misma
      duración que la serie InSAR (cobertura ≥ 60 %, sin huecos > 60 d), y
      las velocidades InSAR SNAPHU (mediana 3×3) sin corregir y con la
      corrección después del desenrollado (gateA.json, unw_snaphu_fig11).

La ventana InSAR misma no sirve para GNSS: las estaciones comparten ahí solo
74 días separados por un hueco de 296 d; se informa, no se grafica.

Insumos: data/gnss/{CLL1,BN16}.SA.tenv3 (NGL IGS20, fijo a SA),
         validation/phase_bias_export/unw_snaphu_fig11/gateA.json
Salida:  docs/phase_bias/figs/fig_gnss_supp.pdf (+ preview PNG) y
         docs/phase_bias/figs/gnss_differential.json (cifras del apéndice).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "validation" / "phase_bias_export" / "unw_snaphu_fig11" / "gateA.json"
OUT = ROOT / "docs" / "phase_bias" / "figs"
U_LOS = np.array([0.618, -0.120, 0.777])  # descendente, θ=39°, heading −169°
COMMON = (2018.9, 2024.6)                 # período común CLL1/BN16
WIN = (2021.97, 2023.0)                   # ventana InSAR

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.linewidth": 0.6,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
})
INK, C_G, C_B, C_P = "#24303a", "#0072B2", "#7a8791", "#009E73"  # C_P = verde "after" de la Fig. 6d


def los(st):
    """Serie LOS diaria (mm) indexada por MJD."""
    out = {}
    for ln in open(ROOT / "data" / "gnss" / f"{st}.SA.tenv3"):
        p = ln.split()
        if p and p[0] == st:
            enu = np.array([float(p[8]), float(p[10]), float(p[12])])
            out[int(p[3])] = (float(p[2]), float(enu @ U_LOS) * 1000.0)
    return out


def fit(t, y, seasonal):
    cols = [np.ones_like(t), t - t.mean()]
    if seasonal:
        for k in (1, 2):
            cols += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]
    A = np.c_[tuple(cols)]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    s2 = r @ r / (len(y) - A.shape[1])
    sig = np.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1])
    rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])          # lag-1
    sig_ac = sig * np.sqrt((1 + rho) / (1 - rho))           # inflación AR(1)
    return dict(rate=float(coef[1]), sigma_formal=float(sig), sigma=float(sig_ac),
                rho1=rho, n=int(len(y))), A @ coef


a, b = los("CLL1"), los("BN16")
mjd = sorted(set(a) & set(b))
t = np.array([a[m][0] for m in mjd])
d = np.array([a[m][1] - b[m][1] for m in mjd])
mc = (t >= COMMON[0]) & (t < COMMON[1])
mw = (t >= WIN[0]) & (t <= WIN[1])
sec, model = fit(t[mc], d[mc], seasonal=True)
win, _ = fit(t[mw], d[mw], seasonal=False)

# σ secular realista: bootstrap de bloques móviles de ~1 año sobre residuos.
tc, dc = t[mc], d[mc]
res = dc - model
L = int(len(res) / (tc[-1] - tc[0]))
rng = np.random.default_rng(0)
boot = []
for _ in range(2000):
    idx = np.concatenate([np.arange(k, k + L)
                          for k in rng.integers(0, len(res) - L, len(res) // L + 1)])
    boot.append(fit(tc, model + res[idx[:len(res)]], True)[0]["rate"])
sec["sigma_boot"] = float(np.std(boot))

# Tasas en ventanas deslizantes de la duración InSAR (paso 30 d).
span = WIN[1] - WIN[0]
slide = []
for t0 in np.arange(COMMON[0], COMMON[1] - span, 30 / 365.25):
    m = (tc >= t0) & (tc <= t0 + span)
    if m.sum() < 0.6 * span * 365.25:
        continue
    edges = np.r_[t0, tc[m], t0 + span]
    if np.diff(edges).max() * 365.25 > 60:
        continue
    slide.append(fit(tc[m], dc[m], seasonal=False)[0]["rate"])
slide = np.array(slide)

g = json.loads(GATE.read_text())["differential"]["insar"]
ins_b, ins_p = g["v_biased"], g["v_post"]

print(f"días comunes: {mc.sum()} (período común), {mw.sum()} (ventana InSAR)")
print(f"GNSS diferencial secular {sec['rate']:+.2f} ± {sec['sigma_boot']:.2f} mm/año "
      f"(1σ bootstrap; formal {sec['sigma_formal']:.2f})")
print(f"GNSS diferencial ventana InSAR (no usable, {win['n']} días): {win['rate']:+.2f}")
print(f"ventanas deslizantes de {span:.2f} años: n={len(slide)}, mediana "
      f"{np.median(slide):+.2f}, P5–P95 {np.percentile(slide, 5):+.2f} a "
      f"{np.percentile(slide, 95):+.2f}, rango {slide.min():+.2f} a {slide.max():+.2f}")
print(f"  fracción de ventanas ≤ InSAR sin corregir: {(slide <= ins_b).mean():.2f}; "
      f"≤ corregida: {(slide <= ins_p).mean():.2f}")
print(f"InSAR diferencial: sin corregir {ins_b:+.2f}, corregida tras desenrollar {ins_p:+.2f}")

# ── figura: 2 paneles, columna y media ─────────────────────────────────────
fig = plt.figure(figsize=(140 / 25.4, 55 / 25.4), layout="constrained")
axs = fig.subplot_mosaic([["a", "a", "b"]])

ax = axs["a"]
ax.axvspan(*WIN, color="#e8edf1", lw=0, zorder=0)
ax.text(np.mean(WIN), 0.97, "InSAR window", transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=6.5, color="#5a6a75")
ax.plot(t[mc], d[mc] - d[mc].mean(), ".", ms=1.2, color="#9aa6af", alpha=0.6,
        rasterized=True, label="daily")
ax.plot(t[mc], model - d[mc].mean(), "-", color=C_G, lw=0.9,
        label="trend + seasonal")
ax.set_xlim(*COMMON)
ax.set_xlabel("Year")
ax.set_ylabel("CLL1 − BN16 LOS (mm)")
ax.legend(loc="lower left", fontsize=6.5, frameon=False, markerscale=4)
ax.spines[["top", "right"]].set_visible(False)

ax = axs["b"]
ax.axhline(0, color="#5a6a75", lw=0.5, ls=":")
ax.errorbar(0, sec["rate"], yerr=2 * sec["sigma_boot"], fmt="o", color=C_G,
            ms=4, lw=0.8, capsize=2)
vp = ax.violinplot(slide, positions=[1], widths=0.7, showextrema=False)
for body in vp["bodies"]:
    body.set_facecolor(C_G); body.set_alpha(0.25); body.set_edgecolor("none")
ax.plot([1, 1], np.percentile(slide, [5, 95]), "-", color=C_G, lw=0.8)
ax.plot(1, np.median(slide), "_", color=C_G, ms=8, mew=1.2)
ax.plot(2, ins_b, "D", color=C_B, ms=4)
ax.plot(3, ins_p, "D", color=C_P, ms=4)
ax.set_xticks(range(4), ["GNSS\nsecular", "GNSS\n1-yr", "InSAR\nuncorr.",
                         "InSAR\ncorr."], fontsize=6.5)
ax.set_xlim(-0.6, 3.6)
ax.set_ylabel("Differential LOS rate (mm yr$^{-1}$)")
ax.spines[["top", "right"]].set_visible(False)

for k, ax in axs.items():
    ax.tick_params(length=2.5, width=0.6)
    ax.text(-0.02 if k == "a" else -0.34, 1.06, k, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="right")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_gnss_supp.pdf", dpi=300)
fig.savefig(OUT / "fig_gnss_supp_preview.png", dpi=300)
(OUT / "gnss_differential.json").write_text(json.dumps(dict(
    secular=sec, window_unusable=win,
    sliding=dict(span_yr=span, n=int(len(slide)), median=float(np.median(slide)),
                 p5=float(np.percentile(slide, 5)), p95=float(np.percentile(slide, 95)),
                 min=float(slide.min()), max=float(slide.max())), insar_uncorrected=ins_b, insar_post=ins_p,
    common_period=COMMON, insar_window=WIN, u_los=U_LOS.tolist(),
    days_common=int(mc.sum()), days_window=int(mw.sum())), indent=2))
print("escrito:", OUT / "fig_gnss_supp.pdf")

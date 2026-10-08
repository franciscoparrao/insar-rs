#!/usr/bin/env python3
"""Figura de cierre del paper #1 (post-pivote): mapas del RMS de la fase de
cierre antes/después de la corrección de phase bias (Ñuble, LiCSAR 083D_12636)
+ coherencia media + estadística estratificada. Lee los crudos exportados por
`cargo run --release --example phase_bias_closure_figs`.

Salida: docs/phase_bias/figs/fig_closure.pdf (+ preview PNG 300 dpi).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "validation" / "phase_bias_export"
OUT = ROOT / "docs" / "phase_bias" / "figs"

# ── estilo global (journal sans-serif, jerarquía tipográfica) ──────────────
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
    "pdf.fonttype": 42,  # TrueType embebido, exigido por Elsevier/Springer
})

# Wong palette (colorblind-safe): antes = bermellón, después = azul.
C_BEFORE, C_AFTER = "#D55E00", "#0072B2"

meta = json.loads((EXP / "meta.json").read_text())
nr, nc, g = meta["rows"], meta["cols"], meta["geo"]
rd = lambda f: np.fromfile(EXP / f, np.float32).reshape(nr, nc)
before, after, mcoh, count = (
    rd("closure_before.f32"), rd("closure_after.f32"), rd("mean_coh.f32"),
    rd("closure_count.f32"),
)
extent = [g["lon0"], g["lon0"] + nc * g["dlon"], g["lat0"] + nr * g["dlat"], g["lat0"]]

# ── estadística estratificada (idéntica al driver Rust: RMS ponderado) ─────
STRATA = [
    ("coh < 0.3\n(cropland / forest)", 0.0, 0.3),
    ("0.3 – 0.5\n(mixed)", 0.3, 0.5),
    ("coh ≥ 0.5\n(urban / stable)", 0.5, np.inf),
]

def agg_rms(m, keep):
    k = keep & (count > 0) & np.isfinite(m)
    if not k.any():
        return np.nan
    return float(np.sqrt(np.sum(m[k].astype(np.float64) ** 2 * count[k]) / np.sum(count[k])))

rows = []
for label, lo, hi in STRATA:
    sel = np.isfinite(mcoh) & (mcoh >= lo) & (mcoh < hi)
    rows.append((label, agg_rms(before, sel), agg_rms(after, sel)))

# ── figura: 2×2, doble columna (180 mm) ────────────────────────────────────
fig = plt.figure(figsize=(180 / 25.4, 150 / 25.4), layout="constrained")
axs = fig.subplot_mosaic([["a", "b"], ["c", "d"]])

vmax = float(np.nanpercentile(before, 99))
im_kw = dict(extent=extent, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
im_a = axs["a"].imshow(before, **im_kw)
axs["b"].imshow(after, **im_kw)
cb = fig.colorbar(im_a, ax=[axs["a"], axs["b"]], location="right", shrink=0.9, pad=0.015)
cb.set_label("closure phase RMS (rad)", fontsize=8)
cb.ax.tick_params(labelsize=7)
cb.outline.set_linewidth(0.6)

im_c = axs["c"].imshow(mcoh, extent=extent, cmap="gray", vmin=0, vmax=0.8,
                       interpolation="nearest")
cbc = fig.colorbar(im_c, ax=axs["c"], location="right", shrink=0.9, pad=0.015)
cbc.set_label("mean coherence", fontsize=8)
cbc.ax.tick_params(labelsize=7)
cbc.outline.set_linewidth(0.6)

# Chillán: el núcleo urbano coherente que ancla la lectura de cobertura.
CHILLAN = (-72.103, -36.606)
for k in ("a", "b", "c"):
    axs[k].plot(*CHILLAN, "o", mfc="none", mec="#009E73", mew=1.2, ms=9)
axs["c"].annotate("Chillán (urban)", xy=CHILLAN, xytext=(-72.06, -36.51),
                  fontsize=7, color="#009E73",
                  arrowprops=dict(arrowstyle="-", color="#009E73", lw=0.8))

for k in ("a", "b", "c"):
    ax = axs[k]
    ax.set_xticks([-72.2, -72.0, -71.8])
    ax.set_yticks([-36.8, -36.6, -36.4])
    ax.tick_params(length=2.5, width=0.6)
    ax.set_xticklabels([]) if k in ("a", "b") else ax.set_xticklabels(
        ["72.2°W", "72.0°W", "71.8°W"])
    ax.set_yticklabels([]) if k == "b" else ax.set_yticklabels(
        ["36.8°S", "36.6°S", "36.4°S"])

# ── panel d: barras antes/después por estrato ──────────────────────────────
ax = axs["d"]
x = np.arange(len(rows))
w = 0.36
b1 = ax.bar(x - w / 2, [r[1] for r in rows], w, color=C_BEFORE, label="before")
b2 = ax.bar(x + w / 2, [r[2] for r in rows], w, color=C_AFTER, label="after")
for xi, (_, vb, va) in zip(x, rows):
    ax.text(xi - w / 2, vb + 0.03, f"{vb:.2f}", ha="center", fontsize=7)
    ax.text(xi + w / 2, va + 0.03, f"{va:.2f}", ha="center", fontsize=7)
    ax.annotate(f"−{100 * (1 - va / vb):.0f}%",
                xy=(xi + w / 2, va / 2), ha="center", fontsize=7,
                color="white", fontweight="bold")
ax.set_xticks(x, [r[0] for r in rows], fontsize=7)
ax.set_ylabel("closure phase RMS (rad)")
ax.set_ylim(0, max(r[1] for r in rows) * 1.25)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=2.5, width=0.6)
ax.legend(frameon=False, fontsize=8, loc="upper right")

for k, ax in axs.items():
    dx = -0.17 if k != "d" else -0.18
    ax.text(dx, 1.10, k, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_closure.pdf")
fig.savefig(OUT / "fig_closure_preview.png", dpi=300)
print("global:", meta["rms_before"], "->", meta["rms_after"])
print("estratos:", [(r[0].split(chr(10))[0], round(r[1], 3), round(r[2], 3)) for r in rows])
print("escrito:", OUT / "fig_closure.pdf")

#!/usr/bin/env python3
"""Figura F5 del paper #1: red temporal del frame LiCSAR 083D_12636 (Ñuble).
Arc diagram de los pares descargados, con los pares podados por LiCSAR
(huecos en spans 1–3) marcados, la ventana hole-free usada para la corrección
sombreada, y los pares ancla (~72 d) destacados.

Lee la estructura de data/licsar_083D_12636/GEOC (un dir por par; el par es
válido si contiene su .geo.diff_unfiltered_pha.tif).

Salida: docs/phase_bias/figs/fig_network.pdf (+ preview PNG 300 dpi).
"""
import datetime as dt
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

ROOT = Path(__file__).resolve().parent.parent
GEOC = ROOT / "data" / "licsar_083D_12636" / "GEOC"
OUT = ROOT / "docs" / "phase_bias" / "figs"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
})

C_SHORT, C_ANCHOR, C_MISS, C_WIN = "#0072B2", "#D55E00", "#CC0000", "#009E73"

d8 = lambda s: dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
pairs, epochs = [], set()
for p in sorted(GEOC.iterdir()):
    m = re.fullmatch(r"(\d{8})_(\d{8})", p.name)
    if not m:
        continue
    a, b = d8(m.group(1)), d8(m.group(2))
    ok = (p / f"{p.name}.geo.diff_unfiltered_pha.tif").exists()
    pairs.append((a, b, ok))
    if ok:
        epochs.update((a, b))
epochs = sorted(epochs)
t0 = epochs[0]
day = lambda d: (d - t0).days

valid = [(a, b) for a, b, ok in pairs if ok]
missing = [(a, b) for a, b, ok in pairs if not ok]
print(f"pares: {len(pairs)} en índice, {len(valid)} válidos, {len(missing)} podados; "
      f"{len(epochs)} épocas {epochs[0]} → {epochs[-1]}")

WIN = (dt.date(2021, 12, 21), dt.date(2022, 12, 28))

fig, ax = plt.subplots(figsize=(180 / 25.4, 62 / 25.4), layout="constrained")

# Ventana hole-free sombreada, detrás de todo.
ax.axvspan(day(WIN[0]), day(WIN[1]), color=C_WIN, alpha=0.10, lw=0, zorder=0)

def draw_arc(a, b, color, lw, alpha, zorder):
    c, w = (day(a) + day(b)) / 2, day(b) - day(a)
    ax.add_patch(Arc((c, 0), w, w * 0.9, theta1=0, theta2=180,
                     color=color, lw=lw, alpha=alpha, zorder=zorder))

for a, b in valid:
    span = (b - a).days
    if span >= 60:
        draw_arc(a, b, C_ANCHOR, 0.9, 0.85, 3)
    else:
        draw_arc(a, b, C_SHORT, 0.5, 0.45, 2)
for a, b in missing:
    draw_arc(a, b, C_MISS, 0.6, 0.5, 1)

ax.plot([day(e) for e in epochs], [0] * len(epochs), ".", ms=2.5,
        color="#24303a", zorder=4)

# Eje x en fechas legibles (cada 3 meses el tick, etiqueta cada 6).
ticks = []
d = dt.date(epochs[0].year, epochs[0].month, 1)
while d <= epochs[-1]:
    ticks.append(d)
    d = dt.date(d.year + (d.month + 2) // 12, (d.month + 2) % 12 + 1, 1)
ax.set_xticks([day(t) for t in ticks],
              [t.strftime("%b\n%Y") if t.month in (1, 7) else "" for t in ticks])
ax.tick_params(length=2.5, width=0.6)

max_span = max((b - a).days for a, b in valid)
ax.set_ylim(-34, max_span * 0.52)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_position(("data", -28))

# Leyenda manual compacta (proxy artists).
from matplotlib.lines import Line2D
handles = [
    Line2D([], [], color=C_SHORT, lw=1, label=f"short pair (6–18 d), n={sum(1 for a,b in valid if (b-a).days<60)}"),
    Line2D([], [], color=C_ANCHOR, lw=1, label=f"long pair (≥60 d, anchor), n={sum(1 for a,b in valid if (b-a).days>=60)}"),
    Line2D([], [], color=C_MISS, lw=1, label=f"pruned by LiCSAR, n={len(missing)}"),
    plt.Rectangle((0, 0), 1, 1, fc=C_WIN, alpha=0.10, label="hole-free window (32 epochs)"),
]
ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left",
          bbox_to_anchor=(0.005, 1.0), handlelength=1.6)

# Anotación de la ventana, en la franja vacía entre las épocas y el eje.
ax.annotate("bias correction applied here", fontsize=7, color=C_WIN,
            xy=((day(WIN[0]) + day(WIN[1])) / 2, -15), ha="center", va="center")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_network.pdf")
fig.savefig(OUT / "fig_network_preview.png", dpi=300)
print("escrito:", OUT / "fig_network.pdf")

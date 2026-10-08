#!/usr/bin/env python3
"""Figura de la revisión post blind b2: validación por hold-out en el ESPACIO
DE LOOPS (K = 5 folds por span; cada cierre retenido se predice como
(aₙ − 1)·Σδ̂ con δ̂ estimado sin él), con enmascarado robusto (σ = 2).

(a) Ñuble, escena completa: cierre medio retenido por span antes (líneas) y
    después de la corrección con cinco juegos de aₙ.
(b) Sintéticos con la red real (ruido 0.7 rad por interferograma): un
    control positivo que sigue el modelo y tres nulos. La prueba falla con el
    nulo de signo opuesto entre spans, pero no distingue la estructura aₙ de
    un cierre igual en ambos spans.

Insumo: validation/phase_bias_export/loop_holdout_robust2/summary.json
(OUTLIER_SIGMA=2 cargo run --release --example phase_bias_loop_holdout).
Salida: docs/phase_bias/figs/fig_loop_holdout.pdf (+ preview PNG 300 dpi).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "validation" / "phase_bias_export" / "loop_holdout_robust2" / "summary.json").read_text())
OUT = ROOT / "docs" / "phase_bias" / "figs"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.linewidth": 0.6,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
})
INK = "#24303a"
C_S2, C_S3 = "#0072B2", "#E69F00"
C_BEFORE, C_AFTER = "#D55E00", "#0072B2"

fig = plt.figure(figsize=(180 / 25.4, 72 / 25.4), layout="constrained")
axs = fig.subplot_mosaic([["a", "b"]], width_ratios=[1.35, 1])

# (a) datos reales por span y juego de aₙ.
ax = axs["a"]
sets = [("scene", "this scene\n(84-d anchor)"), ("fig11_12d", "Maghsoudi 2025\nFig. 11, 12 d"),
        ("turkey2022", "Turkey\n2022"), ("campi_flegrei", "Campi\nFlegrei"), ("azores", "Azores")]
x = np.arange(len(sets))
w = 0.34
for off, span, col in ((-w / 2, "span2", C_S2), (w / 2, "span3", C_S3)):
    after = [S[k][span][1] for k, _ in sets]
    ax.bar(x + off, after, w, color=col, label=f"{span.replace('span', 'span ')}, after")
    ax.axhline(S["scene"][span][0], color=col, lw=0.9, ls="--",
               label=f"{span.replace('span', 'span ')}, before")
ax.axhline(0, color=INK, lw=0.6)
ax.set_xticks(x, [f"{lab}\n({S[k]['coefficients'][0]:.2f}, {S[k]['coefficients'][1]:.2f})" for k, lab in sets],
              fontsize=6.3)
ax.set_ylabel("held-out mean closure (rad)")
ax.set_ylim(-0.066, 0.03)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=2.5, width=0.6)
ax.legend(frameon=False, fontsize=6.3, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          handlelength=1.6, columnspacing=1.0)

# (b) sintéticos.
ax = axs["b"]
scen = [("synthetic_positive_model", "follows\nmodel"), ("synthetic_null_equal", "null: equal\nacross spans"),
        ("synthetic_null_opposite", "null: opposite\nsign"), ("synthetic_null_noise", "null: noise\nonly")]
x = np.arange(len(scen))
before = [S[k]["mean_before"] for k, _ in scen]
after = [S[k]["mean_after"] for k, _ in scen]
ax.bar(x - w / 2, before, w, color=C_BEFORE, label="before")
ax.bar(x + w / 2, after, w, color=C_AFTER, label="after (held-out)")
ax.axhline(0, color=INK, lw=0.6)
notes = ["removed", "removed:\nnot discriminated", "fails, as it should\n(spans cancel\nbefore)", "nothing to\nremove"]
for xi, b, a, note in zip(x, before, after, notes):
    top = max(b, a, 0)
    ax.text(xi, top + 0.004, note, ha="center", va="bottom", fontsize=5.8, color=INK, linespacing=1.15)
ax.set_xticks(x, [lab for _, lab in scen], fontsize=6.5)
ax.set_ylabel("held-out mean closure (rad)")
lo = min(before + after)
ax.set_ylim(lo * 1.35 if lo < 0 else -0.01, max(before + after) + 0.03)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=2.5, width=0.6)
ax.legend(frameon=False, fontsize=6.5, loc="lower left")

for k, ax in axs.items():
    ax.text(-0.09, 1.02, k, transform=ax.transAxes, fontsize=11, ha="right",
            fontweight="bold", va="bottom")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_loop_holdout.pdf")
fig.savefig(OUT / "fig_loop_holdout_preview.png", dpi=300)
print({k: [round(v, 4) for v in (S[k]["span2"] + S[k]["span3"])] for k, _ in sets})
print({k: (round(S[k]["mean_before"], 4), round(S[k]["mean_after"], 4)) for k, _ in scen})
print("escrito:", OUT / "fig_loop_holdout.pdf")

#!/usr/bin/env python3
"""Recalibración de aₙ: estima desde los cierres el único parámetro que estos
identifican, ρ = (1 − a₃)/(1 − a₂). El modelo es invariante a escalar δ por k
y (aₙ − 1) por 1/k, así que con δ′ = (1 − a₂)δ:

    c₂(i) = −(δ′ᵢ + δ′ᵢ₊₁),   c₃(i) = −ρ (δ′ᵢ + δ′ᵢ₊₁ + δ′ᵢ₊₂)

Por loop se usa la media circular de los fasores de cierre sobre la clase
(robusta a las colas del envolvimiento). Para cada ρ de una grilla, δ′ por
mínimos cuadrados; el ρ óptimo minimiza el RSS. IC por bootstrap de bloques
espaciales de 32 px (se re-promedian los fasores con los bloques remuestreados).

Insumos: validation/phase_bias_export/calibration/closures.{f32,json}
Uso: python3 validation/phase_bias_calibration_ratio.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "validation" / "phase_bias_export"
CAL = EXP / "calibration"
meta = json.loads((CAL / "closures.json").read_text())
nr, nc, loops = meta["rows"], meta["cols"], meta["loops"]
cl = np.fromfile(CAL / "closures.f32", np.float32).reshape(len(loops), nr, nc)
wc = np.load(EXP / "wc_grid.npy")
coh = np.fromfile(EXP / "mean_coh.f32", np.float32).reshape(nr, nc)
n_ep = len(meta["epochs"])
U = n_ep - 1
RHO = np.linspace(0.6, 2.4, 181)
BLOCK, N_BOOT = 32, 300
rng = np.random.default_rng(0)
bid = ((np.arange(nr)[:, None] // BLOCK) * (nc // BLOCK + 1) + np.arange(nc)[None, :] // BLOCK)

# Diseño en δ′ (sin ρ) y máscara de filas span 3.
A0 = np.zeros((len(loops), U))
is3 = np.zeros(len(loops), bool)
for k, l in enumerate(loops):
    A0[k, l["start"]:l["start"] + l["span"]] = -1.0
    is3[k] = l["span"] == 3


def fit(c):
    """ρ que minimiza el RSS del modelo sobre el vector de cierres por loop."""
    rss = []
    for r in RHO:
        A = A0.copy()
        A[is3] *= r
        x, *_ = np.linalg.lstsq(A, c, rcond=None)
        rss.append(np.sum((A @ x - c) ** 2))
    rss = np.array(rss)
    return RHO[np.argmin(rss)], rss


def loop_means(mask, blocks_pick=None):
    z = np.exp(1j * cl[:, mask])
    if blocks_pick is None:
        return np.angle(np.nanmean(z, axis=1))
    b = bid[mask]
    zs = np.stack([np.nansum(z[:, b == k], axis=1) for k in np.unique(b)], axis=1)
    ub = np.unique(b)
    pick = rng.integers(0, len(ub), len(ub))
    return np.angle(zs[:, pick].sum(axis=1))


a2_ref = {"scene (72→84 d anchor)": 0.415, "Fig.11 12-d base (≈0.55, 0.30)": 0.55}
out = {}
for name, m in {
    "Cropland": wc == 40, "Grassland": wc == 30, "Tree cover": wc == 10,
    "vegetated (crop+grass+tree)": np.isin(wc, [10, 30, 40]),
    "vegetated, coherent": np.isin(wc, [10, 30, 40]) & (coh >= 0.5) & (coh < 0.99),
}.items():
    c = loop_means(m)
    rho, rss = fit(c)
    boots = [fit(loop_means(m, True))[0] for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ratio_obs = np.mean(c[is3]) / np.mean(c[~is3])
    implied = {k: 1 - rho * (1 - a2) for k, a2 in a2_ref.items()}
    out[name] = dict(n_px=int(m.sum()), rho=float(rho), rho_ci=[float(lo), float(hi)],
                     mean_closure_ratio_span3_span2=float(ratio_obs), implied_a3=implied)
    print(f"{name:<30} n={m.sum():>6}  ρ = {rho:.2f} [{lo:.2f}, {hi:.2f}]  "
          f"(c̄₃/c̄₂ = {ratio_obs:.2f})  a₃ implícito: "
          + ", ".join(f"a₂={a2:.3f}→a₃={1 - rho * (1 - a2):.3f}" for a2 in a2_ref.values()))

print("\nreferencias de ρ = (1−a₃)/(1−a₂):")
for nm, (a2, a3) in {"escena (0.415, 0.472)": (0.415, 0.472), "Turquía (0.47, 0.31)": (0.47, 0.31),
                     "Campi Flegrei (0.50, 0.36)": (0.50, 0.36), "Azores (0.53, 0.33)": (0.53, 0.33),
                     "Fig. 11, base 12 d (≈0.55, 0.30)": (0.55, 0.30)}.items():
    print(f"  {nm:<32} ρ = {(1 - a3) / (1 - a2):.2f}")
(CAL / "ratio.json").write_text(json.dumps(out, indent=2))
print(f"\n→ {CAL / 'ratio.json'}")

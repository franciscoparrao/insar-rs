#!/usr/bin/env python3
"""Compuerta B (paper #1, review blind b2): hold-out en el espacio de loops
por clase WorldCover, con los cuatro juegos de aₙ, y velocidad SBAS de la
pantalla de corrección (definición de "velocidad espuria" como diferencia de
inversiones, no Σδ̂).

Insumos: validation/phase_bias_export/loop_holdout/<set>/ (salida de
`examples/phase_bias_loop_holdout.rs`), wc_grid.npy, mean_coh.f32.

Uso: python3 validation/phase_bias_gateB.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "validation" / "phase_bias_export"
import sys
LH = EXP / (sys.argv[1] if len(sys.argv) > 1 else "loop_holdout")
CLASSES = {40: "Cropland", 30: "Grassland", 10: "Tree cover", 50: "Built-up"}
SETS = [x for x in ["scene", "fig11_12d", "turkey2022", "campi_flegrei", "azores"]]
BLOCK, N_BOOT, MIN_COH = 32, 1000, 0.5

s = json.loads((LH / "summary.json").read_text())
nr, nc = s["rows"], s["cols"]
wc = np.load(EXP / "wc_grid.npy")
coh = np.fromfile(EXP / "mean_coh.f32", np.float32).reshape(nr, nc)
bid = (np.arange(nr)[:, None] // BLOCK) * (nc // BLOCK + 1) + np.arange(nc)[None, :] // BLOCK
rng = np.random.default_rng(0)


def load(setname):
    rd = lambda f: np.fromfile(LH / setname / f, np.float32).reshape(nr, nc).astype(float)
    return {k: rd(f"{k}.f32") for k in
            ("sum_before", "sum_after", "cos_before", "sin_before", "cos_after", "sin_after", "n",
             "vel_screen_mmyr")}


def class_stats(d, m):
    """Cierre medio retenido antes/después (aritmético y circular) sobre la
    máscara, con IC95 de la reducción aritmética por bootstrap de bloques."""
    m = m & (d["n"] > 0)
    b_sum, a_sum, n = d["sum_before"][m], d["sum_after"][m], d["n"][m]
    before, after = b_sum.sum() / n.sum(), a_sum.sum() / n.sum()
    cb = np.arctan2(d["sin_before"][m].sum(), d["cos_before"][m].sum())
    ca = np.arctan2(d["sin_after"][m].sum(), d["cos_after"][m].sum())
    blocks = bid[m]
    ub = np.unique(blocks)
    gb = {b: (b_sum[blocks == b].sum(), a_sum[blocks == b].sum(), n[blocks == b].sum()) for b in ub}
    reds = []
    for _ in range(N_BOOT):
        pick = rng.choice(ub, len(ub))
        sb = sum(gb[b][0] for b in pick); sa = sum(gb[b][1] for b in pick); sn = sum(gb[b][2] for b in pick)
        reds.append(100 * (1 - abs(sa / sn) / abs(sb / sn)))
    lo, hi = np.percentile(reds, [2.5, 97.5])
    return dict(n_px=int(m.sum()), before=float(before), after=float(after),
                reduction=float(100 * (1 - abs(after) / abs(before))), reduction_ci=[float(lo), float(hi)],
                circ_before=float(cb), circ_after=float(ca),
                circ_reduction=float(100 * (1 - abs(ca) / abs(cb))),
                vel_screen=float(np.nanmedian(d["vel_screen_mmyr"][m])))


out = {}
for mask_name, base in (("all_pixels", np.ones((nr, nc), bool)), ("coherent", (coh >= MIN_COH) & (coh < 0.99))):
    print(f"\n=== {mask_name} ===")
    out[mask_name] = {}
    for st in [x for x in SETS if (LH / x).exists()]:
        d = load(st)
        print(f"\n{st} (a = {[round(x, 3) for x in s[st]['coefficients']]})")
        print(f"  {'clase':<11}{'n px':>7}  {'antes':>8} {'después':>8}  {'reducción [IC95]':>22}"
              f"  {'circ antes':>10} {'circ desp':>9} {'red circ':>8}  {'v pantalla':>10}")
        out[mask_name][st] = {}
        for code, name in CLASSES.items():
            r = class_stats(d, base & (wc == code))
            out[mask_name][st][name] = r
            print(f"  {name:<11}{r['n_px']:>7}  {r['before']:+8.4f} {r['after']:+8.4f}  "
                  f"{r['reduction']:6.1f} % [{r['reduction_ci'][0]:5.1f}, {r['reduction_ci'][1]:5.1f}]"
                  f"  {r['circ_before']:+10.4f} {r['circ_after']:+9.4f} {r['circ_reduction']:7.1f} %"
                  f"  {r['vel_screen']:+10.2f}")

print("\n=== sintéticos (red real, ruido 0.7 rad) ===")
for k, v in s.items():
    if k.startswith("synthetic_"):
        red = 100 * (1 - abs(v["mean_after"]) / abs(v["mean_before"])) if v["mean_before"] else float("nan")
        print(f"  {k[10:]:<15} {v['mean_before']:+.4f} → {v['mean_after']:+.4f} ({red:.0f} %) | "
              f"span2 {v['span2'][0]:+.4f}→{v['span2'][1]:+.4f} | span3 {v['span3'][0]:+.4f}→{v['span3'][1]:+.4f}")
out["synthetic"] = {k: v for k, v in s.items() if k.startswith("synthetic_")}
(LH / "gateB.json").write_text(json.dumps(out, indent=2))
print(f"\n→ {LH / 'gateB.json'}")

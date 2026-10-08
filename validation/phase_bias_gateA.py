#!/usr/bin/env python3
"""Compuerta A (paper #1, review blind b2): efecto real de la corrección de
phase bias sobre la velocidad LOS en Ñuble, por clase WorldCover.

Tres caminos (salida de `examples/phase_bias_velocity_unw.rs`):
  vb  sin corregir,
  vc  corregido ANTES del desenrollado (diseño del paper),
  vp  corregido DESPUÉS del desenrollado (misma pantalla restada al
      desenrollado sin corregir, sin re-desenrollar),
más `vel_screen_unref_mmyr.f32` (`examples/phase_bias_screen_velocity.rs`):
velocidad SBAS de la pantalla de corrección sola, por píxel, sin referenciar
— el efecto puro de la corrección, libre de desenrollado y de referencia.

Referencia: el driver referencia a un píxel único que resultó ser un
artefacto (coherencia LiCSAR ≈ 1.000 en todos los pares con fase aleatoria).
Como el referenciado es lineal, aquí se re-referencia a una REGIÓN: la
mediana de los píxeles urbanos coherentes (la clase de menor sesgo).

GNSS: comparación diferencial CLL1 − BN16 (1.6 km, ambas en Chillán, NGL
IGS20 fijo a placa SA), que no depende de la referencia InSAR.

Uso: python3 validation/phase_bias_gateA.py [snaphu|quality]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "validation" / "phase_bias_export"
METHOD = sys.argv[1] if len(sys.argv) > 1 else "snaphu"
RUN = EXP / f"unw_{METHOD}"

CLASSES = {40: "Cropland", 30: "Grassland", 10: "Tree cover", 50: "Built-up"}
MIN_COH = 0.5
BLOCK = 32  # px (~3.2 km)
N_BOOT = 1000
STATIONS = {"CLL1": (-72.0799584, -36.5950712), "BN16": (-72.0950098, -36.6085292)}
U_LOS = np.array([0.618, -0.120, 0.777])  # descendente, θ=39°, heading −169°
WIN = (2021.97, 2023.0)

s = json.loads((RUN / "summary.json").read_text())
nr, nc, g = s["rows"], s["cols"], s["geo"]
rd = lambda p: np.fromfile(p, np.float32).reshape(nr, nc)
vb = rd(RUN / "vel_biased_mpyr.f32") * 1000.0
vc = rd(RUN / "vel_corrected_mpyr.f32") * 1000.0
vp = rd(RUN / "vel_postcorr_mpyr.f32") * 1000.0
screen = rd(RUN / "vel_screen_unref_mmyr.f32")
cyc = rd(RUN / "cycle_changes_layers.f32")
coh = rd(EXP / "mean_coh.f32")
model = rd(EXP / "vel_bias_removed_mmyr.f32")
wc = np.load(EXP / "wc_grid.npy")

ok = np.isfinite(vb) & np.isfinite(vc) & np.isfinite(vp) & (coh >= MIN_COH) & (coh < 0.99)
region = ok & (wc == 50)
for a in (vb, vc, vp):
    a -= np.median(a[region])
screen_rel = screen - np.median(screen[region])

rng = np.random.default_rng(0)
bid = (np.arange(nr)[:, None] // BLOCK) * (nc // BLOCK + 1) + np.arange(nc)[None, :] // BLOCK


def boot_median(vals, blocks):
    ub = np.unique(blocks)
    groups = [vals[blocks == b] for b in ub]
    meds = [np.median(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
            for _ in range(N_BOOT)]
    return [float(x) for x in np.percentile(meds, [2.5, 97.5])]


sat = np.argwhere(coh >= 0.99).tolist()
print(f"método: {METHOD} | coh ≥ {MIN_COH} (excluye {len(sat)} px con coh ≥ 0.99, artefacto) | "
      f"referencia: mediana de {region.sum()} px urbanos coherentes")
print(f"píxeles válidos: {ok.sum()} de {ok.size} ({100*ok.mean():.1f} %)")
nlay = cyc[ok]
print(f"capas (de {s['n_pairs']}) con cambio de ciclo por píxel coherente: mediana {np.median(nlay):.0f}, "
      f"P90 {np.percentile(nlay, 90):.0f}; píxeles con >5 capas: {100*np.mean(nlay > 5):.1f} %\n")

print("mm/año, medianas por clase (IC95 bootstrap de bloques 32 px):")
print(f"{'clase':<11}{'n':>6}  {'puro (pantalla)':>24}  {'post − sin':>10}  {'pre − sin':>10}"
      f"  {'desenr.':>8}  {'Σδ̂ modelo':>10}")
rows = []
for code, name in CLASSES.items():
    m = ok & (wc == code)
    pure_abs = float(np.median(screen[m]))
    pure_ci = boot_median(screen[m], bid[m])
    r = dict(cls=name, n=int(m.sum()),
             pure_unref=pure_abs, pure_unref_ci=pure_ci,
             pure_rel_urban=float(np.median(screen_rel[m])),
             post_minus_biased=float(np.median((vp - vb)[m])),
             pre_minus_biased=float(np.median((vc - vb)[m])),
             unwrap_interaction=float(np.median((vc - vp)[m])),
             model_sum_delta=float(np.nanmedian(model[m])),
             v_biased=float(np.median(vb[m])), v_post=float(np.median(vp[m])),
             v_pre=float(np.median(vc[m])))
    rows.append(r)
    print(f"{name:<11}{r['n']:>6}  {pure_abs:+6.2f} [{pure_ci[0]:+.2f}, {pure_ci[1]:+.2f}]"
          f"  {r['post_minus_biased']:>+10.2f}  {r['pre_minus_biased']:>+10.2f}"
          f"  {r['unwrap_interaction']:>+8.2f}  {r['model_sum_delta']:>+10.2f}")
print("  puro = velocidad SBAS de la pantalla (corregida − sin corregir, sin referencia);\n"
      "  post/pre − sin = cambio de velocidad al corregir tras/antes del desenrollado (ref. urbana);\n"
      "  desenr. = pre − post (lo que agregan los cambios de ciclo).")


def pix(lon, lat):
    return int((lat - g["lat0"]) / g["dlat"]), int((lon - g["lon0"]) / g["dlon"])


def w3(a, rc):
    r, c = rc
    return float(np.nanmedian(a[max(0, r-1):r+2, max(0, c-1):c+2]))


ins = {}
for st, (lon, lat) in STATIONS.items():
    rc = pix(lon, lat)
    ins[st] = dict(rc=list(rc), coh=w3(coh, rc), wc=int(wc[rc]), v_biased=w3(vb, rc),
                   v_post=w3(vp, rc), v_pre=w3(vc, rc), cyc_layers=w3(cyc, rc))


def gnss(st):
    t, e, n, u = [], [], [], []
    for ln in open(ROOT / "data" / "gnss" / f"{st}.SA.tenv3"):
        p = ln.split()
        if p and p[0] == st:
            t.append(float(p[2])); e.append(float(p[8])); n.append(float(p[10])); u.append(float(p[12]))
    t = np.array(t)
    return t, (np.c_[e, n, u] @ U_LOS) * 1000.0


def rate(t, y, seasonal):
    cols = [np.ones_like(t), t - t.mean()]
    if seasonal:
        for k in (1, 2):
            cols += [np.sin(2*np.pi*k*t), np.cos(2*np.pi*k*t)]
    A = np.c_[tuple(cols)]
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    sig = float(np.sqrt(res[0] / (len(y) - A.shape[1]) * np.linalg.inv(A.T @ A)[1, 1])) if len(res) else np.nan
    return float(coef[1]), sig, int(len(y))


gn = {}
for st in STATIONS:
    t, y = gnss(st)
    msec = (t >= 2018.9) & (t < 2024.6)  # período común CLL1/BN16
    mwin = (t >= WIN[0]) & (t <= WIN[1])
    gn[st] = dict(secular=rate(t[msec], y[msec], True), window=rate(t[mwin], y[mwin], False))

print("\nestaciones (mediana 3×3, ref. urbana):")
for st, d in ins.items():
    print(f"  {st}: coh {d['coh']:.2f}, WC {d['wc']}, capas con cambio de ciclo {d['cyc_layers']:.0f} | "
          f"InSAR sin {d['v_biased']:+.1f}, post {d['v_post']:+.1f}, pre {d['v_pre']:+.1f} | "
          f"GNSS secular {gn[st]['secular'][0]:+.1f} ± {gn[st]['secular'][1]:.1f}, "
          f"ventana {gn[st]['window'][0]:+.1f} (n={gn[st]['window'][2]}) mm/año")
dg_sec = gn["CLL1"]["secular"][0] - gn["BN16"]["secular"][0]
dg_win = gn["CLL1"]["window"][0] - gn["BN16"]["window"][0]
di = {k: ins["CLL1"][k] - ins["BN16"][k] for k in ("v_biased", "v_post", "v_pre")}
print(f"\ndiferencial CLL1 − BN16 (independiente de la referencia InSAR):")
print(f"  GNSS secular 2019–2024 {dg_sec:+.1f} | GNSS ventana {dg_win:+.1f} mm/año")
print(f"  InSAR sin corregir {di['v_biased']:+.1f} | post {di['v_post']:+.1f} | pre {di['v_pre']:+.1f} mm/año")

out = dict(method=METHOD, min_coh=MIN_COH, block_px=BLOCK, n_valid=int(ok.sum()),
           saturated_coh_px=sat, reference="median of coherent built-up pixels",
           classes=rows, stations=ins, gnss=gn,
           differential=dict(gnss_secular=dg_sec, gnss_window=dg_win, insar=di),
           cycle_layers_median=float(np.median(nlay)), cycle_layers_p90=float(np.percentile(nlay, 90)))
(RUN / "gateA.json").write_text(json.dumps(out, indent=2))
print(f"\n→ {RUN / 'gateA.json'}")

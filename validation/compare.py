#!/usr/bin/env python3
"""Compara la serie temporal y la velocidad LOS de insar-rs contra MintPy
sobre el dataset Fernandina (inversión SBAS no ponderada, sin correcciones).

Ambos productos se referencian al mismo píxel (REF_Y/REF_X de MintPy) — la
inversión es lineal, así que referenciar-antes-de-invertir (MintPy) equivale a
invertir-y-restar-el-píxel-de-referencia (lo que hacemos aquí con insar-rs).
"""
import json
import sys

import h5py
import numpy as np


def stats(a, b, label, unit):
    """a = insar-rs, b = MintPy; sobre máscara común finita."""
    m = np.isfinite(a) & np.isfinite(b)
    da, db = a[m], b[m]
    diff = da - db
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    maxabs = float(np.max(np.abs(diff)))
    # correlación y pendiente de regresión insar vs mintpy
    r = float(np.corrcoef(da, db)[0, 1])
    slope = float(np.polyfit(db, da, 1)[0])
    print(f"\n[{label}]  (n={m.sum():,} px finitos, {100*m.mean():.1f}% de la grilla)")
    print(f"  RMSE        = {rmse*1000:.4f} mm" if unit == "m" else f"  RMSE        = {rmse*1000:.4f} mm/año")
    print(f"  MAE         = {mae*1000:.4f} {'mm' if unit=='m' else 'mm/año'}")
    print(f"  max|Δ|      = {maxabs*1000:.4f} {'mm' if unit=='m' else 'mm/año'}")
    print(f"  Pearson r   = {r:.6f}")
    print(f"  pendiente   = {slope:.6f}  (ideal 1.0)")
    return dict(rmse=rmse, mae=mae, maxabs=maxabs, r=r, slope=slope, n=int(m.sum()),
                cov=float(m.mean()))


def main():
    export = sys.argv[1] if len(sys.argv) > 1 else "validation/export"
    mintpy = sys.argv[2] if len(sys.argv) > 2 else "data/FernandinaSenDT128/mintpy"

    meta = json.load(open(f"{export}/meta.json"))
    ne, nr, nc = meta["n_epochs"], meta["rows"], meta["cols"]

    # insar-rs (crudo, sin referenciar)
    its = np.fromfile(f"{export}/insar_timeseries.f32", np.float32).reshape(ne, nr, nc)
    ivel = np.fromfile(f"{export}/insar_velocity.f32", np.float32).reshape(nr, nc)

    # MintPy
    with h5py.File(f"{mintpy}/timeseries.h5", "r") as f:
        mts = f["timeseries"][:]
        ry, rx = int(f.attrs["REF_Y"]), int(f.attrs["REF_X"])
        mdates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]
    with h5py.File(f"{mintpy}/velocity.h5", "r") as f:
        mvel = f["velocity"][:]

    print(f"REF pixel (y,x) = ({ry},{rx})")
    print(f"épocas insar={ne}  mintpy={len(mdates)}  "
          f"(coinciden orden: {[d[:4]+'-'+d[4:6]+'-'+d[6:8] for d in mdates] == meta['epochs']})")

    # Referenciar insar-rs al MISMO píxel que MintPy.
    its_ref = its - its[:, ry, rx][:, None, None]
    ivel_ref = ivel - ivel[ry, rx]

    s_ts = stats(its_ref, mts, "Serie temporal LOS", "m")
    s_v = stats(ivel_ref, mvel, "Velocidad LOS", "m/yr")

    # RMSE por época (diagnóstico)
    per_epoch = []
    for e in range(ne):
        m = np.isfinite(its_ref[e]) & np.isfinite(mts[e])
        per_epoch.append(float(np.sqrt(np.mean((its_ref[e][m] - mts[e][m])**2))))
    print(f"\nRMSE por época: min={min(per_epoch)*1000:.3f} mm  "
          f"max={max(per_epoch)*1000:.3f} mm  media={np.mean(per_epoch)*1000:.3f} mm")

    json.dump({"timeseries": s_ts, "velocity": s_v,
               "per_epoch_rmse_m": per_epoch, "ref_pixel": [ry, rx]},
              open(f"{export}/compare_stats.json", "w"), indent=2)

    # Figura: mapas de velocidad + diferencia + scatter
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        vmin, vmax = np.nanpercentile(mvel*1000, [2, 98])
        for a, d, t in [(ax[0,0], ivel_ref*1000, "insar-rs"),
                        (ax[0,1], mvel*1000, "MintPy")]:
            im = a.imshow(d, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            a.set_title(f"Velocidad LOS — {t} (mm/año)"); a.plot(rx, ry, "k^", ms=7)
            plt.colorbar(im, ax=a, shrink=0.8)
        dif = (ivel_ref - mvel)*1000
        im = ax[1,0].imshow(dif, cmap="RdBu_r", vmin=-1, vmax=1)
        ax[1,0].set_title("Diferencia (mm/año)"); plt.colorbar(im, ax=ax[1,0], shrink=0.8)
        m = np.isfinite(ivel_ref) & np.isfinite(mvel)
        ax[1,1].hexbin(mvel[m]*1000, ivel_ref[m]*1000, gridsize=60, cmap="viridis", bins="log")
        lim = [vmin, vmax]; ax[1,1].plot(lim, lim, "r--", lw=1)
        ax[1,1].set_xlabel("MintPy (mm/año)"); ax[1,1].set_ylabel("insar-rs (mm/año)")
        ax[1,1].set_title(f"r={s_v['r']:.5f}, pendiente={s_v['slope']:.4f}")
        fig.tight_layout(); fig.savefig(f"{export}/../validation_velocity.png", dpi=130)
        print(f"\nFigura → validation/validation_velocity.png")
    except Exception as e:
        print(f"(figura omitida: {e})")


if __name__ == "__main__":
    main()

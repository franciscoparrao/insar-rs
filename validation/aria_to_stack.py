#!/usr/bin/env python3
"""Convierte un directorio de productos ARIA S1-GUNW (.nc) en el formato que
ingiere insar-rs: recorta a la zona de interés, apila la fase desenrollada y
la coherencia, y escribe meta.json + phase.f32 + coherence.f32.

Uso: aria_to_stack.py <gunw_dir> --out <export_dir> --lon -70.52 --lat -36.07 --half 0.30
"""
import argparse
import glob
import json
import os
from datetime import datetime

import h5py
import numpy as np

S1_WAVELENGTH = 0.05546576  # banda C, m


def dates_from_name(nm):
    p = os.path.basename(nm).split("-tops-")[1].split("-")[0]
    d1, d2 = p.split("_")  # d1 = secundaria (posterior), d2 = referencia (anterior)
    return datetime.strptime(d2, "%Y%m%d"), datetime.strptime(d1, "%Y%m%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gunw_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lon", type=float, default=-70.52)
    ap.add_argument("--lat", type=float, default=-36.07)
    ap.add_argument("--half", type=float, default=0.30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(f"{args.gunw_dir}/*.nc"))
    lon0, lon1 = args.lon - args.half, args.lon + args.half
    lat0, lat1 = args.lat - args.half, args.lat + args.half

    # Ventana objetivo (ancla geográfica) desde el primer archivo.
    with h5py.File(files[0], "r") as h:
        lon = h["science/grids/data/longitude"][:]
        lat = h["science/grids/data/latitude"][:]
    cx = np.where((lon >= lon0) & (lon <= lon1))[0]
    cy = np.where((lat >= lat0) & (lat <= lat1))[0]
    x0, y0 = cx.min(), cy.min()
    cols, rows = len(cx), len(cy)
    lon_anchor, lat_anchor = float(lon[x0]), float(lat[y0])
    dlon = abs(float(lon[1] - lon[0]))
    print(f"ventana {rows}x{cols} px  lon[{lon[cx.max()]:.3f},{lon_anchor:.3f}] lat[{lat[cy.max()]:.3f},{lat_anchor:.3f}]")

    # Cada archivo se ancla por coordenada al mismo corner (tolera desplazamientos
    # de origen entre productos GUNW de distintas fechas).
    pairs_map = {}  # (ref,sec) -> (file, xf, yf, perp_baseline)
    for f in files:
        ref, sec = dates_from_name(f)
        with h5py.File(f, "r") as h:
            lonf = h["science/grids/data/longitude"][:]
            latf = h["science/grids/data/latitude"][:]
            xf = int(np.argmin(np.abs(lonf - lon_anchor)))
            yf = int(np.argmin(np.abs(latf - lat_anchor)))
            if (xf + cols > len(lonf) or yf + rows > len(latf)
                    or abs(lonf[xf] - lon_anchor) > 1.5 * dlon):
                print("  ! grilla incompatible, salto", os.path.basename(f)[:40]); continue
            pb = float(np.nanmean(h["science/grids/imagingGeometry/perpendicularBaseline"][()]))
        pairs_map[(ref, sec)] = (f, xf, yf, pb)

    epochs = sorted({d for k in pairs_map for d in k})
    eidx = {d: i for i, d in enumerate(epochs)}
    keys = sorted(pairs_map, key=lambda k: (eidx[k[0]], eidx[k[1]]))

    n = len(keys)
    phase = np.empty((n, rows, cols), np.float32)
    coher = np.empty((n, rows, cols), np.float32)
    pairs = []
    for i, (ref, sec) in enumerate(keys):
        f, xf, yf, pb = pairs_map[(ref, sec)]
        with h5py.File(f, "r") as h:
            unw = h["science/grids/data/unwrappedPhase"][yf:yf+rows, xf:xf+cols].astype(np.float32)
            cc = h["science/grids/data/coherence"][yf:yf+rows, xf:xf+cols].astype(np.float32)
            con = h["science/grids/data/connectedComponents"][yf:yf+rows, xf:xf+cols]
        # connectedComponent == 0 marca píxeles no fiables (sin desenrollar).
        # Conservamos TODAS las componentes no-cero (alta cobertura); los saltos
        # de ciclo entre componentes los corrige luego inversion::unwrap_error
        # por cierre de fase.
        unw = np.where(con > 0, unw, np.nan).astype(np.float32)
        phase[i] = unw
        coher[i] = cc
        pairs.append({"reference": eidx[ref], "secondary": eidx[sec], "perp_baseline_m": pb})

    meta = {
        "wavelength_m": S1_WAVELENGTH, "incidence_deg": 39.0,
        "n_epochs": len(epochs), "n_pairs": n, "rows": int(rows), "cols": int(cols),
        "epochs": [d.strftime("%Y-%m-%d") for d in epochs],
        "pairs": pairs,
        "geo": {"lon0": float(lon[x0]), "lat0": float(lat[y0]),
                "dlon": float(lon[x0+1]-lon[x0]), "dlat": float(lat[y0+1]-lat[y0])},
    }
    json.dump(meta, open(f"{args.out}/meta.json", "w"), indent=2)
    phase.tofile(f"{args.out}/phase.f32")
    coher.tofile(f"{args.out}/coherence.f32")
    print(f"OK -> {args.out}: {len(epochs)} épocas, {n} pares, {rows}x{cols}")


if __name__ == "__main__":
    main()

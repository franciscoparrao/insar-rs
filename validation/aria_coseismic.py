#!/usr/bin/env python3
"""Prepara un export **cosísmico** de un producto ARIA S1-GUNW para la
descomposición asc/desc de insar-rs.

Toma el GUNW (un interferograma pre→post que cruza la fecha del sismo),
recorta a la zona de interés, enmascara connectedComponent==0, convierte la
fase desenrollada a desplazamiento LOS (m, hacia el satélite positivo) y extrae
la geometría (incidencia + rumbo). Escribe meta.json + los.f32, el formato que
consume `examples/decompose_coseismic.rs`.

Uso (uno por geometría):
  aria_coseismic.py <gunw.nc> --out validation/venz_asc  --lon -68.74 --lat 10.40 --half 0.4
  aria_coseismic.py <gunw.nc> --out validation/venz_desc --lon -68.74 --lat 10.40 --half 0.4

Notas:
- El signo: por defecto LOS = -λ/(4π)·φ (acortamiento de rango / acercamiento al
  satélite positivo → alzamiento positivo). Ajusta con --flip-sign si tu
  convención del producto difiere; valida el signo contra un dato conocido
  (GNSS, modelo de falla) en el primer uso.
- La incidencia sale del promedio de imagingGeometry. El heading se intenta leer
  de los metadatos de órbita; si no está, usa el default Sentinel-1 según
  ascendente/descendente (se infiere del nombre del producto).
"""
import argparse
import json
import math
import os

import h5py
import numpy as np

S1_WAVELENGTH = 0.05546576  # banda C, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gunw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--half", type=float, default=0.4)
    ap.add_argument("--flip-sign", action="store_true", help="invierte el signo del LOS")
    ap.add_argument("--heading", type=float, default=None, help="rumbo (deg) override")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lon0, lon1 = args.lon - args.half, args.lon + args.half
    lat0, lat1 = args.lat - args.half, args.lat + args.half
    with h5py.File(args.gunw, "r") as h:
        lon = h["science/grids/data/longitude"][:]
        lat = h["science/grids/data/latitude"][:]
        cx = np.where((lon >= lon0) & (lon <= lon1))[0]
        cy = np.where((lat >= lat0) & (lat <= lat1))[0]
        x0, x1, y0, y1 = cx.min(), cx.max() + 1, cy.min(), cy.max() + 1
        rows, cols = y1 - y0, x1 - x0

        unw = h["science/grids/data/unwrappedPhase"][y0:y1, x0:x1].astype(np.float32)
        con = h["science/grids/data/connectedComponents"][y0:y1, x0:x1]
        # Geometría: incidencia media (la grilla de imagingGeometry es gruesa).
        try:
            inc = float(np.nanmean(h["science/grids/imagingGeometry/incidenceAngle"][()]))
        except Exception:
            inc = 39.0

    unw = np.where(con > 0, unw, np.nan).astype(np.float32)
    sign = +1.0 if args.flip_sign else -1.0
    los = (sign * S1_WAVELENGTH / (4.0 * math.pi) * unw).astype(np.float32)

    # Heading: override → metadato → default según asc/desc (del nombre).
    name = os.path.basename(args.gunw)
    is_asc = "-A-" in name or "asc" in name.lower()
    heading = args.heading if args.heading is not None else (-12.0 if is_asc else -168.0)

    meta = {
        "rows": int(rows), "cols": int(cols),
        "wavelength_m": S1_WAVELENGTH,
        "incidence_deg": round(inc, 3),
        "heading_deg": round(heading, 3),
        "geo": {"lon0": float(lon[x0]), "lat0": float(lat[y0]),
                "dlon": float(lon[x0 + 1] - lon[x0]), "dlat": float(lat[y0 + 1] - lat[y0])},
    }
    json.dump(meta, open(f"{args.out}/meta.json", "w"), indent=2)
    los.tofile(f"{args.out}/los.f32")
    valid = int(np.isfinite(los).sum())
    print(f"OK -> {args.out}: {rows}×{cols}, inc={inc:.1f}°, head={heading:.1f}°, "
          f"{valid} px válidos, LOS {np.nanmin(los)*100:.1f}..{np.nanmax(los)*100:.1f} cm "
          f"({'asc' if is_asc else 'desc'})")


if __name__ == "__main__":
    main()

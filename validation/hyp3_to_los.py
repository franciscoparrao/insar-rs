#!/usr/bin/env python3
"""Convierte productos HyP3 INSAR_GAMMA (asc + desc) al formato que consume
`examples/decompose_coseismic.rs`, sobre una **grilla común** en El Canelo.

Los productos HyP3 vienen en UTM 19S con grillas distintas para asc y desc. El
reproyectado + alineado a una grilla lat/lon común se hace con **SurtGIS nativo**
(sin GDAL): `reproject` (proj4rs) + `clip` (grilla de referencia) + `resample`
(alinea ambos al mismo grid). Solo la lectura final de píxeles usa tifffile.

Para cada geometría escribe meta.json (con el vector de vista ENU medio, de
lv_theta/lv_phi) + los.f32 (desplazamiento LOS en m, hacia el satélite positivo).

Convención HyP3 GAMMA: lv_theta = elevación del vector de vista sobre el plano
horizontal (rad); lv_phi = azimut CCW desde el Este (rad). Suelo→satélite:
  ê = (cosθ·cosφ, cosθ·sinφ, sinθ)  en (E, N, U).
LOS = -λ/(4π)·fase_desenrollada  (alzamiento positivo; verificar signo contra
un dato conocido en el primer uso).

Uso: hyp3_to_los.py data/algarrobo_hyp3 --outdir validation
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess

import numpy as np
import tifffile

S1_WAVELENGTH = 0.05546576  # banda C, m
SRC_EPSG = "EPSG:32719"     # UTM 19S (HyP3 sobre Chile central)
BBOX = (-71.708, -33.397, -71.648, -33.337)  # El Canelo: xmin,ymin,xmax,ymax (lon/lat)
TR = 0.0008                 # ~80 m
SURT = os.path.expanduser("~/proyectos/surtgis/target/release/surtgis")


def surt(*args):
    subprocess.run([SURT, *map(str, args)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def reproject(src, dst):
    surt("reproject", src, dst, "--to", "EPSG:4326", "--from", SRC_EPSG, "--pixel-size", TR)


def to_grid(src, ref, dst):
    """Reproyecta a 4326 y alinea a la grilla de `ref` (mismo origen/celda/dims)."""
    tmp = dst + ".ll.tif"
    reproject(src, tmp)
    surt("resample", tmp, dst, "--reference", ref, "-m", "bilinear")
    return tifffile.imread(dst).astype(np.float64)


def read_geo(tif):
    """Origen + tamaño de píxel desde gdalinfo (para georreferenciar la salida)."""
    out = subprocess.run(["gdalinfo", tif], capture_output=True, text=True).stdout
    ox, oy = map(float, re.search(r"Origin = \(([-\d.]+),([-\d.]+)\)", out).groups())
    px, py = map(float, re.search(r"Pixel Size = \(([-\d.]+),([-\d.]+)\)", out).groups())
    return ox, oy, px, py


def classify(folder):
    """asc/desc por hora de adquisición (Chile: tarde≈asc, mañana≈desc)."""
    m = re.search(r"_(\d{8})T(\d{2})\d{4}_", os.path.basename(folder))
    return "asc" if (m and int(m.group(2)) >= 18) else "desc"


def process(folder, label, ref, geo, outdir):
    base = glob.glob(f"{folder}/*_unw_phase.tif")[0].rsplit("_unw_phase.tif", 1)[0]
    unw = to_grid(f"{base}_unw_phase.tif", ref, f"/tmp/{label}_unw.tif")
    corr = to_grid(f"{base}_corr.tif", ref, f"/tmp/{label}_corr.tif")
    theta = to_grid(f"{base}_lv_theta.tif", ref, f"/tmp/{label}_theta.tif")
    phi = to_grid(f"{base}_lv_phi.tif", ref, f"/tmp/{label}_phi.tif")

    nr, nc = unw.shape
    valid = np.isfinite(unw) & (unw != 0) & (corr > 0.3) & np.isfinite(theta) & (theta != 0)
    los = np.where(valid, -S1_WAVELENGTH / (4 * math.pi) * unw, np.nan).astype(np.float32)

    th, ph = theta[valid].mean(), phi[valid].mean()
    ev = (math.cos(th) * math.cos(ph), math.cos(th) * math.sin(ph), math.sin(th))
    ox, oy, px, py = geo

    out = f"{outdir}/algarrobo_{label}_v"
    os.makedirs(out, exist_ok=True)
    meta = {
        "rows": nr, "cols": nc, "wavelength_m": S1_WAVELENGTH,
        "incidence_deg": round(90 - math.degrees(th), 2),
        "los_vector": {"east": ev[0], "north": ev[1], "up": ev[2]},
        "geo": {"lon0": ox, "lat0": oy, "dlon": px, "dlat": py},
    }
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=2)
    los.tofile(f"{out}/los.f32")
    print(f"[{label}] {nr}×{nc}  coh>0.3: {valid.sum()}/{nr*nc} px  "
          f"θ_elev={math.degrees(th):.1f}° ê=({ev[0]:.3f},{ev[1]:.3f},{ev[2]:.3f})  "
          f"LOS {np.nanmin(los)*100:.1f}..{np.nanmax(los)*100:.1f} cm → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hyp3_dir")
    ap.add_argument("--outdir", default="validation")
    args = ap.parse_args()
    folders = sorted(d.rstrip("/") for d in glob.glob(f"{args.hyp3_dir}/*/")
                     if glob.glob(f"{d}/*_unw_phase.tif"))

    # Grilla de referencia común: reproyecta el unw del primer asc y recórtalo a El Canelo.
    asc = next((f for f in folders if classify(f) == "asc"), folders[0])
    base = glob.glob(f"{asc}/*_unw_phase.tif")[0]
    reproject(base, "/tmp/ref_full.tif")
    # --bbox=... (forma con '=' porque el valor empieza en '-')
    surt("clip", "/tmp/ref_full.tif", "/tmp/ref.tif", "--bbox=" + ",".join(map(str, BBOX)))
    geo = read_geo("/tmp/ref.tif")
    print(f"grilla de referencia: origen {geo[0]:.4f},{geo[1]:.4f} px {geo[2]}")

    for f in folders:
        process(f, classify(f), "/tmp/ref.tif", geo, args.outdir)


if __name__ == "__main__":
    main()

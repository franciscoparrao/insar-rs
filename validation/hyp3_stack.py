#!/usr/bin/env python3
"""Ensambla un **stack SBAS** desde varios pares HyP3 INSAR_GAMMA (una red por
geometría) en el formato que ingiere insar-rs (meta.json + phase.f32 +
coherence.f32), sobre la grilla común de El Canelo.

Reproyecta+alinea cada par con SurtGIS nativo (igual que hyp3_to_los) y arma la
red: épocas únicas + pares (reference/secondary). Luego `validate_maule` invierte
SBAS por geometría → velocidad LOS, y `decompose` combina asc+desc.

Excluye los pares de baseline larga (span, VVR>200) si están en el directorio.

Uso: hyp3_stack.py data/algarrobo_hyp3 --outdir validation
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

S1_WAVELENGTH = 0.05546576
SRC_EPSG = "EPSG:32719"
BBOX = (-71.708, -33.397, -71.648, -33.337)
TR = 0.0008
SURT = os.path.expanduser("~/proyectos/surtgis/target/release/surtgis")


def surt(*a):
    subprocess.run([SURT, *map(str, a)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def to_grid(src, ref, dst):
    surt("reproject", src, dst + ".ll.tif", "--to", "EPSG:4326", "--from", SRC_EPSG, "--pixel-size", TR)
    surt("resample", dst + ".ll.tif", dst, "--reference", ref, "-m", "bilinear")
    return tifffile.imread(dst).astype(np.float64)


def read_geo(tif):
    out = subprocess.run(["gdalinfo", tif], capture_output=True, text=True).stdout
    ox, oy = map(float, re.search(r"Origin = \(([-\d.]+),([-\d.]+)\)", out).groups())
    px, py = map(float, re.search(r"Pixel Size = \(([-\d.]+),([-\d.]+)\)", out).groups())
    return ox, oy, px, py


def classify(folder):
    m = re.search(r"_(\d{8})T(\d{2})\d{4}_", os.path.basename(folder))
    return "asc" if (m and int(m.group(2)) >= 18) else "desc"


def pair_dates(folder):
    m = re.findall(r"(\d{8})T\d{6}", os.path.basename(folder))
    return m[0], m[1]  # (ref earlier, sec later)


def baseline_days(folder):
    m = re.search(r"_VV[RP](\d+)_", os.path.basename(folder))  # VVR (viejos) y VVP (nuevos)
    return int(m.group(1)) if m else 0


def fetch_dem(ref):
    surt("stac", "fetch-mosaic", "--catalog", "pc", "--bbox=" + ",".join(map(str, BBOX)),
         "--collection", "cop-dem-glo-30", "/tmp/cop_dem.tif")
    surt("resample", "/tmp/cop_dem.tif", "/tmp/dem_ref.tif", "--reference", ref, "-m", "bilinear")
    return tifffile.imread("/tmp/dem_ref.tif").astype(np.float32)


def build(label, folders, ref, geo, dem, outdir, tag=""):
    folders = sorted(folders, key=lambda f: pair_dates(f))
    epochs = sorted({d for f in folders for d in pair_dates(f)})
    eidx = {d: i for i, d in enumerate(epochs)}
    nr = nc = None
    phase, coher, pairs, thetas, phis = [], [], [], [], []
    skipped = []
    for f in folders:
        base = glob.glob(f"{f}/*_unw_phase.tif")[0].rsplit("_unw_phase.tif", 1)[0]
        try:  # salta pares con TIFFs truncados/ilegibles (descarga incompleta)
            unw = to_grid(f"{base}_unw_phase.tif", ref, f"/tmp/s_unw.tif")
            cc = to_grid(f"{base}_corr.tif", ref, f"/tmp/s_corr.tif")
            th = to_grid(f"{base}_lv_theta.tif", ref, f"/tmp/s_th.tif")
            ph = to_grid(f"{base}_lv_phi.tif", ref, f"/tmp/s_ph.tif")
        except Exception:
            skipped.append(os.path.basename(f)[:40])
            continue
        nr, nc = unw.shape
        m = (unw != 0) & np.isfinite(unw) & (cc > 0.3)
        phase.append(np.where(m, unw, np.nan).astype(np.float32))
        coher.append(np.where(m, cc, np.nan).astype(np.float32))
        r, s = pair_dates(f)
        pairs.append({"reference": eidx[r], "secondary": eidx[s], "perp_baseline_m": 0.0})
        thetas.append(th[m].mean()); phis.append(ph[m].mean())
    if skipped:
        print(f"[{label}] saltados {len(skipped)} pares ilegibles: {', '.join(skipped[:4])}{'…' if len(skipped)>4 else ''}")
    # La red combinada (varios envíos) puede tener clusters de fechas sin par que
    # los una → SBAS falla. Quedarse con el COMPONENTE CONEXO más grande.
    adj = {}
    for p in pairs:
        adj.setdefault(p["reference"], set()).add(p["secondary"])
        adj.setdefault(p["secondary"], set()).add(p["reference"])
    seen, comps = set(), []
    for n in adj:
        if n in seen:
            continue
        stack_, comp = [n], []
        while stack_:
            u = stack_.pop()
            if u in seen:
                continue
            seen.add(u); comp.append(u)
            stack_.extend(adj[u] - seen)
        comps.append(set(comp))
    keep = max(comps, key=len) if comps else set()
    dropped = len(pairs)
    kept_idx = [k for k, p in enumerate(pairs)
                if p["reference"] in keep and p["secondary"] in keep]
    pairs = [pairs[k] for k in kept_idx]
    phase = [phase[k] for k in kept_idx]
    coher = [coher[k] for k in kept_idx]
    thetas = [thetas[k] for k in kept_idx]; phis = [phis[k] for k in kept_idx]
    if dropped - len(pairs):
        print(f"[{label}] red: {len(comps)} componentes; conservo el mayor "
              f"({len(keep)} épocas, {len(pairs)} pares), descarto {dropped-len(pairs)} pares de clusters menores")
    # Reindexa al componente conservado.
    used = sorted(keep)
    remap = {old: new for new, old in enumerate(used)}
    epochs = [epochs[i] for i in used]
    for p in pairs:
        p["reference"] = remap[p["reference"]]
        p["secondary"] = remap[p["secondary"]]

    th, ph = float(np.mean(thetas)), float(np.mean(phis))
    ev = (math.cos(th) * math.cos(ph), math.cos(th) * math.sin(ph), math.sin(th))
    ox, oy, px, py = geo
    out = f"{outdir}/algarrobo_{label}{tag}_stack"
    os.makedirs(out, exist_ok=True)
    meta = {
        "wavelength_m": S1_WAVELENGTH, "incidence_deg": round(90 - math.degrees(th), 2),
        "n_epochs": len(epochs), "n_pairs": len(pairs), "rows": nr, "cols": nc,
        "epochs": [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in epochs], "pairs": pairs,
        "los_vector": {"east": ev[0], "north": ev[1], "up": ev[2]},
        "geo": {"lon0": ox, "lat0": oy, "dlon": px, "dlat": py},
    }
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=2)
    np.stack(phase).tofile(f"{out}/phase.f32")
    np.stack(coher).tofile(f"{out}/coherence.f32")
    dem.tofile(f"{out}/dem.f32")
    print(f"[{label}] stack {len(epochs)} épocas, {len(pairs)} pares, {nr}×{nc}  "
          f"ê=({ev[0]:.3f},{ev[1]:.3f},{ev[2]:.3f}) → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hyp3_dir")
    ap.add_argument("--outdir", default="validation")
    ap.add_argument("--max-bt", type=int, default=200, help="excluir pares de baseline > esto (días)")
    ap.add_argument("--tag", default="", help="sufijo del dir de salida (p.ej. _short)")
    args = ap.parse_args()
    allf = [d.rstrip("/") for d in glob.glob(f"{args.hyp3_dir}/*/") if glob.glob(f"{d}/*_unw_phase.tif")]
    allf = [f for f in allf if baseline_days(f) <= args.max_bt]  # excluye span pairs

    asc = next((f for f in allf if classify(f) == "asc"), allf[0])
    reproject_base = glob.glob(f"{asc}/*_unw_phase.tif")[0]
    surt("reproject", reproject_base, "/tmp/ref_full.tif", "--to", "EPSG:4326", "--from", SRC_EPSG, "--pixel-size", TR)
    surt("clip", "/tmp/ref_full.tif", "/tmp/ref.tif", "--bbox=" + ",".join(map(str, BBOX)))
    geo = read_geo("/tmp/ref.tif")
    dem = fetch_dem("/tmp/ref.tif")
    print(f"grilla ref origen {geo[0]:.4f},{geo[1]:.4f}; DEM {dem.shape} {np.nanmin(dem):.0f}..{np.nanmax(dem):.0f} m")

    for label in ("asc", "desc"):
        fs = [f for f in allf if classify(f) == label]
        if fs:
            build(label, fs, "/tmp/ref.tif", geo, dem, args.outdir, args.tag)


if __name__ == "__main__":
    main()

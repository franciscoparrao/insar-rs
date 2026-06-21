#!/usr/bin/env python3
"""Remuestrea tiles Copernicus GLO-30 DEM a la grilla de un export ARIA,
escribiendo dem.f32 (para la corrección troposférica topo-correlacionada).
Georef analítico desde el nombre del tile (sin GDAL). Requiere imagecodecs.
Uso: aria_add_dem.py <export_dir> <tile1.tif> [tile2.tif ...]"""
import json, sys, numpy as np, tifffile

export = sys.argv[1]
m = json.load(open(f"{export}/meta.json")); nr, nc = m["rows"], m["cols"]; g = m["geo"]
lon = g["lon0"] + np.arange(nc) * g["dlon"]; lat = g["lat0"] + np.arange(nr) * g["dlat"]
LON, LAT = np.meshgrid(lon, lat); dem = np.full((nr, nc), np.nan, np.float32)
for path in sys.argv[2:]:
    # nombre ..._S24_00_W068_00_DEM.tif -> esquina SW (lat -24, lon -68)
    parts = path.split("Copernicus_DSM_COG_10_")[1].split("_")
    la, lo = parts[0], parts[2]  # "S24", "W068"
    slat = -int(la[1:]) if la[0] == "S" else int(la[1:])
    wlon = -int(lo[1:]) if lo[0] == "W" else int(lo[1:])
    a = tifffile.imread(path).astype(np.float32); H, W = a.shape
    sel = (LON >= wlon) & (LON < wlon + 1) & (LAT >= slat) & (LAT < slat + 1)
    col = np.clip(((LON - wlon) * W).astype(int), 0, W - 1)
    row = np.clip(((slat + 1 - LAT) * H).astype(int), 0, H - 1)
    dem[sel] = a[row[sel], col[sel]]
dem = np.where(dem < -1000, np.nan, dem)
dem.tofile(f"{export}/dem.f32")
print(f"dem.f32: {nr}x{nc}, elev[{np.nanmin(dem):.0f},{np.nanmax(dem):.0f}] m, finitos={np.isfinite(dem).mean():.2f}")

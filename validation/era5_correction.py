#!/usr/bin/env python3
"""Corrección troposférica estratificada por reanálisis (G-7/ERA5): fetch +
interpolación horizontal/vertical + proyección a LOS, delegando la física y
la interpolación a `pyaps3` (la misma librería que usa `mintpy.tropo_pyaps3`)
en vez de reimplementarlas — evita duplicar el kernel físico ya validado y
corregido de `insar_core::troposphere::era5` (auditoría 2026-07-05,
hallazgos C-1/A-7/A-8) con una segunda copia en Python que podría divergir.

Genera un GeoTIFF Float32 por época (`disp_YYYYMMDD.tif`, mismo formato que
`io::write_series`/`io::read_series` del lado Rust) con el retardo LOS en
METROS (positivo = más retardo), listo para aplicarse con:

    insar tropo-era5 <series_dir> <output_dir_de_este_script> <output_final>

## Requiere

- Paquetes Python: `cdsapi`, `pyaps3`, `numpy`, `osgeo` (GDAL), `pyproj` —
  todos disponibles en el venv `.venv-mintpy` de este proyecto (pyaps3 es una
  dependencia de MintPy).
- Credenciales de Copernicus CDS en `~/.cdsapirc` (ver
  https://cds.climate.copernicus.eu/how-to-api) — propias del usuario; el
  motor deliberadamente no las gestiona (ver doc de `troposphere::era5`,
  mismo criterio que `validation/hyp3_*.py` para credenciales de Earthdata).
- Un DEM (GeoTIFF, cualquier CRS/resolución — se remuestrea) cubriendo el
  área del stack, p. ej. uno de `data/dem/` — **o**, si la serie viene del
  pipeline ISCE-nativo (coordenadas de radar, sin CRS real: `io::isce` no
  geocodifica), pasar `--geom-dir` apuntando al `geom_reference/` de ISCE
  (mismo grid rango/azimut que la serie) para leer lon/lat/altura desde sus
  `lat.rdr`/`lon.rdr`/`hgt.rdr` en vez de derivarlos del GeoTransform + un
  DEM externo re-muestreado (que no tiene sentido sobre un grid sin CRS).

## Convención de signo (importante, verificada contra el código de pyaps3)

`PyAPS.getdelay()` devuelve el retardo LOS **crudo** en metros (positivo =
más retardo = más camino óptico), ya dividido por `cos(incidencia)` — sin
ninguna negación adicional. Esto coincide EXACTAMENTE con lo que
`correct_era5_series` del lado Rust espera (`d_corregido = d_obs + (retardo_e
− retardo_ref)`, ver su doc). **No** aplicar la negación extra que hace
`mintpy.tropo_pyaps3.get_delay()` (`pha *= -1`): esa negación es una
convención interna de contabilidad de MintPy, no un requisito físico
universal — aplicarla aquí duplicaría el error atmosférico en vez de
corregirlo (el mismo bug de signo que C-1 en el módulo Rust, ya arreglado).

## Uso

    .venv-mintpy/bin/python validation/era5_correction.py \\
        <series_dir> <dem_path> <output_dir> \\
        --incidence-deg 39.0 --hour 18

O, para una serie en coordenadas de radar (pipeline ISCE-nativo):

    .venv-mintpy/bin/python validation/era5_correction.py \\
        <series_dir> --geom-dir <ruta>/merged/geom_reference <output_dir> \\
        --incidence-deg 39.0 --hour 18

`--hour`: hora UTC de adquisición Sentinel-1 (ERA5 es horario). No se puede
derivar de forma confiable desde `disp_YYYYMMDD.tif` (no lleva hora) —
verificar contra los metadatos SAFE/topsStack del track real y ajustar; el
default (18 UTC) es solo un punto de partida razonable para tracks
descendentes sobre Chile continental.
"""
import argparse
import glob
import os

import numpy as np
import pyaps3 as pa
from osgeo import gdal, osr

try:
    import pyproj
except ImportError:  # pragma: no cover - verificado en el venv del proyecto
    pyproj = None

gdal.UseExceptions()

SERIES_PREFIX = "disp_"


def discover_epochs(series_dir):
    """Fechas `YYYYMMDD` desde `disp_YYYYMMDD.tif`, orden ascendente — mismo
    contrato que `io::read_series` del lado Rust (que deriva las épocas de
    los nombres de archivo, no del orden del directorio)."""
    paths = sorted(glob.glob(os.path.join(series_dir, f"{SERIES_PREFIX}*.tif")))
    dates = [os.path.splitext(os.path.basename(p))[0][len(SERIES_PREFIX):] for p in paths]
    if not dates:
        raise SystemExit(f"{series_dir}: no se encontraron {SERIES_PREFIX}YYYYMMDD.tif")
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    return [dates[i] for i in order], paths[order[0]]


def read_grid_geometry(reference_tif):
    """GeoTransform, WKT de proyección y dims (rows, cols) del primer
    GeoTIFF de la serie — la misma grilla para todas las épocas."""
    try:
        ds = gdal.Open(reference_tif)
    except RuntimeError as e:
        raise SystemExit(f"no se pudo abrir {reference_tif} con GDAL: {e}") from e
    gt = ds.GetGeoTransform()
    wkt = ds.GetProjection()
    cols, rows = ds.RasterXSize, ds.RasterYSize
    ds = None
    return gt, wkt, rows, cols


def pixel_lonlat_grid(gt, wkt, rows, cols):
    """Longitud/latitud (grados, WGS84) del CENTRO de cada píxel — lo que
    espera `pa.PyAPS(lat=..., lon=...)`."""
    x0, dx, _, y0, _, dy = gt
    xs = x0 + dx * (np.arange(cols) + 0.5)
    ys = y0 + dy * (np.arange(rows) + 0.5)
    xx, yy = np.meshgrid(xs, ys)

    src = osr.SpatialReference()
    src.ImportFromWkt(wkt)
    if src.IsGeographic():
        return xx.astype(np.float64), yy.astype(np.float64)

    if pyproj is None:
        raise SystemExit("la grilla no está en coordenadas geográficas y pyproj no está instalado")
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    transformer = pyproj.Transformer.from_crs(src.ExportToProj4(), wgs84.ExportToProj4(), always_xy=True)
    lon, lat = transformer.transform(xx, yy)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def read_geom_reference(geom_dir, rows, cols):
    """lon/lat/altura (grados/metros) directamente de `geom_reference/
    {lon,lat,hgt}.rdr` de ISCE — mismo grid rango/azimut que la serie, sin
    pasar por GeoTransform/CRS (que no existe: `io::isce` no geocodifica).
    Valida que las dimensiones coincidan con la serie antes de continuar."""

    def read_rdr(name):
        path = os.path.join(geom_dir, f"{name}.rdr")
        try:
            ds = gdal.Open(path)
        except RuntimeError as e:
            raise SystemExit(f"no se pudo abrir {path} con GDAL: {e}") from e
        arr = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        return arr

    lon = read_rdr("lon").astype(np.float64)
    lat = read_rdr("lat").astype(np.float64)
    hgt = read_rdr("hgt").astype(np.float64)
    for name, arr in (("lon", lon), ("lat", lat), ("hgt", hgt)):
        if arr.shape != (rows, cols):
            raise SystemExit(
                f"{name}.rdr tiene shape {arr.shape}, se esperaba {(rows, cols)} "
                f"(igual a la serie) — ¿geom_dir corresponde a este mismo stack?"
            )
    return lon, lat, hgt


def resample_dem(dem_path, gt, wkt, rows, cols):
    """DEM reproyectado/remuestreado (bilinear) a la grilla de la serie, en
    metros, vía `gdal.Warp` a un dataset en memoria (sin archivos temporales)."""
    x0, dx, _, y0, _, dy = gt
    x1, y1 = x0 + dx * cols, y0 + dy * rows
    xmin, xmax = sorted((x0, x1))
    ymin, ymax = sorted((y0, y1))

    try:
        warped = gdal.Warp(
            "",
            dem_path,
            format="MEM",
            dstSRS=wkt,
            outputBounds=(xmin, ymin, xmax, ymax),
            width=cols,
            height=rows,
            resampleAlg="bilinear",
        )
    except RuntimeError as e:
        raise SystemExit(f"gdal.Warp falló remuestreando {dem_path} a la grilla de la serie: {e}") from e
    dem = warped.GetRasterBand(1).ReadAsArray().astype(np.float64)
    warped = None

    if not np.isfinite(dem).any() or np.nanmax(dem) == np.nanmin(dem) == 0.0:
        print(
            f"advertencia: el DEM remuestreado de {dem_path} parece no cubrir la grilla "
            "de la serie (todo 0/NaN) — revisar que el DEM cubra el área del stack",
        )
    return dem


def write_delay_tif(path, delay_m, gt, wkt):
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, delay_m.shape[1], delay_m.shape[0], 1, gdal.GDT_Float32)
    ds.SetGeoTransform(gt)
    ds.SetProjection(wkt)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(float("nan"))
    band.WriteArray(delay_m.astype(np.float32))
    band.FlushCache()
    ds = None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("series_dir", help="Directorio con la serie a corregir (disp_YYYYMMDD.tif)")
    ap.add_argument(
        "dem_path",
        nargs="?",
        default=None,
        help="GeoTIFF de DEM cubriendo el área (cualquier CRS/resolución). "
        "Omitir si se usa --geom-dir (serie en coordenadas de radar, sin CRS).",
    )
    ap.add_argument("output_dir", help="Directorio de salida para el cubo de retardo (disp_YYYYMMDD.tif)")
    ap.add_argument("--incidence-deg", type=float, required=True, help="Ángulo de incidencia medio (grados)")
    ap.add_argument(
        "--hour",
        default="18",
        help="Hora UTC de adquisición ('HH', default 18) — verificar contra el track real",
    )
    ap.add_argument(
        "--grib-dir",
        default=None,
        help="Directorio para los .grb descargados (default: <output_dir>/grib; se reutilizan si ya existen)",
    )
    ap.add_argument(
        "--geom-dir",
        default=None,
        help="geom_reference/ de ISCE (lon.rdr/lat.rdr/hgt.rdr, mismo grid rango/azimut "
        "que la serie) — alternativa a dem_path para series en coordenadas de radar "
        "(pipeline ISCE-nativo, sin GeoTransform/CRS real).",
    )
    args = ap.parse_args()
    if args.geom_dir is None and args.dem_path is None:
        ap.error("se requiere dem_path o --geom-dir")

    dates, reference_tif = discover_epochs(args.series_dir)
    gt, wkt, rows, cols = read_grid_geometry(reference_tif)
    if args.geom_dir is not None:
        lon, lat, dem = read_geom_reference(args.geom_dir, rows, cols)
    else:
        lon, lat = pixel_lonlat_grid(gt, wkt, rows, cols)
        dem = resample_dem(args.dem_path, gt, wkt, rows, cols)

    snwe = (float(np.nanmin(lat)), float(np.nanmax(lat)), float(np.nanmin(lon)), float(np.nanmax(lon)))
    grib_dir = args.grib_dir or os.path.join(args.output_dir, "grib")
    os.makedirs(grib_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{len(dates)} épocas, grilla {rows}x{cols}, bbox (S,N,W,E)={snwe}, hora UTC={args.hour}")
    grib_files = pa.ECMWFdload(dates, args.hour, grib_dir, model="ERA5", snwe=snwe)

    for date, gribfile in zip(dates, grib_files):
        print(f"procesando {date} ({gribfile})")
        aps = pa.PyAPS(
            gribfile,
            dem=dem,
            lat=lat,
            lon=lon,
            inc=args.incidence_deg,
            grib="ERA5",
            humidity="Q",
            Del="comb",
        )
        # wvl por defecto (4π) -> salida en metros; getdelay ya proyecta a
        # LOS (divide por cos(incidencia)) — ver docstring del módulo.
        delay_m = aps.getdelay()
        out_path = os.path.join(args.output_dir, f"{SERIES_PREFIX}{date}.tif")
        write_delay_tif(out_path, delay_m, gt, wkt)

    print(f"escrito: {args.output_dir}/{SERIES_PREFIX}YYYYMMDD.tif ({len(dates)} épocas)")
    print(f"siguiente paso: insar tropo-era5 {args.series_dir} {args.output_dir} <salida>")


if __name__ == "__main__":
    main()

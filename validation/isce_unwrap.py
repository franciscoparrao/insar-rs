#!/usr/bin/env python3
"""Desenrolla un interferograma topsApp de ISCE (dir ``merged/``) con snaphu MCF
vía la API de ISCE, esquivando el wrapper ``topsApp.py`` que sale exit 1 tras
``runFilter`` sin llegar al paso de unwrap.

Reproduce ``runUnwrapMcf`` (costMode=SMOOTH, initMethod=MCF, initOnly=True) sobre
los productos ya generados (``filt_topophase.flat`` + ``phsig.cor``), sin
necesitar el ``PICKLE/`` del run. Los parámetros geométricos (earthRadius,
altitude) casi no influyen con MCF+initOnly, así que se usan valores nominales
Sentinel-1 en vez de reconstruirlos del producto fino.

Genera ``filt_topophase.unw`` (+ ``.xml``/``.vrt``) legible por
``io::isce::read_isce_unwrapped_stack`` de insar-rs.

Uso:
    isce_unwrap.py <merged_dir> [--range-looks 19] [--azimuth-looks 7]

Ejecutar con el Python del env isce2:
    ~/miniforge3/envs/isce2/bin/python validation/isce_unwrap.py <merged_dir>
"""
import argparse
import os

import numpy as np

import isce  # noqa: F401  (inicializa sys.path de ISCE / contrib)
import isceobj
from contrib.Snaphu.Snaphu import Snaphu

# Valores nominales Sentinel-1 IW. Con MCF + initOnly el costo estadístico no se
# itera, así que earthRadius/altitude sólo entran en el mapeo de fase→ciclos y su
# efecto es despreciable para un AOI pequeño.
S1_WAVELENGTH_M = 0.05546576
EARTH_RADIUS_M = 6_360_000.0   # pegRadCur ~ lat -33
ALTITUDE_M = 700_000.0         # órbita S1 ~693 km
AZFACT = 0.8
RNGFACT = 0.8
MAX_COMPONENTS = 20
DEFOMAX = 2.0


def unwrap(merged_dir, range_looks, azimuth_looks,
           flat="filt_topophase.flat", corr="phsig.cor", out="filt_topophase.unw"):
    wrap_name = os.path.join(merged_dir, flat)
    corr_name = os.path.join(merged_dir, corr)
    unw_name = os.path.join(merged_dir, out)

    img = isceobj.createImage()
    img.load(wrap_name + ".xml")
    width = img.getWidth()

    # snaphu aborta si el interferograma trae NaN/inf (los deja el filtro en
    # bordes/zonas sin datos). Los llevamos a 0, que snaphu enmascara como
    # inválido. Es un producto ISCE regenerable, así que se sanea in situ.
    length = img.getLength()
    flat = np.fromfile(wrap_name, dtype=np.complex64)
    bad = ~np.isfinite(flat.real) | ~np.isfinite(flat.imag)
    if bad.any():
        flat[bad] = 0
        flat.reshape(length, width).tofile(wrap_name)
        print(f"saneados {int(bad.sum())} px NaN/inf → 0 en {os.path.basename(wrap_name)}")

    corr_looks = range_looks * azimuth_looks / (AZFACT * RNGFACT)

    snp = Snaphu()
    snp.setInitOnly(True)
    snp.setInput(wrap_name)
    snp.setOutput(unw_name)
    snp.setWidth(width)
    snp.setCostMode("SMOOTH")
    snp.setEarthRadius(EARTH_RADIUS_M)
    snp.setWavelength(S1_WAVELENGTH_M)
    snp.setAltitude(ALTITUDE_M)
    snp.setCorrfile(corr_name)
    snp.setInitMethod("MCF")
    snp.setCorrLooks(corr_looks)
    snp.setMaxComponents(MAX_COMPONENTS)
    snp.setDefoMaxCycles(DEFOMAX)
    snp.setRangeLooks(range_looks)
    snp.setAzimuthLooks(azimuth_looks)
    snp.setCorFileFormat("FLOAT_DATA")
    snp.prepare()
    snp.unwrap()

    # Header/VRT para que insar-rs (y gdal) lean el .unw como 2-banda BIL.
    out_img = isceobj.Image.createUnwImage()
    out_img.setFilename(unw_name)
    out_img.setWidth(width)
    out_img.setAccessMode("read")
    out_img.renderVRT()
    out_img.createImage()
    out_img.finalizeImage()
    out_img.renderHdr()

    if snp.dumpConnectedComponents:
        conn = isceobj.Image.createImage()
        conn.setFilename(unw_name + ".conncomp")
        conn.setWidth(width)
        conn.setAccessMode("read")
        conn.setDataType("BYTE")
        conn.renderVRT()
        conn.createImage()
        conn.finalizeImage()
        conn.renderHdr()

    print(f"unwrap OK: {unw_name}  ({width} cols)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("merged_dir", help="dir merged/ del run topsApp")
    ap.add_argument("--range-looks", type=int, default=19)
    ap.add_argument("--azimuth-looks", type=int, default=7)
    ap.add_argument("--flat", default="filt_topophase.flat")
    ap.add_argument("--corr", default="phsig.cor")
    ap.add_argument("--out", default="filt_topophase.unw")
    args = ap.parse_args()
    unwrap(args.merged_dir, args.range_looks, args.azimuth_looks,
           args.flat, args.corr, args.out)


if __name__ == "__main__":
    main()

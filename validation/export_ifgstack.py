#!/usr/bin/env python3
"""Exporta las fases desenrolladas de un ifgramStack.h5 de MintPy a un formato
binario simple que el motor insar-rs puede ingerir, para validación 1:1 de la
inversión SBAS.

Salidas (en --out):
  meta.json   : dims, wavelength, épocas (YYYYMMDD), pares [ref,sec] (índices
                en la lista ordenada de épocas), baselines perpendiculares.
  phase.f32   : array float32 little-endian, orden C (par, fila, col), en
                RADIANES. NaN para inválidos.

Solo se exportan los pares con dropIfgram=True (los que MintPy realmente usa).
"""
import argparse
import json
import sys

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ifgstack", help="ruta a inputs/ifgramStack.h5")
    ap.add_argument("--out", required=True, help="directorio de salida")
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)

    with h5py.File(args.ifgstack, "r") as f:
        print("Datasets:", list(f.keys()), file=sys.stderr)
        attrs = dict(f.attrs)
        wavelength = float(attrs["WAVELENGTH"])
        incidence = float(attrs.get("INCIDENCE_ANGLE", attrs.get("CENTER_INCIDENCE_ANGLE", 39.0)))

        unw = f["unwrapPhase"]  # (n_ifg, length, width) float32, radianes
        date12 = f["date"][:]   # (n_ifg, 2) bytes 'YYYYMMDD'
        drop = f["dropIfgram"][:] if "dropIfgram" in f else np.ones(unw.shape[0], bool)
        bperp = f["bperp"][:] if "bperp" in f else np.zeros(unw.shape[0], np.float32)

        n_ifg, length, width = unw.shape
        keep = np.where(drop)[0]
        print(f"ifgs totales={n_ifg}, usados (dropIfgram)={len(keep)}, "
              f"grilla={length}x{width}", file=sys.stderr)

        # Épocas: unión ordenada de todas las fechas usadas.
        def dec(x):
            return x.decode() if isinstance(x, bytes) else str(x)
        dates_all = sorted({dec(date12[i, 0]) for i in keep} |
                           {dec(date12[i, 1]) for i in keep})
        date_idx = {d: k for k, d in enumerate(dates_all)}
        n_epochs = len(dates_all)

        pairs = []
        phase = np.empty((len(keep), length, width), dtype=np.float32)
        for row, i in enumerate(keep):
            d1, d2 = dec(date12[i, 0]), dec(date12[i, 1])
            ref, sec = date_idx[d1], date_idx[d2]
            assert ref < sec, f"par {d1}_{d2} con ref>=sec"
            # bperp del par: MintPy guarda bperp por ifg directamente.
            pairs.append({"reference": ref, "secondary": sec,
                          "perp_baseline_m": float(bperp[i])})
            phase[row] = unw[i]

    # Formato de fecha ISO para insar-rs (YYYY-MM-DD).
    epochs_iso = [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in dates_all]

    meta = {
        "wavelength_m": wavelength,
        "incidence_deg": incidence,
        "n_epochs": n_epochs,
        "n_pairs": len(pairs),
        "rows": int(length),
        "cols": int(width),
        "epochs": epochs_iso,
        "pairs": pairs,
    }
    with open(f"{args.out}/meta.json", "w") as fo:
        json.dump(meta, fo, indent=2)
    phase.tofile(f"{args.out}/phase.f32")
    print(f"OK → {args.out}/meta.json + phase.f32 "
          f"({n_epochs} épocas, {len(pairs)} pares)", file=sys.stderr)


if __name__ == "__main__":
    main()

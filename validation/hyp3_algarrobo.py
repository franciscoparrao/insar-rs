#!/usr/bin/env python3
"""Encola interferogramas Sentinel-1 (INSAR_GAMMA) en **HyP3** (ASF) sobre
Algarrobo / El Canelo, para obtener el **descendente** que ARIA no tiene y así
habilitar la descomposición vertical (asc + desc) de insar-rs.

HyP3 procesa los SLC crudos en la nube y devuelve GeoTIFFs (fase desenrollada,
coherencia, y —clave— los mapas de vector de vista lv_theta/lv_phi) listos para
ingerir. El descendente (track 156) + ascendente (track 18) sobre el mismo
período permiten resolver el alzamiento vertical de la costa.

Por defecto hace DRY-RUN: lista los pares que se enviarían y estima créditos,
sin gastar nada. Para enviar de verdad: --submit. Para esperar + descargar:
--submit --watch.

Auth: usa ~/.netrc (machine urs.earthdata.nasa.gov), igual que asf_search.

Ejemplos:
  python validation/hyp3_algarrobo.py                       # dry-run, plan
  python validation/hyp3_algarrobo.py --submit --watch      # lanza y baja
"""
import argparse
from collections import defaultdict
from datetime import datetime

import asf_search as asf

PT = "POINT(-71.68888 -33.36737)"  # Playa El Canelo, Algarrobo
CREDITS_PER_JOB = 15  # INSAR_GAMMA, aprox (cuota mensual HyP3 ~10.000)


def find_pairs(track, fd, start, end, max_bt, max_pairs):
    """Pares short-baseline consecutivos de SLC en un track."""
    res = asf.search(
        intersectsWith=PT, platform="Sentinel-1", processingLevel="SLC",
        beamMode="IW", flightDirection=fd, relativeOrbit=track,
        start=start, end=end,
    )
    scenes = []
    for r in res:
        p = r.properties
        scenes.append((p["startTime"][:10], p["sceneName"]))
    # una escena por fecha (evita frames duplicados del mismo paso)
    by_date = {}
    for d, nm in scenes:
        by_date.setdefault(d, nm)
    seq = sorted(by_date.items())  # [(fecha, granule), ...] ascendente en tiempo
    pairs = []
    for i in range(len(seq) - 1):
        d1, g1 = seq[i]
        d2, g2 = seq[i + 1]
        bt = (datetime.strptime(d2, "%Y-%m-%d") - datetime.strptime(d1, "%Y-%m-%d")).days
        if bt <= max_bt:
            pairs.append((d1, d2, bt, g1, g2))
    # los más recientes primero, recorta a max_pairs
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:max_pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--end", default="2026-06-26")
    ap.add_argument("--asc-track", type=int, default=18)
    ap.add_argument("--desc-track", type=int, default=156)
    ap.add_argument("--max-bt", type=int, default=24, help="baseline temporal máx (días)")
    ap.add_argument("--max-pairs", type=int, default=4, help="pares por geometría")
    ap.add_argument("--out", default="data/algarrobo_hyp3")
    ap.add_argument("--submit", action="store_true", help="ENVIAR los jobs (gasta créditos)")
    ap.add_argument("--watch", action="store_true", help="esperar y descargar resultados")
    args = ap.parse_args()

    plan = {
        ("asc", "ASCENDING", args.asc_track): find_pairs(
            args.asc_track, "ASCENDING", args.start, args.end, args.max_bt, args.max_pairs),
        ("desc", "DESCENDING", args.desc_track): find_pairs(
            args.desc_track, "DESCENDING", args.start, args.end, args.max_bt, args.max_pairs),
    }

    total = 0
    print(f"=== Plan HyP3 INSAR_GAMMA — El Canelo ({args.start} .. {args.end}) ===")
    for (label, fd, track), pairs in plan.items():
        print(f"\n[{label}] track {track} ({fd}): {len(pairs)} pares")
        for d1, d2, bt, g1, g2 in pairs:
            print(f"  {d1} → {d2}  ({bt}d)  {g1[:27]}…")
        total += len(pairs)
    print(f"\nTOTAL: {total} jobs ≈ {total * CREDITS_PER_JOB} créditos "
          f"(cuota mensual HyP3 ~10.000)")

    if not args.submit:
        print("\nDRY-RUN — nada enviado. Para lanzar: agrega --submit (y --watch para bajar).")
        return

    from hyp3_sdk import HyP3
    hyp3 = HyP3()  # ~/.netrc
    batch = None
    from hyp3_sdk import Batch
    batch = Batch()
    for (label, fd, track), pairs in plan.items():
        for d1, d2, bt, g1, g2 in pairs:
            job = hyp3.submit_insar_job(
                g1, g2, name=f"algarrobo_{label}_{d1}_{d2}",
                include_look_vectors=True,   # lv_theta/lv_phi → geometría para decompose
                include_inc_map=True,
                apply_water_mask=True,
                looks="20x4",
            )
            batch += job
            print(f"enviado [{label}] {d1}→{d2}  job {job.job_id[:8]}")
    print(f"\n{len(batch)} jobs en cola.")

    if args.watch:
        import os
        os.makedirs(args.out, exist_ok=True)
        print("esperando a HyP3 (puede tardar ~20-40 min/lote)…")
        batch = hyp3.watch(batch)
        batch.download_files(args.out)
        print(f"descargado → {args.out}")
    else:
        print("jobs en cola; corre con --watch para esperar y descargar, "
              "o revisa en https://hyp3-api.asf.alaska.edu / search.asf.alaska.edu/#/?topic=on-demand")


if __name__ == "__main__":
    main()

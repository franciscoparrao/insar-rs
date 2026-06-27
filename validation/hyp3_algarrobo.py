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


def find_pairs(track, fd, start, end, max_bt, max_pairs, pairing="sequential", n_epochs=6):
    """Pares de SLC en un track.

    - pairing="sequential": pares short-baseline consecutivos (≤max_bt días),
      recortados a max_pairs (los más recientes).
    - pairing="span": UN par de baseline larga, de la escena más antigua a la
      más reciente de la ventana (desplazamiento acumulado). Ignora max_bt.
    - pairing="sbas": red SBAS redundante sobre las `n_epochs` épocas más
      recientes: pares i→i+1 e i→i+2 (loops para promediar atmósfera).
    """
    res = asf.search(
        intersectsWith=PT, platform="Sentinel-1", processingLevel="SLC",
        beamMode="IW", flightDirection=fd, relativeOrbit=track,
        start=start, end=end,
    )
    # una escena por fecha (evita frames duplicados del mismo paso)
    by_date = {}
    for r in res:
        p = r.properties
        by_date.setdefault(p["startTime"][:10], p["sceneName"])
    seq = sorted(by_date.items())  # [(fecha, granule), ...] ascendente en tiempo
    if len(seq) < 2:
        return []

    def bt_days(a, b):
        return (datetime.strptime(b, "%Y-%m-%d") - datetime.strptime(a, "%Y-%m-%d")).days

    if pairing == "span":
        (d1, g1), (d2, g2) = seq[0], seq[-1]
        return [(d1, d2, bt_days(d1, d2), g1, g2)]

    if pairing == "sbas":
        # N épocas repartidas uniformemente en la ventana (span + pares cortos).
        if len(seq) <= n_epochs:
            ep = seq
        else:
            idx = [round(k * (len(seq) - 1) / (n_epochs - 1)) for k in range(n_epochs)]
            ep = [seq[i] for i in sorted(set(idx))]
        pairs = []
        for i in range(len(ep)):
            for j in (i + 1, i + 2):  # i→i+1 e i→i+2 (redundancia)
                if j < len(ep):
                    (d1, g1), (d2, g2) = ep[i], ep[j]
                    pairs.append((d1, d2, bt_days(d1, d2), g1, g2))
        return pairs

    pairs = []
    for i in range(len(seq) - 1):
        d1, g1 = seq[i]
        d2, g2 = seq[i + 1]
        bt = bt_days(d1, d2)
        if bt <= max_bt:
            pairs.append((d1, d2, bt, g1, g2))
    pairs.sort(key=lambda x: x[1], reverse=True)  # más recientes primero
    return pairs[:max_pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--end", default="2026-06-26")
    ap.add_argument("--asc-track", type=int, default=18)
    ap.add_argument("--desc-track", type=int, default=156)
    ap.add_argument("--max-bt", type=int, default=24, help="baseline temporal máx (días)")
    ap.add_argument("--max-pairs", type=int, default=4, help="pares por geometría")
    ap.add_argument("--pairing", choices=["sequential", "span", "sbas"], default="sequential",
                    help="sbas = red redundante (i+1,i+2) sobre N épocas; span = 1 par largo")
    ap.add_argument("--n-epochs", type=int, default=6, help="épocas para la red SBAS")
    ap.add_argument("--out", default="data/algarrobo_hyp3")
    ap.add_argument("--submit", action="store_true", help="ENVIAR los jobs (gasta créditos)")
    ap.add_argument("--watch", action="store_true", help="esperar y descargar resultados")
    args = ap.parse_args()

    plan = {
        ("asc", "ASCENDING", args.asc_track): find_pairs(
            args.asc_track, "ASCENDING", args.start, args.end, args.max_bt, args.max_pairs,
            args.pairing, args.n_epochs),
        ("desc", "DESCENDING", args.desc_track): find_pairs(
            args.desc_track, "DESCENDING", args.start, args.end, args.max_bt, args.max_pairs,
            args.pairing, args.n_epochs),
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
    from hyp3_sdk import Batch
    batch = Batch()
    for (label, fd, track), pairs in plan.items():
        for d1, d2, bt, g1, g2 in pairs:
            name = f"algarrobo_{label}_{d1}_{d2}"
            # Idempotente: reusa un job ya enviado con este nombre (no re-gasta).
            existing = hyp3.find_jobs(name=name)
            if len(existing) > 0:
                print(f"reuso [{label}] {name} ({len(existing)} job/s ya en HyP3)")
                batch += existing
                continue
            # submit_insar_job devuelve un Batch en hyp3_sdk v7.
            submitted = hyp3.submit_insar_job(
                g1, g2, name=name,
                include_look_vectors=True,   # lv_theta/lv_phi → geometría para decompose
                include_inc_map=True,
                apply_water_mask=True,
                looks="20x4",
            )
            batch += submitted
            print(f"enviado [{label}] {d1}→{d2}  ({name})")
    print(f"\n{len(batch)} jobs en cola/seguimiento.")

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

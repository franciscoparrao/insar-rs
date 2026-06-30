#!/usr/bin/env python3
"""Descarga una pila de **SLC-BURST** de Sentinel-1 sobre El Canelo (Algarrobo)
para el path de stack coregistrado (ISCE → insar-rs), variante liviana por
bursts (~133 MB c/u, vs escenas SLC completas de 4-8 GB).

Un burst (~20×20 km) sobra para El Canelo. Baja la pila ascendente (track 18) y
descendente (track 156) en una ventana, con revisita densa → baselines cortos
naturales para PS-InSAR + SBAS.

Uso: download_bursts_algarrobo.py --n 25 --start 2025-06-01 --end 2026-06-26
Auth: ~/.netrc (Earthdata), igual que el resto del flujo.
"""
import argparse
import os

import asf_search as asf

PT = "POINT(-71.68888 -33.36737)"
TRACKS = {"asc": ("ASCENDING", 18), "desc": ("DESCENDING", 156)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="bursts por geometría (los más recientes)")
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--end", default="2026-06-26")
    ap.add_argument("--out", default="data/algarrobo_bursts")
    args = ap.parse_args()
    sess = asf.ASFSession()  # ~/.netrc

    for label, (fd, track) in TRACKS.items():
        out = f"{args.out}/{label}"
        os.makedirs(out, exist_ok=True)
        res = asf.search(intersectsWith=PT, dataset="SLC-BURST", flightDirection=fd,
                         relativeOrbit=track, start=args.start, end=args.end)
        # una burst-id (mismo subswath/burst) por fecha → pila consistente
        by_date = {}
        for r in res:
            p = r.properties
            by_date.setdefault(p["startTime"][:10], r)
        picked = [by_date[d] for d in sorted(by_date)[-args.n:]]
        print(f"[{label}] track {track}: {len(picked)} bursts ({sorted(by_date)[-args.n]}..{sorted(by_date)[-1]})", flush=True)
        for i, r in enumerate(picked, 1):
            try:
                r.download(path=out, session=sess)
                print(f"  [{i}/{len(picked)}] {r.properties['fileName']}", flush=True)
            except Exception as e:
                print("  ERR", r.properties.get("fileName", "?"), e, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Descarga un subconjunto **ascendente** de ARIA S1-GUNW sobre Algarrobo
(sector El Canelo) para el piloto InSAR.

Solo hay GUNW ascendente (track 18, 2015–2022) en el catálogo ARIA de esta zona;
el descendente existe únicamente como SLC crudo (requiere HyP3/ISCE, fuera del
MVP). Para un piloto tratable se baja una cadena corta de pares de 12 días sobre
el último año disponible → red SBAS conectada (~30 pares, ~1.7 GB).

Uso: download_algarrobo.py            (usa ~/.netrc Earthdata)
"""
import glob
import os
import re
from datetime import datetime

import asf_search as asf

PT = "POINT(-71.68888 -33.36737)"  # Playa El Canelo, Algarrobo
OUT = "data/algarrobo_gunw"
START, END = "2021-05-01", "2022-05-31"
MAX_DT = 12  # días: cadena secuencial corta


def main():
    os.makedirs(OUT, exist_ok=True)
    have = {os.path.basename(f).rsplit(".nc", 1)[0] for f in glob.glob(f"{OUT}/*.nc")}
    res = asf.search(
        intersectsWith=PT, dataset="ARIA S1 GUNW", flightDirection="ASCENDING",
        start=START, end=END,
    )
    todo = []
    for r in res:
        nm = r.properties["sceneName"]
        m = re.search(r"(\d{8})_(\d{8})", nm)
        if not m:
            continue
        ref = datetime.strptime(m.group(1), "%Y%m%d")
        sec = datetime.strptime(m.group(2), "%Y%m%d")
        if abs((ref - sec).days) <= MAX_DT and nm not in have:
            todo.append(r)
    print(f"{len(have)} ya en disco; {len(todo)} pares de ≤{MAX_DT}d por bajar", flush=True)

    sess = asf.ASFSession()  # lee ~/.netrc
    for i, r in enumerate(todo, 1):
        nm = r.properties["sceneName"]
        try:
            r.download(path=OUT, session=sess)
            print(f"[{i}/{len(todo)}] {nm[:50]}", flush=True)
        except Exception as e:
            print("ERR", nm[:40], e, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

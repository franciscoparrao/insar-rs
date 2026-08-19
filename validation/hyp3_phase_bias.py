#!/usr/bin/env python3
"""Encola en **HyP3** (ASF) la red de interferogramas que exige la corrección
de *phase bias* (`insar_core::phase_bias`, Maghsoudi et al. 2022).

Por qué un script aparte de `hyp3_algarrobo.py`: esa red SBAS genérica no
sirve. La corrección impone tres requisitos que hay que cumplir *al pedir los
datos*, no después:

1. **Fase ENVUELTA** (`include_wrapped_phase=True`). El sesgo se estima sobre
   fase envuelta —cierres como argumento del producto complejo— y eso es lo que
   hace la corrección barata. Los productos HyP3 ya bajados no la traen: se
   pidieron sin esa opción y por eso son inservibles para esto.
2. **Spans 1, 2 y 3** en índice de época. Con solo spans 1 y 2 el sistema es
   indeterminado para todo N: N−2 observaciones para N−1 incógnitas. Hacen
   falta dos longitudes de par largo distintas para cerrarlo.
3. **Un par ancla de ~1 año** cuyo span sea múltiplo de mcm(1..M) = 6 épocas,
   para estimar los coeficientes aₙ (Eqs. 8–9). El paper usa 360 días
   justamente porque es divisible por 6, 12 y 18.

El diseño por defecto replica el del paper con la cadencia disponible en Chile:
unidad de 12 días, spans 12/24/36 días, ancla de 360 días (30 épocas).

**El multilooking es la causa del sesgo, no un detalle.** `--looks 20x4` (~80 m)
da más looks y por tanto más sesgo que `10x2`; el filtro Goldstein también lo
altera (Fig. 7 del paper compara filtrado vs no filtrado). Ambos se dejan
explícitos y registrados para que el caso sea reproducible.

Por defecto hace DRY-RUN: imprime la red, los créditos y el disco estimado sin
gastar nada. Para enviar: --submit. Para esperar y bajar: --submit --watch.

Auth: ~/.netrc (machine urs.earthdata.nasa.gov).

Ejemplos:
  python validation/hyp3_phase_bias.py                    # plan, no gasta
  python validation/hyp3_phase_bias.py --submit --watch   # lanza y baja
"""
import argparse
from collections import Counter
from datetime import datetime

import asf_search as asf

PT = "POINT(-71.68888 -33.36737)"  # Playa El Canelo, Algarrobo
CREDITS_PER_JOB = 15  # INSAR_GAMMA
MB_PER_JOB = 250      # con wrapped phase; los productos ya bajados pesan ~200


def lcm_range(m):
    """mcm(1..m) — el span que debe tener el ancla para teselar con toda
    cadena de span ≤ m."""
    from math import gcd
    out = 1
    for k in range(1, m + 1):
        out = out * k // gcd(out, k)
    return out


def scene_dates(track, fd, start, end):
    """{fecha: granule} en el track, una escena por fecha."""
    res = asf.search(
        intersectsWith=PT, platform="Sentinel-1", processingLevel="SLC",
        beamMode="IW", flightDirection=fd, relativeOrbit=track,
        start=start, end=end,
    )
    by_date = {}
    for r in res:
        p = r.properties
        by_date.setdefault(p["startTime"][:10], p["sceneName"])
    return by_date


def pick_epochs(by_date, unit_days, n_epochs):
    """Submuestrea las fechas disponibles a una grilla ~regular de `unit_days`.

    Greedy desde la más antigua: acepta en cada paso la fecha cuyo intervalo
    respecto de la última aceptada esté más cerca de `unit_days`. Con datos de
    6 días esto produce una cadena de 12 días saltando una de cada dos; donde el
    origen ya tiene un hueco de 12, el paso resultante puede ser 12 o 18, y esa
    irregularidad es justamente la razón de indexar los spans por ORDEN DE
    ÉPOCA y no por días de calendario.
    """
    seq = sorted(by_date)
    if not seq:
        return []
    picked = [seq[0]]
    while len(picked) < n_epochs:
        last = datetime.fromisoformat(picked[-1])
        cands = [
            (abs((datetime.fromisoformat(d) - last).days - unit_days), d)
            for d in seq
            if (datetime.fromisoformat(d) - last).days > 0
        ]
        if not cands:
            break
        picked.append(min(cands)[1])
    return picked


def build_network(epochs, max_span, anchor_span):
    """Pares (i, j) en índice de época: cadena daisy de spans 1..max_span más
    las anclas de span `anchor_span`. Devuelve lista de (i, j)."""
    n = len(epochs)
    pairs = [(i, i + s) for s in range(1, max_span + 1) for i in range(n - s)]
    anchors = [(i, i + anchor_span) for i in range(n - anchor_span)]
    return pairs, anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-08-01")
    ap.add_argument("--track", type=int, default=18)
    ap.add_argument("--direction", default="ASCENDING", choices=["ASCENDING", "DESCENDING"])
    ap.add_argument("--unit-days", type=int, default=12,
                    help="intervalo de la cadena unitaria (12 = diseño del paper "
                         "con la cadencia chilena; 6 da más sesgo pero el ancla "
                         "de 1 año cuesta el doble de jobs)")
    ap.add_argument("--n-epochs", type=int, default=31,
                    help="épocas de la red (31 × 12 días ≈ 360 días)")
    ap.add_argument("--max-span", type=int, default=3,
                    help="span máximo en índice de época; DEBE ser ≥ 3")
    ap.add_argument("--anchor-span", type=int, default=30,
                    help="span del ancla en épocas; debe ser múltiplo de mcm(1..max_span)")
    ap.add_argument("--looks", default="20x4", choices=["20x4", "10x2"])
    ap.add_argument("--phase-filter", type=float, default=0.6,
                    help="parámetro del filtro Goldstein (afecta el cierre, Fig. 7 del paper)")
    ap.add_argument("--out", default="data/algarrobo_phasebias")
    ap.add_argument("--submit", action="store_true", help="ENVIAR los jobs")
    ap.add_argument("--watch", action="store_true", help="esperar y descargar")
    args = ap.parse_args()

    if args.max_span < 3:
        ap.error("max_span < 3 deja el sistema indeterminado para todo N "
                 "(N−2 observaciones para N−1 incógnitas). Usa 3 o más.")
    step = lcm_range(args.max_span)
    if args.anchor_span % step:
        ap.error(f"anchor_span={args.anchor_span} no es múltiplo de mcm(1..{args.max_span})={step}: "
                 f"las cadenas no teselan el ancla y los coeficientes no se pueden estimar.")
    if args.anchor_span >= args.n_epochs:
        ap.error(f"anchor_span={args.anchor_span} ≥ n_epochs={args.n_epochs}: no cabe ningún ancla.")

    by_date = scene_dates(args.track, args.direction, args.start, args.end)
    epochs = pick_epochs(by_date, args.unit_days, args.n_epochs)
    if len(epochs) < args.n_epochs:
        print(f"AVISO: solo {len(epochs)} épocas disponibles de las {args.n_epochs} pedidas.")
    pairs, anchors = build_network(epochs, args.max_span, args.anchor_span)

    # --- Reporte del plan ---
    gaps = [(datetime.fromisoformat(b) - datetime.fromisoformat(a)).days
            for a, b in zip(epochs, epochs[1:])]
    total_days = (datetime.fromisoformat(epochs[-1]) - datetime.fromisoformat(epochs[0])).days
    print(f"=== Red para phase bias — track {args.track} {args.direction} ===")
    print(f"  {len(epochs)} épocas: {epochs[0]} .. {epochs[-1]}  ({total_days} días)")
    print(f"  intervalos reales entre épocas: {dict(sorted(Counter(gaps).items()))} días")
    print(f"  looks {args.looks} · filtro Goldstein {args.phase_filter}")

    by_span = Counter(j - i for i, j in pairs)
    print(f"\n  cadena daisy (spans 1..{args.max_span}):")
    for s in sorted(by_span):
        med = sorted((datetime.fromisoformat(epochs[i + s]) - datetime.fromisoformat(epochs[i])).days
                     for i in range(len(epochs) - s))[(len(epochs) - s) // 2]
        print(f"    span {s}: {by_span[s]:>3} pares  (~{med} días)")
    anchor_days = [(datetime.fromisoformat(epochs[j]) - datetime.fromisoformat(epochs[i])).days
                   for i, j in anchors]
    print(f"    ancla span {args.anchor_span}: {len(anchors):>3} pares  "
          f"(~{anchor_days[0] if anchor_days else 0} días)")

    all_pairs = pairs + anchors
    n = len(all_pairs)
    n_unknowns = len(epochs) - 1
    n_obs = sum(len(epochs) - s for s in range(2, args.max_span + 1))
    print(f"\n  sistema de cierres: {n_obs} observaciones / {n_unknowns} incógnitas "
          f"→ {'SOBREDETERMINADO ✓' if n_obs >= n_unknowns else 'INDETERMINADO ✗'}")
    print(f"\nTOTAL: {n} jobs ≈ {n * CREDITS_PER_JOB} créditos (cuota mensual ~10.000) "
          f"· ~{n * MB_PER_JOB / 1024:.1f} GB de descarga")

    if not args.submit:
        print("\nDRY-RUN — nada enviado. Para lanzar: --submit (y --watch para bajar).")
        return

    from hyp3_sdk import Batch, HyP3
    hyp3 = HyP3()
    batch = Batch()
    for i, j in all_pairs:
        d1, d2 = epochs[i], epochs[j]
        name = f"pbias_{args.track}_{d1}_{d2}"
        existing = hyp3.find_jobs(name=name)
        if len(existing) > 0:
            print(f"reuso {name}")
            batch += existing
            continue
        batch += hyp3.submit_insar_job(
            by_date[d1], by_date[d2], name=name,
            include_wrapped_phase=True,   # ← el requisito que faltaba
            include_look_vectors=True,
            include_inc_map=True,
            apply_water_mask=True,
            looks=args.looks,
            phase_filter_parameter=args.phase_filter,
        )
        print(f"enviado {d1}→{d2} (span {j - i})")
    print(f"\n{len(batch)} jobs en cola/seguimiento.")

    if args.watch:
        import os
        os.makedirs(args.out, exist_ok=True)
        print("esperando a HyP3 (puede tardar horas para un lote de este tamaño)…")
        batch = hyp3.watch(batch)
        batch.download_files(args.out)
        print(f"descargado → {args.out}")
    else:
        print("jobs en cola; corre con --watch para esperar y descargar.")


if __name__ == "__main__":
    main()

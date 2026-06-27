#!/usr/bin/env bash
# Piloto El Canelo (Algarrobo) — cadena ASCENDENTE GUNW track 18.
# Requiere: data/algarrobo_gunw/*.nc (validation/download_algarrobo.py).
# Produce: serie SBAS LOS + velocidad + coherencia temporal sobre El Canelo.
#
# Nota: ascendente solo da LOS (mezcla vertical + Este). El vertical (decompose)
# requiere descendente, que en esta zona hay que generar con HyP3 desde SLC.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv-mintpy/bin/python
EXPORT=validation/algarrobo_asc_export
# AOI El Canelo + pueblo de Algarrobo (cobertura para PS + píxel de referencia).
LON=-71.678; LAT=-33.367; HALF=0.030

echo ">> apilando GUNW ascendente → $EXPORT"
$PY validation/aria_to_stack.py data/algarrobo_gunw --out "$EXPORT" \
    --lon "$LON" --lat "$LAT" --half "$HALF"

echo ">> inversión SBAS + velocidad + coherencia (DERAMP para quitar rampa orbital)"
DERAMP=1 cargo run --release -q -p insar-core --example validate_maule -- "$EXPORT"

echo ">> listo: $EXPORT/{velocity,tcoh,series}.f32"
echo "   (velocidad LOS ascendente; para vertical, sumar descendente vía HyP3)"

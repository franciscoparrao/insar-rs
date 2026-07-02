#!/usr/bin/env bash
# Coregistra una pila de mini-SAFE (burst2stack) con ISCE topsStack y genera la
# red de interferogramas para insar-rs. Un run por geometría (asc/desc).
#
# Prerequisitos ya satisfechos en el repo:
#   - SAFEs:   data/algarrobo_safe/{asc,desc}/*.SAFE   (burst2stack)
#   - órbitas: data/algarrobo_safe/orbits/*.EOF        (python -m eof)
#   - DEM:     data/algarrobo_isce/dem/dem.wgs84       (cubre el AOI)
#   - aux:     data/algarrobo_safe/aux                 (vacío; ok para InSAR)
#
# Uso:
#   validation/run_topsstack.sh asc   IW1
#   validation/run_topsstack.sh desc  IW2
# Luego ejecutar en orden los run_files generados:
#   cd data/algarrobo_stack/<geom> && bash run_files/run_01_* ... (o el helper de abajo)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISCE_STACK="$HOME/miniforge3/envs/isce2/share/isce2"
PY="$HOME/miniforge3/envs/isce2/bin/python"
export PYTHONPATH="$ISCE_STACK:${PYTHONPATH:-}"
# isce2/bin primero → los SentinelWrapper.py (shebang python3) usan el python del env.
export PATH="$HOME/miniforge3/envs/isce2/bin:$ISCE_STACK/topsStack:$PATH"

# Modo exec: corre en orden todos los run_files de una geometría ya configurada.
# Cada run_NN es un script con una línea por tarea; se ejecutan secuencialmente.
if [[ "${1:-}" == "exec" ]]; then
  GEOM="${2:?asc|desc}"
  WORK="$ROOT/data/algarrobo_stack/$GEOM"
  cd "$WORK"
  for rf in $(ls run_files/run_[0-9]* | sort -t_ -k2 -n); do
    echo ">> $(date +%H:%M:%S) $rf  [libre: $(df --output=avail -BG /home | tail -1 | tr -d ' ')]"
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      eval "$line"
    done < "$rf"
    # Limpieza-sobre-la-marcha: topsStack procesa bursts a resolución completa;
    # los intermedios (coreg SLC ~25GB, igrams por-burst ~20-40GB) no caben junto
    # al final. Se borran apenas dejan de ser necesarios (merged/ ya es AOI-crop).
    case "$rf" in
      *run_08_generate_burst_igram)   # coreg SLC ya consumidos por todos los igrams
        rm -rf coreg_secondarys secondarys; echo "   limpio coreg_secondarys" ;;
      *run_09_merge_burst_igram)      # igrams por-burst ya fusionados a merged/
        rm -rf interferograms; echo "   limpio interferograms por-burst" ;;
    esac
  done
  echo ">> stack $GEOM COMPLETO: merged/interferograms/*/filt_fine.unw"
  exit 0
fi

GEOM="${1:?asc|desc}"
SWATH="${2:?IW1|IW2|IW3}"

SLC="$ROOT/data/algarrobo_safe/$GEOM"
ORBITS="$ROOT/data/algarrobo_safe/orbits"
AUX="$ROOT/data/algarrobo_safe/aux"
DEM="$ROOT/data/algarrobo_isce/dem/dem.wgs84"
WORK="$ROOT/data/algarrobo_stack/$GEOM"
BBOX="-33.397 -33.337 -71.708 -71.648"   # S N W E (El Canelo)
NUM_CONN="${3:-2}"                        # vecinos por fecha (SBAS). 2 = disco-seguro
                                          # (~47 pares); 4 no cabe a res completa.

mkdir -p "$WORK" "$AUX"

# ISCE 2.6.3 no reconoce Sentinel-1D ("unknown mission id S1D") → excluir esas
# fechas. S1A/S1C sí funcionan. Auto-detecta las fechas de SAFEs S1D.
EXCL=$(ls -d "$SLC"/S1D_*.SAFE 2>/dev/null | grep -oE '_20[0-9]{6}T' | grep -oE '20[0-9]{6}' | sort -u | paste -sd,)
EXCL_ARG=()
if [[ -n "$EXCL" ]]; then
  echo ">> excluyendo fechas S1D (no soportadas por ISCE 2.6.3): $EXCL"
  EXCL_ARG=(-x "$EXCL")
fi

echo ">> stackSentinel $GEOM $SWATH  (bbox: $BBOX, conn: $NUM_CONN)"
"$PY" "$ISCE_STACK/topsStack/stackSentinel.py" \
  -s "$SLC" -o "$ORBITS" -a "$AUX" -d "$DEM" \
  -w "$WORK" -W interferogram -C geometry \
  -n "${SWATH#IW}" -b "$BBOX" -c "$NUM_CONN" -p vv "${EXCL_ARG[@]}"

echo ">> run_files generados en $WORK/run_files/"
ls "$WORK/run_files/" | grep -v '\.job$' | sort
echo ">> Ejecutar en orden con: validation/run_topsstack.sh exec $GEOM"

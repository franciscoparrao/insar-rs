#!/usr/bin/env bash
# Benchmark end-to-end insar-rs vs MintPy sobre Fernandina (tabla T2 del paper).
# Tarea común: de los .unw ISCE en disco a una velocidad LOS.
#   insar-rs: validate_fernandina_isce (lee ISCE nativo, invierte, velocidad).
#   MintPy:   load_data + reference_point + ifgram_inversion -w no
#             + timeseries2velocity (cluster=none; hilos vía OpenBLAS).
# 1 calentamiento + N repeticiones por configuración; /usr/bin/time -v por
# etapa. Los intermedios de MintPy se borran antes de cada repetición
# (load_data se salta el paso si ifgramStack.h5 ya existe).
# Uso: validation/bench_fernandina.sh [N=5] [out=validation/bench_out]
set -euo pipefail
N=${1:-5}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(realpath -m "${2:-$ROOT/validation/bench_out}")
D=$ROOT/data/FernandinaSenDT128
MP=$D/mintpy
mkdir -p "$OUT"
cargo build --release -q -p insar-core --example validate_fernandina_isce \
  --manifest-path "$ROOT/Cargo.toml"
BIN=$ROOT/target/release/examples/validate_fernandina_isce
PY=$ROOT/.venv-mintpy/bin

run_rs() {  # $1 hilos, $2 etiqueta de repetición
  RAYON_NUM_THREADS=$1 /usr/bin/time -v -o "$OUT/rs_t$1_$2.time" \
    "$BIN" "$D/merged/interferograms" "$D/baselines" "$OUT/v_rs.f32" \
    > "$OUT/rs_t$1_$2.log" 2>&1
}

run_mp() {  # $1 hilos BLAS, $2 etiqueta
  export OMP_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1 MKL_NUM_THREADS=$1
  ( cd "$MP"
    rm -f inputs/ifgramStack.h5 inputs/geometryRadar.h5 avgSpatialCoh.h5 \
          maskConnComp.h5 timeseries.h5 numInvIfgram.h5 temporalCoherence.h5 \
          "$OUT/v_mp.h5"
    st() { /usr/bin/time -v -o "$OUT/mp_t$1_$2_$3.time" "${@:4}" \
             >> "$OUT/mp_t$1_$2.log" 2>&1; }
    st "$1" "$2" 1load  "$PY/smallbaselineApp.py" smallbaselineApp.cfg --dostep load_data
    st "$1" "$2" 2ref   "$PY/smallbaselineApp.py" smallbaselineApp.cfg --dostep reference_point
    st "$1" "$2" 3inv   "$PY/ifgram_inversion.py" inputs/ifgramStack.h5 -w no
    st "$1" "$2" 4vel   "$PY/timeseries2velocity.py" timeseries.h5 -o "$OUT/v_mp.h5" )
}

for t in 16 1; do
  run_rs "$t" warm
  for i in $(seq 1 "$N"); do run_rs "$t" "$i"; done
done
for t in 16 1; do
  run_mp "$t" warm
  for i in $(seq 1 "$N"); do run_mp "$t" "$i"; done
done
du -sb "$MP/inputs/ifgramStack.h5" "$MP/inputs/geometryRadar.h5" "$MP/timeseries.h5" \
  > "$OUT/intermediates.txt"
"$PY/python" "$ROOT/validation/bench_summary.py" "$OUT"

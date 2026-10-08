# Benchmarks

> Medidos 2026-06-15 con criterion (`cargo bench -p insar-core`) sobre stacks
> sintéticos del tamaño del caso Fernandina (98 épocas, ~285 pares, grilla
> 450×600 = 270 k píxeles). Single-thread salvo donde rayon paraleliza por filas.

## Núcleo SBAS (criterion, tiempo de cómputo aislado)

| Operación | Tamaño | Tiempo (mediana) |
|-----------|--------|------------------|
| `invert_sbas` | 98 ep, 150×200 (~30 k px) | ~0.38 s |
| `invert_sbas` | 98 ep, 450×600 (270 k px) | ~2.9 s |
| `estimate_velocity` | 98 ep, 450×600 | ~94 ms |
| `amplitude_dispersion` | 98 ep, 450×600 | ~0.20 s |

La inversión escala ~lineal con el número de píxeles (la SVD de la matriz de
diseño se factoriza una sola vez y se reutiliza; el costo por píxel es la
multiplicación por la pseudoinversa).

## Comparación de wall-clock vs MintPy (caso Fernandina real)

Sobre el dataset real (288 interferogramas ISCE, 270 k píxeles):

| Etapa | insar-rs | MintPy 1.6.2 |
|-------|----------|--------------|
| Lectura del stack | 3.3 s (lee 288 `.unw` ISCE) | parte de `load_data` (~29 s a HDF5) |
| Inversión + velocidad | **1.8 s** | **55.7 s** (`ifgram_inversion.py -w no`) |

**Caveat de comparación honesta**: los 55.7 s de MintPy incluyen I/O del stack
HDF5 de 626 MB y el cálculo de coherencia temporal (métrica de calidad), además
de correr multi-thread; insar-rs aquí no calcula esa métrica de calidad y opera
sobre datos ya en memoria. No es por tanto un cómputo idéntico. Aun descontando
el I/O, insar-rs resuelve la inversión en **~1–2 órdenes de magnitud menos de
tiempo**, consistente con la ventaja de Rust nativo + reutilización de la SVD.
La cifra reproducible y limpia es la de criterion (cómputo aislado).

## Recursos end-to-end vs MintPy — medición vigente (2026-10-08, tabla T2 del paper)

`validation/bench_fernandina.sh 5` (resumen: `validation/bench_summary.py`).
Mediana [IQR] de 5 repeticiones tras 1 calentamiento descartado; caché de
disco caliente para ambos. insar-rs en commit 9da455f (post v0.2.0), Rust
1.96.1; MintPy 1.6.3 con `cluster = none` (default), `ifgram_inversion.py -w no`,
paralelismo solo vía OpenBLAS (`OMP/OPENBLAS_NUM_THREADS`). Intermedios de
MintPy borrados antes de cada repetición (si no, `load_data` se salta).

| | insar-rs 16 h | insar-rs 1 h | MintPy 1 h | MintPy 16 h |
|---|---|---|---|---|
| Pared (s) | **0.89** [0.85–1.00] | 3.56 [3.47–3.58] | **37.4** [37.2–37.4] | 45.0 [43.9–55.0] |
| CPU (s) | 5.3 | 3.5 | 37.4 | 272 |
| RAM pico (MB) | 404 | 403 | 1 401 | 1 439 |
| Intermedios | 0 | 0 | 770 MB | 770 MB |
| Despliegue | binario 3.8 MB | ídem | venv 0.90 GB | ídem |

Etapas MintPy (1 h): load 14.3 s, reference_point 6.9 s, inversión 15.0 s,
velocidad 1.1 s. Con 16 hilos la inversión sube a 20.9 s (≈1100 % CPU):
OpenBLAS multihilo no compensa en matrices de este tamaño, así que la
referencia justa de MintPy es la de 1 hilo. insar-rs, inversión + velocidad
aisladas: 0.39 s (16 h) / 2.8 s (1 h).

Razones: a hilos iguales ≈10× en la tarea completa; mejor contra mejor ≈42×.
Las cifras de agosto (5.0 s, 118 s) mezclaban caché fría y una sola corrida;
quedan abajo solo como historial.

## Recursos end-to-end vs MintPy (medido 2026-08-29, histórico, superado)

Misma máquina (16 hilos lógicos), mismo dataset Fernandina real (288
interferogramas ISCE, 270 k píxeles). Tarea completa: de los archivos ISCE en
disco a la velocidad LOS.

| | insar-rs (16 hilos) | insar-rs (1 hilo) | MintPy 1.6.2 |
|---|---|---|---|
| Pipeline completo (wall) | **5.0 s** | 8.6 s | ~118 s (load 47.9 + ref_point 17.8 + inversión 52.5) |
| RAM pico | **404 MB** | 403 MB | **1 439 MB** (inversión); 147 MB (load) |
| Intermedios en disco | 0 B | 0 B | 626 MB (ifgramStack.h5) + 105 MB (timeseries.h5) |
| Despliegue | binario `insar` **3.5 MB** | ídem | venv **982 MB** (Python+deps) |

Medido con `/usr/bin/time -v` (Maximum resident set size). La inversión de
MintPy corrió multi-thread (320 s de CPU en 52.5 s de pared).

**Caveats de comparación honesta** (van al paper tal cual):
- La inversión de MintPy calcula además coherencia temporal (métrica de
  calidad que insar-rs no computa aquí) y escribe sus resultados a HDF5.
- El load de MintPy es un paso de conversión que insar-rs no necesita (lee
  ISCE nativo directo a memoria); compararlo como parte de la tarea es justo a
  nivel de flujo de trabajo, no de algoritmo.
- La cifra algorítmica limpia sigue siendo la de criterion (arriba):
  inversión aislada ~2.9 s vs 52.5 s de `ifgram_inversion.py -w no`.
- La RAM de insar-rs está dominada por el stack en memoria (Array3 completo);
  MintPy pagina por bloques desde HDF5 pero su pico de inversión igual es 3.6×.
- **LiCSBAS: no medido aún** — su pipeline exige la cadena de pasos 11–13
  sobre estructura de frame LiCSAR; queda para revisión si un reviewer lo
  pide (mismo frame Ñuble serviría).

## Reproducir

```bash
cargo bench -p insar-core                 # núcleo (criterion, con reportes HTML)
# wall-clock real:
cargo run --release -p insar-core --example validate_fernandina_isce -- \
  data/FernandinaSenDT128/merged/interferograms data/FernandinaSenDT128/baselines /tmp/v.f32
```

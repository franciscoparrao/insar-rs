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

## Reproducir

```bash
cargo bench -p insar-core                 # núcleo (criterion, con reportes HTML)
# wall-clock real:
cargo run --release -p insar-core --example validate_fernandina_isce -- \
  data/FernandinaSenDT128/merged/interferograms data/FernandinaSenDT128/baselines /tmp/v.f32
```

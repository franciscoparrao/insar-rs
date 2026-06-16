# insar-rs

Motor de análisis de series temporales InSAR (Sentinel-1) en Rust: deformación
del terreno por Persistent Scatterers (PS-InSAR) y Small-Baseline Subset (SBAS).

> Estado: MVP v0.1 en desarrollo. Ver `PLAN.md` para arquitectura y progreso.

## Estructura

- `crates/core` — `insar-core`: tipos, I/O de stacks (GeoTIFF + **lector ISCE
  nativo** sin GDAL), selección PS, red SBAS, desenrollado mínimo, inversión de
  series LOS, corrección atmosférica simple.
- `crates/cli` — binario `insar`: subcomandos `info`, `ps`, `network`, `run`, `isce`.
- `crates/python` — `insar_rs`: bindings PyO3 (numpy). Ver `crates/python/README.md`.

## Validación

Paridad numérica con MintPy sobre el dataset Fernandina (Sentinel-1): serie
temporal RMSE 0.0029 mm, velocidad RMSE 0.0070 mm/año, r = 1.000000. Detalle en
[`docs/validation.md`](docs/validation.md); rendimiento en
[`docs/benchmarks.md`](docs/benchmarks.md).

## Build

```bash
cargo build --release
```

Requiere el repo hermano [`surtgis`](../surtgis) en `../surtgis` (writer/reader
GeoTIFF nativo, sin GDAL).

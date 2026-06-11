# insar-rs

Motor de análisis de series temporales InSAR (Sentinel-1) en Rust: deformación
del terreno por Persistent Scatterers (PS-InSAR) y Small-Baseline Subset (SBAS).

> Estado: MVP v0.1 en desarrollo. Ver `PLAN.md` para arquitectura y progreso.

## Estructura

- `crates/core` — `insar-core`: tipos, I/O de stacks, selección PS, red SBAS,
  desenrollado mínimo, inversión de series LOS, corrección atmosférica simple.
- `crates/cli` — binario `insar`: subcomandos `info`, `ps`, `network`, `run`.

## Build

```bash
cargo build --release
```

Requiere el repo hermano [`surtgis`](../surtgis) en `../surtgis` (writer/reader
GeoTIFF nativo, sin GDAL).

## Validación

Paridad numérica contra MintPy sobre un stack público (ver `PLAN.md`, Fase 4).

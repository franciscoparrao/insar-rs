# insar-core

Motor de análisis de series temporales InSAR (deformación del terreno) en Rust
puro: **PS-InSAR** (Persistent Scatterers) y **SBAS** (Small-Baseline Subset)
sobre datos Sentinel-1. Núcleo del proyecto [insar-rs](https://github.com/franciscoparrao/insar-rs).

El campo (ISCE, StaMPS, MintPy, EZ-InSAR) es Python/MATLAB. `insar-core` es un
motor nativo Rust: sin runtime pesado, paralelo (Rayon), consumible como
librería, desde la CLI `insar`, o vía bindings Python (`insar-rs`).

## Qué hace

- **I/O de stacks**: formato propio en memoria (`Array3`, eje 0 = tiempo/par) +
  lector nativo de la salida de **ISCE** (interferogramas `.unw`/`.int` vía
  `.vrt`, mmap de solo lectura).
- **Selección de PS** por dispersión de amplitud; construcción de la **red SBAS**.
- **Desenrollado de fase** quality-guided (flood-fill propio) + backend **SNAPHU**
  opcional (shell-out).
- **Corrección de fase**: deramp (plano/cuadrático), errores de desenrollado por
  cierre de fase, APS/troposfera (ERA5).
- **Inversión de la serie LOS**: OLS/WLS + **L1-IRLS** robusto, estimación de
  **error de DEM**, coherencia temporal, bootstrap.
- **Descomposición** LOS asc+desc → vertical/este.
- **Features** por píxel para ML; pipeline end-to-end `run_sbas` / `run_sbas_isce`.

## Uso

```rust
use insar_core::pipeline::{IsceSbasConfig, run_sbas_isce};

// End-to-end desde un directorio de interferogramas ISCE desenrollados.
let cfg = IsceSbasConfig::new("merged/interferograms".into());
let out = run_sbas_isce(&cfg)?;
// out.velocity (m/año), out.series (épocas × filas × cols, m),
// out.temporal_coherence (máscara de calidad), out.dem_error_m, ...
```

Convenciones: NoData = `NaN`; desplazamiento LOS `d = −λ/(4π)·φ`; serie relativa
a la primera época (compatible con MintPy).

## Validación

Paridad numérica con **MintPy** sobre Fernandina (Sentinel-1 DT128): serie
RMSE 0.0029 mm, velocidad 0.0070 mm/año, r = 1.000000. Ver
[`docs/validation.md`](https://github.com/franciscoparrao/insar-rs/blob/main/docs/validation.md).

## Licencia

MIT OR Apache-2.0.

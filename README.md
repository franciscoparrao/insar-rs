# insar-rs

Motor de análisis de series temporales InSAR (Sentinel-1) en Rust: deformación
del terreno por Small-Baseline Subset (SBAS) y selección de Persistent
Scatterers, con validación numérica contra MintPy.

> Estado: v0.1. Ver `PLAN.md` (arquitectura y progreso), `CHANGELOG.md` y
> `docs/auditoria-2026-07-02.md` (hoja de ruta).

## Capacidades (insar-core, 14 módulos)

| Módulo | Qué hace |
|---|---|
| `types` | Contratos: stacks (ifg/amplitud/desenrollado), pares, series, metadata |
| `io` | Formato de stack propio (`stack.json` + GeoTIFF, coherencia opcional) — writer/reader nativo vía SurtGIS, **sin GDAL** |
| `io::isce` | **Lector ISCE nativo** (VRT + raw): `.unw` (NoData enmascarado), `.int` (CFloat32), `.cor`, `.unw.conncomp`, `los.rdr` (geometría por píxel), baselines topsStack |
| `ps` | Amplitude dispersion y selección de candidatos PS |
| `network` | Red small-baseline (doble umbral), matriz de diseño, conectividad |
| `unwrap` | Desenrollado 2D quality-guided con coherencia y umbral `min_quality`; **backend SNAPHU opcional** (`unwrap::snaphu`, shell-out) |
| `unwrap_error` | Corrección de saltos 2π por **cierre de fase** (con verificación de efectividad) + QC `nonzero_closure_count` |
| `inversion` | SBAS OLS / **WLS por coherencia** / **L1 robusto (IRLS)**; **error de DEM** (∝ B⊥); velocidad + SE formal + **bootstrap**; **modelo temporal** (polinomio + estacional + saltos); coherencia temporal; referenciado |
| `atmosphere` | Filtro APS espacio-temporal (pasa-alto temporal en **tiempo real** — robusto a gaps) |
| `troposphere` | Corrección estratificada fase-elevación (Doin 2009); **`gacos`** (mapas ZTD, end-to-end) y `era5` (kernel físico) |
| `postprocess` | Deramp (plano/cuadrática), `coherence_mask`, re-exports de referenciado y γ_temp |
| `decompose` | LOS → (Up, East) con geometría escalar o **por píxel** (incidencia/heading de `los.rdr`) |
| `features` | Descriptores por píxel para ML (tabla determinista lista para smelt-ml, con coordenadas para CV espacial) |
| `pipeline` | `run_sbas` end-to-end con el orden físico documentado y productos de QC |

## Estructura

- `crates/core` — `insar-core`: el motor (tabla de arriba).
- `crates/cli` — binario `insar`: `info`, `ps`, `network`, `run`
  (`--min-quality`, `--unwrap-backend {flood-fill,snaphu}`, `--snaphu-bin`,
  `--deramp`, `--no-closure-correction`), `isce`
  (`--wls`, `--robust`, `--dem-error-range`, `--deramp`), `decompose`
  (LOS asc+desc → up/east), `features` (tabla ML por píxel, `--csv`),
  `deramp` (standalone sobre una serie ya escrita).
- `crates/python` — `insar_rs`: bindings PyO3/numpy (cómputo sin GIL,
  excepciones idiomáticas): `invert_sbas`, `estimate_velocity[_uncertainty]`,
  `amplitude_dispersion`, `temporal_coherence`, `sbas_from_isce`,
  `decompose_asc_desc`, `decompose_per_pixel`, `extract_features`,
  `remove_ramp`, `correct_unwrap_errors`. Ver `crates/python/README.md`.

## Validación

Paridad numérica con MintPy sobre el dataset Fernandina (Sentinel-1, camino
OLS): serie temporal RMSE 0.034 µm, velocidad RMSE 0.013 µm/año,
r = 1.000000. Detalle en [`docs/validation.md`](docs/validation.md);
rendimiento en [`docs/benchmarks.md`](docs/benchmarks.md).

## Build

```bash
cargo build --release          # binario insar
cargo test  --workspace        # 166 tests (163 unit + 1 e2e + 2 smoke CLI)
```

Requiere el repo hermano [`surtgis`](../surtgis) en `../surtgis`
(writer/reader GeoTIFF nativo, sin GDAL). Los tests y examples usan además
`../geostat-rs`, `../smelt` y `../swarm-abm` (dev-deps del ecosistema).

Bindings Python:

```bash
pip install maturin
maturin develop -m crates/python/Cargo.toml
pytest crates/python/tests/
```

## Reproducibilidad y cita

Release **v0.2.0** publicado en los registries oficiales:

```bash
# Rust (motor): https://crates.io/crates/insar-core
cargo add insar-core

# Python (bindings): https://pypi.org/project/insar-rs/
pip install insar-rs
```

La validación es reproducible: el ejemplo `validate_fernandina` recompone la
paridad con MintPy (serie RMSE 0.034 µm, velocidad 0.013 µm/año,
r = 1.000000) sobre el stack público de Fernandina; metodología en
[`docs/validation.md`](docs/validation.md).

Copia archivada con DOI en Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21924287.svg)](https://doi.org/10.5281/zenodo.21924287)

- **Concept DOI** (todas las versiones, citar este): [`10.5281/zenodo.21924287`](https://doi.org/10.5281/zenodo.21924287)
- **Version DOI** (v0.2.0): [`10.5281/zenodo.21924288`](https://doi.org/10.5281/zenodo.21924288)

Para citar, ver [`CITATION.cff`](CITATION.cff) (badge "Cite this repository" de
GitHub) o la metadata en [`.zenodo.json`](.zenodo.json).

## Licencia

MIT OR Apache-2.0 (ver `LICENSE-MIT` / `LICENSE-APACHE`).

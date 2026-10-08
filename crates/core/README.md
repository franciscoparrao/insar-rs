# insar-core

A pure-Rust engine for InSAR time-series analysis of ground deformation:
**PS-InSAR** (Persistent Scatterers) and **SBAS** (Small-Baseline Subset) on
Sentinel-1 data. Core of the [insar-rs](https://github.com/franciscoparrao/insar-rs)
project.

The field (ISCE, StaMPS, MintPy, EZ-InSAR) is Python/MATLAB. `insar-core` is
a native Rust engine: no heavy runtime, parallel (Rayon), usable as a
library, from the `insar` CLI, or through the Python bindings (`insar-rs`).

## What it does

- **Stack I/O**: own in-memory format (`Array3`, axis 0 = time/pair), a
  native reader for **ISCE** output (`.unw`/`.int` interferograms through
  `.vrt`, read-only mmap) and a reader for **LiCSAR/COMET** products (wrapped
  phase and coherence, AOI crop, date range).
- **PS selection** by amplitude dispersion; **SBAS network** construction.
- **Phase unwrapping**: quality-guided (own flood-fill) plus an optional
  **SNAPHU** backend (shell-out).
- **Non-closure phase-bias correction** (Maghsoudi et al. 2022): estimated
  from wrapped closures, arbitrary span and incomplete networks, robust
  masking of outlier closures; by default the correction is subtracted after
  unwrapping (`PhaseBiasStage::AfterUnwrap`).
- **Phase corrections**: deramp (plane/quadratic), unwrapping errors by phase
  closure, APS, and troposphere (stratified, GACOS, ERA5).
- **LOS time-series inversion**: OLS/WLS and robust **L1-IRLS**, **DEM
  error** estimation, temporal coherence, bootstrap.
- **Decomposition** of LOS asc+desc into vertical/east.
- Per-pixel **features** for ML; end-to-end pipelines `run_sbas` /
  `run_sbas_isce`.

## Usage

```rust
use insar_core::pipeline::{IsceSbasConfig, run_sbas_isce};

// End-to-end from a directory of unwrapped ISCE interferograms.
let cfg = IsceSbasConfig::new("merged/interferograms".into());
let out = run_sbas_isce(&cfg)?;
// out.velocity (m/yr), out.series (epochs × rows × cols, m),
// out.temporal_coherence (quality mask), out.dem_error_m, ...
```

The phase-bias correction needs wrapped phase, so it runs in `run_sbas`
(configure `SbasPipelineConfig::phase_bias`) or directly through
`phase_bias::correct_phase_bias`; it does not apply to `run_sbas_isce`, which
reads already-unwrapped `.unw` files.

Conventions: NoData = `NaN`; LOS displacement `d = −λ/(4π)·φ`; series
relative to the first epoch (MintPy-compatible).

## Validation

Numerical parity with **MintPy** 1.6.3 on Fernandina (Sentinel-1 DT128):
series RMSE 0.034 µm, velocity RMSE 0.013 µm/yr, r = 1.000000, after
aligning two MintPy conventions. See
[`docs/validation.md`](https://github.com/franciscoparrao/insar-rs/blob/main/docs/validation.md).

## License

MIT OR Apache-2.0.

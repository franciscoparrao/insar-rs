# insar-rs (Python)

Python bindings for the SBAS engine of
[insar-rs](https://github.com/franciscoparrao/insar-rs) (Rust). Fast InSAR
time-series inversion, usable from numpy.

## Install

```bash
pip install insar-rs
```

Development build:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy
cd crates/python
maturin develop --release
```

## Usage

```python
import insar_rs, numpy as np

# End-to-end from ISCE interferograms (unwrapped .unw). Automatically
# references to the pixel with the highest mean coherence.
vel, vel_std, series, coherence, epochs = insar_rs.sbas_from_isce(
    "merged/interferograms", baselines_dir="baselines")
# vel: (rows, cols) m/yr ; vel_std: (rows, cols) m/yr (standard error)
# series: (epochs, rows, cols) m
# coherence: (rows, cols) temporal coherence [0, 1] (quality mask)
# epochs: ['YYYY-MM-DD', ...]

# Or on your own unwrapped-phase arrays (n_pairs, rows, cols) in radians:
ts  = insar_rs.invert_sbas(phase, refs, secs, epoch_days, wavelength_m=0.0555)
vel = insar_rs.estimate_velocity(ts, epoch_days)

# Persistent Scatterers:
da  = insar_rs.amplitude_dispersion(amp_stack)  # (n_epochs, rows, cols) -> (rows, cols)

# LOS asc+desc decomposition -> (Up, East); scalar or per-pixel geometry:
up, east = insar_rs.decompose_asc_desc(los_asc, 39.0, 349.0, los_desc, 39.0, 191.0)
up, east = insar_rs.decompose_per_pixel([los_asc, los_desc], [inc_asc, inc_desc], [head_asc, head_desc])

# Per-pixel ML descriptors (dict of arrays, deterministic schema):
features = insar_rs.extract_features(series, epoch_days)  # {"velocity": ..., "acceleration": ..., ...}

# Deramp (plane/quadratic) and 2π jump correction by phase closure:
flat = insar_rs.remove_ramp(vel, "linear")
corrected_phase, n_corrected, n_uncorrected = insar_rs.correct_unwrap_errors(phase, refs, secs)
```

The non-closure phase-bias correction and the LiCSAR reader are not exposed
in the bindings yet; use the Rust crate `insar-core` for them.

Conventions: NoData = NaN; LOS displacement `d = −λ/(4π)·φ`; series relative
to the first epoch. Validated against MintPy (see
[`docs/validation.md`](https://github.com/franciscoparrao/insar-rs/blob/main/docs/validation.md)).

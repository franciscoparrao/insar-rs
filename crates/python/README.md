# insar-rs (Python)

Bindings Python del motor SBAS de [insar-rs](../..) (Rust). Inversión de series
temporales InSAR rápida, consumible desde numpy.

## Instalar (desarrollo)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy
cd crates/python
maturin develop --release
```

## Uso

```python
import insar_rs, numpy as np

# End-to-end desde interferogramas ISCE (.unw desenrollados). Referencia
# automáticamente al píxel de máxima coherencia media.
vel, vel_std, series, coherence, epochs = insar_rs.sbas_from_isce(
    "merged/interferograms", baselines_dir="baselines")
# vel: (filas, cols) m/año ; vel_std: (filas, cols) m/año (error estándar)
# series: (épocas, filas, cols) m
# coherence: (filas, cols) coherencia temporal [0,1] (máscara de calidad)
# epochs: ['YYYY-MM-DD', ...]

# O sobre arrays propios de fase desenrollada (n_pares, filas, cols) en radianes:
ts  = insar_rs.invert_sbas(phase, refs, secs, epoch_days, wavelength_m=0.0555)
vel = insar_rs.estimate_velocity(ts, epoch_days)

# Persistent Scatterers:
da  = insar_rs.amplitude_dispersion(amp_stack)  # (n_épocas, filas, cols) -> (filas, cols)

# Descomposición LOS asc+desc -> (Up, East); geometría escalar o por píxel:
up, east = insar_rs.decompose_asc_desc(los_asc, 39.0, 349.0, los_desc, 39.0, 191.0)
up, east = insar_rs.decompose_per_pixel([los_asc, los_desc], [inc_asc, inc_desc], [head_asc, head_desc])

# Descriptores por píxel para ML (dict de arrays, esquema determinista):
features = insar_rs.extract_features(series, epoch_days)  # {"velocity": ..., "acceleration": ..., ...}

# Deramp (plano/cuadrática) y corrección de saltos 2π por cierre de fase:
flat = insar_rs.remove_ramp(vel, "linear")
corrected_phase, n_corrected, n_uncorrected = insar_rs.correct_unwrap_errors(phase, refs, secs)
```

Convenciones: NoData = NaN; desplazamiento LOS `d = −λ/(4π)·φ`; serie relativa a
la primera época. Validado contra MintPy (ver `../../docs/validation.md`).

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
vel, series, coherence, epochs = insar_rs.sbas_from_isce(
    "merged/interferograms", baselines_dir="baselines")
# vel: (filas, cols) m/año ; series: (épocas, filas, cols) m
# coherence: (filas, cols) coherencia temporal [0,1] (máscara de calidad)
# epochs: ['YYYY-MM-DD', ...]

# O sobre arrays propios de fase desenrollada (n_pares, filas, cols) en radianes:
ts  = insar_rs.invert_sbas(phase, refs, secs, epoch_days, wavelength_m=0.0555)
vel = insar_rs.estimate_velocity(ts, epoch_days)

# Persistent Scatterers:
da  = insar_rs.amplitude_dispersion(amp_stack)  # (n_épocas, filas, cols) -> (filas, cols)
```

Convenciones: NoData = NaN; desplazamiento LOS `d = −λ/(4π)·φ`; serie relativa a
la primera época. Validado contra MintPy (ver `../../docs/validation.md`).

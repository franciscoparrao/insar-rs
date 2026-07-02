"""Tests de regresión de los bindings insar_rs.

Mismo caso sintético que los tests Rust del core: 4 épocas cada 12 días,
red consecutivos + saltos de 2, desplazamiento lineal v = -0.05 m/año.

Correr con:
    maturin develop -m crates/python/Cargo.toml   # (o pip install .)
    pytest crates/python/tests/
"""

import numpy as np
import pytest

insar_rs = pytest.importorskip("insar_rs")

WAVELENGTH = 0.05546576  # Sentinel-1 C-band (m)
V_TRUE = -0.05  # m/año

REFS = [0, 1, 2, 0, 1]
SECS = [1, 2, 3, 2, 3]
EPOCH_DAYS = [0, 12, 24, 36]


def true_displacements():
    t = np.array(EPOCH_DAYS, dtype=np.float64) / 365.25
    return V_TRUE * t


def synthetic_phase(rows=2, cols=3):
    """Fases desenrolladas exactas: phi = -4*pi/lambda * (d_sec - d_ref)."""
    d = true_displacements()
    n_pairs = len(REFS)
    phase = np.zeros((n_pairs, rows, cols), dtype=np.float32)
    for k, (r, s) in enumerate(zip(REFS, SECS)):
        phase[k, :, :] = -4.0 * np.pi / WAVELENGTH * (d[s] - d[r])
    return phase


def test_invert_sbas_recupera_serie_lineal():
    series = insar_rs.invert_sbas(synthetic_phase(), REFS, SECS, EPOCH_DAYS)
    assert series.shape == (4, 2, 3)
    assert series.dtype == np.float32
    d = true_displacements()
    np.testing.assert_allclose(series[:, 0, 0], d, atol=1e-5)
    # Relativa a la primera época.
    assert series[0].max() == 0.0


def test_estimate_velocity_y_uncertainty():
    series = insar_rs.invert_sbas(synthetic_phase(), REFS, SECS, EPOCH_DAYS)
    vel = insar_rs.estimate_velocity(series, EPOCH_DAYS)
    assert vel.shape == (2, 3)
    np.testing.assert_allclose(vel, V_TRUE, atol=1e-5)
    se = insar_rs.estimate_velocity_uncertainty(series, EPOCH_DAYS)
    assert se.shape == (2, 3)
    assert (se < 1e-5).all(), "ajuste exacto => SE ~ 0"


def test_temporal_coherence_perfecta():
    phase = synthetic_phase()
    series = insar_rs.invert_sbas(phase, REFS, SECS, EPOCH_DAYS)
    gamma = insar_rs.temporal_coherence(phase, series, REFS, SECS, EPOCH_DAYS)
    np.testing.assert_allclose(gamma, 1.0, atol=1e-5)


def test_amplitude_dispersion():
    amp = np.full((6, 2, 2), 100.0, dtype=np.float32)
    amp += np.random.default_rng(7).normal(0, 1e-3, amp.shape).astype(np.float32)
    da = insar_rs.amplitude_dispersion(amp)
    assert da.shape == (2, 2)
    assert (da < 0.01).all(), "amplitud estable => dispersion ~ 0"


def test_nan_propaga_sin_romper():
    phase = synthetic_phase()
    phase[:, 0, 1] = np.nan  # pixel sin observaciones
    series = insar_rs.invert_sbas(phase, REFS, SECS, EPOCH_DAYS)
    assert np.isnan(series[:, 0, 1]).all()
    assert np.isfinite(series[1:, 0, 0]).all()


def test_errores_idiomaticos():
    # Largo inconsistente => ValueError.
    with pytest.raises(ValueError):
        insar_rs.invert_sbas(synthetic_phase(), [0, 1], SECS, EPOCH_DAYS)
    with pytest.raises(ValueError):
        insar_rs.temporal_coherence(
            synthetic_phase(),
            np.zeros((4, 2, 3), dtype=np.float32),
            [0, 1],
            SECS,
            EPOCH_DAYS,
        )
    # Red desconectada => ValueError (InvalidNetwork).
    with pytest.raises(ValueError):
        insar_rs.invert_sbas(
            synthetic_phase()[:2], [0, 2], [1, 3], EPOCH_DAYS
        )
    # Directorio ISCE inexistente => OSError (idiomatico: except OSError).
    with pytest.raises(OSError):
        insar_rs.sbas_from_isce("/directorio/que/no/existe")

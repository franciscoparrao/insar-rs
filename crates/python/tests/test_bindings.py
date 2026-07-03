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


def los_vector(incidence_deg, heading_deg):
    """Espejo de insar_core::decompose::LosVector::from_incidence_heading."""
    th = np.radians(incidence_deg)
    a = np.radians(heading_deg)
    return {"up": np.cos(th), "east": -np.sin(th) * np.cos(a)}


def test_decompose_asc_desc_recupera_up_east():
    up_true, east_true = 1.0, 0.5
    g_asc = los_vector(39.0, 349.0)
    g_desc = los_vector(39.0, 191.0)
    los_asc = np.full((2, 3), up_true * g_asc["up"] + east_true * g_asc["east"], dtype=np.float32)
    los_desc = np.full((2, 3), up_true * g_desc["up"] + east_true * g_desc["east"], dtype=np.float32)

    up, east = insar_rs.decompose_asc_desc(los_asc, 39.0, 349.0, los_desc, 39.0, 191.0)
    assert up.shape == (2, 3)
    assert east.shape == (2, 3)
    np.testing.assert_allclose(up, up_true, atol=1e-4)
    np.testing.assert_allclose(east, east_true, atol=1e-4)


def test_decompose_per_pixel_geometria_constante_coincide_con_asc_desc():
    up_true, east_true = 1.0, 0.5
    g_asc = los_vector(39.0, 349.0)
    g_desc = los_vector(39.0, 191.0)
    los_asc = np.full((2, 3), up_true * g_asc["up"] + east_true * g_asc["east"], dtype=np.float32)
    los_desc = np.full((2, 3), up_true * g_desc["up"] + east_true * g_desc["east"], dtype=np.float32)

    inc = np.full((2, 3), 39.0, dtype=np.float32)
    head_asc = np.full((2, 3), 349.0, dtype=np.float32)
    head_desc = np.full((2, 3), 191.0, dtype=np.float32)

    up, east = insar_rs.decompose_per_pixel(
        [los_asc, los_desc], [inc, inc], [head_asc, head_desc]
    )
    np.testing.assert_allclose(up, up_true, atol=1e-4)
    np.testing.assert_allclose(east, east_true, atol=1e-4)


def test_extract_features_esquema_y_velocidad():
    epoch_days = [0, 12, 24, 36, 48]
    v_true = -0.05
    t = np.array(epoch_days, dtype=np.float64) / 365.25
    series = (v_true * t).astype(np.float32).reshape(-1, 1, 1)

    features = insar_rs.extract_features(series, epoch_days)
    assert set(features) == {
        "velocity",
        "velocity_std",
        "acceleration",
        "linearity_r2",
        "residual_rms",
        "cumulative",
        "seasonal_amplitude",
        "seasonal_phase",
        "max_step",
    }
    assert features["velocity"].shape == (1, 1)
    np.testing.assert_allclose(features["velocity"][0, 0], v_true, atol=1e-4)

    # seasonal=False achica el esquema (determinista, no depende de los datos).
    reduced = insar_rs.extract_features(series, epoch_days, seasonal=False)
    assert "seasonal_amplitude" not in reduced


def test_remove_ramp_quita_plano_exacto():
    rows, cols = 5, 6
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    data = (0.1 * r + 0.2 * c + 3.0).astype(np.float32)

    flat = insar_rs.remove_ramp(data, "linear")
    assert flat.shape == data.shape
    np.testing.assert_allclose(flat, 0.0, atol=1e-4)

    with pytest.raises(ValueError):
        insar_rs.remove_ramp(data, "cubic")


def test_correct_unwrap_errors_sin_saltos_no_corrige():
    phase = synthetic_phase()
    corrected_phase, corrected, detected_uncorrected = insar_rs.correct_unwrap_errors(
        phase, REFS, SECS
    )
    assert corrected_phase.shape == phase.shape
    assert corrected == 0
    assert detected_uncorrected == 0
    np.testing.assert_allclose(corrected_phase, phase, atol=1e-5)


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

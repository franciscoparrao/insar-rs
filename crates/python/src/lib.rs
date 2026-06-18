//! Bindings Python (PyO3) del motor SBAS de insar-rs.
//!
//! Expone la inversión, la velocidad, la amplitude dispersion y el camino
//! end-to-end desde un directorio ISCE. Las matrices se intercambian como
//! arrays de numpy `float32` con el layout de insar-core (eje 0 = tiempo/par).
//
// pyo3 0.22 genera funciones `unsafe` cuyo cuerpo (macro) dispara el lint
// `unsafe_op_in_unsafe_fn` de edition 2024. Se silencia a nivel de crate
// (workaround estándar hasta pyo3 ≥0.23).
#![allow(unsafe_op_in_unsafe_fn)]
// Ruido inherente a la macro de pyo3 en la capa de binding.
#![allow(clippy::useless_conversion, clippy::type_complexity)]

use chrono::NaiveDate;
use ndarray::{Array3, Axis};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use insar_core::inversion::{estimate_velocity as core_velocity, invert_sbas as core_invert};
use insar_core::io::isce::{IsceLoadConfig, read_isce_coherence, read_isce_unwrapped_stack};
use insar_core::ps::amplitude_dispersion as core_ad;
use insar_core::types::{
    AmplitudeStack, DisplacementSeries, Epoch, IfgPair, SENTINEL1_WAVELENGTH_M, StackMeta,
    UnwrappedStack, VelocityMap,
};

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Fecha base arbitraria; solo importan las diferencias (years_since).
fn epoch_from_day(day: i64) -> Epoch {
    let base = NaiveDate::from_ymd_opt(2000, 1, 1).unwrap();
    Epoch(base + chrono::Duration::days(day))
}

fn dummy_meta(wavelength_m: f64, incidence_deg: f64) -> StackMeta {
    StackMeta {
        transform: surtgis_transform(),
        crs: None,
        wavelength_m,
        incidence_deg,
        heading_deg: None,
    }
}

fn surtgis_transform() -> surtgis_core::GeoTransform {
    surtgis_core::GeoTransform::new(0.0, 0.0, 1.0, -1.0)
}

/// Invierte la serie temporal de desplazamiento LOS (SBAS no ponderado).
///
/// phase: array (n_pares, filas, cols) float32 en radianes desenrollados.
/// refs/secs: índices de época (en `epoch_days`) de cada par.
/// epoch_days: día de cada época (offsets enteros; solo importan diferencias).
/// Devuelve (n_épocas, filas, cols) float32 en metros, relativa a la época 0.
#[pyfunction]
#[pyo3(signature = (phase, refs, secs, epoch_days, wavelength_m=SENTINEL1_WAVELENGTH_M))]
fn invert_sbas<'py>(
    py: Python<'py>,
    phase: PyReadonlyArray3<'py, f32>,
    refs: Vec<usize>,
    secs: Vec<usize>,
    epoch_days: Vec<i64>,
    wavelength_m: f64,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let data: Array3<f32> = phase.as_array().to_owned();
    if refs.len() != secs.len() || refs.len() != data.len_of(Axis(0)) {
        return Err(PyValueError::new_err(
            "refs, secs y el eje 0 de phase deben tener la misma longitud",
        ));
    }
    let pairs: Vec<IfgPair> = refs
        .iter()
        .zip(&secs)
        .map(|(&r, &s)| IfgPair { reference: r, secondary: s, perp_baseline_m: 0.0 })
        .collect();
    let stack = UnwrappedStack {
        data,
        epochs: epoch_days.iter().map(|&d| epoch_from_day(d)).collect(),
        pairs,
        meta: dummy_meta(wavelength_m, 39.0),
    };
    let series = core_invert(&stack, None).map_err(err)?;
    Ok(series.data.into_pyarray_bound(py))
}

/// Velocidad media LOS (m/año) por ajuste lineal de la serie.
/// series: (n_épocas, filas, cols) float32; epoch_days: día de cada época.
#[pyfunction]
fn estimate_velocity<'py>(
    py: Python<'py>,
    series: PyReadonlyArray3<'py, f32>,
    epoch_days: Vec<i64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let data: Array3<f32> = series.as_array().to_owned();
    if epoch_days.len() != data.len_of(Axis(0)) {
        return Err(PyValueError::new_err(
            "epoch_days debe tener tantas entradas como el eje 0 de series",
        ));
    }
    let ds = DisplacementSeries {
        data,
        epochs: epoch_days.iter().map(|&d| epoch_from_day(d)).collect(),
        meta: dummy_meta(SENTINEL1_WAVELENGTH_M, 39.0),
    };
    let vel: VelocityMap = core_velocity(&ds).map_err(err)?;
    Ok(vel.data.into_pyarray_bound(py))
}

/// Amplitude dispersion D_A = σ/μ por píxel sobre el eje temporal.
/// amp: (n_épocas, filas, cols) float32. Devuelve (filas, cols) float32.
#[pyfunction]
fn amplitude_dispersion<'py>(
    py: Python<'py>,
    amp: PyReadonlyArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let data: Array3<f32> = amp.as_array().to_owned();
    let n = data.len_of(Axis(0));
    let stack = AmplitudeStack {
        data,
        epochs: (0..n as i64).map(epoch_from_day).collect(),
        meta: dummy_meta(SENTINEL1_WAVELENGTH_M, 39.0),
    };
    let disp = core_ad(&stack).map_err(err)?;
    Ok(disp.into_pyarray_bound(py))
}

/// Incertidumbre (error estándar) de la velocidad LOS por píxel (m/año).
/// series: (n_épocas, filas, cols) float32; epoch_days: día de cada época.
#[pyfunction]
fn estimate_velocity_uncertainty<'py>(
    py: Python<'py>,
    series: PyReadonlyArray3<'py, f32>,
    epoch_days: Vec<i64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let data: Array3<f32> = series.as_array().to_owned();
    if epoch_days.len() != data.len_of(Axis(0)) {
        return Err(PyValueError::new_err(
            "epoch_days debe tener tantas entradas como el eje 0 de series",
        ));
    }
    let ds = DisplacementSeries {
        data,
        epochs: epoch_days.iter().map(|&d| epoch_from_day(d)).collect(),
        meta: dummy_meta(SENTINEL1_WAVELENGTH_M, 39.0),
    };
    let se = insar_core::inversion::estimate_velocity_uncertainty(&ds).map_err(err)?;
    Ok(se.into_pyarray_bound(py))
}

/// Coherencia temporal (Pepe & Lanari 2006), métrica de calidad de la
/// inversión en [0, 1]. `phase`: (n_pares, filas, cols) radianes; `series`:
/// (n_épocas, filas, cols) m. Devuelve (filas, cols) float32.
#[pyfunction]
#[pyo3(signature = (phase, series, refs, secs, epoch_days, wavelength_m=SENTINEL1_WAVELENGTH_M))]
fn temporal_coherence<'py>(
    py: Python<'py>,
    phase: PyReadonlyArray3<'py, f32>,
    series: PyReadonlyArray3<'py, f32>,
    refs: Vec<usize>,
    secs: Vec<usize>,
    epoch_days: Vec<i64>,
    wavelength_m: f64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let pairs: Vec<IfgPair> = refs
        .iter()
        .zip(&secs)
        .map(|(&r, &s)| IfgPair { reference: r, secondary: s, perp_baseline_m: 0.0 })
        .collect();
    let epochs: Vec<Epoch> = epoch_days.iter().map(|&d| epoch_from_day(d)).collect();
    let stack = UnwrappedStack {
        data: phase.as_array().to_owned(),
        epochs: epochs.clone(),
        pairs,
        meta: dummy_meta(wavelength_m, 39.0),
    };
    let ds = DisplacementSeries {
        data: series.as_array().to_owned(),
        epochs,
        meta: dummy_meta(wavelength_m, 39.0),
    };
    let gamma = insar_core::inversion::temporal_coherence(&stack, &ds).map_err(err)?;
    Ok(gamma.into_pyarray_bound(py))
}

/// Lee un directorio de interferogramas ISCE, referencia al píxel de máxima
/// coherencia media (elimina el offset por interferograma del desenrollado),
/// invierte y estima velocidad y coherencia temporal.
///
/// Devuelve `(velocity, velocity_std, series, temporal_coherence, epochs)`:
///   velocity:           (filas, cols) float32 m/año
///   velocity_std:       (filas, cols) float32 m/año (error estándar)
///   series:             (n_épocas, filas, cols) float32 m
///   temporal_coherence: (filas, cols) float32 en [0, 1]
///   epochs:             lista de fechas ISO 'YYYY-MM-DD'
#[pyfunction]
#[pyo3(signature = (ifg_dir, baselines_dir=None, wavelength_m=SENTINEL1_WAVELENGTH_M, incidence_deg=39.0))]
fn sbas_from_isce<'py>(
    py: Python<'py>,
    ifg_dir: String,
    baselines_dir: Option<String>,
    wavelength_m: f64,
    incidence_deg: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray2<f32>>,
    Vec<String>,
)> {
    let config = IsceLoadConfig {
        baselines_dir: baselines_dir.map(std::path::PathBuf::from),
        wavelength_m,
        incidence_deg,
        ..Default::default()
    };
    let dir = std::path::Path::new(&ifg_dir);
    let mut stack = read_isce_unwrapped_stack(dir, &config).map_err(err)?;
    let epochs: Vec<String> = stack.epochs.iter().map(|e| e.0.to_string()).collect();

    // Referencia espacial al píxel de máxima coherencia media (o el centro).
    let (n_rows, n_cols) = stack.dims();
    let (ref_r, ref_c) = match read_isce_coherence(dir, &config) {
        Ok(coh) => {
            let n_pairs = coh.shape()[0];
            let mut best = (n_rows / 2, n_cols / 2, f32::MIN);
            for r in 0..n_rows {
                for c in 0..n_cols {
                    let (mut sum, mut n) = (0.0_f64, 0u32);
                    for k in 0..n_pairs {
                        let v = coh[[k, r, c]];
                        if v.is_finite() {
                            sum += v as f64;
                            n += 1;
                        }
                    }
                    if n > 0 {
                        let mean = (sum / n as f64) as f32;
                        if mean > best.2 {
                            best = (r, c, mean);
                        }
                    }
                }
            }
            (best.0, best.1)
        }
        Err(_) => (n_rows / 2, n_cols / 2),
    };
    insar_core::inversion::reference_to_pixel(&mut stack, ref_r, ref_c).map_err(err)?;

    let series = core_invert(&stack, None).map_err(err)?;
    let velocity = core_velocity(&series).map_err(err)?;
    let vel_std = insar_core::inversion::estimate_velocity_uncertainty(&series).map_err(err)?;
    let gamma = insar_core::inversion::temporal_coherence(&stack, &series).map_err(err)?;
    Ok((
        velocity.data.into_pyarray_bound(py),
        vel_std.into_pyarray_bound(py),
        series.data.into_pyarray_bound(py),
        gamma.into_pyarray_bound(py),
        epochs,
    ))
}

#[pymodule]
fn insar_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(invert_sbas, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_velocity, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_velocity_uncertainty, m)?)?;
    m.add_function(wrap_pyfunction!(amplitude_dispersion, m)?)?;
    m.add_function(wrap_pyfunction!(temporal_coherence, m)?)?;
    m.add_function(wrap_pyfunction!(sbas_from_isce, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

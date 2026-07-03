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
use ndarray::{Array2, Array3, Axis};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use insar_core::InsarError;
use insar_core::decompose::{
    LosVector, PerPixelGeometry, decompose_asc_desc as core_decompose_asc_desc,
    decompose_per_pixel as core_decompose_per_pixel,
};
use insar_core::features::{FeatureConfig, extract_features as core_extract_features};
use insar_core::inversion::{estimate_velocity as core_velocity, invert_sbas as core_invert};
use insar_core::io::isce::{IsceLoadConfig, read_isce_coherence, read_isce_unwrapped_stack};
use insar_core::postprocess::{RampKind, remove_ramp as core_remove_ramp};
use insar_core::ps::amplitude_dispersion as core_ad;
use insar_core::types::{
    AmplitudeStack, DisplacementSeries, Epoch, IfgPair, SENTINEL1_WAVELENGTH_M, StackMeta,
    UnwrappedStack, VelocityMap,
};
use insar_core::unwrap_error::correct_unwrap_errors as core_correct_unwrap_errors;

/// Mapea [`InsarError`] a la excepción Python idiomática: fallas de archivo
/// → `IOError` (capturable con `except OSError`), fallas numéricas →
/// `RuntimeError`, y datos/parámetros inválidos → `ValueError`.
fn err(e: InsarError) -> PyErr {
    match &e {
        InsarError::Io(_) | InsarError::Raster(_) => PyIOError::new_err(e.to_string()),
        InsarError::Inversion(_) => PyRuntimeError::new_err(e.to_string()),
        InsarError::DimensionMismatch(_)
        | InsarError::InvalidNetwork(_)
        | InsarError::Metadata(_)
        | InsarError::UnsupportedFormat(_) => PyValueError::new_err(e.to_string()),
    }
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
    // La entrada ya es owned: el cómputo corre sin el GIL (otros threads
    // Python siguen vivos mientras rayon trabaja).
    let series = py.allow_threads(|| core_invert(&stack, None)).map_err(err)?;
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
    let vel: VelocityMap = py.allow_threads(|| core_velocity(&ds)).map_err(err)?;
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
    let disp = py.allow_threads(|| core_ad(&stack)).map_err(err)?;
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
    let se = py
        .allow_threads(|| insar_core::inversion::estimate_velocity_uncertainty(&ds))
        .map_err(err)?;
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
    // Misma validación temprana que invert_sbas: sin ella, zip trunca en
    // silencio al vector más corto.
    if refs.len() != secs.len() || refs.len() != phase.as_array().len_of(Axis(0)) {
        return Err(PyValueError::new_err(
            "refs, secs y el eje 0 de phase deben tener la misma longitud",
        ));
    }
    if epoch_days.len() != series.as_array().len_of(Axis(0)) {
        return Err(PyValueError::new_err(
            "epoch_days debe tener tantas entradas como el eje 0 de series",
        ));
    }
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
    let gamma = py
        .allow_threads(|| insar_core::inversion::temporal_coherence(&stack, &ds))
        .map_err(err)?;
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
    // Todo el trabajo pesado (I/O de disco de cientos de pares + inversión)
    // corre SIN el GIL: otros threads Python siguen vivos mientras tanto.
    let (velocity, vel_std, series, gamma, epochs) = py
        .allow_threads(|| -> Result<_, InsarError> {
            let dir = std::path::Path::new(&ifg_dir);
            let mut stack = read_isce_unwrapped_stack(dir, &config)?;
            let epochs: Vec<String> =
                stack.epochs.iter().map(|e| e.0.to_string()).collect();

            // Referencia espacial al píxel de máxima coherencia media (o el
            // centro). La selección vive en el core (compartida con la CLI y
            // el pipeline).
            let (n_rows, n_cols) = stack.dims();
            let (ref_r, ref_c) = read_isce_coherence(dir, &config)
                .ok()
                .and_then(|coh| insar_core::inversion::select_reference_pixel(&coh))
                .unwrap_or((n_rows / 2, n_cols / 2));
            insar_core::inversion::reference_to_pixel(&mut stack, ref_r, ref_c)?;

            let series = core_invert(&stack, None)?;
            let velocity = core_velocity(&series)?;
            let vel_std = insar_core::inversion::estimate_velocity_uncertainty(&series)?;
            let gamma = insar_core::inversion::temporal_coherence(&stack, &series)?;
            Ok((velocity, vel_std, series, gamma, epochs))
        })
        .map_err(err)?;
    Ok((
        velocity.data.into_pyarray_bound(py),
        vel_std.into_pyarray_bound(py),
        series.data.into_pyarray_bound(py),
        gamma.into_pyarray_bound(py),
        epochs,
    ))
}

/// Descompone LOS ascendente + descendente en (Up, East) — geometría escalar
/// (un único ángulo de incidencia/heading por par asc/desc, ver
/// [`decompose_per_pixel`] para geometría por píxel).
///
/// los_asc/los_desc: (filas, cols) float32, misma grilla. Devuelve (up, east).
#[pyfunction]
fn decompose_asc_desc<'py>(
    py: Python<'py>,
    los_asc: PyReadonlyArray2<'py, f32>,
    incidence_asc_deg: f64,
    heading_asc_deg: f64,
    los_desc: PyReadonlyArray2<'py, f32>,
    incidence_desc_deg: f64,
    heading_desc_deg: f64,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let asc: Array2<f32> = los_asc.as_array().to_owned();
    let desc: Array2<f32> = los_desc.as_array().to_owned();
    let geom_asc = LosVector::from_incidence_heading(incidence_asc_deg, heading_asc_deg);
    let geom_desc = LosVector::from_incidence_heading(incidence_desc_deg, heading_desc_deg);
    let decomposed = py
        .allow_threads(|| core_decompose_asc_desc(&asc, geom_asc, &desc, geom_desc))
        .map_err(err)?;
    Ok((decomposed.up.into_pyarray_bound(py), decomposed.east.into_pyarray_bound(py)))
}

/// Descompone N mapas LOS en (Up, East) con geometría **por píxel**
/// (incidencia/heading variables, p. ej. desde `los.rdr` de ISCE) —
/// diferenciador frente a la geometría escalar de [`decompose_asc_desc`].
///
/// los/incidence_deg/heading_deg: listas de igual longitud N (≥2), cada
/// elemento (filas, cols) float32, misma grilla. Devuelve (up, east).
#[pyfunction]
fn decompose_per_pixel<'py>(
    py: Python<'py>,
    los: Vec<PyReadonlyArray2<'py, f32>>,
    incidence_deg: Vec<PyReadonlyArray2<'py, f32>>,
    heading_deg: Vec<PyReadonlyArray2<'py, f32>>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    if los.len() != incidence_deg.len() || los.len() != heading_deg.len() {
        return Err(PyValueError::new_err("los, incidence_deg y heading_deg deben tener la misma longitud"));
    }
    let los_owned: Vec<Array2<f32>> = los.iter().map(|a| a.as_array().to_owned()).collect();
    let inc_owned: Vec<Array2<f32>> = incidence_deg.iter().map(|a| a.as_array().to_owned()).collect();
    let head_owned: Vec<Array2<f32>> = heading_deg.iter().map(|a| a.as_array().to_owned()).collect();

    let decomposed = py
        .allow_threads(|| {
            let los_refs: Vec<&Array2<f32>> = los_owned.iter().collect();
            let geoms: Vec<PerPixelGeometry<'_>> = inc_owned
                .iter()
                .zip(head_owned.iter())
                .map(|(incidence_deg, heading_deg)| PerPixelGeometry { incidence_deg, heading_deg })
                .collect();
            core_decompose_per_pixel(&los_refs, &geoms)
        })
        .map_err(err)?;
    Ok((decomposed.up.into_pyarray_bound(py), decomposed.east.into_pyarray_bound(py)))
}

/// Descriptores por píxel para ML (velocidad, aceleración, estacionalidad,
/// R², saltos) — ver `insar_core::features` para el detalle de cada campo.
///
/// series: (n_épocas, filas, cols) float32 m; epoch_days: día de cada época.
/// Devuelve un dict `{nombre_feature: array (filas, cols) float32}`, con las
/// claves exactas de `FeatureMaps::feature_names()` (el esquema depende solo
/// de `seasonal`/`acceleration`, nunca de los datos).
#[pyfunction]
#[pyo3(signature = (series, epoch_days, wavelength_m=SENTINEL1_WAVELENGTH_M, seasonal=true, acceleration=true, min_valid_epochs=5))]
fn extract_features<'py>(
    py: Python<'py>,
    series: PyReadonlyArray3<'py, f32>,
    epoch_days: Vec<i64>,
    wavelength_m: f64,
    seasonal: bool,
    acceleration: bool,
    min_valid_epochs: usize,
) -> PyResult<Py<PyDict>> {
    let data: Array3<f32> = series.as_array().to_owned();
    if epoch_days.len() != data.len_of(Axis(0)) {
        return Err(PyValueError::new_err(
            "epoch_days debe tener tantas entradas como el eje 0 de series",
        ));
    }
    let ds = DisplacementSeries {
        data,
        epochs: epoch_days.iter().map(|&d| epoch_from_day(d)).collect(),
        meta: dummy_meta(wavelength_m, 39.0),
    };
    let config = FeatureConfig { seasonal, acceleration, min_valid_epochs };
    let maps = py.allow_threads(|| core_extract_features(&ds, None, &config)).map_err(err)?;

    let names = maps.feature_names();
    let dict = PyDict::new_bound(py);
    for name in &names {
        let value = match *name {
            "velocity" => &maps.velocity,
            "velocity_std" => &maps.velocity_std,
            "acceleration" => &maps.acceleration,
            "linearity_r2" => &maps.linearity_r2,
            "residual_rms" => &maps.residual_rms,
            "cumulative" => &maps.cumulative,
            "seasonal_amplitude" => &maps.seasonal_amplitude,
            "seasonal_phase" => &maps.seasonal_phase,
            "max_step" => &maps.max_step,
            "temporal_coherence" => maps.temporal_coherence.as_ref().expect("nombre presente solo si Some"),
            other => unreachable!("feature_names() produjo un nombre desconocido: {other}"),
        };
        dict.set_item(name, value.clone().into_pyarray_bound(py))?;
    }
    Ok(dict.into())
}

/// Remueve una rampa (plano o cuadrática) ajustada por LSQ sobre `data`
/// (con `mask` opcional excluyendo píxeles del ajuste). `kind`: `"linear"`
/// o `"quadratic"`. Devuelve una copia de `data` con la rampa restada.
#[pyfunction]
#[pyo3(signature = (data, kind, mask=None))]
fn remove_ramp<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    kind: &str,
    mask: Option<PyReadonlyArray2<'py, bool>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let kind = match kind {
        "linear" => RampKind::Linear,
        "quadratic" => RampKind::Quadratic,
        other => {
            return Err(PyValueError::new_err(format!(
                "kind debe ser 'linear' o 'quadratic', recibido: '{other}'"
            )));
        }
    };
    let mut owned: Array2<f32> = data.as_array().to_owned();
    let mask_owned: Option<Array2<bool>> = mask.map(|m| m.as_array().to_owned());
    py.allow_threads(|| core_remove_ramp(&mut owned, kind, mask_owned.as_ref())).map_err(err)?;
    Ok(owned.into_pyarray_bound(py))
}

/// Corrige saltos de 2π por cierre de fase sobre la red SBAS (Yunjun et al.
/// 2019). phase: (n_pares, filas, cols) float32 radianes desenrollados;
/// refs/secs: índices de época de cada par.
/// Devuelve `(phase_corregida, corrected, detected_uncorrected)` — los dos
/// últimos son el conteo de píxeles corregidos vs. detectados pero no
/// corregidos (redes ralas: ver `UnwrapCorrectionReport`).
#[pyfunction]
fn correct_unwrap_errors<'py>(
    py: Python<'py>,
    phase: PyReadonlyArray3<'py, f32>,
    refs: Vec<usize>,
    secs: Vec<usize>,
) -> PyResult<(Bound<'py, PyArray3<f32>>, usize, usize)> {
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
    let n = data.len_of(Axis(0));
    let mut stack = UnwrappedStack {
        data,
        epochs: (0..n as i64).map(epoch_from_day).collect(),
        pairs,
        meta: dummy_meta(SENTINEL1_WAVELENGTH_M, 39.0),
    };
    let report = py.allow_threads(|| core_correct_unwrap_errors(&mut stack)).map_err(err)?;
    Ok((stack.data.into_pyarray_bound(py), report.corrected, report.detected_uncorrected))
}

#[pymodule]
fn insar_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(invert_sbas, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_velocity, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_velocity_uncertainty, m)?)?;
    m.add_function(wrap_pyfunction!(amplitude_dispersion, m)?)?;
    m.add_function(wrap_pyfunction!(temporal_coherence, m)?)?;
    m.add_function(wrap_pyfunction!(sbas_from_isce, m)?)?;
    m.add_function(wrap_pyfunction!(decompose_asc_desc, m)?)?;
    m.add_function(wrap_pyfunction!(decompose_per_pixel, m)?)?;
    m.add_function(wrap_pyfunction!(extract_features, m)?)?;
    m.add_function(wrap_pyfunction!(remove_ramp, m)?)?;
    m.add_function(wrap_pyfunction!(correct_unwrap_errors, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

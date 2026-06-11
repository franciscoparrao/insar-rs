//! Selección de Persistent Scatterers por amplitude dispersion
//! (Ferretti et al. 2001): D_A = σ_A / μ_A por píxel sobre el eje temporal.
//! Umbral típico: 0.25 (estricto) a 0.4 (laxo).
//!
//! Convenciones numéricas:
//! - σ_A es desviación estándar **muestral** (divisor n−1).
//! - NoData = NaN: cualquier NaN en la serie temporal de un píxel, una media
//!   ~0 (|μ| ≤ `f32::EPSILON`) o menos de 2 épocas → D_A = NaN (se propaga,
//!   no se aborta).
//! - Acumulación interna en `f64` (two-pass) para estabilidad; salida `f32`.

use ndarray::parallel::prelude::*;
use ndarray::{Array2, ArrayView1, Axis, s};

use crate::error::{InsarError, Result};
use crate::types::{AmplitudeStack, PsCandidate};

/// Calcula el mapa de amplitude dispersion (filas × cols). Píxeles con
/// amplitud media ~0 o NaN en la serie → NaN.
///
/// # Errores
/// [`InsarError::DimensionMismatch`] si el stack tiene 0 épocas.
pub fn amplitude_dispersion(stack: &AmplitudeStack) -> Result<Array2<f32>> {
    let (n_epochs, n_rows, n_cols) = stack.data.dim();
    if n_epochs == 0 {
        return Err(InsarError::DimensionMismatch(
            "amplitude_dispersion: el stack de amplitudes tiene 0 épocas".to_string(),
        ));
    }

    let mut dispersion = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);

    // Paralelización por filas: cada fila del mapa de salida es independiente.
    dispersion
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(row, mut out_row)| {
            for col in 0..n_cols {
                let series = stack.data.slice(s![.., row, col]);
                out_row[col] = pixel_dispersion(series);
            }
        });

    Ok(dispersion)
}

/// D_A para la serie temporal de un píxel. NaN si la serie contiene NaN,
/// tiene menos de 2 épocas, o su media es ~0.
fn pixel_dispersion(series: ArrayView1<'_, f32>) -> f32 {
    let n = series.len();
    if n < 2 {
        return f32::NAN;
    }

    // Primer paso: detección de NaN + suma (acumulación en f64).
    let mut sum = 0.0_f64;
    for &v in series.iter() {
        if v.is_nan() {
            return f32::NAN;
        }
        sum += f64::from(v);
    }
    let mean = sum / n as f64;

    // Media ~0: D_A no está definido (división por cero).
    if mean.abs() <= f64::from(f32::EPSILON) {
        return f32::NAN;
    }

    // Segundo paso: suma de cuadrados de desviaciones (varianza muestral, n−1).
    let mut ss = 0.0_f64;
    for &v in series.iter() {
        let d = f64::from(v) - mean;
        ss += d * d;
    }
    let std = (ss / (n - 1) as f64).sqrt();

    (std / mean) as f32
}

/// Selecciona candidatos con `amp_dispersion <= threshold`, ordenados de
/// menor a mayor dispersión (más estable primero). Los NaN se excluyen.
pub fn select_ps(dispersion: &Array2<f32>, threshold: f32) -> Vec<PsCandidate> {
    let mut candidates: Vec<PsCandidate> = dispersion
        .indexed_iter()
        .filter(|&(_, &d)| !d.is_nan() && d <= threshold)
        .map(|((row, col), &d)| PsCandidate { row, col, amp_dispersion: d })
        .collect();

    // Sin NaN tras el filtro → total_cmp da un orden total estable.
    candidates.sort_by(|a, b| a.amp_dispersion.total_cmp(&b.amp_dispersion));
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Epoch, SENTINEL1_WAVELENGTH_M, StackMeta};
    use ndarray::{Array3, array};
    use surtgis_core::GeoTransform;

    fn meta() -> StackMeta {
        StackMeta {
            transform: GeoTransform::new(0.0, 0.0, 30.0, -30.0),
            crs: None,
            wavelength_m: SENTINEL1_WAVELENGTH_M,
            incidence_deg: 39.0,
            heading_deg: None,
        }
    }

    fn epochs(n: usize) -> Vec<Epoch> {
        (0..n)
            .map(|i| {
                let d = chrono::NaiveDate::from_ymd_opt(2023, 1, 1).unwrap()
                    + chrono::Duration::days(12 * i as i64);
                Epoch(d)
            })
            .collect()
    }

    fn stack_from(data: Array3<f32>) -> AmplitudeStack {
        let n = data.shape()[0];
        AmplitudeStack { data, epochs: epochs(n), meta: meta() }
    }

    #[test]
    fn serie_constante_da_dispersion_cero() {
        // 3 épocas, 1×1, amplitud constante 5 → σ = 0 → D_A = 0.
        let stack = stack_from(Array3::from_elem((3, 1, 1), 5.0));
        let d = amplitude_dispersion(&stack).unwrap();
        assert_eq!(d[[0, 0]], 0.0);
    }

    #[test]
    fn serie_conocida_verificable_a_mano() {
        // Serie [1, 2, 3]: μ = 2, σ muestral = sqrt(((1)²+(0)²+(1)²)/2) = 1
        // → D_A = 1/2 = 0.5 exacto.
        let data = Array3::from_shape_vec((3, 1, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let d = amplitude_dispersion(&stack_from(data)).unwrap();
        assert!((d[[0, 0]] - 0.5).abs() < 1e-7, "D_A = {}", d[[0, 0]]);
    }

    #[test]
    fn serie_conocida_cuatro_epocas() {
        // Serie [2, 4, 4, 6]: μ = 4, σ² muestral = (4+0+0+4)/3 = 8/3,
        // σ = sqrt(8/3) ≈ 1.6329932 → D_A ≈ 0.4082483.
        let data = Array3::from_shape_vec((4, 1, 1), vec![2.0, 4.0, 4.0, 6.0]).unwrap();
        let d = amplitude_dispersion(&stack_from(data)).unwrap();
        let expected = (8.0_f64 / 3.0).sqrt() as f32 / 4.0;
        assert!((d[[0, 0]] - expected).abs() < 1e-7, "D_A = {}", d[[0, 0]]);
    }

    #[test]
    fn nan_en_la_serie_propaga_nan() {
        // Píxel (0,0) limpio; píxel (0,1) con un NaN en la época 1.
        let mut data = Array3::from_elem((3, 1, 2), 5.0);
        data[[1, 0, 1]] = f32::NAN;
        let d = amplitude_dispersion(&stack_from(data)).unwrap();
        assert_eq!(d[[0, 0]], 0.0);
        assert!(d[[0, 1]].is_nan());
    }

    #[test]
    fn media_cero_da_nan() {
        // Amplitudes todas 0 → μ = 0 → D_A indefinido → NaN.
        let stack = stack_from(Array3::zeros((3, 1, 1)));
        let d = amplitude_dispersion(&stack).unwrap();
        assert!(d[[0, 0]].is_nan());
    }

    #[test]
    fn una_sola_epoca_da_nan() {
        // σ muestral indefinida con n = 1 (< 2 épocas válidas).
        let stack = stack_from(Array3::from_elem((1, 2, 2), 5.0));
        let d = amplitude_dispersion(&stack).unwrap();
        assert!(d.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn cero_epocas_es_error_de_dimension() {
        let stack = stack_from(Array3::zeros((0, 2, 2)));
        match amplitude_dispersion(&stack) {
            Err(InsarError::DimensionMismatch(_)) => {}
            other => panic!("se esperaba DimensionMismatch, se obtuvo {other:?}"),
        }
    }

    #[test]
    fn dimensiones_de_salida_correctas() {
        let stack = stack_from(Array3::from_elem((3, 4, 7), 1.0));
        let d = amplitude_dispersion(&stack).unwrap();
        assert_eq!(d.dim(), (4, 7));
    }

    #[test]
    fn select_ps_filtra_y_ordena_ascendente() {
        // NaN excluido; 0.5 > umbral excluido; resto ordenado por D_A.
        let disp = array![[0.30, f32::NAN], [0.10, 0.50], [0.25, 0.40]];
        let ps = select_ps(&disp, 0.4);
        let got: Vec<(usize, usize, f32)> =
            ps.iter().map(|p| (p.row, p.col, p.amp_dispersion)).collect();
        assert_eq!(
            got,
            vec![(1, 0, 0.10), (2, 0, 0.25), (0, 0, 0.30), (2, 1, 0.40)]
        );
    }

    #[test]
    fn select_ps_umbral_es_inclusivo() {
        let disp = array![[0.25]];
        assert_eq!(select_ps(&disp, 0.25).len(), 1);
        assert_eq!(select_ps(&disp, 0.24).len(), 0);
    }

    #[test]
    fn select_ps_mapa_todo_nan_devuelve_vacio() {
        let disp = Array2::<f32>::from_elem((3, 3), f32::NAN);
        assert!(select_ps(&disp, 1.0).is_empty());
    }

    #[test]
    fn pipeline_ps_sintetico_completo() {
        // 2×2: (0,0) estable [10,10,10]; (0,1) variable [1,2,3] → D_A=0.5;
        // (1,0) con NaN; (1,1) cero → NaN. Umbral 0.4 → solo (0,0).
        let data = Array3::from_shape_vec(
            (3, 2, 2),
            vec![
                10.0, 1.0, 5.0, 0.0, // época 0
                10.0, 2.0, f32::NAN, 0.0, // época 1
                10.0, 3.0, 5.0, 0.0, // época 2
            ],
        )
        .unwrap();
        let d = amplitude_dispersion(&stack_from(data)).unwrap();
        let ps = select_ps(&d, 0.4);
        assert_eq!(ps.len(), 1);
        assert_eq!((ps[0].row, ps[0].col), (0, 0));
        assert_eq!(ps[0].amp_dispersion, 0.0);
    }
}

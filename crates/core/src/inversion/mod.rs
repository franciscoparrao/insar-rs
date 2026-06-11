//! Inversión SBAS de la serie temporal de desplazamiento LOS por mínimos
//! cuadrados (SVD de nalgebra), conversión fase→desplazamiento y estimación
//! de velocidad media.
//!
//! Variante de incrementos de Berardino et al. (2002): las incógnitas son
//! los desplazamientos entre épocas consecutivas; la serie acumulada se
//! reconstruye relativa a la primera época.

use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, Array3, Axis};
use rayon::prelude::*;

use crate::error::{InsarError, Result};
use crate::network;
use crate::types::{DisplacementSeries, PsCandidate, UnwrappedStack, VelocityMap};

/// Convierte fase desenrollada (radianes) a desplazamiento LOS (metros):
/// `d = -λ/(4π)·φ` (alejamiento del sensor = negativo).
pub fn phase_to_displacement(phase_rad: f64, wavelength_m: f64) -> f64 {
    -(wavelength_m / (4.0 * PI)) * phase_rad
}

/// Invierte la serie temporal de desplazamiento LOS por píxel.
/// - `ps = Some(...)`: invierte solo en los candidatos PS; el resto queda NaN.
/// - `ps = None`: invierte toda la grilla (modo SBAS clásico).
///
/// La serie resultante es relativa a la primera época (desplazamiento 0).
/// Error si la red es desconectada ([`crate::network::is_connected`]).
///
/// Manejo de NoData (MVP): un píxel con cualquier fase no finita (NaN/inf)
/// en cualquiera de sus pares queda con la serie completa en NaN.
/// Nota para después: soportar NaN por par requeriría re-factorizar la
/// matriz de diseño por píxel (subconjunto de filas), no solo cachear la SVD.
pub fn invert_sbas(
    stack: &UnwrappedStack,
    ps: Option<&[PsCandidate]>,
) -> Result<DisplacementSeries> {
    stack.validate()?;

    let n_epochs = stack.epochs.len();
    let n_pairs = stack.pairs.len();
    let (n_rows, n_cols) = stack.dims();

    if n_epochs < 2 {
        return Err(InsarError::DimensionMismatch(format!(
            "se requieren al menos 2 épocas para invertir la serie ({n_epochs} recibidas)"
        )));
    }

    if !network::is_connected(&stack.pairs, n_epochs) {
        return Err(InsarError::InvalidNetwork(format!(
            "la red de {n_pairs} pares sobre {n_epochs} épocas es desconectada; \
             la inversión SBAS quedaría ambigua entre subsets"
        )));
    }

    // Candidatos PS fuera de la grilla: error explícito (sin panic).
    if let Some(cands) = ps
        && let Some(p) = cands.iter().find(|p| p.row >= n_rows || p.col >= n_cols)
    {
        return Err(InsarError::DimensionMismatch(format!(
            "candidato PS ({}, {}) fuera de la grilla {n_rows}×{n_cols}",
            p.row, p.col
        )));
    }

    // Matriz de diseño y su pseudoinversa: se calculan UNA vez y se
    // reutilizan en todos los píxeles (la SVD es el costo dominante).
    let a = network::design_matrix(&stack.pairs, n_epochs)?;
    let n_unknowns = n_epochs - 1;
    let a_mat = DMatrix::<f64>::from_fn(n_pairs, n_unknowns, |i, j| a[[i, j]]);
    let svd = a_mat.svd(true, true);
    // Tolerancia estilo rcond de numpy/LAPACK: σ_max · max(m,n) · ε_f64.
    // Con red conexa A es de rango columna completo; si no lo fuera, la
    // pseudoinversa entrega la solución de norma mínima.
    let s_max = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let eps = s_max * (n_pairs.max(n_unknowns) as f64) * f64::EPSILON;
    let pinv = svd
        .pseudo_inverse(eps)
        .map_err(|e| InsarError::Inversion(format!("pseudoinversa SVD: {e}")))?;

    // Columnas a invertir por fila (toda la grilla si ps = None).
    let cols_by_row: Vec<Vec<usize>> = match ps {
        None => (0..n_rows).map(|_| (0..n_cols).collect()).collect(),
        Some(cands) => {
            let mut by_row = vec![Vec::new(); n_rows];
            for p in cands {
                by_row[p.row].push(p.col);
            }
            by_row
        }
    };

    let mut out = Array3::<f32>::from_elem((n_epochs, n_rows, n_cols), f32::NAN);
    let phases = stack.data.view();
    let wavelength_m = stack.meta.wavelength_m;

    // Paralelización por filas: cada fila escribe su vista (épocas × cols).
    let mut row_views: Vec<_> = out.axis_iter_mut(Axis(1)).collect();
    row_views.par_iter_mut().enumerate().for_each(|(r, out_row)| {
        let mut b = DVector::<f64>::zeros(n_pairs);
        for &c in &cols_by_row[r] {
            let mut valid = true;
            for k in 0..n_pairs {
                let phi = phases[[k, r, c]];
                if !phi.is_finite() {
                    valid = false;
                    break;
                }
                b[k] = phase_to_displacement(phi as f64, wavelength_m);
            }
            if !valid {
                continue; // serie completa NaN para este píxel (MVP)
            }
            // x = incrementos de desplazamiento entre épocas consecutivas.
            let x = &pinv * &b;
            out_row[[0, c]] = 0.0;
            let mut acc = 0.0_f64;
            for e in 1..n_epochs {
                acc += x[e - 1];
                out_row[[e, c]] = acc as f32;
            }
        }
    });

    Ok(DisplacementSeries {
        data: out,
        epochs: stack.epochs.clone(),
        meta: stack.meta.clone(),
    })
}

/// Velocidad media LOS (m/año) por ajuste lineal de la serie de cada píxel.
/// Píxeles con < 2 épocas válidas → NaN (en MVP: cualquier NaN en la serie
/// deja el píxel en NaN). Error si la serie tiene menos de 2 épocas.
pub fn estimate_velocity(series: &DisplacementSeries) -> Result<VelocityMap> {
    let n_epochs = series.n_layers();
    let (n_rows, n_cols) = series.dims();

    if series.epochs.len() != n_epochs {
        return Err(InsarError::DimensionMismatch(format!(
            "{} épocas declaradas vs {n_epochs} capas en la serie",
            series.epochs.len()
        )));
    }
    if n_epochs < 2 {
        return Err(InsarError::DimensionMismatch(format!(
            "se requieren al menos 2 épocas para estimar velocidad ({n_epochs} recibidas)"
        )));
    }

    // Tiempo en años decimales relativo a la primera época.
    let t: Vec<f64> = series
        .epochs
        .iter()
        .map(|e| e.years_since(&series.epochs[0]))
        .collect();
    let t_mean = t.iter().sum::<f64>() / n_epochs as f64;
    // Pendiente LSQ: v = Σ(t_e − t̄)·d_e / Σ(t_e − t̄)².
    let denom: f64 = t.iter().map(|&ti| (ti - t_mean).powi(2)).sum();
    if denom <= 0.0 {
        return Err(InsarError::Inversion(
            "todas las épocas tienen la misma fecha; el ajuste lineal es indeterminado"
                .to_string(),
        ));
    }

    let mut out = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let data = series.data.view();

    let mut row_views: Vec<_> = out.axis_iter_mut(Axis(0)).collect();
    row_views.par_iter_mut().enumerate().for_each(|(r, out_row)| {
        for c in 0..n_cols {
            let mut sxy = 0.0_f64;
            let mut valid = true;
            for e in 0..n_epochs {
                let d = data[[e, r, c]];
                if !d.is_finite() {
                    valid = false;
                    break;
                }
                sxy += (t[e] - t_mean) * d as f64;
            }
            if valid {
                out_row[c] = (sxy / denom) as f32;
            }
        }
    });

    Ok(VelocityMap { data: out, meta: series.meta.clone() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        DisplacementSeries, Epoch, IfgPair, PsCandidate, StackMeta, UnwrappedStack,
        SENTINEL1_WAVELENGTH_M,
    };
    use ndarray::Array3;
    use surtgis_core::GeoTransform;

    #[test]
    fn fase_a_desplazamiento_signo_y_magnitud() {
        // Un ciclo completo de fase (2π) equivale a λ/2 de desplazamiento LOS.
        let d = phase_to_displacement(2.0 * PI, SENTINEL1_WAVELENGTH_M);
        assert!((d.abs() - SENTINEL1_WAVELENGTH_M / 2.0).abs() < 1e-12);
        // Fase positiva (aumento de camino) = alejamiento = negativo.
        assert!(d < 0.0);
    }

    // ---------- helpers ----------

    fn meta() -> StackMeta {
        StackMeta {
            transform: GeoTransform::new(0.0, 0.0, 30.0, -30.0),
            crs: None,
            wavelength_m: SENTINEL1_WAVELENGTH_M,
            incidence_deg: 39.0,
            heading_deg: None,
        }
    }

    /// Épocas cada 12 días a partir de 2023-01-01.
    fn epochs_12d(n: usize) -> Vec<Epoch> {
        let start: chrono::NaiveDate = "2023-01-01".parse().unwrap();
        (0..n)
            .map(|i| Epoch(start + chrono::Duration::days(12 * i as i64)))
            .collect()
    }

    fn pair(i: usize, j: usize) -> IfgPair {
        IfgPair { reference: i, secondary: j, perp_baseline_m: 0.0 }
    }

    /// Red de 4 épocas: pares consecutivos + saltos de 2.
    fn pairs_4ep() -> Vec<IfgPair> {
        vec![pair(0, 1), pair(1, 2), pair(2, 3), pair(0, 2), pair(1, 3)]
    }

    const V_TRUE: f64 = -0.05; // m/año, desplazamiento lineal sintético

    /// Desplazamientos verdaderos por época, relativos a la primera.
    fn true_displacements(epochs: &[Epoch]) -> Vec<f64> {
        epochs.iter().map(|e| V_TRUE * e.years_since(&epochs[0])).collect()
    }

    /// Stack sintético exacto: φ_par = −4π/λ · (d_sec − d_ref), igual en
    /// todos los píxeles de la grilla rows×cols.
    fn synthetic_stack(rows: usize, cols: usize) -> UnwrappedStack {
        let epochs = epochs_12d(4);
        let pairs = pairs_4ep();
        let d = true_displacements(&epochs);
        let mut data = Array3::<f32>::zeros((pairs.len(), rows, cols));
        for (k, p) in pairs.iter().enumerate() {
            let dd = d[p.secondary] - d[p.reference];
            let phi = (-4.0 * PI / SENTINEL1_WAVELENGTH_M * dd) as f32;
            data.index_axis_mut(ndarray::Axis(0), k).fill(phi);
        }
        UnwrappedStack { data, epochs, pairs, meta: meta() }
    }

    // ---------- invert_sbas ----------

    #[test]
    fn inversion_recupera_desplazamiento_lineal() {
        let stack = synthetic_stack(2, 3);
        let series = invert_sbas(&stack, None).unwrap();

        assert_eq!(series.data.shape(), &[4, 2, 3]);
        let d = true_displacements(&stack.epochs);
        for e in 0..4 {
            for r in 0..2 {
                for c in 0..3 {
                    let got = series.data[[e, r, c]] as f64;
                    assert!(
                        (got - d[e]).abs() < 1e-5,
                        "época {e}, píxel ({r},{c}): {got} vs {}",
                        d[e]
                    );
                }
            }
        }
        // Relativa a la primera época: exactamente 0.
        assert_eq!(series.data[[0, 0, 0]], 0.0);
    }

    #[test]
    fn velocidad_recupera_v_sintetica() {
        let stack = synthetic_stack(2, 3);
        let series = invert_sbas(&stack, None).unwrap();
        let vel = estimate_velocity(&series).unwrap();

        assert_eq!(vel.data.shape(), &[2, 3]);
        for r in 0..2 {
            for c in 0..3 {
                let v = vel.data[[r, c]] as f64;
                assert!(
                    (v - V_TRUE).abs() < 1e-6,
                    "píxel ({r},{c}): v = {v} vs {V_TRUE}"
                );
            }
        }
    }

    #[test]
    fn ps_some_invierte_solo_candidatos() {
        let stack = synthetic_stack(2, 3);
        let cands = [PsCandidate { row: 1, col: 2, amp_dispersion: 0.1 }];
        let series = invert_sbas(&stack, Some(&cands)).unwrap();

        let d = true_displacements(&stack.epochs);
        for e in 0..4 {
            // El candidato se invierte correctamente...
            assert!((series.data[[e, 1, 2]] as f64 - d[e]).abs() < 1e-5);
            // ...y todo el resto de la grilla queda NaN.
            for r in 0..2 {
                for c in 0..3 {
                    if (r, c) != (1, 2) {
                        assert!(series.data[[e, r, c]].is_nan(), "({e},{r},{c}) no es NaN");
                    }
                }
            }
        }
    }

    #[test]
    fn ps_fuera_de_grilla_es_error() {
        let stack = synthetic_stack(2, 3);
        let cands = [PsCandidate { row: 5, col: 0, amp_dispersion: 0.1 }];
        let err = invert_sbas(&stack, Some(&cands)).unwrap_err();
        assert!(matches!(err, InsarError::DimensionMismatch(_)));
    }

    #[test]
    fn pixel_con_nan_queda_todo_nan() {
        let mut stack = synthetic_stack(2, 3);
        stack.data[[2, 0, 1]] = f32::NAN; // un solo par contaminado

        let series = invert_sbas(&stack, None).unwrap();
        // El píxel contaminado: serie completa NaN, incluida la época 0.
        for e in 0..4 {
            assert!(series.data[[e, 0, 1]].is_nan(), "época {e} no es NaN");
        }
        // Un vecino limpio se invierte normalmente.
        let d = true_displacements(&stack.epochs);
        for e in 0..4 {
            assert!((series.data[[e, 0, 0]] as f64 - d[e]).abs() < 1e-5);
        }
    }

    #[test]
    fn red_desconectada_es_error() {
        // {0,1} y {2,3} sin puente.
        let pairs = vec![pair(0, 1), pair(2, 3)];
        let stack = UnwrappedStack {
            data: Array3::zeros((2, 2, 2)),
            epochs: epochs_12d(4),
            pairs,
            meta: meta(),
        };
        let err = invert_sbas(&stack, None).unwrap_err();
        assert!(matches!(err, InsarError::InvalidNetwork(_)));
    }

    // ---------- estimate_velocity ----------

    #[test]
    fn velocidad_propaga_nan_de_la_serie() {
        let stack = synthetic_stack(2, 2);
        let mut series = invert_sbas(&stack, None).unwrap();
        series.data[[3, 1, 0]] = f32::NAN;

        let vel = estimate_velocity(&series).unwrap();
        assert!(vel.data[[1, 0]].is_nan());
        assert!((vel.data[[0, 0]] as f64 - V_TRUE).abs() < 1e-6);
    }

    #[test]
    fn velocidad_con_una_epoca_es_error() {
        let series = DisplacementSeries {
            data: Array3::zeros((1, 2, 2)),
            epochs: epochs_12d(1),
            meta: meta(),
        };
        let err = estimate_velocity(&series).unwrap_err();
        assert!(matches!(err, InsarError::DimensionMismatch(_)));
    }
}

//! Inversión SBAS de la serie temporal de desplazamiento LOS por mínimos
//! cuadrados (SVD de nalgebra), conversión fase→desplazamiento y estimación
//! de velocidad media.
//!
//! Variante de incrementos de Berardino et al. (2002): las incógnitas son
//! los desplazamientos entre épocas consecutivas; la serie acumulada se
//! reconstruye relativa a la primera época.

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, Array3, Axis};
use rayon::prelude::*;

use crate::error::{InsarError, Result};
use crate::network;
use crate::types::{DisplacementSeries, IfgPair, PsCandidate, UnwrappedStack, VelocityMap};

/// Convierte fase desenrollada (radianes) a desplazamiento LOS (metros):
/// `d = -λ/(4π)·φ` (alejamiento del sensor = negativo).
pub fn phase_to_displacement(phase_rad: f64, wavelength_m: f64) -> f64 {
    -(wavelength_m / (4.0 * PI)) * phase_rad
}

/// Inversa de [`phase_to_displacement`]: `φ = -(4π/λ)·d`.
pub fn displacement_to_phase(displacement_m: f64, wavelength_m: f64) -> f64 {
    -(4.0 * PI / wavelength_m) * displacement_m
}

/// Referencia espacial del stack: resta, en cada par, la fase del píxel
/// `(row, col)` a todos los píxeles. Elimina el offset constante por
/// interferograma que deja el desenrollado (cada `.unw` tiene una referencia
/// de fase arbitraria), dejando la serie relativa a ese píxel. Sin este paso
/// los offsets aparecen como residuos y degradan la coherencia temporal.
///
/// Si el píxel de referencia no tiene fase finita en un par, ese par no se
/// puede referenciar y queda NaN (se descartará por píxel en la inversión).
/// Error si `(row, col)` está fuera de la grilla.
pub fn reference_to_pixel(stack: &mut UnwrappedStack, row: usize, col: usize) -> Result<()> {
    let (n_rows, n_cols) = stack.dims();
    if row >= n_rows || col >= n_cols {
        return Err(InsarError::DimensionMismatch(format!(
            "píxel de referencia ({row}, {col}) fuera de la grilla {n_rows}×{n_cols}"
        )));
    }
    for k in 0..stack.n_layers() {
        let mut layer = stack.data.index_axis_mut(Axis(0), k);
        let ref_phase = layer[[row, col]];
        if ref_phase.is_finite() {
            layer.mapv_inplace(|v| v - ref_phase);
        } else {
            layer.fill(f32::NAN);
        }
    }
    Ok(())
}

/// Invierte la serie temporal de desplazamiento LOS por píxel.
/// - `ps = Some(...)`: invierte solo en los candidatos PS; el resto queda NaN.
/// - `ps = None`: invierte toda la grilla (modo SBAS clásico).
///
/// La serie resultante es relativa a la primera época (desplazamiento 0).
/// Error si la red es desconectada ([`crate::network::is_connected`]).
///
/// Manejo de NoData por par: cada píxel se invierte con los pares que tienen
/// fase finita. Si todos sus pares son válidos se usa la SVD cacheada de la
/// matriz de diseño completa (camino rápido); si faltan algunos, se resuelve
/// con una matriz de diseño reducida a los pares válidos (camino por píxel).
/// Un píxel queda con la serie completa en NaN solo si sus pares válidos no
/// conectan todas las épocas (red desconectada → serie indeterminada).
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

    // Inversión por píxel con pseudoinversa cacheada por patrón de validez:
    // los píxeles con todos los pares válidos usan la SVD completa (camino
    // rápido); los demás se agrupan por su conjunto de pares válidos y comparten
    // una pseudoinversa reducida, evitando una SVD por píxel.
    let n_words = n_pairs.div_ceil(64);
    let mask_bit = |key: &mut [u64], k: usize| key[k / 64] |= 1u64 << (k % 64);

    // Pasada 1: patrones de máscara parcial distintos (en paralelo por fila).
    let unique_masks: HashSet<Vec<u64>> = cols_by_row
        .par_iter()
        .enumerate()
        .map(|(r, cols)| {
            let mut set: HashSet<Vec<u64>> = HashSet::new();
            for &c in cols {
                let mut key = vec![0u64; n_words];
                let mut n_valid = 0usize;
                for k in 0..n_pairs {
                    if phases[[k, r, c]].is_finite() {
                        mask_bit(&mut key, k);
                        n_valid += 1;
                    }
                }
                if n_valid != n_pairs {
                    set.insert(key);
                }
            }
            set
        })
        .reduce(HashSet::new, |mut a, b| {
            a.extend(b);
            a
        });

    // Pasada 2: pseudoinversa reducida por patrón (en paralelo). `None` si la
    // red reducida queda desconectada o con menos pares que incógnitas.
    let solvers: HashMap<Vec<u64>, Option<DMatrix<f64>>> = unique_masks
        .into_par_iter()
        .map(|mask| {
            let valid_idx: Vec<usize> = (0..n_pairs)
                .filter(|&k| mask[k / 64] & (1u64 << (k % 64)) != 0)
                .collect();
            let solver = reduced_pinv(&valid_idx, &stack.pairs, n_epochs, n_unknowns);
            (mask, solver)
        })
        .collect();

    // Pasada 3: inversión por píxel (en paralelo por fila).
    let mut row_views: Vec<_> = out.axis_iter_mut(Axis(1)).collect();
    row_views.par_iter_mut().enumerate().for_each(|(r, out_row)| {
        let mut b_vals: Vec<f64> = Vec::with_capacity(n_pairs);
        let mut key = vec![0u64; n_words];
        for &c in &cols_by_row[r] {
            b_vals.clear();
            key.iter_mut().for_each(|w| *w = 0);
            for k in 0..n_pairs {
                let phi = phases[[k, r, c]];
                if phi.is_finite() {
                    mask_bit(&mut key, k);
                    b_vals.push(phase_to_displacement(phi as f64, wavelength_m));
                }
            }
            // x = incrementos de desplazamiento entre épocas consecutivas.
            let x = if b_vals.len() == n_pairs {
                Some(&pinv * DVector::from_column_slice(&b_vals)) // camino rápido
            } else {
                match solvers.get(&key) {
                    Some(Some(rp)) => Some(rp * DVector::from_column_slice(&b_vals)),
                    _ => None, // sin pares válidos o red reducida desconectada
                }
            };
            if let Some(x) = x {
                out_row[[0, c]] = 0.0;
                let mut acc = 0.0_f64;
                for e in 1..n_epochs {
                    acc += x[e - 1];
                    out_row[[e, c]] = acc as f32;
                }
            }
        }
    });

    Ok(DisplacementSeries {
        data: out,
        epochs: stack.epochs.clone(),
        meta: stack.meta.clone(),
    })
}

/// Pseudoinversa de la matriz de diseño reducida al subconjunto de pares
/// válidos (`valid_idx`, ascendentes). Devuelve `None` (→ serie NaN) si hay
/// menos pares que incógnitas o si los pares no conectan todas las épocas
/// (red reducida desconectada → sistema rank-deficiente).
fn reduced_pinv(
    valid_idx: &[usize],
    pairs: &[IfgPair],
    n_epochs: usize,
    n_unknowns: usize,
) -> Option<DMatrix<f64>> {
    if valid_idx.len() < n_unknowns {
        return None;
    }
    let reduced: Vec<IfgPair> = valid_idx.iter().map(|&k| pairs[k]).collect();
    if !network::is_connected(&reduced, n_epochs) {
        return None;
    }
    let m = reduced.len();
    // Matriz de diseño reducida: fila i = par válido i, 1.0 en columnas
    // [reference, secondary) (incrementos entre épocas consecutivas).
    let a = DMatrix::<f64>::from_fn(m, n_unknowns, |i, j| {
        let p = &reduced[i];
        if p.reference <= j && j < p.secondary { 1.0 } else { 0.0 }
    });
    let svd = a.svd(true, true);
    let s_max = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let eps = s_max * (m.max(n_unknowns) as f64) * f64::EPSILON;
    svd.pseudo_inverse(eps).ok()
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

/// Incertidumbre (error estándar) de la velocidad LOS por píxel (m/año), del
/// ajuste lineal OLS: `SE(v) = sqrt( (SSR/(n−2)) / Σ(t−t̄)² )`, con SSR la suma
/// de residuos al cuadrado del ajuste. Requiere ≥3 épocas (n−2 grados de
/// libertad); con menos → error. Píxeles con la serie no finita → NaN.
pub fn estimate_velocity_uncertainty(series: &DisplacementSeries) -> Result<Array2<f32>> {
    let n_epochs = series.n_layers();
    let (n_rows, n_cols) = series.dims();

    if series.epochs.len() != n_epochs {
        return Err(InsarError::DimensionMismatch(format!(
            "{} épocas declaradas vs {n_epochs} capas en la serie",
            series.epochs.len()
        )));
    }
    if n_epochs < 3 {
        return Err(InsarError::DimensionMismatch(format!(
            "se requieren al menos 3 épocas para la incertidumbre de velocidad ({n_epochs} recibidas)"
        )));
    }

    let t: Vec<f64> = series
        .epochs
        .iter()
        .map(|e| e.years_since(&series.epochs[0]))
        .collect();
    let t_mean = t.iter().sum::<f64>() / n_epochs as f64;
    let sxx: f64 = t.iter().map(|&ti| (ti - t_mean).powi(2)).sum();
    if sxx <= 0.0 {
        return Err(InsarError::Inversion(
            "todas las épocas tienen la misma fecha; el ajuste lineal es indeterminado".to_string(),
        ));
    }

    let mut out = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let data = series.data.view();
    let mut row_views: Vec<_> = out.axis_iter_mut(Axis(0)).collect();
    row_views.par_iter_mut().enumerate().for_each(|(r, out_row)| {
        for c in 0..n_cols {
            // d̄ y la pendiente v = Σ(t−t̄)d / Σ(t−t̄)²; aborta si hay NaN.
            let (mut d_mean, mut sxy, mut valid) = (0.0_f64, 0.0_f64, true);
            for e in 0..n_epochs {
                let d = data[[e, r, c]];
                if !d.is_finite() {
                    valid = false;
                    break;
                }
                d_mean += d as f64;
                sxy += (t[e] - t_mean) * d as f64;
            }
            if !valid {
                continue;
            }
            d_mean /= n_epochs as f64;
            let v = sxy / sxx;
            // SSR = Σ (d − d̄ − v·(t − t̄))².
            let mut ssr = 0.0_f64;
            for e in 0..n_epochs {
                let resid = data[[e, r, c]] as f64 - d_mean - v * (t[e] - t_mean);
                ssr += resid * resid;
            }
            let var_v = (ssr / (n_epochs - 2) as f64) / sxx;
            out_row[c] = var_v.sqrt() as f32;
        }
    });

    Ok(out)
}

/// Coherencia temporal (Pepe & Lanari 2006): consistencia entre las fases
/// observadas de cada par y las reconstruidas desde la serie invertida. Rango
/// [0, 1] (1 = ajuste perfecto). Es la métrica de calidad estándar para
/// enmascarar píxeles poco fiables (p. ej. `γ_temp < 0.7`).
///
/// `γ = (1/M)·| Σ_k exp(j·(φ_obs_k − φ_model_k)) |`, donde
/// `φ_model_k = displacement_to_phase(d_sec − d_ref)` y M es el número de pares
/// con fase observada finita y serie finita en ambas épocas. El exponencial
/// complejo es 2π-periódico, así que no requiere desenrollar el residuo.
/// Píxeles sin pares válidos → NaN.
pub fn temporal_coherence(
    stack: &UnwrappedStack,
    series: &DisplacementSeries,
) -> Result<Array2<f32>> {
    let (n_rows, n_cols) = stack.dims();
    if series.dims() != (n_rows, n_cols) {
        return Err(InsarError::DimensionMismatch(format!(
            "serie {:?} vs stack {:?}",
            series.dims(),
            (n_rows, n_cols)
        )));
    }
    if series.epochs.len() != stack.epochs.len() {
        return Err(InsarError::DimensionMismatch(format!(
            "{} épocas en la serie vs {} en el stack",
            series.epochs.len(),
            stack.epochs.len()
        )));
    }

    let wavelength_m = stack.meta.wavelength_m;
    let phases = stack.data.view();
    let disp = series.data.view();
    let pairs = &stack.pairs;

    let mut out = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut row_views: Vec<_> = out.axis_iter_mut(Axis(0)).collect();
    row_views.par_iter_mut().enumerate().for_each(|(r, out_row)| {
        for c in 0..n_cols {
            let (mut re, mut im, mut m) = (0.0_f64, 0.0_f64, 0usize);
            for (k, p) in pairs.iter().enumerate() {
                let obs = phases[[k, r, c]];
                let d_sec = disp[[p.secondary, r, c]];
                let d_ref = disp[[p.reference, r, c]];
                if obs.is_finite() && d_sec.is_finite() && d_ref.is_finite() {
                    let model = displacement_to_phase((d_sec - d_ref) as f64, wavelength_m);
                    let dphi = obs as f64 - model;
                    re += dphi.cos();
                    im += dphi.sin();
                    m += 1;
                }
            }
            if m > 0 {
                out_row[c] = ((re * re + im * im).sqrt() / m as f64) as f32;
            }
        }
    });

    Ok(out)
}

#[cfg(test)]
mod tests {
    // En los tests `e` indexa tanto la serie 3D como el vector de verdad d[e];
    // el bucle por rango es el más legible aquí.
    #![allow(clippy::needless_range_loop)]
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
    fn inversion_con_par_faltante_recupera() {
        // pairs_4ep: idx0=(0,1) idx1=(1,2) idx2=(2,3) idx3=(0,2) idx4=(1,3).
        let mut stack = synthetic_stack(1, 2);
        let d = true_displacements(&stack.epochs);
        // Píxel (0,0): elimina el par redundante (0,2); los restantes
        // [(0,1),(1,2),(2,3),(1,3)] aún conectan las 4 épocas → debe recuperar.
        stack.data[[3, 0, 0]] = f32::NAN;
        let series = invert_sbas(&stack, None).unwrap();
        for e in 0..4 {
            let got = series.data[[e, 0, 0]] as f64;
            assert!((got - d[e]).abs() < 1e-5, "par faltante, época {e}: {got} vs {}", d[e]);
            // Píxel (0,1) intacto: camino rápido, también recupera.
            assert!((series.data[[e, 0, 1]] as f64 - d[e]).abs() < 1e-5);
        }
    }

    #[test]
    fn inversion_nan_si_pares_validos_desconectan_red() {
        let mut stack = synthetic_stack(1, 2);
        let d = true_displacements(&stack.epochs);
        // Píxel (0,0): elimina los dos pares que tocan la época 3: (2,3) y (1,3).
        // Quedan (0,1),(1,2),(0,2) → época 3 aislada → serie NaN.
        stack.data[[2, 0, 0]] = f32::NAN;
        stack.data[[4, 0, 0]] = f32::NAN;
        let series = invert_sbas(&stack, None).unwrap();
        for e in 0..4 {
            assert!(series.data[[e, 0, 0]].is_nan(), "época {e} debería ser NaN");
            // Píxel (0,1) intacto recupera normal.
            assert!((series.data[[e, 0, 1]] as f64 - d[e]).abs() < 1e-5);
        }
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
    fn pixel_sin_pares_validos_queda_todo_nan() {
        let mut stack = synthetic_stack(2, 3);
        // Todos los pares del píxel (0,1) contaminados → sin observaciones.
        for k in 0..stack.pairs.len() {
            stack.data[[k, 0, 1]] = f32::NAN;
        }

        let series = invert_sbas(&stack, None).unwrap();
        // El píxel sin pares válidos: serie completa NaN, incluida la época 0.
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

    // ---------- estimate_velocity_uncertainty ----------

    #[test]
    fn incertidumbre_cero_en_ajuste_perfecto() {
        // Serie lineal exacta → residuos 0 → SE(v) = 0.
        let stack = synthetic_stack(2, 2);
        let series = invert_sbas(&stack, None).unwrap();
        let se = estimate_velocity_uncertainty(&series).unwrap();
        for &s in se.iter() {
            assert!(s < 1e-6, "SE = {s}");
        }
    }

    #[test]
    fn incertidumbre_positiva_con_residuo() {
        // Perturbar una época rompe la linealidad → SE > 0.
        let stack = synthetic_stack(1, 1);
        let mut series = invert_sbas(&stack, None).unwrap();
        series.data[[2, 0, 0]] += 0.01; // 1 cm fuera de la recta
        let se = estimate_velocity_uncertainty(&series).unwrap();
        assert!(se[[0, 0]] > 0.0);
    }

    #[test]
    fn incertidumbre_menos_de_tres_epocas_es_error() {
        let series = DisplacementSeries {
            data: Array3::zeros((2, 2, 2)),
            epochs: epochs_12d(2),
            meta: meta(),
        };
        assert!(matches!(
            estimate_velocity_uncertainty(&series).unwrap_err(),
            InsarError::DimensionMismatch(_)
        ));
    }

    // ---------- reference_to_pixel ----------

    #[test]
    fn referencia_elimina_offset_por_par() {
        let mut stack = synthetic_stack(2, 2);
        // Añade un offset constante distinto a cada par (todos los píxeles).
        for k in 0..stack.pairs.len() {
            let off = 3.0 + k as f32;
            stack.data.index_axis_mut(ndarray::Axis(0), k).mapv_inplace(|v| v + off);
        }
        reference_to_pixel(&mut stack, 0, 0).unwrap();
        // El píxel de referencia queda en 0 para todos los pares.
        for k in 0..stack.pairs.len() {
            assert!(stack.data[[k, 0, 0]].abs() < 1e-6);
        }
        // Tras referenciar, la inversión recupera el desplazamiento lineal
        // (el offset por par desaparece). Píxeles uniformes → serie 0.
        let series = invert_sbas(&stack, None).unwrap();
        for e in 0..4 {
            assert!(series.data[[e, 1, 1]].abs() < 1e-5);
        }
    }

    #[test]
    fn referencia_fuera_de_grilla_es_error() {
        let mut stack = synthetic_stack(2, 2);
        assert!(matches!(
            reference_to_pixel(&mut stack, 5, 0).unwrap_err(),
            InsarError::DimensionMismatch(_)
        ));
    }

    // ---------- temporal_coherence ----------

    #[test]
    fn coherencia_temporal_uno_en_ajuste_perfecto() {
        // Stack sintético exacto: la serie invertida reconstruye las fases
        // observadas sin residuo → γ_temp = 1 en todos los píxeles.
        let stack = synthetic_stack(2, 3);
        let series = invert_sbas(&stack, None).unwrap();
        let gamma = temporal_coherence(&stack, &series).unwrap();
        assert_eq!(gamma.shape(), &[2, 3]);
        for &g in gamma.iter() {
            assert!((g - 1.0).abs() < 1e-5, "γ = {g}");
        }
    }

    #[test]
    fn coherencia_temporal_baja_con_residuo() {
        // Corromper un par en un píxel introduce un residuo de fase → γ < 1.
        let mut stack = synthetic_stack(1, 2);
        let series = invert_sbas(&stack, None).unwrap();
        // El píxel (0,0) recibe un offset grande en un par tras invertir con
        // la serie limpia: rompemos la consistencia obs vs modelo.
        stack.data[[0, 0, 0]] += 2.0; // +2 rad en el par 0
        let gamma = temporal_coherence(&stack, &series).unwrap();
        assert!(gamma[[0, 0]] < 0.95, "γ corrupto = {}", gamma[[0, 0]]);
        assert!((gamma[[0, 1]] - 1.0).abs() < 1e-5, "γ limpio = {}", gamma[[0, 1]]);
    }

    #[test]
    fn coherencia_temporal_nan_sin_pares_validos() {
        let mut stack = synthetic_stack(1, 1);
        let series = invert_sbas(&stack, None).unwrap();
        for k in 0..stack.pairs.len() {
            stack.data[[k, 0, 0]] = f32::NAN;
        }
        let gamma = temporal_coherence(&stack, &series).unwrap();
        assert!(gamma[[0, 0]].is_nan());
    }

    #[test]
    fn coherencia_temporal_dim_mismatch_es_error() {
        let stack = synthetic_stack(2, 3);
        let series = DisplacementSeries {
            data: Array3::zeros((4, 2, 2)), // cols distinto
            epochs: epochs_12d(4),
            meta: meta(),
        };
        assert!(matches!(
            temporal_coherence(&stack, &series).unwrap_err(),
            InsarError::DimensionMismatch(_)
        ));
    }
}

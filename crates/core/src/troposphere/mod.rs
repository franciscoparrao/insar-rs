//! Corrección troposférica **topo-correlacionada** (fase-elevación).
//!
//! El retardo troposférico de gran escala correlaciona con la elevación: el
//! aire húmedo se concentra a baja altura, así que la fase InSAR suele tener
//! una componente lineal (o polinomial) en la altura del terreno, ajena a la
//! deformación. Se ajusta la relación fase-elevación sobre píxeles válidos y se
//! remueve **solo** la componente dependiente de la elevación (Doin et al. 2009;
//! Bekaert et al. 2015).
//!
//! La rampa orbital se trata aparte ([`crate::postprocess::remove_ramp`]); aquí
//! se puede incluir un plano en el AJUSTE (`with_ramp`) para que no sesgue la
//! pendiente fase-elevación, pero solo se resta el término topográfico.

use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, Axis};

use crate::error::{InsarError, Result};
use crate::types::DisplacementSeries;

/// Ajusta la relación fase-elevación sobre los píxeles válidos (finitos en
/// `data` y `dem`, restringidos por `mask` si se da) y resta la componente
/// dependiente de la elevación de todos los píxeles válidos, in situ.
///
/// - `dem`: elevación por píxel en metros, misma grilla que `data`.
/// - `degree`: 1 (lineal en h) o 2 (cuadrático).
/// - `with_ramp`: si `true`, incluye un plano `a·x + b·y` en el ajuste para
///   separar la rampa orbital de la señal topográfica (no se resta el plano).
///
/// Error si las dimensiones no coinciden, `degree ∉ {1,2}`, o hay menos píxeles
/// válidos que coeficientes.
pub fn correct_topo_correlated(
    data: &mut Array2<f32>,
    dem: &Array2<f32>,
    mask: Option<&Array2<bool>>,
    degree: usize,
    with_ramp: bool,
) -> Result<()> {
    let (nr, nc) = data.dim();
    if dem.dim() != (nr, nc) {
        return Err(InsarError::DimensionMismatch(format!(
            "DEM {:?} vs datos {:?}",
            dem.dim(),
            (nr, nc)
        )));
    }
    if mask.is_some_and(|m| m.dim() != (nr, nc)) {
        return Err(InsarError::DimensionMismatch("máscara vs datos".into()));
    }
    if degree != 1 && degree != 2 {
        return Err(InsarError::Metadata(format!("degree {degree} inválido (1 o 2)")));
    }

    // Normaliza la elevación (centrada y escalada) para buen condicionamiento,
    // usando los píxeles que entran al ajuste.
    let mut hs: Vec<f64> = Vec::new();
    for r in 0..nr {
        for c in 0..nc {
            if data[[r, c]].is_finite()
                && dem[[r, c]].is_finite()
                && mask.is_none_or(|m| m[[r, c]])
            {
                hs.push(dem[[r, c]] as f64);
            }
        }
    }
    if hs.is_empty() {
        return Err(InsarError::Inversion("sin píxeles válidos para fase-elevación".into()));
    }
    let h_mean = hs.iter().sum::<f64>() / hs.len() as f64;
    let h_scale = hs
        .iter()
        .map(|h| (h - h_mean).abs())
        .fold(0.0_f64, f64::max)
        .max(1e-6);
    let hnorm = |h: f64| (h - h_mean) / h_scale;

    // Base: [1, h, (h² si deg2), (x, y si with_ramp)]. Las columnas topográficas
    // son la 1 (h) y, si deg2, la 2 (h²).
    let n_topo = degree; // 1 o 2 columnas topográficas
    let n_coef = 1 + n_topo + if with_ramp { 2 } else { 0 };
    let basis = |h: f64, x: f64, y: f64| -> Vec<f64> {
        let mut b = Vec::with_capacity(n_coef);
        b.push(1.0);
        b.push(h);
        if degree == 2 {
            b.push(h * h);
        }
        if with_ramp {
            b.push(x);
            b.push(y);
        }
        b
    };
    let norm_xy = |r: usize, c: usize| (c as f64 / (nc.max(2) - 1) as f64, r as f64 / (nr.max(2) - 1) as f64);

    // Sistema de mínimos cuadrados sobre los píxeles de ajuste.
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut rhs: Vec<f64> = Vec::new();
    for r in 0..nr {
        for c in 0..nc {
            let v = data[[r, c]];
            let h = dem[[r, c]];
            if !v.is_finite() || !h.is_finite() || mask.is_some_and(|m| !m[[r, c]]) {
                continue;
            }
            let (x, y) = norm_xy(r, c);
            rows.push(basis(hnorm(h as f64), x, y));
            rhs.push(v as f64);
        }
    }
    if rows.len() < n_coef {
        return Err(InsarError::Inversion(format!(
            "{} píxeles válidos para {n_coef} coeficientes",
            rows.len()
        )));
    }

    let a = DMatrix::<f64>::from_fn(rows.len(), n_coef, |i, j| rows[i][j]);
    let b = DVector::from_column_slice(&rhs);
    let svd = a.svd(true, true);
    let s_max = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let eps = s_max * (rows.len().max(n_coef) as f64) * f64::EPSILON;
    let coef = svd
        .solve(&b, eps)
        .map_err(|e| InsarError::Inversion(format!("ajuste fase-elevación: {e}")))?;

    // Coeficientes topográficos: k1 en col 1, k2 (si deg2) en col 2.
    let k1 = coef[1];
    let k2 = if degree == 2 { coef[2] } else { 0.0 };

    // Resta solo la componente topográfica de todos los píxeles válidos.
    for r in 0..nr {
        for c in 0..nc {
            let v = data[[r, c]];
            let h = dem[[r, c]];
            if !v.is_finite() || !h.is_finite() {
                continue;
            }
            let hn = hnorm(h as f64);
            let topo = k1 * hn + k2 * hn * hn;
            data[[r, c]] = (v as f64 - topo) as f32;
        }
    }
    Ok(())
}

/// Aplica [`correct_topo_correlated`] a cada época de la serie (el DEM y la
/// máscara se comparten entre épocas).
pub fn correct_topo_series(
    series: &mut DisplacementSeries,
    dem: &Array2<f32>,
    mask: Option<&Array2<bool>>,
    degree: usize,
    with_ramp: bool,
) -> Result<()> {
    for e in 0..series.n_layers() {
        let mut layer = series.data.index_axis(Axis(0), e).to_owned();
        correct_topo_correlated(&mut layer, dem, mask, degree, with_ramp)?;
        series.data.index_axis_mut(Axis(0), e).assign(&layer);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// DEM sintético: rampa de elevación 0..3000 m diagonal.
    fn dem(nr: usize, nc: usize) -> Array2<f32> {
        Array2::from_shape_fn((nr, nc), |(r, c)| {
            (1000.0 * (r as f32 / nr as f32 + c as f32 / nc as f32)) * 3.0
        })
    }

    #[test]
    fn remueve_senal_topo_correlada() {
        let (nr, nc) = (30, 30);
        let h = dem(nr, nc);
        // fase = 0.002 rad/m · elevación + constante
        let mut d = Array2::from_shape_fn((nr, nc), |(r, c)| 0.002 * h[[r, c]] + 5.0);
        correct_topo_correlated(&mut d, &h, None, 1, false).unwrap();
        // Tras corregir, la fase no debe correlacionar con la elevación.
        let hv: Vec<f64> = h.iter().map(|&x| x as f64).collect();
        let dv: Vec<f64> = d.iter().map(|&x| x as f64).collect();
        let hm = hv.iter().sum::<f64>() / hv.len() as f64;
        let dm = dv.iter().sum::<f64>() / dv.len() as f64;
        let cov: f64 = hv.iter().zip(&dv).map(|(h, d)| (h - hm) * (d - dm)).sum();
        let vh: f64 = hv.iter().map(|h| (h - hm).powi(2)).sum();
        let slope = cov / vh;
        assert!(slope.abs() < 1e-6, "pendiente fase-elevación residual {slope}");
    }

    #[test]
    fn conserva_deformacion_no_topo() {
        // DEM que varía con la COLUMNA; deformación en escalón por FILAS
        // (ortogonal al gradiente de elevación) → debe sobrevivir al corregir.
        let (nr, nc) = (40, 40);
        let h = Array2::from_shape_fn((nr, nc), |(_, c)| 3000.0 * c as f32 / nc as f32);
        let mut d = Array2::from_shape_fn((nr, nc), |(r, c)| 0.001 * h[[r, c]]);
        for r in 0..20 {
            for c in 0..40 {
                d[[r, c]] += 3.0; // deformación en la mitad superior (filas)
            }
        }
        correct_topo_correlated(&mut d, &h, None, 1, false).unwrap();
        let top = d.slice(ndarray::s![0..20, ..]).mean().unwrap();
        let bot = d.slice(ndarray::s![20..40, ..]).mean().unwrap();
        assert!((top - bot).abs() > 2.0, "el escalón de deformación se perdió: {}", top - bot);
    }

    #[test]
    fn cuadratico_remueve_topo_no_lineal() {
        let (nr, nc) = (30, 30);
        let h = dem(nr, nc);
        let hmax = h.iter().cloned().fold(0.0f32, f32::max) as f64;
        let mut d = Array2::from_shape_fn((nr, nc), |(r, c)| {
            let hh = h[[r, c]] as f64 / hmax;
            (0.5 * hh + 2.0 * hh * hh) as f32
        });
        correct_topo_correlated(&mut d, &h, None, 2, false).unwrap();
        let rng = d.iter().cloned().fold(f32::MIN, f32::max)
            - d.iter().cloned().fold(f32::MAX, f32::min);
        assert!(rng < 1e-3, "residuo cuadrático {rng}");
    }

    #[test]
    fn dims_invalidas_es_error() {
        let mut d = Array2::<f32>::zeros((10, 10));
        let h = Array2::<f32>::zeros((10, 9));
        assert!(matches!(
            correct_topo_correlated(&mut d, &h, None, 1, false).unwrap_err(),
            InsarError::DimensionMismatch(_)
        ));
        let h2 = Array2::<f32>::zeros((10, 10));
        assert!(correct_topo_correlated(&mut d, &h2, None, 3, false).is_err());
    }
}

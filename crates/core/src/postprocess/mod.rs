//! Post-procesamiento de productos: eliminación de rampas (deramp).
//!
//! Las velocidades y series InSAR suelen contener una tendencia de larga
//! longitud de onda (rampa) por errores orbitales y atmósfera de gran escala,
//! ajena a la deformación local. `remove_ramp` ajusta un plano (o superficie
//! cuadrática) sobre los píxeles válidos y lo resta, aislando la señal local.

use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, Axis};

use crate::error::{InsarError, Result};
use crate::types::DisplacementSeries;

/// Tipo de superficie a ajustar y restar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RampKind {
    /// Plano: `a·x + b·y + c`.
    Linear,
    /// Cuadrática: `a·x + b·y + c + d·x² + e·y² + f·x·y`.
    Quadratic,
}

impl RampKind {
    fn n_coef(self) -> usize {
        match self {
            RampKind::Linear => 3,
            RampKind::Quadratic => 6,
        }
    }
    /// Términos de la base en coordenadas normalizadas (x, y) ∈ [0, 1].
    fn basis(self, x: f64, y: f64) -> [f64; 6] {
        match self {
            RampKind::Linear => [x, y, 1.0, 0.0, 0.0, 0.0],
            RampKind::Quadratic => [x, y, 1.0, x * x, y * y, x * y],
        }
    }
}

/// Ajusta una rampa sobre los píxeles finitos (restringidos por `mask` si se
/// indica) y la resta de TODOS los píxeles finitos de `data`, in situ.
///
/// Coordenadas normalizadas a [0, 1] para buen condicionamiento. `mask`, si se
/// da, selecciona qué píxeles entran al AJUSTE (p. ej. solo coherentes); la
/// resta se aplica a todo píxel finito. Error si hay menos píxeles válidos que
/// coeficientes o si las dimensiones de `mask` no coinciden.
pub fn remove_ramp(
    data: &mut Array2<f32>,
    kind: RampKind,
    mask: Option<&Array2<bool>>,
) -> Result<()> {
    let (nr, nc) = data.dim();
    if let Some(m) = mask
        && m.dim() != (nr, nc)
    {
        return Err(InsarError::DimensionMismatch(format!(
            "máscara {:?} vs datos {:?}",
            m.dim(),
            (nr, nc)
        )));
    }
    if nr < 2 || nc < 2 {
        return Err(InsarError::DimensionMismatch(
            "se requiere al menos 2×2 para ajustar una rampa".into(),
        ));
    }

    let n_coef = kind.n_coef();
    let norm = |r: usize, c: usize| (c as f64 / (nc - 1) as f64, r as f64 / (nr - 1) as f64);

    // Acumula filas del sistema de mínimos cuadrados sobre píxeles de ajuste.
    let mut rows: Vec<[f64; 6]> = Vec::new();
    let mut rhs: Vec<f64> = Vec::new();
    for r in 0..nr {
        for c in 0..nc {
            let v = data[[r, c]];
            if !v.is_finite() {
                continue;
            }
            if mask.is_some_and(|m| !m[[r, c]]) {
                continue;
            }
            let (x, y) = norm(r, c);
            rows.push(kind.basis(x, y));
            rhs.push(v as f64);
        }
    }
    if rows.len() < n_coef {
        return Err(InsarError::Inversion(format!(
            "{} píxeles válidos para ajustar una rampa de {n_coef} coeficientes",
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
        .map_err(|e| InsarError::Inversion(format!("ajuste de rampa: {e}")))?;

    // Resta la rampa de todos los píxeles finitos.
    for r in 0..nr {
        for c in 0..nc {
            let v = data[[r, c]];
            if !v.is_finite() {
                continue;
            }
            let (x, y) = norm(r, c);
            let basis = kind.basis(x, y);
            let ramp: f64 = (0..n_coef).map(|j| coef[j] * basis[j]).sum();
            data[[r, c]] = (v as f64 - ramp) as f32;
        }
    }
    Ok(())
}

/// Aplica [`remove_ramp`] a cada época de la serie (deramp por época), que
/// remueve la atmósfera de gran escala de cada adquisición antes de estimar la
/// velocidad. La misma `mask` espacial se usa en todas las épocas.
pub fn deramp_series(
    series: &mut DisplacementSeries,
    kind: RampKind,
    mask: Option<&Array2<bool>>,
) -> Result<()> {
    let n_epochs = series.n_layers();
    for e in 0..n_epochs {
        let mut layer = series.data.index_axis_mut(Axis(0), e).to_owned();
        remove_ramp(&mut layer, kind, mask)?;
        series.data.index_axis_mut(Axis(0), e).assign(&layer);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn remueve_plano_exacto() {
        // Campo = plano puro 3·x − 2·y + 1; tras deramp queda ~0.
        let (nr, nc) = (20, 25);
        let mut d = Array2::<f32>::zeros((nr, nc));
        for r in 0..nr {
            for c in 0..nc {
                let x = c as f64 / (nc - 1) as f64;
                let y = r as f64 / (nr - 1) as f64;
                d[[r, c]] = (3.0 * x - 2.0 * y + 1.0) as f32;
            }
        }
        remove_ramp(&mut d, RampKind::Linear, None).unwrap();
        for &v in d.iter() {
            assert!(v.abs() < 1e-4, "residuo {v}");
        }
    }

    #[test]
    fn conserva_senal_local_sobre_rampa() {
        // Plano + un pico local: el pico debe sobrevivir al deramp.
        let (nr, nc) = (40, 40);
        let mut d = Array2::<f32>::zeros((nr, nc));
        for r in 0..nr {
            for c in 0..nc {
                let x = c as f64 / (nc - 1) as f64;
                let y = r as f64 / (nr - 1) as f64;
                d[[r, c]] = (5.0 * x + 4.0 * y) as f32;
            }
        }
        d[[20, 20]] += 10.0; // pico local
        remove_ramp(&mut d, RampKind::Linear, None).unwrap();
        // El pico domina el residuo (la rampa se fue, el pico no).
        let (mut mr, mut mc, mut mv) = (0, 0, 0.0f32);
        for r in 0..nr {
            for c in 0..nc {
                if d[[r, c]].abs() > mv {
                    mv = d[[r, c]].abs();
                    mr = r;
                    mc = c;
                }
            }
        }
        assert_eq!((mr, mc), (20, 20));
        assert!(mv > 5.0, "pico atenuado: {mv}");
    }

    #[test]
    fn mascara_excluye_del_ajuste() {
        // Plano + región anómala; enmascararla evita que sesgue la rampa.
        let (nr, nc) = (30, 30);
        let mut d = Array2::<f32>::zeros((nr, nc));
        let mut m = Array2::from_elem((nr, nc), true);
        for r in 0..nr {
            for c in 0..nc {
                let x = c as f64 / (nc - 1) as f64;
                let y = r as f64 / (nr - 1) as f64;
                d[[r, c]] = (2.0 * x - 3.0 * y + 0.5) as f32;
                if r < 6 && c < 6 {
                    d[[r, c]] += 50.0; // anomalía fuerte
                    m[[r, c]] = false; // excluida del ajuste
                }
            }
        }
        remove_ramp(&mut d, RampKind::Linear, Some(&m)).unwrap();
        // Fuera de la anomalía, el plano se removió bien.
        let mut maxabs = 0.0f32;
        for r in 10..nr {
            for c in 10..nc {
                maxabs = maxabs.max(d[[r, c]].abs());
            }
        }
        assert!(maxabs < 1e-3, "residuo fuera de anomalía {maxabs}");
    }

    #[test]
    fn pocos_pixeles_es_error() {
        let mut d = Array2::<f32>::from_elem((1, 1), 1.0);
        assert!(remove_ramp(&mut d, RampKind::Linear, None).is_err());
    }
}

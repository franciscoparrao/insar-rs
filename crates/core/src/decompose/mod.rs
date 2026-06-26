//! Descomposición de desplazamiento **LOS multi-geometría** en componentes
//! **vertical (Up)** y **horizontal Este-Oeste (East)**.
//!
//! Un solo interferograma mide la proyección del desplazamiento 3D sobre la
//! línea de vista (LOS): `d_los = u · ê`, donde `ê` es el vector unitario que
//! apunta del **suelo al satélite** (componentes ENU). Con dos geometrías de
//! mirada distintas — típicamente una órbita **ascendente** y una
//! **descendente** — se puede resolver el desplazamiento por píxel.
//!
//! Sentinel-1 (órbita casi polar) es prácticamente **ciego al Norte-Sur**: la
//! componente N del vector de vista es pequeña (~0.1). Por eso aquí se resuelve
//! para **(Up, East)** despreciando el Norte, que es la práctica estándar
//! (Wright et al. 2004; Fialko et al. 2001). Para una falla de rumbo casi E-O
//! con deslizamiento dextral, el movimiento horizontal es además casi puro E-O,
//! de modo que la aproximación es especialmente buena.
//!
//! Convención: `d_los > 0` = desplazamiento **hacia** el satélite (acortamiento
//! de rango), de modo que un alzamiento del terreno (`Up > 0`) produce
//! `d_los > 0`. El llamador debe entregar el LOS con ese signo (la conversión
//! `d = -λ/(4π)·φ` de [`crate::inversion::phase_to_displacement`] lo cumple bajo
//! la convención de signo del motor).

use ndarray::Array2;

use crate::error::{InsarError, Result};

/// Vector unitario de la línea de vista (**suelo → satélite**), en componentes
/// Este-Norte-Up (ENU).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LosVector {
    pub east: f64,
    pub north: f64,
    pub up: f64,
}

impl LosVector {
    /// Construye el vector de vista desde el **ángulo de incidencia** (grados,
    /// medido desde la vertical) y el **rumbo de la órbita** (`heading`, grados,
    /// azimut de la dirección de vuelo medido en sentido horario desde el Norte).
    ///
    /// Asume SAR **right-looking** (Sentinel-1). Para Sentinel-1 los rumbos
    /// típicos son ~`-12°` (≈348°) en ascendente y ~`-168°` (≈192°) en
    /// descendente, lo que da componentes Este de signo opuesto (la base de la
    /// descomposición).
    pub fn from_incidence_heading(incidence_deg: f64, heading_deg: f64) -> Self {
        let th = incidence_deg.to_radians();
        let a = heading_deg.to_radians();
        // Suelo→satélite: horizontal de magnitud sin(θ) en dirección
        // -(right-of-flight) = (-cosα, sinα); vertical cos(θ) hacia arriba.
        LosVector {
            east: -th.sin() * a.cos(),
            north: th.sin() * a.sin(),
            up: th.cos(),
        }
    }
}

/// Resultado de la descomposición: mapas de desplazamiento vertical y E-O,
/// misma grilla que la entrada. NoData = `NaN`.
#[derive(Clone, Debug)]
pub struct Decomposed {
    /// Desplazamiento vertical (Up positivo = alzamiento), mismas unidades que el LOS.
    pub up: Array2<f32>,
    /// Desplazamiento horizontal Este (positivo hacia el Este).
    pub east: Array2<f32>,
}

/// Descompone dos o más mapas de desplazamiento LOS (una por geometría) en
/// **(Up, East)** por píxel, despreciando la componente Norte.
///
/// - `geoms[i]`: vector de vista de la geometría `i`.
/// - `los[i]`: mapa de desplazamiento LOS de la geometría `i` (misma grilla).
///
/// Con 2 geometrías el sistema es exacto (2×2); con más, se resuelve por mínimos
/// cuadrados. La matriz de diseño es constante en toda la escena (un único
/// vector de vista por geometría), de modo que su pseudoinversa se factoriza
/// **una sola vez** y el costo por píxel es una multiplicación 2×N.
///
/// Un píxel queda `NaN` si falta dato (`NaN`) en cualquiera de las geometrías.
///
/// Error si: menos de 2 geometrías, `geoms.len() != los.len()`, grillas
/// inconsistentes, o las geometrías son casi colineales (no separan Up de East,
/// p.ej. dos ascendentes) — el determinante normal cae bajo tolerancia.
pub fn decompose(geoms: &[LosVector], los: &[&Array2<f32>]) -> Result<Decomposed> {
    if geoms.len() < 2 {
        return Err(InsarError::Metadata(format!(
            "se requieren ≥2 geometrías, se dieron {}",
            geoms.len()
        )));
    }
    if geoms.len() != los.len() {
        return Err(InsarError::DimensionMismatch(format!(
            "{} geometrías vs {} mapas LOS",
            geoms.len(),
            los.len()
        )));
    }
    let (nr, nc) = los[0].dim();
    for (i, m) in los.iter().enumerate() {
        if m.dim() != (nr, nc) {
            return Err(InsarError::DimensionMismatch(format!(
                "mapa LOS {i} {:?} vs {:?}",
                m.dim(),
                (nr, nc)
            )));
        }
    }

    // Diseño A (N×2) con columnas [up, east]; normales M = AᵀA (2×2) constantes.
    let n = geoms.len();
    let (mut m00, mut m01, mut m11) = (0.0f64, 0.0f64, 0.0f64);
    for g in geoms {
        m00 += g.up * g.up;
        m01 += g.up * g.east;
        m11 += g.east * g.east;
    }
    let det = m00 * m11 - m01 * m01;
    // Tolerancia relativa a la escala de los coeficientes (todos ≤1).
    if det.abs() < 1e-9 * (m00 * m11).max(1e-30) {
        return Err(InsarError::Inversion(
            "geometrías casi colineales: no separan Up de East (¿asc+asc o desc+desc?)".into(),
        ));
    }
    // Inversa de M (2×2).
    let (i00, i01, i11) = (m11 / det, -m01 / det, m00 / det);

    let ups: Vec<f64> = geoms.iter().map(|g| g.up).collect();
    let easts: Vec<f64> = geoms.iter().map(|g| g.east).collect();

    let mut up = Array2::<f32>::from_elem((nr, nc), f32::NAN);
    let mut east = Array2::<f32>::from_elem((nr, nc), f32::NAN);
    for r in 0..nr {
        for c in 0..nc {
            // Lado derecho b = Aᵀd; NaN si falta cualquier geometría.
            let (mut b0, mut b1) = (0.0f64, 0.0f64);
            let mut ok = true;
            for i in 0..n {
                let d = los[i][[r, c]];
                if !d.is_finite() {
                    ok = false;
                    break;
                }
                b0 += ups[i] * d as f64;
                b1 += easts[i] * d as f64;
            }
            if !ok {
                continue;
            }
            up[[r, c]] = (i00 * b0 + i01 * b1) as f32;
            east[[r, c]] = (i01 * b0 + i11 * b1) as f32;
        }
    }
    Ok(Decomposed { up, east })
}

/// Conveniencia para el caso típico de dos geometrías (ascendente + descendente).
pub fn decompose_asc_desc(
    los_asc: &Array2<f32>,
    geom_asc: LosVector,
    los_desc: &Array2<f32>,
    geom_desc: LosVector,
) -> Result<Decomposed> {
    decompose(&[geom_asc, geom_desc], &[los_asc, los_desc])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // Geometrías Sentinel-1 representativas (incidencia 39°).
    fn asc() -> LosVector {
        LosVector::from_incidence_heading(39.0, -12.0)
    }
    fn desc() -> LosVector {
        LosVector::from_incidence_heading(39.0, -168.0)
    }

    #[test]
    fn vectores_de_vista_tienen_este_opuesto() {
        let (a, d) = (asc(), desc());
        // Up positivo y casi igual; Este de signo opuesto (la base de todo).
        assert!(a.up > 0.0 && d.up > 0.0);
        assert!((a.up - d.up).abs() < 1e-9);
        assert!(a.east * d.east < 0.0, "este asc {} desc {}", a.east, d.east);
        // Norte pequeño (Sentinel ~ciego al N-S).
        assert!(a.north.abs() < 0.2 && d.north.abs() < 0.2);
        // Vector unitario.
        assert!((a.east * a.east + a.north * a.north + a.up * a.up - 1.0).abs() < 1e-9);
    }

    #[test]
    fn recupera_up_y_east_sinteticos() {
        // Campo conocido: alzamiento y desplazamiento Este por píxel.
        let (nr, nc) = (12, 15);
        let true_up = Array2::from_shape_fn((nr, nc), |(r, c)| 0.10 + 0.01 * r as f32 - 0.005 * c as f32);
        let true_east = Array2::from_shape_fn((nr, nc), |(_r, c)| -0.04 + 0.003 * c as f32);
        let (ga, gd) = (asc(), desc());
        // Proyección directa a LOS de cada geometría (Norte = 0).
        let proj = |g: LosVector, up: &Array2<f32>, e: &Array2<f32>| {
            Array2::from_shape_fn((nr, nc), |(r, c)| {
                (g.up * up[[r, c]] as f64 + g.east * e[[r, c]] as f64) as f32
            })
        };
        let la = proj(ga, &true_up, &true_east);
        let ld = proj(gd, &true_up, &true_east);

        let out = decompose_asc_desc(&la, ga, &ld, gd).unwrap();
        for r in 0..nr {
            for c in 0..nc {
                assert!((out.up[[r, c]] - true_up[[r, c]]).abs() < 1e-4);
                assert!((out.east[[r, c]] - true_east[[r, c]]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn nan_se_propaga_por_pixel() {
        let (nr, nc) = (4, 4);
        let mut la = Array2::<f32>::zeros((nr, nc));
        let ld = Array2::<f32>::zeros((nr, nc));
        la[[1, 1]] = f32::NAN;
        let out = decompose_asc_desc(&la, asc(), &ld, desc()).unwrap();
        assert!(out.up[[1, 1]].is_nan() && out.east[[1, 1]].is_nan());
        assert!(out.up[[0, 0]].is_finite() && out.east[[0, 0]].is_finite());
    }

    #[test]
    fn geometrias_colineales_es_error() {
        // Dos ascendentes ≈ misma geometría → no separan Up de East.
        let z = Array2::<f32>::zeros((3, 3));
        let g = asc();
        assert!(matches!(
            decompose(&[g, g], &[&z, &z]).unwrap_err(),
            InsarError::Inversion(_)
        ));
    }

    #[test]
    fn dims_o_conteo_invalidos_es_error() {
        let z = Array2::<f32>::zeros((3, 3));
        // Menos de 2 geometrías.
        assert!(decompose(&[asc()], &[&z]).is_err());
        // Conteo geom vs mapas.
        assert!(decompose(&[asc(), desc()], &[&z]).is_err());
        // Grillas inconsistentes.
        let z2 = Array2::<f32>::zeros((3, 4));
        assert!(matches!(
            decompose_asc_desc(&z, asc(), &z2, desc()).unwrap_err(),
            InsarError::DimensionMismatch(_)
        ));
    }

    #[test]
    fn tres_geometrias_minimos_cuadrados() {
        // Con 3 geometrías (incl. una segunda asc), el LSQ debe recuperar igual.
        let (nr, nc) = (6, 6);
        let true_up = Array2::from_elem((nr, nc), 0.08f32);
        let true_east = Array2::from_elem((nr, nc), -0.03f32);
        let g = [asc(), desc(), LosVector::from_incidence_heading(43.0, -12.0)];
        let proj = |gi: LosVector| {
            Array2::from_shape_fn((nr, nc), |(r, c)| {
                (gi.up * true_up[[r, c]] as f64 + gi.east * true_east[[r, c]] as f64) as f32
            })
        };
        let maps = [proj(g[0]), proj(g[1]), proj(g[2])];
        let refs: Vec<&Array2<f32>> = maps.iter().collect();
        let out = decompose(&g, &refs).unwrap();
        assert!((out.up[[0, 0]] - 0.08).abs() < 1e-4);
        assert!((out.east[[0, 0]] - (-0.03)).abs() < 1e-4);
    }
}

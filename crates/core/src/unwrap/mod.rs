//! Desenrollado de fase 2D mínimo (alcance MVP): flood-fill guiado por un
//! mapa de calidad (coherencia si está disponible), integrando saltos ±2π
//! entre vecinos. No es un reemplazo de SNAPHU — ver PLAN.md.
//!
//! ## Algoritmo
//!
//! Quality-guided flood-fill clásico:
//! 1. Semilla = píxel de máxima calidad (o el centro de la imagen si no hay
//!    mapa de calidad, en cuyo caso la calidad es uniforme).
//! 2. Se crece con una cola de prioridad (max-heap sobre la calidad del
//!    candidato): siempre se visita primero el vecino pendiente de mayor
//!    calidad. Vecindad de 4 (arriba/abajo/izquierda/derecha).
//! 3. Al visitar un vecino `v` desde el píxel `u` ya resuelto:
//!    `unw[v] = unw[u] + wrap_diff(wrapped[v] − wrapped[u])`, donde
//!    `wrap_diff` lleva la diferencia al rango (−π, π].
//!
//! ## Convenciones y limitaciones
//!
//! - NoData = NaN: un píxel con fase NaN (o calidad NaN) no se visita y queda
//!   NaN en la salida.
//! - Si los NaN parten la imagen en islas desconectadas, cada isla se
//!   desenrolla por separado, re-sembrando en el píxel de mayor calidad aún
//!   no visitado. **Cada isla queda referida a su propia semilla**: el valor
//!   en la semilla es exactamente la fase envuelta en ese punto (sin
//!   normalizar), por lo que existe un offset global 2πk independiente por
//!   isla — ambigüedad inherente al problema de desenrollado.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use ndarray::{Array2, Array3, Axis};
use rayon::prelude::*;

use crate::error::{InsarError, Result};
use crate::types::{IfgStack, UnwrappedStack};

pub mod snaphu;

/// Fase envuelta (arg(z)) de la capa `k` de un `IfgStack`; NaN si `re`/`im`
/// no son finitos. Compartida entre [`unwrap_stack_min_quality`] y el
/// backend [`snaphu::unwrap_stack_snaphu`].
fn wrapped_phase_layer(stack: &IfgStack, k: usize) -> Array2<f32> {
    stack.data.index_axis(Axis(0), k).map(|z| {
        if z.re.is_finite() && z.im.is_finite() { z.im.atan2(z.re) } else { f32::NAN }
    })
}

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// Lleva una diferencia de fase al rango (−π, π].
#[inline]
fn wrap_diff(d: f32) -> f32 {
    let d = f64::from(d);
    (d - TWO_PI * ((d - std::f64::consts::PI) / TWO_PI).ceil()) as f32
}

/// Candidato en la cola de prioridad: vecino pendiente con su calidad y el
/// valor desenrollado propuesto (calculado desde el predecesor ya resuelto).
struct Candidate {
    quality: f32,
    row: usize,
    col: usize,
    value: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.quality.total_cmp(&other.quality) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    /// Max-heap por calidad; los empates dan igual (orden indistinto).
    fn cmp(&self, other: &Self) -> Ordering {
        self.quality.total_cmp(&other.quality)
    }
}

/// Desenrolla un interferograma 2D. `wrapped` en radianes (-π, π].
/// `quality`: mapa opcional (mayor = mejor); si es None se usa calidad
/// uniforme y semilla en el centro de la imagen. NaN se propaga.
pub fn unwrap_2d(wrapped: &Array2<f32>, quality: Option<&Array2<f32>>) -> Result<Array2<f32>> {
    unwrap_2d_min_quality(wrapped, quality, None)
}

/// Como [`unwrap_2d`], pero con umbral de calidad mínima: los píxeles con
/// `quality < min_quality` se tratan como NoData (no se desenrollan y quedan
/// NaN), evitando que píxeles de coherencia ~0 se integren tarde al árbol y
/// propaguen errores a sus vecinos aguas abajo. Las islas que el umbral
/// genere se manejan igual que las de NaN (re-siembra por isla).
///
/// Error [`InsarError::Metadata`] si se pide `min_quality` sin mapa de
/// calidad (no hay nada que umbralizar).
pub fn unwrap_2d_min_quality(
    wrapped: &Array2<f32>,
    quality: Option<&Array2<f32>>,
    min_quality: Option<f32>,
) -> Result<Array2<f32>> {
    if min_quality.is_some() && quality.is_none() {
        return Err(InsarError::Metadata(
            "min_quality requiere un mapa de calidad (coherencia)".into(),
        ));
    }
    let (rows, cols) = wrapped.dim();
    if rows == 0 || cols == 0 {
        return Err(InsarError::DimensionMismatch(format!(
            "imagen vacía: {rows}×{cols}"
        )));
    }
    if let Some(q) = quality
        && q.dim() != wrapped.dim()
    {
        return Err(InsarError::DimensionMismatch(format!(
            "quality {:?} vs wrapped {:?}",
            q.dim(),
            wrapped.dim()
        )));
    }

    // Un píxel es válido (visitable) si su fase es finita, su calidad no es
    // NaN, y su calidad alcanza el umbral mínimo (si se configuró).
    let is_valid = |r: usize, c: usize| -> bool {
        wrapped[[r, c]].is_finite()
            && quality.is_none_or(|q| {
                let v = q[[r, c]];
                !v.is_nan() && min_quality.is_none_or(|thr| v >= thr)
            })
    };
    // Calidad efectiva: uniforme (1.0) si no hay mapa.
    let qual = |r: usize, c: usize| -> f32 {
        quality.map_or(1.0, |q| q[[r, c]])
    };

    let mut unw = Array2::from_elem((rows, cols), f32::NAN);
    let mut visited = Array2::from_elem((rows, cols), false);

    let n_valid = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (r, c)))
        .filter(|&(r, c)| is_valid(r, c))
        .count();
    if n_valid == 0 {
        // Imagen completamente NaN: salida completamente NaN.
        return Ok(unw);
    }

    let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
    let mut n_visited = 0usize;
    let mut first_island = true;

    // Encola los vecinos (4-conectividad) válidos y no visitados de (r, c).
    let push_neighbors = |heap: &mut BinaryHeap<Candidate>,
                          unw: &Array2<f32>,
                          visited: &Array2<bool>,
                          r: usize,
                          c: usize| {
        let base = unw[[r, c]];
        let w_u = wrapped[[r, c]];
        let neighbors = [
            (r.wrapping_sub(1), c),
            (r + 1, c),
            (r, c.wrapping_sub(1)),
            (r, c + 1),
        ];
        for (nr, nc) in neighbors {
            if nr < rows && nc < cols && !visited[[nr, nc]] && is_valid(nr, nc) {
                heap.push(Candidate {
                    quality: qual(nr, nc),
                    row: nr,
                    col: nc,
                    value: base + wrap_diff(wrapped[[nr, nc]] - w_u),
                });
            }
        }
    };

    while n_visited < n_valid {
        // Sembrar (primera isla) o re-sembrar (islas separadas por NaN).
        let seed = find_seed(
            rows,
            cols,
            quality.is_none() && first_island,
            &visited,
            &is_valid,
            &qual,
        );
        first_island = false;
        let Some((sr, sc)) = seed else {
            // No debería ocurrir (n_visited < n_valid garantiza candidatos),
            // pero salimos limpiamente en vez de iterar para siempre.
            break;
        };

        // La semilla queda con su fase envuelta tal cual (referencia de la isla).
        unw[[sr, sc]] = wrapped[[sr, sc]];
        visited[[sr, sc]] = true;
        n_visited += 1;
        push_neighbors(&mut heap, &unw, &visited, sr, sc);

        // Crecimiento guiado por calidad dentro de la isla.
        while let Some(cand) = heap.pop() {
            if visited[[cand.row, cand.col]] {
                continue; // entrada obsoleta (encolada más de una vez)
            }
            unw[[cand.row, cand.col]] = cand.value;
            visited[[cand.row, cand.col]] = true;
            n_visited += 1;
            push_neighbors(&mut heap, &unw, &visited, cand.row, cand.col);
        }
    }

    Ok(unw)
}

/// Elige la semilla: centro de la imagen para la primera isla sin mapa de
/// calidad (si es válido), o el píxel válido no visitado de máxima calidad.
fn find_seed(
    rows: usize,
    cols: usize,
    prefer_center: bool,
    visited: &Array2<bool>,
    is_valid: &dyn Fn(usize, usize) -> bool,
    qual: &dyn Fn(usize, usize) -> f32,
) -> Option<(usize, usize)> {
    if prefer_center {
        let (cr, cc) = (rows / 2, cols / 2);
        if is_valid(cr, cc) && !visited[[cr, cc]] {
            return Some((cr, cc));
        }
    }
    let mut best: Option<((usize, usize), f32)> = None;
    for r in 0..rows {
        for c in 0..cols {
            if visited[[r, c]] || !is_valid(r, c) {
                continue;
            }
            let q = qual(r, c);
            match best {
                Some((_, bq)) if q.total_cmp(&bq) != Ordering::Greater => {}
                _ => best = Some(((r, c), q)),
            }
        }
    }
    best.map(|(pos, _)| pos)
}

/// Desenrolla cada interferograma del stack (paralelizable por capa).
/// `coherence`: stack opcional con el mismo layout que `stack.data`.
pub fn unwrap_stack(stack: &IfgStack, coherence: Option<&Array3<f32>>) -> Result<UnwrappedStack> {
    unwrap_stack_min_quality(stack, coherence, None)
}

/// Como [`unwrap_stack`], con umbral de calidad mínima por píxel (ver
/// [`unwrap_2d_min_quality`]). Requiere `coherence` si `min_quality` es Some.
pub fn unwrap_stack_min_quality(
    stack: &IfgStack,
    coherence: Option<&Array3<f32>>,
    min_quality: Option<f32>,
) -> Result<UnwrappedStack> {
    if min_quality.is_some() && coherence.is_none() {
        return Err(InsarError::Metadata(
            "min_quality requiere el stack de coherencia".into(),
        ));
    }
    if let Some(coh) = coherence
        && coh.dim() != stack.data.dim()
    {
        return Err(InsarError::DimensionMismatch(format!(
            "coherence {:?} vs stack {:?}",
            coh.dim(),
            stack.data.dim()
        )));
    }

    let (n_layers, rows, cols) = stack.data.dim();

    // Una capa por tarea rayon; cada capa corre unwrap_2d secuencial.
    let layers: Vec<Array2<f32>> = (0..n_layers)
        .into_par_iter()
        .map(|k| {
            let wrapped = wrapped_phase_layer(stack, k);
            let qual = coherence.map(|coh| coh.index_axis(Axis(0), k).to_owned());
            unwrap_2d_min_quality(&wrapped, qual.as_ref(), min_quality)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut data = Array3::from_elem((n_layers, rows, cols), f32::NAN);
    for (k, layer) in layers.into_iter().enumerate() {
        data.index_axis_mut(Axis(0), k).assign(&layer);
    }

    let out = UnwrappedStack {
        data,
        epochs: stack.epochs.clone(),
        pairs: stack.pairs.clone(),
        meta: stack.meta.clone(),
    };
    out.validate()?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Epoch, IfgPair, StackMeta, SENTINEL1_WAVELENGTH_M};
    use ndarray::Array2;
    use num_complex::Complex32;
    use std::f32::consts::PI;
    use surtgis_core::GeoTransform;

    /// Rampa de fase continua: `phi = a·fila + b·col`.
    fn ramp(rows: usize, cols: usize, a: f32, b: f32) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |(r, c)| a * r as f32 + b * c as f32)
    }

    /// Envuelve fase a (−π, π] con atan2(sin, cos).
    fn wrap(phase: &Array2<f32>) -> Array2<f32> {
        phase.map(|&p| p.sin().atan2(p.cos()))
    }

    fn meta() -> StackMeta {
        StackMeta {
            transform: GeoTransform::new(0.0, 0.0, 30.0, -30.0),
            crs: None,
            wavelength_m: SENTINEL1_WAVELENGTH_M,
            incidence_deg: 39.0,
            heading_deg: None,
        }
    }

    fn epochs3() -> Vec<Epoch> {
        ["2023-01-01", "2023-01-13", "2023-01-25"]
            .iter()
            .map(|s| Epoch(s.parse().unwrap()))
            .collect()
    }

    /// Compara desenrollado contra la rampa verdadera, ambos referidos a un
    /// píxel de referencia (anula el offset global 2πk). `select` restringe
    /// la comparación (ej. a una sola isla, que tiene su propio offset).
    fn assert_matches_ramp_where(
        unw: &Array2<f32>,
        truth: &Array2<f32>,
        reference: (usize, usize),
        tol: f32,
        select: impl Fn(usize, usize) -> bool,
    ) {
        let (rr, rc) = reference;
        let u0 = unw[[rr, rc]];
        let t0 = truth[[rr, rc]];
        assert!(u0.is_finite(), "referencia {reference:?} quedó NaN");
        for ((r, c), &u) in unw.indexed_iter() {
            if !truth[[r, c]].is_nan() && select(r, c) {
                let err = ((u - u0) - (truth[[r, c]] - t0)).abs();
                assert!(
                    err < tol,
                    "píxel ({r},{c}): err={err} (unw={u}, truth={})",
                    truth[[r, c]]
                );
            }
        }
    }

    fn assert_matches_ramp(
        unw: &Array2<f32>,
        truth: &Array2<f32>,
        reference: (usize, usize),
        tol: f32,
    ) {
        assert_matches_ramp_where(unw, truth, reference, tol, |_, _| true);
    }

    #[test]
    fn wrap_diff_rango() {
        // Valores lejos del borde ±π (el borde exacto es ambiguo en f32:
        // f32::consts::PI ≠ π, y ±π difieren en exactamente 2π).
        assert!((wrap_diff(0.0)).abs() < 1e-6);
        assert!((wrap_diff(3.0) - 3.0).abs() < 1e-6);
        assert!((wrap_diff(-3.0) + 3.0).abs() < 1e-6);
        assert!((wrap_diff(2.0 * PI + 0.3) - 0.3).abs() < 1e-5);
        assert!((wrap_diff(-2.0 * PI - 0.3) + 0.3).abs() < 1e-5);
        assert!((wrap_diff(5.0 * PI - 0.7) - (PI - 0.7)).abs() < 1e-5);
        // Propiedades: resultado en (−π, π] (con ε de f32) y congruente
        // con la entrada módulo 2π.
        for i in -100..=100 {
            let x = 0.137 * i as f32;
            let w = wrap_diff(x);
            assert!(w > -PI - 1e-5 && w <= PI + 1e-5, "fuera de rango: {x} -> {w}");
            let k = ((x - w) / (2.0 * PI)).round();
            assert!(((x - w) - k * 2.0 * PI).abs() < 1e-4, "no congruente: {x} -> {w}");
        }
    }

    #[test]
    fn rampa_lineal_se_recupera() {
        // Rampa que cruza varios ciclos 2π: rango ≈ 0.45·23 + 0.35·19 ≈ 17 rad.
        let truth = ramp(24, 20, 0.45, 0.35);
        let wrapped = wrap(&truth);
        let unw = unwrap_2d(&wrapped, None).unwrap();
        // Semilla = centro (12, 10); comparar diferencias respecto a ella.
        assert_matches_ramp(&unw, &truth, (12, 10), 1e-4);
        // La semilla conserva su fase envuelta tal cual.
        assert_eq!(unw[[12, 10]], wrapped[[12, 10]]);
    }

    #[test]
    fn franja_nan_genera_dos_islas_independientes() {
        let rows = 16;
        let cols = 21;
        let mut truth = ramp(rows, cols, 0.5, 0.4);
        // Franja vertical NaN (cols 9..12) que parte la imagen en dos islas.
        for r in 0..rows {
            for c in 9..12 {
                truth[[r, c]] = f32::NAN;
            }
        }
        let wrapped = wrap(&truth);
        let unw = unwrap_2d(&wrapped, None).unwrap();

        // NaN preservado en la franja.
        for r in 0..rows {
            for c in 9..12 {
                assert!(unw[[r, c]].is_nan(), "({r},{c}) debía ser NaN");
            }
        }
        // Cada isla se verifica POR SEPARADO con su propia referencia:
        // los offsets 2πk entre islas son independientes (ambigüedad inherente).
        assert_matches_ramp_where(&unw, &truth, (8, 4), 1e-4, |_, c| c < 9); // izquierda
        assert_matches_ramp_where(&unw, &truth, (8, 16), 1e-4, |_, c| c >= 12); // derecha

        // Ningún píxel válido quedó sin visitar.
        for ((r, c), &u) in unw.indexed_iter() {
            if !truth[[r, c]].is_nan() {
                assert!(u.is_finite(), "({r},{c}) válido quedó NaN");
            }
        }
    }

    #[test]
    fn quality_dirige_la_semilla_y_el_orden() {
        // Rampa con varios ciclos; calidad máxima en (2, 3) → semilla ahí.
        let truth = ramp(12, 12, 0.45, 0.35);
        let wrapped = wrap(&truth);
        let mut quality = Array2::from_elem((12, 12), 0.3_f32);
        quality[[2, 3]] = 5.0;
        let unw = unwrap_2d(&wrapped, Some(&quality)).unwrap();
        // La semilla es exactamente el píxel de máxima calidad: su valor de
        // salida es la fase envuelta sin alterar. Con la semilla en el centro
        // (uniforme) este píxel quedaría desplazado en −2πk ≠ 0, así que el
        // test detecta si quality fue ignorada.
        assert_eq!(unw[[2, 3]], wrapped[[2, 3]]);
        assert_matches_ramp(&unw, &truth, (2, 3), 1e-4);
    }

    #[test]
    fn quality_nan_bloquea_pixel() {
        let truth = ramp(8, 8, 0.3, 0.2);
        let wrapped = wrap(&truth);
        let mut quality = Array2::from_elem((8, 8), 1.0_f32);
        quality[[4, 4]] = f32::NAN;
        let unw = unwrap_2d(&wrapped, Some(&quality)).unwrap();
        assert!(unw[[4, 4]].is_nan());
        assert_matches_ramp(
            &unw,
            &{
                let mut t = truth.clone();
                t[[4, 4]] = f32::NAN;
                t
            },
            (0, 0),
            1e-4,
        );
    }

    #[test]
    fn min_quality_bloquea_pixeles_de_baja_coherencia() {
        // Franja de calidad 0.1 con umbral 0.3: los píxeles quedan NaN y las
        // dos islas restantes se desenrollan bien (cada una con su offset).
        let rows = 12;
        let cols = 15;
        let truth = ramp(rows, cols, 0.5, 0.4);
        let wrapped = wrap(&truth);
        let mut quality = Array2::from_elem((rows, cols), 0.9_f32);
        for r in 0..rows {
            quality[[r, 7]] = 0.1;
        }

        let unw = unwrap_2d_min_quality(&wrapped, Some(&quality), Some(0.3)).unwrap();
        for r in 0..rows {
            assert!(unw[[r, 7]].is_nan(), "({r},7) bajo el umbral debía ser NaN");
        }
        assert_matches_ramp_where(&unw, &truth, (5, 3), 1e-4, |_, c| c < 7);
        assert_matches_ramp_where(&unw, &truth, (5, 11), 1e-4, |_, c| c > 7);

        // Umbral sin mapa de calidad → error claro.
        assert!(matches!(
            unwrap_2d_min_quality(&wrapped, None, Some(0.3)),
            Err(InsarError::Metadata(_))
        ));
    }

    #[test]
    fn imagen_toda_nan_devuelve_nan() {
        let wrapped = Array2::from_elem((5, 5), f32::NAN);
        let unw = unwrap_2d(&wrapped, None).unwrap();
        assert!(unw.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn errores_de_dimensiones() {
        // Imagen vacía.
        let empty = Array2::<f32>::zeros((0, 4));
        assert!(matches!(
            unwrap_2d(&empty, None),
            Err(InsarError::DimensionMismatch(_))
        ));
        // Quality con dims distintas.
        let wrapped = Array2::<f32>::zeros((4, 4));
        let quality = Array2::<f32>::zeros((4, 5));
        assert!(matches!(
            unwrap_2d(&wrapped, Some(&quality)),
            Err(InsarError::DimensionMismatch(_))
        ));
    }

    fn stack_2_capas(rows: usize, cols: usize) -> (IfgStack, Vec<Array2<f32>>) {
        let truths = vec![ramp(rows, cols, 0.45, 0.35), ramp(rows, cols, -0.3, 0.5)];
        let mut data = Array3::from_elem((2, rows, cols), Complex32::new(0.0, 0.0));
        for (k, t) in truths.iter().enumerate() {
            for ((r, c), &p) in t.indexed_iter() {
                data[[k, r, c]] = Complex32::new(p.cos(), p.sin());
            }
        }
        let stack = IfgStack {
            data,
            epochs: epochs3(),
            pairs: vec![
                IfgPair { reference: 0, secondary: 1, perp_baseline_m: 40.0 },
                IfgPair { reference: 1, secondary: 2, perp_baseline_m: -25.0 },
            ],
            meta: meta(),
        };
        (stack, truths)
    }

    #[test]
    fn unwrap_stack_dos_capas() {
        let (stack, truths) = stack_2_capas(16, 14);
        let out = unwrap_stack(&stack, None).unwrap();
        assert_eq!(out.data.dim(), (2, 16, 14));
        assert_eq!(out.pairs.len(), 2);
        assert_eq!(out.epochs, stack.epochs);
        for (k, truth) in truths.iter().enumerate() {
            let layer = out.data.index_axis(Axis(0), k).to_owned();
            assert_matches_ramp(&layer, truth, (8, 7), 1e-4);
        }
    }

    #[test]
    fn unwrap_stack_con_coherencia_y_nan_complejo() {
        let (mut stack, truths) = stack_2_capas(10, 10);
        // Píxel inválido (re/im NaN) en la capa 0 → NaN en la salida.
        stack.data[[0, 5, 5]] = Complex32::new(f32::NAN, f32::NAN);
        let coherence = Array3::from_elem((2, 10, 10), 0.8_f32);
        let out = unwrap_stack(&stack, Some(&coherence)).unwrap();
        assert!(out.data[[0, 5, 5]].is_nan());
        assert!(out.data[[1, 5, 5]].is_finite());
        let layer1 = out.data.index_axis(Axis(0), 1).to_owned();
        assert_matches_ramp(&layer1, &truths[1], (3, 3), 1e-4);
    }

    #[test]
    fn unwrap_stack_coherencia_dims_invalidas() {
        let (stack, _) = stack_2_capas(8, 8);
        let coherence = Array3::from_elem((2, 8, 9), 0.8_f32);
        assert!(matches!(
            unwrap_stack(&stack, Some(&coherence)),
            Err(InsarError::DimensionMismatch(_))
        ));
    }
}

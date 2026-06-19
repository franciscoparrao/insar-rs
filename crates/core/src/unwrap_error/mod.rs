//! Corrección de errores de desenrollado por **cierre de fase**
//! (Yunjun et al. 2019).
//!
//! Cada interferograma desenrollado puede tener saltos de ciclo entero (2π)
//! entre componentes conexas independientes (cada `.unw` de ISCE/GUNW asigna
//! una referencia de fase arbitraria por componente). Estos saltos rompen la
//! consistencia de la red: para un triplete de pares que forma un lazo cerrado
//! (i→j, j→k, i→k), la suma de fases (el *cierre*) debería ser ~0; un salto de
//! ciclo lo lleva a un múltiplo entero de 2π.
//!
//! Usando los lazos de cierre de la red SBAS se estima, por par y píxel, el
//! entero de corrección U que minimiza la inconsistencia de cierre, y se
//! corrige la fase: `φ' = φ − 2π·U`. No requiere adyacencia espacial entre
//! componentes — solo la redundancia temporal de la red.

use std::collections::HashMap;
use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, Axis};
use rayon::prelude::*;

use crate::error::Result;
use crate::types::{IfgPair, UnwrappedStack};

/// `2π` en `f64` (factor de un ciclo de fase entero).
const TWO_PI: f64 = 2.0 * PI;

/// Lazos de cierre de la red: cada lazo es un triplete de pares (con signo)
/// cuya fase debe cerrar en ~0. `matrix` es (n_lazos × n_pares) con entradas
/// en {−1, 0, +1}.
#[derive(Debug, Clone)]
pub struct ClosureLoops {
    /// Matriz de diseño de lazos (n_lazos × n_pares), entradas ±1 / 0.
    pub matrix: Array2<f64>,
    /// Índices de los pares que participan en cada lazo (para el camino rápido).
    pub loops: Vec<[usize; 3]>,
}

/// Construye los lazos de cierre de tripletes a partir de los pares: para cada
/// terna de épocas (a<b<c) tal que existen los pares (a,b), (b,c) y (a,c), el
/// lazo es `φ_ab + φ_bc − φ_ac`.
pub fn build_closure_loops(pairs: &[IfgPair]) -> ClosureLoops {
    let n_pairs = pairs.len();

    // Mapa (reference, secondary) -> índice de par. Si hay pares duplicados,
    // se queda con el primero (la red SBAS no debería tenerlos).
    let mut index: HashMap<(usize, usize), usize> = HashMap::with_capacity(n_pairs);
    for (k, p) in pairs.iter().enumerate() {
        index.entry((p.reference, p.secondary)).or_insert(k);
    }

    // Conjunto ordenado de épocas que efectivamente aparecen en algún par.
    let mut epochs: Vec<usize> = pairs
        .iter()
        .flat_map(|p| [p.reference, p.secondary])
        .collect();
    epochs.sort_unstable();
    epochs.dedup();

    // Para cada terna a<b<c con los tres pares presentes: un lazo.
    let mut loops: Vec<[usize; 3]> = Vec::new();
    for (ia, &a) in epochs.iter().enumerate() {
        for (ib, &b) in epochs.iter().enumerate().skip(ia + 1) {
            // (a,b) debe existir para que cualquier lazo con esta base aporte.
            let Some(&idx_ab) = index.get(&(a, b)) else {
                continue;
            };
            for &c in epochs.iter().skip(ib + 1) {
                let (Some(&idx_bc), Some(&idx_ac)) =
                    (index.get(&(b, c)), index.get(&(a, c)))
                else {
                    continue;
                };
                loops.push([idx_ab, idx_bc, idx_ac]);
            }
        }
    }

    // Matriz de diseño (n_lazos × n_pares): +1 en (a,b) y (b,c), −1 en (a,c).
    let mut matrix = Array2::<f64>::zeros((loops.len(), n_pairs));
    for (l, &[ab, bc, ac]) in loops.iter().enumerate() {
        matrix[[l, ab]] += 1.0;
        matrix[[l, bc]] += 1.0;
        matrix[[l, ac]] -= 1.0;
    }

    ClosureLoops { matrix, loops }
}

/// Corrige los errores de desenrollado del stack in situ por cierre de fase.
///
/// Para cada píxel calcula el cierre entero `n = round(closure / 2π)` de cada
/// lazo; si todos son 0 no hay error. Si no, resuelve el entero de corrección
/// por par (solución de norma mínima, redondeada) y aplica `φ −= 2π·U`.
/// Los pares no finitos en un píxel se excluyen de sus lazos.
///
/// Devuelve el número de píxeles corregidos.
pub fn correct_unwrap_errors(stack: &mut UnwrappedStack) -> Result<usize> {
    let n_pairs = stack.n_layers();

    let closure = build_closure_loops(&stack.pairs);
    let n_loops = closure.loops.len();

    // Sin redundancia temporal no hay cierres que evaluar: nada que corregir.
    if n_loops == 0 {
        return Ok(0);
    }

    // Pseudoinversa L2 de la matriz de lazos C (n_lazos × n_pares). `pinv` es
    // (n_pares × n_lazos) y entrega la solución de norma mínima U = pinv·n_int
    // del sistema sobre-/sub-determinado C·U = n_int. Se calcula UNA vez (la
    // SVD es el costo dominante) y se reutiliza en todos los píxeles.
    //
    // Nota: U es la solución L2 redondeada a enteros. La solución L1 (entera,
    // p. ej. programación lineal) sería más robusta a outliers de cierre, pero
    // esta L2 redondeada es un buen primer orden — el enfoque clásico de
    // Yunjun et al. (2019) para corrección de errores de desenrollado.
    let c_mat = DMatrix::<f64>::from_fn(n_loops, n_pairs, |i, j| closure.matrix[[i, j]]);
    let svd = c_mat.svd(true, true);
    // Tolerancia estilo rcond de numpy/LAPACK: σ_max · max(m,n) · ε_f64
    // (idéntica a la usada en la inversión SBAS).
    let s_max = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let eps = s_max * (n_loops.max(n_pairs) as f64) * f64::EPSILON;
    let pinv = match svd.pseudo_inverse(eps) {
        Ok(p) => p,
        // C es entera y bien condicionada; si la SVD fallara no hay corrección
        // fiable que aplicar, así que devolvemos 0 sin tocar el stack.
        Err(_) => return Ok(0),
    };

    let loops = &closure.loops;

    // Paraleliza por filas (igual que `invert_sbas`), reusando buffers por
    // fila para no reasignar por píxel. El contador por fila se combina con un
    // map+reduce de rayon (suma).
    let mut row_views: Vec<_> = stack.data.axis_iter_mut(Axis(1)).collect();
    let corrected: usize = row_views
        .par_iter_mut()
        .map(|row| {
            // Buffers reutilizados a lo largo de toda la fila.
            let mut phi = vec![0.0_f64; n_pairs];
            let mut finite = vec![false; n_pairs];
            let mut n_int = DVector::<f64>::zeros(n_loops);
            let mut local = 0usize;

            for c in 0..row.shape()[1] {
                // (a) Lee las fases del píxel.
                for k in 0..n_pairs {
                    let v = row[[k, c]];
                    if v.is_finite() {
                        phi[k] = v as f64;
                        finite[k] = true;
                    } else {
                        phi[k] = 0.0;
                        finite[k] = false;
                    }
                }

                // (b) Cierre entero por lazo. Lazos con algún par NaN no aportan
                // (n_l = 0): se excluyen de la inversión.
                let mut any_closure = false;
                for (l, &[ab, bc, ac]) in loops.iter().enumerate() {
                    let nl = if finite[ab] && finite[bc] && finite[ac] {
                        let c_l = phi[ab] + phi[bc] - phi[ac];
                        (c_l / TWO_PI).round()
                    } else {
                        0.0
                    };
                    n_int[l] = nl;
                    if nl != 0.0 {
                        any_closure = true;
                    }
                }

                // (c) Todos los cierres 0 → píxel sin error de desenrollado.
                if !any_closure {
                    continue;
                }

                // (d) U = pinv · n_int, redondeada a enteros (solución L2).
                let u = &pinv * &n_int;

                // (e) Aplica φ_k −= 2π·U_k en pares con corrección y fase finita.
                let mut pixel_corrected = false;
                for k in 0..n_pairs {
                    let uk = u[k].round();
                    if uk != 0.0 && finite[k] {
                        let new = phi[k] - TWO_PI * uk;
                        row[[k, c]] = new as f32;
                        pixel_corrected = true;
                    }
                }
                if pixel_corrected {
                    local += 1;
                }
            }
            local
        })
        .sum();

    Ok(corrected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Epoch, StackMeta, UnwrappedStack, SENTINEL1_WAVELENGTH_M};
    use ndarray::Array3;
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
        let start: chrono::NaiveDate = "2023-01-01".parse().unwrap();
        (0..n)
            .map(|i| Epoch(start + chrono::Duration::days(12 * i as i64)))
            .collect()
    }

    fn pair(i: usize, j: usize) -> IfgPair {
        IfgPair { reference: i, secondary: j, perp_baseline_m: 0.0 }
    }

    // ---------- build_closure_loops ----------

    #[test]
    fn build_loops_red_de_tres_epocas() {
        // Pares en orden (0,1),(1,2),(0,2) → exactamente 1 lazo con la fila
        // [+1, +1, −1] y loops = [[0, 1, 2]].
        let pairs = vec![pair(0, 1), pair(1, 2), pair(0, 2)];
        let cl = build_closure_loops(&pairs);

        assert_eq!(cl.loops.len(), 1);
        assert_eq!(cl.loops[0], [0, 1, 2]);
        assert_eq!(cl.matrix.shape(), &[1, 3]);
        assert_eq!(cl.matrix[[0, 0]], 1.0);
        assert_eq!(cl.matrix[[0, 1]], 1.0);
        assert_eq!(cl.matrix[[0, 2]], -1.0);
    }

    #[test]
    fn build_loops_sin_redundancia_es_vacio() {
        // 2 épocas, 1 par → no hay terna posible → 0 lazos, matriz (0 × 1).
        let pairs = vec![pair(0, 1)];
        let cl = build_closure_loops(&pairs);
        assert_eq!(cl.loops.len(), 0);
        assert_eq!(cl.matrix.shape(), &[0, 1]);
    }

    // ---------- correct_unwrap_errors ----------

    /// Potencial sintético por época (relativo a la época 0). Las fases
    /// consistentes de cada par son `φ_ij = pot[j] − pot[i]`, lo que garantiza
    /// cierre 0 en todo triángulo (es un campo gradiente).
    const POT: [f32; 4] = [0.0, 1.0, 1.7, 2.4];

    /// Stack 6 pares = red completa de 4 épocas
    /// [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)] × rows×cols, fases consistentes.
    /// La redundancia (par 0 en 3 lazos) permite que la solución L2 redondeada
    /// localice un salto de ciclo sobre el par (0,1).
    fn consistent_stack(rows: usize, cols: usize) -> UnwrappedStack {
        let pairs = vec![
            pair(0, 1),
            pair(0, 2),
            pair(0, 3),
            pair(1, 2),
            pair(1, 3),
            pair(2, 3),
        ];
        let mut data = Array3::<f32>::zeros((pairs.len(), rows, cols));
        for (k, p) in pairs.iter().enumerate() {
            data.index_axis_mut(Axis(0), k)
                .fill(POT[p.secondary] - POT[p.reference]);
        }
        UnwrappedStack { data, epochs: epochs(4), pairs, meta: meta() }
    }

    #[test]
    fn corrige_salto_de_2pi_inyectado() {
        // Inyecta +2π en el par 0 = (0,1) de dos píxeles; el resto del stack es
        // consistente. Tras corregir, esos pares recuperan su valor original
        // y el contador = nº de píxeles inyectados (2). Un píxel sin salto
        // queda intacto.
        let mut stack = consistent_stack(2, 3);
        let orig = stack.data.clone();

        // Inyecta salto en píxeles (0,0) y (1,2), par 0.
        stack.data[[0, 0, 0]] += TWO_PI as f32;
        stack.data[[0, 1, 2]] += TWO_PI as f32;

        let n = correct_unwrap_errors(&mut stack).unwrap();
        assert_eq!(n, 2, "deberían corregirse exactamente 2 píxeles");

        // Los pares inyectados recuperan su valor original.
        assert!(
            (stack.data[[0, 0, 0]] - orig[[0, 0, 0]]).abs() < 1e-3,
            "píxel (0,0) par 0: {} vs {}",
            stack.data[[0, 0, 0]],
            orig[[0, 0, 0]]
        );
        assert!(
            (stack.data[[0, 1, 2]] - orig[[0, 1, 2]]).abs() < 1e-3,
            "píxel (1,2) par 0: {} vs {}",
            stack.data[[0, 1, 2]],
            orig[[0, 1, 2]]
        );

        // Un píxel sin salto queda intacto en todos sus pares.
        for k in 0..stack.pairs.len() {
            assert!(
                (stack.data[[k, 0, 1]] - orig[[k, 0, 1]]).abs() < 1e-6,
                "píxel limpio (0,1) par {k} cambió"
            );
        }
    }

    #[test]
    fn cierre_consistente_no_corrige_nada() {
        // Sin saltos: todos los cierres redondean a 0 → 0 píxeles corregidos
        // y stack intacto.
        let mut stack = consistent_stack(3, 3);
        let orig = stack.data.clone();
        let n = correct_unwrap_errors(&mut stack).unwrap();
        assert_eq!(n, 0);
        assert_eq!(stack.data, orig);
    }

    #[test]
    fn red_sin_lazos_devuelve_cero() {
        // 2 épocas, 1 par → sin redundancia → Ok(0), stack intacto.
        let mut stack = UnwrappedStack {
            data: Array3::<f32>::from_elem((1, 2, 2), 1.5),
            epochs: epochs(2),
            pairs: vec![pair(0, 1)],
            meta: meta(),
        };
        let orig = stack.data.clone();
        let n = correct_unwrap_errors(&mut stack).unwrap();
        assert_eq!(n, 0);
        assert_eq!(stack.data, orig);
    }

    #[test]
    fn nan_en_un_par_no_rompe() {
        // Un píxel con un par NaN: los lazos que lo incluyen se ignoran; no
        // hay panic y el píxel no se "corrige" espuriamente.
        let mut stack = consistent_stack(1, 2);
        stack.data[[0, 0, 0]] = f32::NAN; // par 0 NaN en píxel (0,0)
        let orig = stack.data.clone();

        let n = correct_unwrap_errors(&mut stack).unwrap();
        // Los lazos que usan el par 0 se ignoran; los demás cierran en 0 →
        // píxel sin corrección y sin panic.
        assert_eq!(n, 0);
        // El NaN persiste y el resto queda intacto.
        assert!(stack.data[[0, 0, 0]].is_nan());
        for k in 1..stack.pairs.len() {
            assert_eq!(stack.data[[k, 0, 0]], orig[[k, 0, 0]]);
        }
        // El píxel finito vecino tampoco cambia (cierre consistente).
        for k in 0..stack.pairs.len() {
            assert_eq!(stack.data[[k, 0, 1]], orig[[k, 0, 1]]);
        }
    }
}

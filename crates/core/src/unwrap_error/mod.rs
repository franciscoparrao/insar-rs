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
//! entero de corrección U que minimiza la inconsistencia de cierre, se
//! **verifica** que aplicarlo reduzca efectivamente los cierres (en redes de
//! baja redundancia la solución L2 redondeada puede ser nula: esos píxeles se
//! reportan como detectados-sin-corregir, ver [`UnwrapCorrectionReport`]) y
//! se corrige la fase: `φ' = φ − 2π·U`. No requiere adyacencia espacial entre
//! componentes — solo la redundancia temporal de la red.

use std::collections::{HashMap, HashSet};
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

/// Nº de lazos de cierre con ambigüedad entera ≠ 0 por píxel — el producto de
/// QC estándar de errores de desenrollado (equivalente a
/// `numTriNonzeroIntAmbiguity` de MintPy, Yunjun et al. 2019). Un píxel con
/// conteo > 0 tiene al menos un salto de ciclo entre sus pares; sirve como
/// máscara de calidad antes/después de [`correct_unwrap_errors`].
///
/// Solo se evalúan los lazos **activos** (los 3 pares con fase finita); un
/// píxel sin ningún lazo activo queda NaN (sin información de cierre, no
/// "cierre perfecto"). Error si el stack es inválido.
pub fn nonzero_closure_count(stack: &UnwrappedStack) -> Result<Array2<f32>> {
    stack.validate()?;
    let n_pairs = stack.n_layers();
    let (n_rows, n_cols) = stack.dims();

    let closure = build_closure_loops(&stack.pairs);
    let loops = &closure.loops;
    let phases = stack.data.view();

    let mut out = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut row_views: Vec<_> = out.axis_iter_mut(Axis(0)).collect();
    row_views.par_iter_mut().enumerate().for_each(|(r, out_row)| {
        let mut phi = vec![0.0_f64; n_pairs];
        let mut finite = vec![false; n_pairs];
        for c in 0..n_cols {
            for k in 0..n_pairs {
                let v = phases[[k, r, c]];
                finite[k] = v.is_finite();
                phi[k] = if finite[k] { v as f64 } else { 0.0 };
            }
            let mut active = 0usize;
            let mut nonzero = 0usize;
            for &[ab, bc, ac] in loops {
                if finite[ab] && finite[bc] && finite[ac] {
                    active += 1;
                    if ((phi[ab] + phi[bc] - phi[ac]) / TWO_PI).round() != 0.0 {
                        nonzero += 1;
                    }
                }
            }
            if active > 0 {
                out_row[c] = nonzero as f32;
            }
        }
    });

    Ok(out)
}

/// Resultado de [`correct_unwrap_errors`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UnwrapCorrectionReport {
    /// Píxeles donde la corrección redujo la inconsistencia de cierre y se
    /// aplicó al stack.
    pub corrected: usize,
    /// Píxeles con cierre ≠ 0 detectado cuya corrección L2-redondeada NO
    /// redujo la inconsistencia: se dejaron **intactos**. Es el caso típico
    /// de redes de baja redundancia (pares en 1-2 lazos), donde la solución
    /// de norma mínima reparte el salto en fracciones que redondean a cero.
    /// Un valor alto indica que la red no puede localizar los saltos: esos
    /// píxeles deben enmascararse (p. ej. vía coherencia temporal) o la red
    /// densificarse.
    pub detected_uncorrected: usize,
}

/// Solver de corrección para un patrón de pares finitos: índices de los lazos
/// activos (los 3 pares finitos) y pseudoinversa de la matriz de lazos
/// restringida a esas filas.
type MaskSolver = Option<(Vec<usize>, DMatrix<f64>)>;

/// Pseudoinversa L2 de la matriz de lazos restringida a `active` (filas).
/// `None` si no hay lazos activos o la SVD falla (sin corrección fiable).
fn closure_pinv(matrix: &Array2<f64>, active: &[usize], n_pairs: usize) -> MaskSolver {
    if active.is_empty() {
        return None;
    }
    let c_mat = DMatrix::<f64>::from_fn(active.len(), n_pairs, |i, j| matrix[[active[i], j]]);
    let svd = c_mat.svd(true, true);
    // Tolerancia estilo rcond de numpy/LAPACK: σ_max · max(m,n) · ε_f64
    // (idéntica a la usada en la inversión SBAS).
    let s_max = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let eps = s_max * (active.len().max(n_pairs) as f64) * f64::EPSILON;
    svd.pseudo_inverse(eps).ok().map(|p| (active.to_vec(), p))
}

/// Corrige los errores de desenrollado del stack in situ por cierre de fase.
///
/// Para cada píxel calcula el cierre entero `n = round(closure / 2π)` de cada
/// lazo **activo** (los 3 pares con fase finita); si todos son 0 no hay
/// error. Si no, resuelve el entero de corrección por par (solución de norma
/// mínima L2 sobre los lazos activos, redondeada) y **verifica** que la
/// corrección reduzca la inconsistencia total de cierre (`Σ|n_l|`) antes de
/// aplicar `φ −= 2π·U`:
///
/// - Si la reduce → se aplica (cuenta en `corrected`).
/// - Si no (típico de redes ralas: la solución fraccional redondea a 0) → el
///   píxel queda intacto y cuenta en `detected_uncorrected`. Sin esta
///   verificación la corrección sería silenciosamente nula: el cierre se
///   detectaría pero nada cambiaría, sin señal al caller.
///
/// Los pares no finitos de un píxel excluyen sus lazos de la estimación (la
/// pseudoinversa se recalcula por patrón de pares finitos y se cachea, igual
/// que en `invert_sbas`): un lazo sin dato es ausencia de información, no un
/// "cierre perfecto".
///
/// Nota: U es la solución L2 redondeada a enteros. La solución entera L1
/// (p. ej. ILP) sería más robusta a outliers de cierre — es la recomendación
/// de Yunjun et al. (2019) y queda como mejora futura; la verificación de
/// efectividad de arriba acota el daño de la aproximación L2.
pub fn correct_unwrap_errors(stack: &mut UnwrappedStack) -> Result<UnwrapCorrectionReport> {
    stack.validate()?;
    let n_pairs = stack.n_layers();

    let closure = build_closure_loops(&stack.pairs);
    let n_loops = closure.loops.len();

    // Sin redundancia temporal no hay cierres que evaluar: nada que corregir.
    if n_loops == 0 {
        return Ok(UnwrapCorrectionReport::default());
    }

    let loops = &closure.loops;
    let phases = stack.data.view();

    // Camino rápido (todos los pares finitos): todos los lazos activos.
    let full_active: Vec<usize> = (0..n_loops).collect();
    let Some((_, full_pinv)) = closure_pinv(&closure.matrix, &full_active, n_pairs) else {
        // C es entera y bien condicionada; si la SVD fallara no hay corrección
        // fiable que aplicar: stack intacto.
        return Ok(UnwrapCorrectionReport::default());
    };

    // Pasada 1: patrones de pares finitos con al menos un NaN (por fila, en
    // paralelo) — mismo esquema de máscara en bits que `invert_sbas`.
    let n_words = n_pairs.div_ceil(64);
    let mask_bit = |key: &mut [u64], k: usize| key[k / 64] |= 1u64 << (k % 64);
    let (n_rows, n_cols) = stack.dims();
    let unique_masks: HashSet<Vec<u64>> = (0..n_rows)
        .into_par_iter()
        .map(|r| {
            let mut set: HashSet<Vec<u64>> = HashSet::new();
            for c in 0..n_cols {
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

    // Pasada 2: solver por patrón (en paralelo): lazos activos = los que
    // tienen sus 3 pares finitos; pinv restringida a esas filas.
    let solvers: HashMap<Vec<u64>, MaskSolver> = unique_masks
        .into_par_iter()
        .map(|mask| {
            let is_set = |k: usize| mask[k / 64] & (1u64 << (k % 64)) != 0;
            let active: Vec<usize> = loops
                .iter()
                .enumerate()
                .filter(|(_, trio)| trio.iter().all(|&k| is_set(k)))
                .map(|(l, _)| l)
                .collect();
            let solver = closure_pinv(&closure.matrix, &active, n_pairs);
            (mask, solver)
        })
        .collect();

    // Pasada 3: corrección por píxel (en paralelo por fila), reusando buffers
    // por fila. Los contadores se combinan con map+reduce de rayon.
    let mut row_views: Vec<_> = stack.data.axis_iter_mut(Axis(1)).collect();
    let (corrected, detected_uncorrected) = row_views
        .par_iter_mut()
        .map(|row| {
            // Buffers reutilizados a lo largo de toda la fila.
            let mut phi = vec![0.0_f64; n_pairs];
            let mut finite = vec![false; n_pairs];
            let mut key = vec![0u64; n_words];
            let mut n_buf: Vec<f64> = Vec::with_capacity(n_loops);
            let mut u_round = vec![0.0_f64; n_pairs];
            let mut local = (0usize, 0usize);

            for c in 0..row.shape()[1] {
                // (a) Lee las fases del píxel y arma la máscara de finitos.
                key.iter_mut().for_each(|w| *w = 0);
                let mut n_valid = 0usize;
                for k in 0..n_pairs {
                    let v = row[[k, c]];
                    if v.is_finite() {
                        phi[k] = v as f64;
                        finite[k] = true;
                        mask_bit(&mut key, k);
                        n_valid += 1;
                    } else {
                        phi[k] = 0.0;
                        finite[k] = false;
                    }
                }

                // (b) Solver según patrón: completo (camino rápido) o cacheado.
                let (active, pinv) = if n_valid == n_pairs {
                    (&full_active, &full_pinv)
                } else {
                    match solvers.get(&key) {
                        Some(Some((a, p))) => (a, p),
                        // Sin lazos activos: no hay información de cierre.
                        _ => continue,
                    }
                };

                // (c) Cierre entero por lazo activo.
                n_buf.clear();
                let mut sum_abs_old = 0.0_f64;
                for &l in active {
                    let [ab, bc, ac] = loops[l];
                    let nl = ((phi[ab] + phi[bc] - phi[ac]) / TWO_PI).round();
                    sum_abs_old += nl.abs();
                    n_buf.push(nl);
                }

                // (d) Todos los cierres 0 → píxel sin error de desenrollado.
                if sum_abs_old == 0.0 {
                    continue;
                }

                // (e) U = pinv · n_int, redondeada a enteros (solución L2).
                let u = pinv * DVector::from_column_slice(&n_buf);
                for k in 0..n_pairs {
                    u_round[k] = u[k].round();
                }

                // (f) Verificación: la corrección aplica φ −= 2π·U, así que el
                // cierre de cada lazo activo pasa a n_l − (C·U)_l. Si Σ|·| no
                // baja, la corrección no mejora nada → píxel intacto.
                let mut sum_abs_new = 0.0_f64;
                for (i, &l) in active.iter().enumerate() {
                    let [ab, bc, ac] = loops[l];
                    let delta = u_round[ab] + u_round[bc] - u_round[ac];
                    sum_abs_new += (n_buf[i] - delta).abs();
                }
                if sum_abs_new >= sum_abs_old {
                    local.1 += 1; // detectado pero no corregible con esta red
                    continue;
                }

                // (g) Aplica φ_k −= 2π·U_k en pares con corrección y fase finita.
                for k in 0..n_pairs {
                    if u_round[k] != 0.0 && finite[k] {
                        row[[k, c]] = (phi[k] - TWO_PI * u_round[k]) as f32;
                    }
                }
                local.0 += 1;
            }
            local
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

    Ok(UnwrapCorrectionReport { corrected, detected_uncorrected })
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

        let rep = correct_unwrap_errors(&mut stack).unwrap();
        assert_eq!(rep.corrected, 2, "deberían corregirse exactamente 2 píxeles");
        assert_eq!(rep.detected_uncorrected, 0);

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
        let rep = correct_unwrap_errors(&mut stack).unwrap();
        assert_eq!(rep, UnwrapCorrectionReport::default());
        assert_eq!(stack.data, orig);
    }

    #[test]
    fn red_sin_lazos_devuelve_cero() {
        // 2 épocas, 1 par → sin redundancia → reporte vacío, stack intacto.
        let mut stack = UnwrappedStack {
            data: Array3::<f32>::from_elem((1, 2, 2), 1.5),
            epochs: epochs(2),
            pairs: vec![pair(0, 1)],
            meta: meta(),
        };
        let orig = stack.data.clone();
        let rep = correct_unwrap_errors(&mut stack).unwrap();
        assert_eq!(rep, UnwrapCorrectionReport::default());
        assert_eq!(stack.data, orig);
    }

    #[test]
    fn nan_en_un_par_no_rompe() {
        // Un píxel con un par NaN: los lazos que lo incluyen se excluyen del
        // solver; no hay panic y el píxel no se "corrige" espuriamente.
        let mut stack = consistent_stack(1, 2);
        stack.data[[0, 0, 0]] = f32::NAN; // par 0 NaN en píxel (0,0)
        let orig = stack.data.clone();

        let rep = correct_unwrap_errors(&mut stack).unwrap();
        // Los lazos que usan el par 0 se excluyen; los demás cierran en 0 →
        // píxel sin corrección y sin panic.
        assert_eq!(rep, UnwrapCorrectionReport::default());
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

    #[test]
    fn red_minima_detecta_pero_no_corrige() {
        // Red de 3 pares = 1 solo lazo: un salto de 2π da U = ±1/3 en cada
        // par → round → 0 → la corrección L2 es nula. Antes esto pasaba en
        // silencio; ahora el píxel queda intacto y se reporta como
        // detectado-sin-corregir.
        let pairs = vec![pair(0, 1), pair(1, 2), pair(0, 2)];
        let mut data = Array3::<f32>::zeros((3, 1, 2));
        for (k, p) in pairs.iter().enumerate() {
            data.index_axis_mut(Axis(0), k)
                .fill(POT[p.secondary] - POT[p.reference]);
        }
        let mut stack =
            UnwrappedStack { data, epochs: epochs(3), pairs, meta: meta() };
        stack.data[[0, 0, 0]] += TWO_PI as f32; // salto en el par (0,1)
        let orig = stack.data.clone();

        let rep = correct_unwrap_errors(&mut stack).unwrap();
        assert_eq!(rep.corrected, 0);
        assert_eq!(rep.detected_uncorrected, 1);
        // El píxel con el salto queda INTACTO (no semi-corregido).
        assert_eq!(stack.data, orig);
    }

    #[test]
    fn salto_se_corrige_con_otro_par_nan() {
        // Píxel con el par (2,3) NaN y salto de 2π en el par (0,1): los dos
        // lazos activos que contienen (0,1) bastan para localizar el salto.
        // Antes, los lazos con NaN entraban a la pinv como "cierre 0" falso y
        // sesgaban U hacia cero; ahora se excluyen del solver.
        let mut stack = consistent_stack(1, 2);
        let orig = stack.data.clone();
        stack.data[[5, 0, 0]] = f32::NAN; // par (2,3) NaN
        stack.data[[0, 0, 0]] += TWO_PI as f32; // salto en el par (0,1)

        let rep = correct_unwrap_errors(&mut stack).unwrap();
        assert_eq!(rep.corrected, 1, "el salto debía corregirse");
        assert_eq!(rep.detected_uncorrected, 0);
        assert!(
            (stack.data[[0, 0, 0]] - orig[[0, 0, 0]]).abs() < 1e-3,
            "par (0,1) no recuperó su valor: {} vs {}",
            stack.data[[0, 0, 0]],
            orig[[0, 0, 0]]
        );
        // El NaN persiste; los demás pares del píxel quedan intactos.
        assert!(stack.data[[5, 0, 0]].is_nan());
        for k in 1..5 {
            assert_eq!(stack.data[[k, 0, 0]], orig[[k, 0, 0]], "par {k} cambió");
        }
        // El píxel limpio vecino no cambia.
        for k in 0..stack.pairs.len() {
            assert_eq!(stack.data[[k, 0, 1]], orig[[k, 0, 1]]);
        }
    }

    #[test]
    fn stack_invalido_es_error_no_panic() {
        // Pares declarados ≠ capas → error de validate, sin panic.
        let mut stack = consistent_stack(1, 1);
        stack.pairs.pop();
        assert!(correct_unwrap_errors(&mut stack).is_err());
    }

    // ---------- nonzero_closure_count ----------

    #[test]
    fn conteo_de_cierres_detecta_salto_y_respeta_nan() {
        let mut stack = consistent_stack(2, 2);
        // Píxel (0,0): salto de 2π en el par (0,1) → los 2 lazos que lo
        // contienen (tripletes 0-1-2 y 0-1-3) no cierran. Píxel (1,1): par
        // (2,3) NaN → solo quedan 2 lazos activos, consistentes.
        stack.data[[0, 0, 0]] += TWO_PI as f32;
        stack.data[[5, 1, 1]] = f32::NAN;

        let qc = nonzero_closure_count(&stack).unwrap();
        assert_eq!(qc[[0, 0]], 2.0, "2 lazos contienen el par (0,1)");
        assert_eq!(qc[[0, 1]], 0.0, "píxel consistente");
        assert_eq!(qc[[1, 1]], 0.0, "lazos activos consistentes con par NaN");

        // Píxel sin ningún lazo activo → NaN.
        let mut isolated = consistent_stack(1, 1);
        for k in 0..isolated.pairs.len() {
            isolated.data[[k, 0, 0]] = f32::NAN;
        }
        let qc = nonzero_closure_count(&isolated).unwrap();
        assert!(qc[[0, 0]].is_nan());
    }
}

//! Inversión SBAS de la serie temporal de desplazamiento LOS por mínimos
//! cuadrados (SVD de nalgebra), conversión fase→desplazamiento y estimación
//! de velocidad media.
//!
//! Variante de incrementos de Berardino et al. (2002): las incógnitas son
//! los desplazamientos entre épocas consecutivas; la serie acumulada se
//! reconstruye relativa a la primera época.
//!
//! ## Extensiones ([`invert_sbas_ext`])
//!
//! - **WLS por coherencia** ([`WeightScheme`]): cada observación (par) se
//!   pondera según su coherencia — el default recomendado es el inverso de la
//!   varianza de fase (aprox. Cramér-Rao, Tough et al. 1995; el esquema `var`
//!   de MintPy, Yunjun et al. 2019). Se resuelve por ecuaciones normales
//!   `AᵀWA·x = AᵀW·b` con Cholesky por píxel.
//! - **Error de DEM** ([`DemErrorConfig`]): agrega a la matriz de diseño la
//!   columna `g_k = −B⊥_k/(R·sinθ)` (en desplazamiento por metro de Δz) y
//!   estima el residuo topográfico Δz por píxel junto con la serie
//!   (Fattahi & Amelung 2013). Requiere baselines perpendiculares no nulas.

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, Array3, Axis};
use rayon::prelude::*;

use crate::error::{InsarError, Result};
use crate::network;
use crate::types::{DisplacementSeries, Epoch, IfgPair, PsCandidate, UnwrappedStack, VelocityMap};

/// Esquema de pesos WLS por observación, derivado de la coherencia del par.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightScheme {
    /// Sin pesos (OLS, comportamiento v0.1).
    #[default]
    Unit,
    /// `w = γ` — peso lineal en coherencia (esquema `coh` de MintPy).
    Coherence,
    /// `w = 2γ²/(1−γ²)` — inverso de la varianza de fase (aprox. Cramér-Rao,
    /// Tough et al. 1995; el default `var` de MintPy). Recomendado.
    InversePhaseVariance,
}

/// Geometría para estimar el error de DEM junto con la serie.
#[derive(Debug, Clone, Copy)]
pub struct DemErrorConfig {
    /// Rango oblicuo (slant range) medio en metros. Sentinel-1 IW ≈ 850 km.
    pub slant_range_m: f64,
}

/// Configuración de la inversión robusta L1 por IRLS (mínimos cuadrados
/// re-ponderados iterativamente): en cada iteración `w_k ← w_base_k /
/// max(|r_k|, ε)`, lo que converge a la solución de norma L1 — mucho más
/// robusta a errores de desenrollado residuales que L2 (Lauknes et al. 2011).
/// MintPy no la trae de serie.
///
/// **Nota de identificabilidad**: la robustez L1 exige redundancia real. Si
/// un par corrupto es la suma exacta de otros DOS pares de la red (p. ej.
/// `(0,2) = (0,1) + (1,2)` cuando ambos consecutivos están presentes y nada
/// más cubre esos incrementos), el trade-off es 1:1 y el outlier es solo
/// débilmente identificable (el mínimo L1 gana por un margen ~0 y el
/// suavizado ε puede repartir el error). Con redes donde cada incremento
/// aparece en ≥2 pares además del corrupto (p. ej. consecutivos + saltos de
/// 2 y 3), el margen es ≥2:1 y el outlier se aísla limpiamente.
#[derive(Debug, Clone, Copy)]
pub struct IrlsConfig {
    /// Máximo de iteraciones IRLS (default 20).
    pub max_iterations: usize,
    /// Convergencia: ‖Δx‖∞ bajo esta cota (m) detiene la iteración (1e-6).
    pub tolerance_m: f64,
    /// Piso del residuo |r| en el re-peso, evita 1/0 (default 1e-4 m).
    pub epsilon_m: f64,
}

impl Default for IrlsConfig {
    fn default() -> Self {
        Self { max_iterations: 20, tolerance_m: 1e-6, epsilon_m: 1e-4 }
    }
}

/// Configuración del solver SBAS extendido ([`invert_sbas_ext`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct SbasSolverConfig {
    /// Pesos por observación (requiere stack de coherencia si ≠ `Unit`).
    pub weighting: WeightScheme,
    /// `Some(_)` → estima el error de DEM Δz por píxel (requiere baselines
    /// perpendiculares no nulas en los pares).
    pub dem_error: Option<DemErrorConfig>,
    /// `Some(_)` → inversión robusta L1 por IRLS sobre los pesos base
    /// (funciona con o sin coherencia). `None` = L2 puro.
    pub robust: Option<IrlsConfig>,
}

/// Solución de [`invert_sbas_ext`].
#[derive(Debug, Clone)]
pub struct SbasSolution {
    /// Serie de desplazamiento LOS relativa a la primera época.
    pub series: DisplacementSeries,
    /// Error de DEM Δz en metros por píxel (`Some` si se configuró
    /// [`SbasSolverConfig::dem_error`]).
    pub dem_error_m: Option<Array2<f32>>,
}

/// Peso WLS de una observación a partir de su coherencia. γ se clampea a
/// [0.01, 0.99] para evitar pesos degenerados (0 o ∞).
fn weight_from_coherence(gamma: f32, scheme: WeightScheme) -> f64 {
    let g = f64::from(gamma).clamp(0.01, 0.99);
    match scheme {
        WeightScheme::Unit => 1.0,
        WeightScheme::Coherence => g,
        WeightScheme::InversePhaseVariance => 2.0 * g * g / (1.0 - g * g),
    }
}

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
///
/// Devuelve el número de pares (de `stack.n_layers()`) que quedaron
/// completamente en NaN por no tener fase finita en el píxel de referencia —
/// el caller debe advertir si este número es alto en relación al total, ya
/// que cada uno de esos pares se pierde por completo para *todos* los
/// píxeles del stack, no solo para el de referencia.
pub fn reference_to_pixel(stack: &mut UnwrappedStack, row: usize, col: usize) -> Result<usize> {
    let (n_rows, n_cols) = stack.dims();
    if row >= n_rows || col >= n_cols {
        return Err(InsarError::DimensionMismatch(format!(
            "píxel de referencia ({row}, {col}) fuera de la grilla {n_rows}×{n_cols}"
        )));
    }
    let mut n_lost = 0usize;
    for k in 0..stack.n_layers() {
        let mut layer = stack.data.index_axis_mut(Axis(0), k);
        let ref_phase = layer[[row, col]];
        if ref_phase.is_finite() {
            layer.mapv_inplace(|v| v - ref_phase);
        } else {
            layer.fill(f32::NAN);
            n_lost += 1;
        }
    }
    Ok(n_lost)
}

/// Invierte la serie temporal de desplazamiento LOS por píxel (modo v0.1:
/// OLS sin pesos, sin error de DEM). Wrapper de [`invert_sbas_ext`].
/// - `ps = Some(...)`: invierte solo en los candidatos PS; el resto queda NaN.
/// - `ps = None`: invierte toda la grilla (modo SBAS clásico).
///
/// La serie resultante es relativa a la primera época (desplazamiento 0).
/// Error si la red es desconectada ([`crate::network::is_connected`]).
pub fn invert_sbas(
    stack: &UnwrappedStack,
    ps: Option<&[PsCandidate]>,
) -> Result<DisplacementSeries> {
    invert_sbas_ext(stack, ps, None, &SbasSolverConfig::default()).map(|s| s.series)
}

/// Inversión SBAS extendida: WLS por coherencia y/o estimación del error de
/// DEM (ver doc del módulo).
///
/// - `coherence`: stack de coherencia alineado con `stack.data` (pares ×
///   filas × cols). Obligatorio si `config.weighting != Unit`. Con WLS, un
///   par con coherencia no finita se excluye de la inversión de ese píxel.
/// - `config.dem_error = Some(_)`: la matriz de diseño gana la columna
///   `g_k = −B⊥_k/(R·sinθ)` y se estima Δz por píxel; error
///   [`InsarError::Metadata`] si todas las baselines son 0 (sin información).
///
/// Manejo de NoData por par: cada píxel se invierte con los pares de fase
/// finita. En OLS (`Unit`), los píxeles completos usan la pseudoinversa
/// cacheada (camino rápido) y los parciales comparten pseudoinversas
/// reducidas por patrón de validez. En WLS los pesos cambian por píxel, así
/// que se resuelven ecuaciones normales `AᵀWA·x = AᵀW·b` (Cholesky) por
/// píxel. Un píxel queda NaN si sus pares válidos no conectan todas las
/// épocas o si el sistema normal no es definido positivo (p. ej. columna de
/// DEM colineal tras el enmascarado).
pub fn invert_sbas_ext(
    stack: &UnwrappedStack,
    ps: Option<&[PsCandidate]>,
    coherence: Option<&Array3<f32>>,
    config: &SbasSolverConfig,
) -> Result<SbasSolution> {
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

    if let Some(coh) = coherence
        && coh.dim() != stack.data.dim()
    {
        return Err(InsarError::DimensionMismatch(format!(
            "coherencia {:?} vs stack {:?}",
            coh.dim(),
            stack.data.dim()
        )));
    }
    if config.weighting != WeightScheme::Unit && coherence.is_none() {
        return Err(InsarError::Metadata(
            "weighting != Unit requiere el stack de coherencia (pares × filas × cols)".into(),
        ));
    }

    // Columna de error de DEM (desplazamiento por metro de Δz), si se pide.
    // Se ESCALA a norma unitaria antes de armar la matriz de diseño: sus
    // elementos crudos (`g_k = -B⊥_k/(R·sinθ)`, ~2e-4, ver doc de
    // `dem_error_column`) son ~1e4 veces menores que los de la matriz de
    // incrementos (0/1) — sin escalar, eso infla el número de condición de
    // `AᵀWA` en ~1e7-1e8 incluso antes de cualquier colinealidad real,
    // reduciendo el margen que le queda a la guarda de `solve_normal_eqs`
    // para detectar casos genuinamente singulares (p. ej. deriva orbital
    // lineal, `B⊥_k ≈ β·Δt_k`, que hace la columna DEM colineal con la
    // matriz de incrementos). `dem_scale` se aplica de vuelta en
    // `write_solution` al extraer Δz; costo cero por píxel, la escala es
    // global. `dem_error_column` garantiza al menos una baseline no nula, así
    // que la norma siempre es > 0 cuando `dem_col` es `Some`.
    let dem_col: Option<Vec<f64>> = match &config.dem_error {
        Some(d) => Some(dem_error_column(&stack.pairs, stack.meta.incidence_deg, d)?),
        None => None,
    };
    let dem_scale = dem_col
        .as_ref()
        .map(|g| g.iter().map(|v| v * v).sum::<f64>().sqrt());
    let dem_col: Option<Vec<f64>> = dem_col.map(|g| {
        let scale = dem_scale.unwrap();
        g.into_iter().map(|v| v / scale).collect()
    });
    let n_incr = n_epochs - 1;
    let n_unknowns = n_incr + usize::from(dem_col.is_some());

    // Matriz de diseño extendida (una vez): [A | g].
    let a_ext = build_design_ext(&stack.pairs, n_epochs, dem_col.as_deref())?;

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
    let mut dem_out = dem_col
        .as_ref()
        .map(|_| Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN));
    let phases = stack.data.view();
    let wavelength_m = stack.meta.wavelength_m;

    // Vistas por fila de la serie y del mapa de Δz para escribir en paralelo.
    let mut out_rows: Vec<_> = out.axis_iter_mut(Axis(1)).collect();
    let mut dem_rows: Vec<Option<ndarray::ArrayViewMut1<'_, f32>>> = match dem_out.as_mut() {
        Some(d) => d.axis_iter_mut(Axis(0)).map(Some).collect(),
        None => (0..n_rows).map(|_| None).collect(),
    };

    // Escribe la solución x (incrementos [+ Δz]) en el píxel (fila implícita,
    // c). `x[n_incr]` viene en unidades de la columna DEM ESCALADA
    // (`dem_scale`, ver arriba) — se des-escala aquí, el único lugar donde
    // Δz se extrae del vector solución.
    let write_solution = |x: &DVector<f64>,
                          out_row: &mut ndarray::ArrayViewMut2<'_, f32>,
                          dem_row: &mut Option<ndarray::ArrayViewMut1<'_, f32>>,
                          c: usize| {
        out_row[[0, c]] = 0.0;
        let mut acc = 0.0_f64;
        for e in 1..n_epochs {
            acc += x[e - 1];
            out_row[[e, c]] = acc as f32;
        }
        if let Some(dem) = dem_row {
            // Columna escalada g' = g/scale ⇒ el coeficiente ajustado es
            // `scale·Δz` (no `Δz/scale`): des-escalar es DIVIDIR por `scale`.
            dem[c] = (x[n_incr] / dem_scale.unwrap_or(1.0)) as f32;
        }
    };

    // El camino por píxel se necesita si los pesos varían por píxel (WLS por
    // coherencia) o si hay re-ponderación iterativa (IRLS): en ambos casos no
    // existe una pseudoinversa cacheable.
    let per_pixel = config.weighting != WeightScheme::Unit || config.robust.is_some();

    if !per_pixel {
        // ------- Camino OLS: pseudoinversa cacheada por patrón de validez -------
        // La SVD se calcula UNA vez y se reutiliza en todos los píxeles.
        let svd = a_ext.clone().svd(true, true);
        // Tolerancia estilo rcond de numpy/LAPACK: σ_max · max(m,n) · ε_f64.
        // Con red conexa A es de rango columna completo; si no lo fuera, la
        // pseudoinversa entrega la solución de norma mínima.
        let s_max = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
        let eps = s_max * (n_pairs.max(n_unknowns) as f64) * f64::EPSILON;
        let pinv = svd
            .pseudo_inverse(eps)
            .map_err(|e| InsarError::Inversion(format!("pseudoinversa SVD: {e}")))?;

        let n_words = n_pairs.div_ceil(64);
        let mask_bit = |key: &mut [u64], k: usize| key[k / 64] |= 1u64 << (k % 64);

        // Pasada 1: patrones de máscara parcial distintos (paralelo por fila).
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

        // Pasada 2: pseudoinversa reducida por patrón (en paralelo). `None` si
        // la red reducida queda desconectada o con menos pares que incógnitas.
        let solvers: HashMap<Vec<u64>, Option<DMatrix<f64>>> = unique_masks
            .into_par_iter()
            .map(|mask| {
                let valid_idx: Vec<usize> = (0..n_pairs)
                    .filter(|&k| mask[k / 64] & (1u64 << (k % 64)) != 0)
                    .collect();
                let solver =
                    reduced_pinv(&valid_idx, &stack.pairs, n_epochs, dem_col.as_deref());
                (mask, solver)
            })
            .collect();

        // Pasada 3: inversión por píxel (paralelo por fila).
        out_rows
            .par_iter_mut()
            .zip(dem_rows.par_iter_mut())
            .enumerate()
            .for_each(|(r, (out_row, dem_row))| {
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
                    // x = incrementos entre épocas consecutivas [+ Δz].
                    let x = if b_vals.len() == n_pairs {
                        Some(&pinv * DVector::from_column_slice(&b_vals)) // camino rápido
                    } else {
                        match solvers.get(&key) {
                            Some(Some(rp)) => Some(rp * DVector::from_column_slice(&b_vals)),
                            _ => None, // sin pares válidos o red reducida desconectada
                        }
                    };
                    if let Some(x) = x {
                        write_solution(&x, out_row, dem_row, c);
                    }
                }
            });
    } else {
        // ------- Camino por píxel: ecuaciones normales (Cholesky) -------
        // Pesos base = coherencia (WLS) o 1 (Unit); IRLS opcional re-pondera
        // por 1/max(|r|, ε) hasta converger a la solución L1.
        let scheme = config.weighting;
        let need_coh = scheme != WeightScheme::Unit;
        let coh_view = coherence.map(|c| c.view());
        let robust = config.robust;
        let pairs = &stack.pairs;

        out_rows
            .par_iter_mut()
            .zip(dem_rows.par_iter_mut())
            .enumerate()
            .for_each(|(r, (out_row, dem_row))| {
                let mut nmat = DMatrix::<f64>::zeros(n_unknowns, n_unknowns);
                let mut yvec = DVector::<f64>::zeros(n_unknowns);
                let mut valid_idx: Vec<usize> = Vec::with_capacity(n_pairs);
                let mut reduced: Vec<IfgPair> = Vec::with_capacity(n_pairs);
                let mut b_vals: Vec<f64> = Vec::with_capacity(n_pairs);
                let mut w_base: Vec<f64> = Vec::with_capacity(n_pairs);
                let mut w_iter: Vec<f64> = Vec::with_capacity(n_pairs);

                for &c in &cols_by_row[r] {
                    // Par válido: fase finita y, si hay pesos por coherencia,
                    // coherencia finita (sin ella no hay peso definido).
                    valid_idx.clear();
                    b_vals.clear();
                    w_base.clear();
                    for k in 0..n_pairs {
                        let phi = phases[[k, r, c]];
                        if !phi.is_finite() {
                            continue;
                        }
                        let w = match &coh_view {
                            Some(coh) if need_coh => {
                                let g = coh[[k, r, c]];
                                if !g.is_finite() {
                                    continue;
                                }
                                weight_from_coherence(g, scheme)
                            }
                            _ => 1.0,
                        };
                        valid_idx.push(k);
                        b_vals.push(phase_to_displacement(phi as f64, wavelength_m));
                        w_base.push(w);
                    }
                    if valid_idx.len() < n_unknowns {
                        continue; // píxel indeterminado → NaN
                    }
                    // Conectividad de la red reducida (mismo criterio que OLS).
                    reduced.clear();
                    reduced.extend(valid_idx.iter().map(|&k| pairs[k]));
                    if !network::is_connected(&reduced, n_epochs) {
                        continue;
                    }

                    // Solución inicial con los pesos base. N no definida
                    // positiva (p. ej. columna DEM colineal) → NaN, sin panic.
                    let Some(mut x) = solve_normal_eqs(
                        &a_ext, &valid_idx, &b_vals, &w_base, &mut nmat, &mut yvec,
                    ) else {
                        continue;
                    };

                    // IRLS hacia la norma L1: w ← w_base/max(|r|, ε).
                    if let Some(irls) = &robust {
                        for _ in 0..irls.max_iterations {
                            w_iter.clear();
                            for (idx, &k) in valid_idx.iter().enumerate() {
                                let mut pred = 0.0_f64;
                                for j in 0..n_unknowns {
                                    pred += a_ext[(k, j)] * x[j];
                                }
                                let resid = (b_vals[idx] - pred).abs().max(irls.epsilon_m);
                                w_iter.push(w_base[idx] / resid);
                            }
                            let Some(x_new) = solve_normal_eqs(
                                &a_ext, &valid_idx, &b_vals, &w_iter, &mut nmat, &mut yvec,
                            ) else {
                                break; // conserva la última solución estable
                            };
                            let delta = (0..n_unknowns)
                                .map(|j| (x_new[j] - x[j]).abs())
                                .fold(0.0_f64, f64::max);
                            x = x_new;
                            if delta < irls.tolerance_m {
                                break;
                            }
                        }
                    }

                    write_solution(&x, out_row, dem_row, c);
                }
            });
    }

    drop(out_rows);
    drop(dem_rows);

    Ok(SbasSolution {
        series: DisplacementSeries {
            data: out,
            epochs: stack.epochs.clone(),
            meta: stack.meta.clone(),
        },
        dem_error_m: dem_out,
    })
}

/// Umbral relativo mínimo de pivote de Cholesky (`l_ii` más chico / más
/// grande del factor `L`, con `N = L·Lᵀ`) para aceptar la solución de
/// [`solve_normal_eqs`]. El número de condición de `N` escala
/// aproximadamente como el CUADRADO de esta razón, así que `1e-6` aquí
/// equivale a rechazar condicionamientos peores que ~1e12 — el umbral
/// estándar de "numéricamente singular" en `f64` (cf. las tolerancias
/// estilo LAPACK ya usadas en este módulo para la pseudoinversa SVD).
///
/// Sin esta guarda, Cholesky puede "tener éxito" con un pivote de puro
/// redondeo (~1e-16) cuando el sistema normal es matemáticamente singular
/// — el caso más común es la columna de error de DEM colineal con la matriz
/// de incrementos (deriva orbital lineal, `B⊥_k ≈ β·Δt_k` ⇒ `g ∝ A·Δt`) — y
/// entregar un Δz de magnitud arbitraria que además corrompe los
/// incrementos, sin ningún error ni NaN.
const CHOLESKY_MIN_RELATIVE_PIVOT: f64 = 1e-6;

/// Arma y resuelve las ecuaciones normales ponderadas `(AᵀWA)·x = AᵀW·b`
/// sobre las filas `valid_idx` de `a_ext` (con `b`/`w` alineados a
/// `valid_idx`), reutilizando los buffers `nmat`/`yvec`. `None` si el sistema
/// normal no es definido positivo (Cholesky falla) o si su condicionamiento
/// es demasiado pobre ([`CHOLESKY_MIN_RELATIVE_PIVOT`]).
fn solve_normal_eqs(
    a_ext: &DMatrix<f64>,
    valid_idx: &[usize],
    b: &[f64],
    w: &[f64],
    nmat: &mut DMatrix<f64>,
    yvec: &mut DVector<f64>,
) -> Option<DVector<f64>> {
    let n_unknowns = nmat.nrows();
    nmat.fill(0.0);
    yvec.fill(0.0);
    for (idx, &k) in valid_idx.iter().enumerate() {
        let (wi, bi) = (w[idx], b[idx]);
        for i in 0..n_unknowns {
            let ai = a_ext[(k, i)];
            if ai == 0.0 {
                continue;
            }
            yvec[i] += wi * ai * bi;
            for j in i..n_unknowns {
                nmat[(i, j)] += wi * ai * a_ext[(k, j)];
            }
        }
    }
    // Espejo del triángulo superior.
    for i in 1..n_unknowns {
        for j in 0..i {
            nmat[(i, j)] = nmat[(j, i)];
        }
    }
    nmat.clone().cholesky().and_then(|ch| {
        let diag = ch.l().diagonal();
        let max_l = diag.iter().copied().fold(0.0_f64, f64::max);
        let min_l = diag.iter().copied().fold(f64::INFINITY, f64::min);
        if max_l <= 0.0 || min_l / max_l < CHOLESKY_MIN_RELATIVE_PIVOT {
            None
        } else {
            Some(ch.solve(yvec))
        }
    })
}

/// Columna de error de DEM en unidades de desplazamiento LOS por metro de Δz:
/// `g_k = −B⊥_k/(R·sinθ)` (la fase topográfica es `(4π/λ)·B⊥/(R·sinθ)·Δz` y
/// la conversión a desplazamiento aporta el `−λ/(4π)`).
fn dem_error_column(
    pairs: &[IfgPair],
    incidence_deg: f64,
    config: &DemErrorConfig,
) -> Result<Vec<f64>> {
    if !(config.slant_range_m.is_finite() && config.slant_range_m > 0.0) {
        return Err(InsarError::Metadata(format!(
            "slant_range_m inválido: {} (Sentinel-1 IW ≈ 850000)",
            config.slant_range_m
        )));
    }
    if !(incidence_deg.is_finite() && (0.0..90.0).contains(&incidence_deg) && incidence_deg > 0.0)
    {
        return Err(InsarError::Metadata(format!(
            "incidence_deg inválido para error de DEM: {incidence_deg}"
        )));
    }
    if !pairs.iter().any(|p| p.perp_baseline_m != 0.0) {
        return Err(InsarError::Metadata(
            "estimación de error de DEM sin información: todas las baselines \
             perpendiculares son 0 (¿faltó configurar baselines_dir al leer el stack?)"
                .into(),
        ));
    }
    let sin_theta = incidence_deg.to_radians().sin();
    Ok(pairs
        .iter()
        .map(|p| -p.perp_baseline_m / (config.slant_range_m * sin_theta))
        .collect())
}

/// Matriz de diseño extendida `[A | g]` como `DMatrix`: la parte A viene de
/// [`network::design_matrix`] (única fuente de verdad de la parametrización
/// por incrementos); `g` es la columna de error de DEM si se estima Δz.
fn build_design_ext(
    pairs: &[IfgPair],
    n_epochs: usize,
    dem_col: Option<&[f64]>,
) -> Result<DMatrix<f64>> {
    let a = network::design_matrix(pairs, n_epochs)?;
    let n_incr = n_epochs - 1;
    let n_unknowns = n_incr + usize::from(dem_col.is_some());
    Ok(DMatrix::<f64>::from_fn(pairs.len(), n_unknowns, |i, j| {
        if j < n_incr {
            a[[i, j]]
        } else {
            dem_col.expect("j >= n_incr solo si hay columna DEM")[i]
        }
    }))
}

/// Pseudoinversa de la matriz de diseño (extendida) reducida al subconjunto
/// de pares válidos (`valid_idx`, ascendentes). Devuelve `None` (→ serie NaN)
/// si hay menos pares que incógnitas o si los pares no conectan todas las
/// épocas (red reducida desconectada → sistema rank-deficiente).
fn reduced_pinv(
    valid_idx: &[usize],
    pairs: &[IfgPair],
    n_epochs: usize,
    dem_col: Option<&[f64]>,
) -> Option<DMatrix<f64>> {
    let n_unknowns = (n_epochs - 1) + usize::from(dem_col.is_some());
    if valid_idx.len() < n_unknowns {
        return None;
    }
    let reduced: Vec<IfgPair> = valid_idx.iter().map(|&k| pairs[k]).collect();
    if !network::is_connected(&reduced, n_epochs) {
        return None;
    }
    let reduced_dem: Option<Vec<f64>> =
        dem_col.map(|g| valid_idx.iter().map(|&k| g[k]).collect());
    let a = build_design_ext(&reduced, n_epochs, reduced_dem.as_deref()).ok()?;
    let m = reduced.len();
    let svd = a.svd(true, true);
    let s_max = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let eps = s_max * (m.max(n_unknowns) as f64) * f64::EPSILON;
    svd.pseudo_inverse(eps).ok()
}

/// Píxel de referencia sugerido: prioriza la **validez temporal** (número de
/// pares con coherencia finita) por sobre la coherencia media, y usa esta
/// última solo para desempatar entre píxeles igualmente válidos; empates
/// finales → menor (fila, col) para determinismo. `coh`: pares × filas ×
/// cols. `None` si ningún píxel tiene coherencia finita. Compartido por el
/// pipeline, la CLI y los bindings.
///
/// Priorizar `n` antes que la media evita un modo de falla visto en
/// producción: un píxel de borde con coherencia finita en solo 2 de 100
/// pares (media 0.99 sobre esos 2) le ganaba a uno finito en los 100 pares
/// (media 0.95) — y como [`reference_to_pixel`] rellena con NaN cada capa
/// donde la referencia no es finita, elegir ese píxel de borde destruye 98
/// interferogramas completos para *todos* los píxeles del stack.
///
/// `region`: si se da, restringe la búsqueda a los píxeles `true` de esta
/// máscara (mismas dims que `coh`). Necesario porque el stack de entrada
/// puede cubrir un área mucho más grande que el AOI real de interés — el
/// bbox de `stackSentinel.py`/topsStack, por ejemplo, solo filtra qué bursts
/// entran, no recorta el producto final — y sin restricción la auto-selección
/// puede caer arbitrariamente lejos del área que importa, con coherencia alta
/// pero sin relación con lo que se está midiendo (visto en producción:
/// referencia a 25 km del AOI, sobre un vacío de DEM). `None` también si
/// `region` no coincide en dimensiones con `coh` o no deja ningún píxel.
pub fn select_reference_pixel(
    coh: &Array3<f32>,
    region: Option<&Array2<bool>>,
) -> Option<(usize, usize)> {
    let (n_pairs, n_rows, n_cols) = coh.dim();
    if region.is_some_and(|m| m.dim() != (n_rows, n_cols)) {
        return None;
    }
    (0..n_rows)
        .into_par_iter()
        .map(|r| {
            // Clave de comparación (n_válidos, media): el orden lexicográfico
            // de tuplas hace que `n_válidos` domine y `media` solo desempate.
            let mut best: Option<((u32, f32), (usize, usize))> = None;
            for c in 0..n_cols {
                if region.is_some_and(|m| !m[[r, c]]) {
                    continue;
                }
                let (mut sum, mut n) = (0.0_f64, 0u32);
                for k in 0..n_pairs {
                    let v = coh[[k, r, c]];
                    if v.is_finite() {
                        sum += f64::from(v);
                        n += 1;
                    }
                }
                if n > 0 {
                    let mean = (sum / f64::from(n)) as f32;
                    let key = (n, mean);
                    if best.is_none_or(|(bk, _)| key > bk) {
                        best = Some((key, (r, c)));
                    }
                }
            }
            best
        })
        .reduce(
            || None,
            |a, b| match (a, b) {
                (None, x) | (x, None) => x,
                (Some(x), Some(y)) => {
                    // Mayor (n_válidos, media) gana; empate → menor (fila, col).
                    if y.0 > x.0 || (y.0 == x.0 && y.1 < x.1) { Some(y) } else { Some(x) }
                }
            },
        )
        .map(|(_, rc)| rc)
}

/// Velocidad media LOS (m/año) por ajuste lineal de la serie de cada píxel.
/// Píxeles con < 2 épocas válidas → NaN (en MVP: cualquier NaN en la serie
/// deja el píxel en NaN). Error si la serie tiene menos de 2 épocas.
pub fn estimate_velocity(series: &DisplacementSeries) -> Result<VelocityMap> {
    series.validate()?;
    let n_epochs = series.n_layers();
    let (n_rows, n_cols) = series.dims();

    if n_epochs < 2 {
        return Err(InsarError::DimensionMismatch(format!(
            "se requieren al menos 2 épocas para estimar velocidad ({n_epochs} recibidas)"
        )));
    }

    // Tiempo en años decimales relativo a la primera época.
    let t = series.epoch_years();
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
    series.validate()?;
    let n_epochs = series.n_layers();
    let (n_rows, n_cols) = series.dims();

    if n_epochs < 3 {
        return Err(InsarError::DimensionMismatch(format!(
            "se requieren al menos 3 épocas para la incertidumbre de velocidad ({n_epochs} recibidas)"
        )));
    }

    let t = series.epoch_years();
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

/// SplitMix64: PRNG determinista de 64 bits (Steele et al. 2014), sin
/// dependencia externa. Suficiente para el remuestreo bootstrap.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Incertidumbre de la velocidad LOS por **bootstrap de épocas** (m/año):
/// para cada píxel se remuestrean con reemplazo las parejas `(t_e, d_e)` de
/// la serie, se ajusta la pendiente en cada remuestreo, y se reporta la
/// desviación estándar de las pendientes. A diferencia del SE formal de
/// [`estimate_velocity_uncertainty`] (que asume residuos i.i.d.), el
/// bootstrap es honesto frente a residuos correlacionados — p. ej. la
/// covarianza que induce la propia inversión SBAS o la atmósfera residual.
///
/// Determinista: el mismo `seed` produce el mismo resultado (el generador se
/// siembra por píxel con `seed` mezclado con (fila, col), independiente del
/// orden de los threads). Remuestreos degenerados (todas las épocas
/// sorteadas con la misma fecha) se descartan; un píxel necesita ≥2
/// remuestreos válidos para reportar SE (si no → NaN). Píxeles con NaN en la
/// serie → NaN (política MVP, como [`estimate_velocity`]).
///
/// Errores: serie con < 3 épocas, `n_resamples < 2`, o fechas todas iguales.
pub fn estimate_velocity_bootstrap(
    series: &DisplacementSeries,
    n_resamples: usize,
    seed: u64,
) -> Result<Array2<f32>> {
    series.validate()?;
    let n_epochs = series.n_layers();
    let (n_rows, n_cols) = series.dims();

    if n_epochs < 3 {
        return Err(InsarError::DimensionMismatch(format!(
            "se requieren al menos 3 épocas para el bootstrap ({n_epochs} recibidas)"
        )));
    }
    if n_resamples < 2 {
        return Err(InsarError::Metadata(format!(
            "n_resamples debe ser ≥ 2 (recibido {n_resamples}; típico 100-1000)"
        )));
    }

    let t = series.epoch_years();
    if t.windows(2).all(|w| w[0] == w[1]) {
        return Err(InsarError::Inversion(
            "todas las épocas tienen la misma fecha; el ajuste lineal es indeterminado".into(),
        ));
    }

    let mut out = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let data = series.data.view();
    let mut row_views: Vec<_> = out.axis_iter_mut(Axis(0)).collect();
    row_views.par_iter_mut().enumerate().for_each(|(r, out_row)| {
        let mut d_px = vec![0.0_f64; n_epochs];
        let mut slopes: Vec<f64> = Vec::with_capacity(n_resamples);
        for c in 0..n_cols {
            // Política MVP: cualquier NaN en la serie del píxel → NaN.
            let mut valid = true;
            for e in 0..n_epochs {
                let d = data[[e, r, c]];
                if !d.is_finite() {
                    valid = false;
                    break;
                }
                d_px[e] = d as f64;
            }
            if !valid {
                continue;
            }

            // Semilla por píxel independiente del orden de los threads.
            let mut rng = seed
                ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (c as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);

            slopes.clear();
            for _ in 0..n_resamples {
                // Remuestreo con reemplazo de las parejas (t_e, d_e).
                let (mut st, mut sd, mut stt, mut std_) = (0.0_f64, 0.0, 0.0, 0.0);
                for _ in 0..n_epochs {
                    let e = (splitmix64(&mut rng) % n_epochs as u64) as usize;
                    st += t[e];
                    sd += d_px[e];
                    stt += t[e] * t[e];
                    std_ += t[e] * d_px[e];
                }
                let n = n_epochs as f64;
                let sxx = stt - st * st / n;
                if sxx <= f64::EPSILON * stt.max(1.0) {
                    continue; // remuestreo degenerado (misma fecha repetida)
                }
                slopes.push((std_ - st * sd / n) / sxx);
            }
            if slopes.len() < 2 {
                continue;
            }
            let m = slopes.iter().sum::<f64>() / slopes.len() as f64;
            let var = slopes.iter().map(|v| (v - m).powi(2)).sum::<f64>()
                / (slopes.len() - 1) as f64;
            out_row[c] = var.sqrt() as f32;
        }
    });

    Ok(out)
}

/// Modelo temporal para el ajuste de la serie por píxel: polinomio +
/// componentes periódicas + saltos (estilo `timeseries2velocity` de MintPy).
/// El fit lineal puro sesga la velocidad donde hay estacionalidad
/// (termoelástica, hidrológica) o saltos cosísmicos; este modelo los absorbe.
#[derive(Debug, Clone)]
pub struct TemporalModel {
    /// Orden del polinomio (≥ 1; 1 = offset + velocidad).
    pub polynomial_order: usize,
    /// Periodos de las componentes sin/cos, en años (p. ej. `[1.0, 0.5]`
    /// para anual + semianual).
    pub periods_yr: Vec<f64>,
    /// Fechas de salto (Heaviside `H(t ≥ fecha)`, p. ej. un sismo). Deben
    /// caer estrictamente dentro del rango de épocas (si no, la columna es
    /// constante y colinear con el offset).
    pub steps: Vec<Epoch>,
}

impl Default for TemporalModel {
    fn default() -> Self {
        Self { polynomial_order: 1, periods_yr: Vec::new(), steps: Vec::new() }
    }
}

/// Resultado de [`fit_temporal_model`].
#[derive(Debug, Clone)]
pub struct TemporalFit {
    /// Coeficiente lineal (velocidad, m/año) con la georreferencia de la serie.
    pub velocity: VelocityMap,
    /// SE formal de la velocidad (m/año): `sqrt(σ̂²·[(GᵀG)⁻¹]_vv)`.
    pub velocity_std: Array2<f32>,
    /// Todos los coeficientes: (n_coef, filas, cols), orden de `names`.
    pub coefficients: Array3<f32>,
    /// Nombre de cada coeficiente (offset, velocity, poly2…, cos/sin_Tyr,
    /// step_YYYYMMDD).
    pub names: Vec<String>,
}

/// Pseudoinversa de `m` vía SVD con tolerancia rcond estilo LAPACK/numpy
/// (`s_max · max(filas, cols) · f64::EPSILON`). `None` si `m` es
/// rank-deficiente (menor valor singular ≤ esa tolerancia) o si la SVD no
/// converge. Compartido con [`crate::features::extract_features`] — antes
/// cada uno recalculaba la misma tolerancia por separado.
pub(crate) fn rcond_pseudo_inverse(m: DMatrix<f64>) -> Option<DMatrix<f64>> {
    let (n_rows, n_cols) = (m.nrows(), m.ncols());
    let svd = m.svd(true, true);
    let s_max = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let eps = s_max * (n_rows.max(n_cols) as f64) * f64::EPSILON;
    let s_min = svd.singular_values.iter().copied().fold(f64::MAX, f64::min);
    if s_min <= eps {
        return None;
    }
    svd.pseudo_inverse(eps).ok()
}

/// Ajusta el [`TemporalModel`] a la serie de cada píxel por mínimos cuadrados
/// (pseudoinversa de la matriz temporal G, compartida por todos los píxeles).
///
/// Política NaN (MVP): un píxel con cualquier época no finita queda NaN en
/// todos los productos. Errores: modelo inválido (orden 0, periodo ≤ 0,
/// salto fuera del rango de épocas) o menos épocas que `n_coef + 1` (se
/// exige al menos 1 grado de libertad para σ̂²).
pub fn fit_temporal_model(
    series: &DisplacementSeries,
    model: &TemporalModel,
) -> Result<TemporalFit> {
    series.validate()?;
    let n_epochs = series.n_layers();
    let (n_rows, n_cols) = series.dims();
    if model.polynomial_order == 0 {
        return Err(InsarError::Metadata(
            "polynomial_order debe ser ≥ 1 (1 = offset + velocidad)".into(),
        ));
    }
    if let Some(p) = model.periods_yr.iter().find(|p| !(p.is_finite() && **p > 0.0)) {
        return Err(InsarError::Metadata(format!("periodo inválido: {p} años")));
    }

    let t = series.epoch_years();
    let (t_first, t_last) = (series.epochs[0], series.epochs[n_epochs - 1]);
    for s in &model.steps {
        if *s <= t_first || *s > t_last {
            return Err(InsarError::Metadata(format!(
                "salto {} fuera del rango de épocas ({} – {}): su columna \
                 Heaviside sería constante (colineal con el offset)",
                s.0, t_first.0, t_last.0
            )));
        }
    }

    // Nombres y columnas del diseño temporal G (n_epochs × n_coef).
    let mut names: Vec<String> = vec!["offset".into(), "velocity".into()];
    for k in 2..=model.polynomial_order {
        names.push(format!("poly{k}"));
    }
    for p in &model.periods_yr {
        names.push(format!("cos_{p}yr"));
        names.push(format!("sin_{p}yr"));
    }
    for s in &model.steps {
        names.push(format!("step_{}", s.0.format("%Y%m%d")));
    }
    let n_coef = names.len();
    if n_epochs < n_coef + 1 {
        return Err(InsarError::DimensionMismatch(format!(
            "{n_epochs} épocas para {n_coef} coeficientes: se requiere al menos \
             n_coef + 1 (grados de libertad para σ̂²)"
        )));
    }

    let step_t: Vec<f64> = model
        .steps
        .iter()
        .map(|s| s.years_since(&series.epochs[0]))
        .collect();
    let g = DMatrix::<f64>::from_fn(n_epochs, n_coef, |e, j| {
        let te = t[e];
        if j <= model.polynomial_order {
            te.powi(j as i32) // j=0 → 1 (offset), j=1 → t (velocidad), …
        } else if j < 1 + model.polynomial_order + 2 * model.periods_yr.len() {
            let jj = j - 1 - model.polynomial_order;
            let period = model.periods_yr[jj / 2];
            let arg = 2.0 * PI * te / period;
            if jj.is_multiple_of(2) { arg.cos() } else { arg.sin() }
        } else {
            let s = j - 1 - model.polynomial_order - 2 * model.periods_yr.len();
            if te >= step_t[s] { 1.0 } else { 0.0 }
        }
    });

    // Pseudoinversa (compartida por todos los píxeles) + varianza formal del
    // coeficiente de velocidad: [(GᵀG)⁻¹]_vv.
    let pinv = rcond_pseudo_inverse(g.clone()).ok_or_else(|| {
        InsarError::Inversion(
            "matriz temporal G rank-deficiente: el modelo no es identificable \
             con estas épocas (¿periodo ≫ span temporal o saltos degenerados?)"
                .into(),
        )
    })?;
    let gtg_inv = (g.transpose() * &g)
        .try_inverse()
        .ok_or_else(|| InsarError::Inversion("GᵀG no invertible".into()))?;
    let var_vel_factor = gtg_inv[(1, 1)]; // columna 1 = velocidad

    let mut coeffs = Array3::<f32>::from_elem((n_coef, n_rows, n_cols), f32::NAN);
    let mut vel = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut vel_std = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let data = series.data.view();
    let dof = (n_epochs - n_coef) as f64;

    // Escritura en paralelo por fila: vistas disjuntas de los tres productos
    // avanzan juntas (mismo patrón que invert_sbas_ext).
    let mut coeff_rows: Vec<_> = coeffs.axis_iter_mut(Axis(1)).collect();
    let mut vel_rows: Vec<_> = vel.axis_iter_mut(Axis(0)).collect();
    let mut std_rows: Vec<_> = vel_std.axis_iter_mut(Axis(0)).collect();
    coeff_rows
        .par_iter_mut()
        .zip(vel_rows.par_iter_mut())
        .zip(std_rows.par_iter_mut())
        .enumerate()
        .for_each(|(r, ((coeff_row, vel_row), std_row))| {
            let mut d = DVector::<f64>::zeros(n_epochs);
            for c in 0..n_cols {
                let mut valid = true;
                for e in 0..n_epochs {
                    let v = data[[e, r, c]];
                    if !v.is_finite() {
                        valid = false;
                        break;
                    }
                    d[e] = f64::from(v);
                }
                if !valid {
                    continue;
                }
                let x = &pinv * &d;
                for (j, xv) in x.iter().enumerate() {
                    coeff_row[[j, c]] = *xv as f32;
                }
                vel_row[c] = x[1] as f32;
                // σ̂² = SSR/(n − p); SE(v) = sqrt(σ̂²·[(GᵀG)⁻¹]_vv).
                let mut ssr = 0.0_f64;
                for e in 0..n_epochs {
                    let mut pred = 0.0_f64;
                    for j in 0..n_coef {
                        pred += g[(e, j)] * x[j];
                    }
                    ssr += (d[e] - pred).powi(2);
                }
                std_row[c] = ((ssr / dof) * var_vel_factor).sqrt() as f32;
            }
        });
    drop(coeff_rows);
    drop(vel_rows);
    drop(std_rows);

    Ok(TemporalFit {
        velocity: VelocityMap { data: vel, meta: series.meta.clone() },
        velocity_std: vel_std,
        coefficients: coeffs,
        names,
    })
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
    // Misma validación de entrada que `invert_sbas`: sin ella, un stack con
    // pares/capas inconsistentes o una serie con capas ≠ épocas haría panic
    // en el indexado de ndarray (violando la convención sin-panic).
    stack.validate()?;
    if series.n_layers() != series.epochs.len() {
        return Err(InsarError::DimensionMismatch(format!(
            "{} épocas declaradas vs {} capas en la serie",
            series.epochs.len(),
            series.n_layers()
        )));
    }
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

    // ---------- invert_sbas_ext: WLS por coherencia ----------

    #[test]
    fn wls_con_pesos_uniformes_reproduce_ols() {
        // Coherencia constante → pesos iguales → WLS ≡ OLS.
        let stack = synthetic_stack(2, 3);
        let ols = invert_sbas(&stack, None).unwrap();
        let coh = Array3::from_elem(stack.data.dim(), 0.8_f32);
        let cfg = SbasSolverConfig {
            weighting: WeightScheme::InversePhaseVariance,
            ..Default::default()
        };
        let wls = invert_sbas_ext(&stack, None, Some(&coh), &cfg).unwrap();
        for (a, b) in wls.series.data.iter().zip(ols.data.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        assert!(wls.dem_error_m.is_none());
    }

    #[test]
    fn wls_pondera_a_la_baja_par_ruidoso() {
        // Par redundante (0,2) corrupto (+3 rad) con coherencia 0.05 frente a
        // 0.95 del resto: WLS debe acercarse a la verdad mucho más que OLS.
        let mut stack = synthetic_stack(1, 1);
        stack.data[[3, 0, 0]] += 3.0; // par (0,2)
        let d = true_displacements(&stack.epochs);

        let ols = invert_sbas(&stack, None).unwrap();
        let mut coh = Array3::from_elem(stack.data.dim(), 0.95_f32);
        coh[[3, 0, 0]] = 0.05;
        let cfg = SbasSolverConfig {
            weighting: WeightScheme::InversePhaseVariance,
            ..Default::default()
        };
        let wls = invert_sbas_ext(&stack, None, Some(&coh), &cfg).unwrap();

        let max_err = |s: &Array3<f32>| {
            (0..4)
                .map(|e| (s[[e, 0, 0]] as f64 - d[e]).abs())
                .fold(0.0_f64, f64::max)
        };
        let e_ols = max_err(&ols.data);
        let e_wls = max_err(&wls.series.data);
        assert!(e_ols > 1e-4, "el par corrupto debía sesgar OLS: {e_ols}");
        assert!(e_wls < e_ols / 5.0, "WLS no mejoró: wls={e_wls}, ols={e_ols}");
    }

    #[test]
    fn wls_sin_coherencia_es_error() {
        let stack = synthetic_stack(1, 1);
        let cfg = SbasSolverConfig {
            weighting: WeightScheme::Coherence,
            ..Default::default()
        };
        assert!(matches!(
            invert_sbas_ext(&stack, None, None, &cfg).unwrap_err(),
            InsarError::Metadata(_)
        ));
    }

    #[test]
    fn wls_coherencia_nan_excluye_el_par() {
        // Coherencia NaN en el par redundante (0,2) de un píxel: se excluye
        // (como un par faltante en OLS); la red restante conecta y recupera.
        let stack = synthetic_stack(1, 2);
        let d = true_displacements(&stack.epochs);
        let mut coh = Array3::from_elem(stack.data.dim(), 0.9_f32);
        coh[[3, 0, 0]] = f32::NAN;
        let cfg = SbasSolverConfig {
            weighting: WeightScheme::InversePhaseVariance,
            ..Default::default()
        };
        let sol = invert_sbas_ext(&stack, None, Some(&coh), &cfg).unwrap();
        for e in 0..4 {
            for c in 0..2 {
                let got = sol.series.data[[e, 0, c]] as f64;
                assert!((got - d[e]).abs() < 1e-5, "época {e}, col {c}: {got} vs {}", d[e]);
            }
        }
    }

    // ---------- invert_sbas_ext: error de DEM ----------

    const SLANT_RANGE: f64 = 850_000.0;

    /// Stack sintético con residuo topográfico: la fase de cada par incluye
    /// `φ_topo = displacement_to_phase(g_k·Δz)` con `g_k = −B⊥_k/(R·sinθ)`.
    /// Las B⊥ se eligen fuera del espacio columna de A (identificable).
    fn synthetic_stack_dem(dz_m: f64) -> UnwrappedStack {
        let mut stack = synthetic_stack(1, 2);
        let bperp = [50.0, -30.0, 80.0, -20.0, 60.0];
        for (k, p) in stack.pairs.iter_mut().enumerate() {
            p.perp_baseline_m = bperp[k];
        }
        let sin_theta = stack.meta.incidence_deg.to_radians().sin();
        for k in 0..bperp.len() {
            let g = -bperp[k] / (SLANT_RANGE * sin_theta);
            let phi_topo = displacement_to_phase(g * dz_m, SENTINEL1_WAVELENGTH_M) as f32;
            stack
                .data
                .index_axis_mut(ndarray::Axis(0), k)
                .mapv_inplace(|v| v + phi_topo);
        }
        stack
    }

    #[test]
    fn dem_error_se_recupera_y_limpia_la_serie() {
        let dz = 30.0;
        let stack = synthetic_stack_dem(dz);
        let d = true_displacements(&stack.epochs);

        let cfg = SbasSolverConfig {
            dem_error: Some(DemErrorConfig { slant_range_m: SLANT_RANGE }),
            ..Default::default()
        };
        let sol = invert_sbas_ext(&stack, None, None, &cfg).unwrap();
        let dem = sol.dem_error_m.expect("mapa Δz presente");
        for c in 0..2 {
            assert!(
                (dem[[0, c]] as f64 - dz).abs() < 0.05,
                "Δz col {c} = {} vs {dz}",
                dem[[0, c]]
            );
            for e in 0..4 {
                let got = sol.series.data[[e, 0, c]] as f64;
                assert!((got - d[e]).abs() < 1e-4, "época {e}: {got} vs {}", d[e]);
            }
        }

        // Sin la columna de DEM, el residuo topográfico sesga la serie.
        let ols = invert_sbas(&stack, None).unwrap();
        let bias = (0..4)
            .map(|e| (ols.data[[e, 0, 0]] as f64 - d[e]).abs())
            .fold(0.0_f64, f64::max);
        assert!(bias > 5e-4, "se esperaba sesgo sin corrección de DEM: {bias}");
    }

    #[test]
    fn dem_error_con_wls_tambien_recupera() {
        let dz = -12.0;
        let stack = synthetic_stack_dem(dz);
        let coh = Array3::from_elem(stack.data.dim(), 0.85_f32);
        let cfg = SbasSolverConfig {
            weighting: WeightScheme::InversePhaseVariance,
            dem_error: Some(DemErrorConfig { slant_range_m: SLANT_RANGE }),
            robust: None,
        };
        let sol = invert_sbas_ext(&stack, None, Some(&coh), &cfg).unwrap();
        let dem = sol.dem_error_m.expect("mapa Δz presente");
        assert!((dem[[0, 0]] as f64 - dz).abs() < 0.05, "Δz = {}", dem[[0, 0]]);
    }

    #[test]
    fn dem_error_sin_baselines_es_error() {
        // pairs_4ep tiene B⊥ = 0 → sin información de Δz → error claro.
        let stack = synthetic_stack(1, 1);
        let cfg = SbasSolverConfig {
            dem_error: Some(DemErrorConfig { slant_range_m: SLANT_RANGE }),
            ..Default::default()
        };
        assert!(matches!(
            invert_sbas_ext(&stack, None, None, &cfg).unwrap_err(),
            InsarError::Metadata(_)
        ));
    }

    /// Regresión A-12: si las baselines perpendiculares son EXACTAMENTE
    /// proporcionales al tiempo (deriva orbital lineal, `B⊥_k = β·Δt_k` —
    /// físicamente común), la columna de error de DEM cae exactamente en el
    /// espacio columna de la matriz de incrementos y el sistema normal
    /// (camino WLS/IRLS, `solve_normal_eqs`) es matemáticamente singular.
    /// Antes de la guarda de condicionamiento, Cholesky podía "tener éxito"
    /// con un pivote de puro redondeo y entregar un Δz de magnitud
    /// arbitraria que además corrompe los incrementos. Debe dar NaN
    /// (píxel indeterminado), no un número — con o sin ese número siendo
    /// "razonable" por casualidad.
    #[test]
    fn dem_error_colineal_con_incrementos_por_deriva_orbital_da_nan() {
        let epochs = epochs_12d(4);
        let mut pairs = pairs_4ep();
        let beta_per_day = 50.0 / 12.0; // m/día de deriva orbital sintética
        for p in pairs.iter_mut() {
            let dt_days = epochs[p.secondary].days_since(&epochs[p.reference]) as f64;
            p.perp_baseline_m = beta_per_day * dt_days;
        }
        let d = true_displacements(&epochs);
        let mut data = Array3::<f32>::zeros((pairs.len(), 1, 1));
        for (k, p) in pairs.iter().enumerate() {
            let dd = d[p.secondary] - d[p.reference];
            data[[k, 0, 0]] = (-4.0 * PI / SENTINEL1_WAVELENGTH_M * dd) as f32;
        }
        let stack = UnwrappedStack { data, epochs, pairs, meta: meta() };

        let coh = Array3::from_elem(stack.data.dim(), 0.85_f32);
        let cfg = SbasSolverConfig {
            weighting: WeightScheme::InversePhaseVariance,
            dem_error: Some(DemErrorConfig { slant_range_m: SLANT_RANGE }),
            robust: None,
        };
        let sol = invert_sbas_ext(&stack, None, Some(&coh), &cfg).unwrap();
        let dem = sol.dem_error_m.expect("mapa Δz presente");
        assert!(dem[[0, 0]].is_nan(), "Δz debería quedar NaN por colinealidad, no {}", dem[[0, 0]]);
    }

    // ---------- invert_sbas_ext: inversión robusta L1 (IRLS) ----------

    /// Stack sintético con red COMPLETA de 4 épocas (6 pares): cada
    /// incremento aparece en 3 pares → margen L1 ≥ 2:1 frente a un outlier
    /// en cualquier par (ver nota de identificabilidad en [`IrlsConfig`]).
    fn synthetic_stack_complete(rows: usize, cols: usize) -> UnwrappedStack {
        let epochs = epochs_12d(4);
        let pairs = vec![
            pair(0, 1),
            pair(0, 2),
            pair(0, 3),
            pair(1, 2),
            pair(1, 3),
            pair(2, 3),
        ];
        let d = true_displacements(&epochs);
        let mut data = Array3::<f32>::zeros((pairs.len(), rows, cols));
        for (k, p) in pairs.iter().enumerate() {
            let dd = d[p.secondary] - d[p.reference];
            let phi = (-4.0 * PI / SENTINEL1_WAVELENGTH_M * dd) as f32;
            data.index_axis_mut(ndarray::Axis(0), k).fill(phi);
        }
        UnwrappedStack { data, epochs, pairs, meta: meta() }
    }

    #[test]
    fn irls_rechaza_outlier_sin_conocer_la_coherencia() {
        // Par (0,3) corrupto (+3 rad) SIN información de coherencia, en una
        // red completa (margen 2:1): L2 reparte el error entre los
        // incrementos; IRLS (L1) concentra el residuo en el par corrupto y
        // recupera la verdad.
        let mut stack = synthetic_stack_complete(1, 2);
        stack.data[[2, 0, 0]] += 3.0; // par (0,3), píxel (0,0)
        let d = true_displacements(&stack.epochs);

        let ols = invert_sbas(&stack, None).unwrap();
        let cfg = SbasSolverConfig {
            robust: Some(IrlsConfig::default()),
            ..Default::default()
        };
        let l1 = invert_sbas_ext(&stack, None, None, &cfg).unwrap();

        let max_err = |s: &Array3<f32>, c: usize| {
            (0..4)
                .map(|e| (s[[e, 0, c]] as f64 - d[e]).abs())
                .fold(0.0_f64, f64::max)
        };
        let e_ols = max_err(&ols.data, 0);
        let e_l1 = max_err(&l1.series.data, 0);
        assert!(e_ols > 1e-4, "el outlier debía sesgar L2: {e_ols}");
        assert!(e_l1 < e_ols / 10.0, "L1 no aisló el outlier: l1={e_l1}, l2={e_ols}");
        // El píxel limpio queda igual de bien que con L2.
        assert!(max_err(&l1.series.data, 1) < 1e-5);
    }

    #[test]
    fn irls_sin_outliers_coincide_con_l2() {
        // Datos consistentes: los residuos son ~0 y el re-peso 1/max(|r|, ε)
        // es uniforme → IRLS converge a la misma solución que L2.
        let stack = synthetic_stack(2, 3);
        let l2 = invert_sbas(&stack, None).unwrap();
        let cfg = SbasSolverConfig {
            robust: Some(IrlsConfig::default()),
            ..Default::default()
        };
        let l1 = invert_sbas_ext(&stack, None, None, &cfg).unwrap();
        for (a, b) in l1.series.data.iter().zip(l2.data.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn irls_combina_con_wls_y_dem_error() {
        // Humo: IRLS + pesos por coherencia + columna de DEM conviven.
        let dz = 15.0;
        let stack = synthetic_stack_dem(dz);
        let coh = Array3::from_elem(stack.data.dim(), 0.9_f32);
        let cfg = SbasSolverConfig {
            weighting: WeightScheme::InversePhaseVariance,
            dem_error: Some(DemErrorConfig { slant_range_m: SLANT_RANGE }),
            robust: Some(IrlsConfig::default()),
        };
        let sol = invert_sbas_ext(&stack, None, Some(&coh), &cfg).unwrap();
        let dem = sol.dem_error_m.expect("mapa Δz presente");
        assert!((dem[[0, 0]] as f64 - dz).abs() < 0.05, "Δz = {}", dem[[0, 0]]);
    }

    // ---------- estimate_velocity_bootstrap ----------

    #[test]
    fn bootstrap_es_determinista_y_coherente_con_el_se_formal() {
        // Serie lineal + ruido determinista: el SE bootstrap debe ser del
        // mismo orden que el SE formal (residuos i.i.d. de verdad aquí).
        let n = 12;
        let epochs = epochs_12d(n);
        let mut seed = 7_u64;
        let mut lcg = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 40) as f64 / (1u64 << 24) as f64) - 0.5
        };
        let mut data = Array3::<f32>::zeros((n, 2, 2));
        for e in 0..n {
            let t = epochs[e].years_since(&epochs[0]);
            for r in 0..2 {
                for c in 0..2 {
                    data[[e, r, c]] = (V_TRUE * t + 0.002 * lcg()) as f32;
                }
            }
        }
        let series = DisplacementSeries { data, epochs, meta: meta() };

        let b1 = estimate_velocity_bootstrap(&series, 200, 42).unwrap();
        let b2 = estimate_velocity_bootstrap(&series, 200, 42).unwrap();
        assert_eq!(b1, b2, "mismo seed → mismo resultado");

        let formal = estimate_velocity_uncertainty(&series).unwrap();
        for r in 0..2 {
            for c in 0..2 {
                let (b, f) = (b1[[r, c]] as f64, formal[[r, c]] as f64);
                assert!(b > 0.0, "SE bootstrap debe ser > 0: {b}");
                assert!(
                    b / f < 3.0 && f / b < 3.0,
                    "bootstrap {b} vs formal {f}: difieren en más de 3×"
                );
            }
        }
    }

    #[test]
    fn bootstrap_propaga_nan_y_valida_entradas() {
        let stack = synthetic_stack(2, 2);
        let mut series = invert_sbas(&stack, None).unwrap();
        series.data[[2, 1, 0]] = f32::NAN;
        let se = estimate_velocity_bootstrap(&series, 50, 1).unwrap();
        assert!(se[[1, 0]].is_nan());
        assert!(se[[0, 0]].is_finite());

        // n_resamples < 2 → error.
        assert!(estimate_velocity_bootstrap(&series, 1, 1).is_err());
    }

    // ---------- fit_temporal_model ----------

    /// Serie sintética: v·t + A·sin(2πt) (+ salto opcional en t ≥ t_step).
    fn seasonal_series(n: usize, amp: f64, step: Option<(usize, f64)>) -> DisplacementSeries {
        let epochs = epochs_12d(n);
        let data = Array3::from_shape_fn((n, 2, 2), |(e, _, _)| {
            let t = epochs[e].years_since(&epochs[0]);
            let mut d = V_TRUE * t + amp * (2.0 * PI * t).sin();
            if let Some((e_step, jump)) = step
                && e >= e_step
            {
                d += jump;
            }
            d as f32
        });
        DisplacementSeries { data, epochs, meta: meta() }
    }

    #[test]
    fn modelo_estacional_desbiasa_la_velocidad() {
        // 40 épocas × 12 días ≈ 1.28 años: ciclo anual incompleto → el fit
        // lineal puro queda sesgado; el modelo con periodo anual lo absorbe.
        let series = seasonal_series(40, 0.02, None);

        let plain = estimate_velocity(&series).unwrap();
        let bias_plain = (plain.data[[0, 0]] as f64 - V_TRUE).abs();

        let model = TemporalModel { periods_yr: vec![1.0], ..Default::default() };
        let fit = fit_temporal_model(&series, &model).unwrap();
        let bias_model = (fit.velocity.data[[0, 0]] as f64 - V_TRUE).abs();

        assert!(bias_plain > 1e-3, "el fit lineal debía quedar sesgado: {bias_plain}");
        assert!(bias_model < 1e-5, "el modelo estacional no desbiasó: {bias_model}");
        assert_eq!(
            fit.names,
            vec!["offset", "velocity", "cos_1yr", "sin_1yr"],
            "nombres de coeficientes"
        );
        // La amplitud estacional se recupera en el coeficiente sin.
        let sin_coef = fit.coefficients[[3, 0, 0]] as f64;
        assert!((sin_coef - 0.02).abs() < 1e-4, "amplitud sin = {sin_coef}");
    }

    #[test]
    fn modelo_con_salto_recupera_velocidad_y_magnitud() {
        // Salto cosísmico de 5 cm en la época 20 de 40.
        let series = seasonal_series(40, 0.0, Some((20, 0.05)));
        let step_epoch = series.epochs[20];
        let model = TemporalModel { steps: vec![step_epoch], ..Default::default() };
        let fit = fit_temporal_model(&series, &model).unwrap();

        assert!((fit.velocity.data[[0, 0]] as f64 - V_TRUE).abs() < 1e-5);
        let jump = fit.coefficients[[2, 0, 0]] as f64; // offset, velocity, step
        assert!((jump - 0.05).abs() < 1e-4, "salto recuperado = {jump}");
        // SE ~ 0 en ajuste exacto.
        assert!(fit.velocity_std[[0, 0]] < 1e-5);
    }

    #[test]
    fn modelo_invalido_es_error() {
        let series = seasonal_series(10, 0.0, None);
        // Orden 0.
        let m = TemporalModel { polynomial_order: 0, ..Default::default() };
        assert!(fit_temporal_model(&series, &m).is_err());
        // Periodo inválido.
        let m = TemporalModel { periods_yr: vec![0.0], ..Default::default() };
        assert!(fit_temporal_model(&series, &m).is_err());
        // Salto fuera del rango (antes de la primera época).
        let early = Epoch("2020-01-01".parse().unwrap());
        let m = TemporalModel { steps: vec![early], ..Default::default() };
        assert!(fit_temporal_model(&series, &m).is_err());
        // Más coeficientes que épocas.
        let m = TemporalModel {
            periods_yr: vec![1.0, 0.5, 0.25, 0.125],
            ..Default::default()
        };
        assert!(fit_temporal_model(&series, &m).is_err());
    }

    // ---------- select_reference_pixel ----------

    #[test]
    fn referencia_elige_maxima_coherencia_media() {
        let mut coh = Array3::from_elem((2, 3, 4), 0.5_f32);
        coh[[0, 1, 2]] = 0.9;
        coh[[1, 1, 2]] = 0.9;
        assert_eq!(select_reference_pixel(&coh, None), Some((1, 2)));

        // Todo NaN → None.
        let nan = Array3::from_elem((2, 2, 2), f32::NAN);
        assert_eq!(select_reference_pixel(&nan, None), None);
    }

    #[test]
    fn referencia_respeta_region() {
        // El píxel de mejor coherencia global (1,2) queda fuera de la región;
        // dentro de la región, el mejor es (0,0).
        let mut coh = Array3::from_elem((2, 3, 4), 0.5_f32);
        coh[[0, 1, 2]] = 0.9;
        coh[[1, 1, 2]] = 0.9;
        coh[[0, 0, 0]] = 0.7;
        coh[[1, 0, 0]] = 0.7;

        let mut region = Array2::from_elem((3, 4), false);
        region[[0, 0]] = true;
        region[[0, 1]] = true;
        assert_eq!(select_reference_pixel(&coh, Some(&region)), Some((0, 0)));

        // Región sin ningún píxel true → None.
        let vacia = Array2::from_elem((3, 4), false);
        assert_eq!(select_reference_pixel(&coh, Some(&vacia)), None);

        // Región con dims distintas a `coh` → None (no se ignora en silencio).
        let mismatch = Array2::from_elem((2, 2), true);
        assert_eq!(select_reference_pixel(&coh, Some(&mismatch)), None);
    }

    /// Regresión: un píxel de borde finito en solo 2 de 10 pares (media 0.99
    /// sobre esos 2) NO debe ganarle a uno finito en los 10 pares completos
    /// (media 0.95) — la validez temporal domina sobre la coherencia media.
    /// Antes de este fix, el borde ganaba y `reference_to_pixel` destruía los
    /// 8 pares restantes para todo el stack.
    #[test]
    fn referencia_prioriza_validez_temporal_sobre_media() {
        let n_pairs = 10;
        let mut coh = Array3::from_elem((n_pairs, 2, 2), f32::NAN);
        // (0, 0): válido en los 10 pares, coherencia 0.95.
        for k in 0..n_pairs {
            coh[[k, 0, 0]] = 0.95;
        }
        // (0, 1): "borde" válido en solo 2 pares, coherencia 0.99 en esos 2.
        coh[[0, 0, 1]] = 0.99;
        coh[[1, 0, 1]] = 0.99;

        assert_eq!(select_reference_pixel(&coh, None), Some((0, 0)));
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
        let n_lost = reference_to_pixel(&mut stack, 0, 0).unwrap();
        assert_eq!(n_lost, 0, "el píxel de referencia es finito en todos los pares");
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

    /// Regresión: `reference_to_pixel` debe reportar cuántos pares quedaron
    /// en NaN por no tener fase finita en el píxel de referencia — antes este
    /// dato se perdía y la aniquilación de capas completas pasaba inadvertida.
    #[test]
    fn referencia_reporta_pares_perdidos_por_nan() {
        let mut stack = synthetic_stack(2, 2);
        // El píxel de referencia (0,0) queda NaN en el primer par únicamente.
        stack.data[[0, 0, 0]] = f32::NAN;
        let n_lost = reference_to_pixel(&mut stack, 0, 0).unwrap();
        assert_eq!(n_lost, 1);
        // Ese par queda enteramente NaN (todos los píxeles, no solo (0,0)).
        assert!(stack.data.index_axis(ndarray::Axis(0), 0).iter().all(|v| v.is_nan()));
        // El resto de los pares sí quedó referenciado (no NaN).
        for k in 1..stack.pairs.len() {
            assert!(stack.data[[k, 0, 0]].abs() < 1e-6);
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

    #[test]
    fn coherencia_temporal_stack_invalido_es_error_no_panic() {
        // Stack con más pares declarados que capas: sin validate() esto haría
        // panic en phases[[k, r, c]] con k fuera de rango.
        let mut stack = synthetic_stack(2, 3);
        let series = invert_sbas(&stack, None).unwrap();
        stack.pairs.push(pair(0, 3)); // 6 pares declarados, 5 capas
        assert!(matches!(
            temporal_coherence(&stack, &series).unwrap_err(),
            InsarError::DimensionMismatch(_)
        ));
    }

    #[test]
    fn coherencia_temporal_serie_inconsistente_es_error_no_panic() {
        // Serie con capas ≠ épocas declaradas: sin la validación, el acceso
        // disp[[p.secondary, ...]] podría indexar fuera de rango.
        let stack = synthetic_stack(2, 3);
        let series = DisplacementSeries {
            data: Array3::zeros((2, 2, 3)), // 2 capas
            epochs: epochs_12d(4),          // 4 épocas declaradas
            meta: meta(),
        };
        assert!(matches!(
            temporal_coherence(&stack, &series).unwrap_err(),
            InsarError::DimensionMismatch(_)
        ));
    }
}

//! Validación anti-circularidad de la corrección de phase bias sobre Ñuble
//! (Issue 1 y 4 del review C&G simulado). Tres experimentos:
//!
//! 1. **Calibración del nulo**: la corrección resuelve N−1 δ por píxel contra
//!    2N−5 cierres — parte de la caída del RMS es absorción de grados de
//!    libertad, no sesgo removido. Se mide corrigiendo K stacks de fase
//!    aleatoria i.i.d. (sin sesgo, misma red) con los mismos aₙ.
//! 2. **Hold-out espacial**: el sesgo real es espacialmente suave (lo dicta la
//!    cobertura); el sobreajuste por píxel no. Se corrige cada píxel con el
//!    δ̂ promedio de su bloque B×B EXCLUYENDO el propio píxel (donut): la
//!    reducción que sobrevive es genuinamente out-of-sample.
//! 3. **Sensibilidad a los aₙ**: reducción con los coeficientes estimados de
//!    la escena vs los tres sets publicados (Turquía, Campi Flegrei, Azores).
//!
//! Uso: cargo run --release --example phase_bias_validation -- <dir_GEOC>

use std::fs;
use std::path::PathBuf;

use chrono::NaiveDate;
use insar_core::io::licsar::{Aoi, LicsarLoadConfig, read_licsar_stack};
use insar_core::phase_bias::{
    PhaseBiasConfig, closure_rms, correct_phase_bias, estimate_bias_terms,
    estimate_coefficients,
};
use insar_core::types::IfgStack;
use ndarray::{Array3, Axis};
use num_complex::Complex32;

const NULL_RUNS: usize = 20;
const NULL_GRID: usize = 128;
const BLOCK: usize = 8;

/// LCG determinista (Numerical Recipes) — sin dependencia de `rand`.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32)
    }
}

fn reduction(before: f64, after: f64) -> f64 {
    100.0 * (1.0 - after / before)
}

fn corrected_reduction(
    stack: &IfgStack,
    coefficients: Vec<f64>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let cfg = PhaseBiasConfig { coefficients: Some(coefficients), ..Default::default() };
    let (before, _) = closure_rms(stack, cfg.max_span)?;
    let mut s = stack.clone();
    correct_phase_bias(&mut s, &cfg)?;
    let (after, _) = closure_rms(&s, cfg.max_span)?;
    Ok(reduction(before, after))
}

/// Media del bloque B×B del píxel, excluyendo el píxel mismo (donut). NaN si
/// el bloque no aporta ningún vecino válido.
fn donut_block_mean(d: &Array3<f32>) -> Array3<f32> {
    let (n_slots, rows, cols) = d.dim();
    let mut out = Array3::<f32>::from_elem(d.dim(), f32::NAN);
    for s in 0..n_slots {
        let plane = d.index_axis(Axis(0), s);
        let mut oplane = out.index_axis_mut(Axis(0), s);
        for br in (0..rows).step_by(BLOCK) {
            for bc in (0..cols).step_by(BLOCK) {
                let (r1, c1) = ((br + BLOCK).min(rows), (bc + BLOCK).min(cols));
                let (mut sum, mut n) = (0.0f64, 0u32);
                for r in br..r1 {
                    for c in bc..c1 {
                        let v = plane[[r, c]];
                        if v.is_finite() {
                            sum += v as f64;
                            n += 1;
                        }
                    }
                }
                for r in br..r1 {
                    for c in bc..c1 {
                        let v = plane[[r, c]];
                        // Donut: excluir la contribución propia si existe.
                        let (s2, n2) = if v.is_finite() {
                            (sum - v as f64, n - 1)
                        } else {
                            (sum, n)
                        };
                        if n2 > 0 {
                            oplane[[r, c]] = (s2 / n2 as f64) as f32;
                        }
                    }
                }
            }
        }
    }
    out
}

/// Resta a cada interferograma el sesgo del modelo evaluado con los δ dados:
/// bias(par de span n) = coef(n) · Σ δ[slots de la cadena].
fn apply_bias_terms(stack: &mut IfgStack, delta: &Array3<f32>, coefficients: &[f64]) {
    let (_, rows, cols) = stack.data.dim();
    for (k, p) in stack.pairs.clone().iter().enumerate() {
        let span = p.secondary - p.reference;
        let coef = if span == 1 {
            1.0
        } else if span - 2 < coefficients.len() {
            coefficients[span - 2]
        } else {
            continue; // span fuera del modelo: intacto
        };
        let mut layer = stack.data.index_axis_mut(Axis(0), k);
        for r in 0..rows {
            for c in 0..cols {
                let mut acc = 0.0f64;
                let mut ok = true;
                for slot in p.reference..p.secondary {
                    let v = delta[[slot, r, c]];
                    if v.is_finite() {
                        acc += v as f64;
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    let bias = (coef * acc) as f32;
                    layer[[r, c]] *= Complex32::from_polar(1.0, -bias);
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(
        std::env::args().nth(1).unwrap_or_else(|| "data/licsar_083D_12636/GEOC".to_string()),
    );
    let d = |s: &str| NaiveDate::parse_from_str(s, "%Y%m%d").unwrap();
    let aoi = Aoi { min_lon: -72.20, max_lon: -71.80, min_lat: -36.80, max_lat: -36.40 };

    // Coeficientes de escena (red completa, ancla span 12).
    let est = estimate_coefficients(
        &read_licsar_stack(&dir, &LicsarLoadConfig { aoi: Some(aoi), ..Default::default() })?,
        &PhaseBiasConfig { anchor_days: 72.0, ..Default::default() },
    )?;
    println!("coeficientes de escena: {:?}", est.coefficients);

    // Ventana hole-free.
    let load = LicsarLoadConfig {
        aoi: Some(aoi),
        date_range: Some((d("20211221"), d("20221228"))),
        ..Default::default()
    };
    let stack = read_licsar_stack(&dir, &load)?;
    let n_epochs = stack.epochs.len();
    let n_obs = (n_epochs - 2) + (n_epochs - 3);
    let n_unknowns = n_epochs - 1;
    let dof_null = 100.0 * (1.0 - (1.0 - n_unknowns as f64 / n_obs as f64).sqrt());
    println!(
        "red: {} épocas, {} cierres, {} incógnitas/píxel → nulo analítico (ruido blanco): {:.1}%",
        n_epochs, n_obs, n_unknowns, dof_null
    );

    // ── 1. Reducción real (in-sample, la del paper hasta ahora) ────────────
    let r_real = corrected_reduction(&stack, est.coefficients.clone())?;
    println!("\n[1] reducción real in-sample: {r_real:.1}%");

    // ── 2. Nulo empírico: fase aleatoria, misma red, mismos aₙ ─────────────
    let mut null_reds = Vec::with_capacity(NULL_RUNS);
    for seed in 0..NULL_RUNS as u64 {
        let mut rng = Lcg(0x9E3779B97F4A7C15 ^ (seed * 0xD1B54A32D192ED03 + 1));
        let mut data =
            Array3::<Complex32>::zeros((stack.pairs.len(), NULL_GRID, NULL_GRID));
        for v in data.iter_mut() {
            let phi = (rng.next_f32() * 2.0 - 1.0) * std::f32::consts::PI;
            *v = Complex32::from_polar(1.0, phi);
        }
        let null_stack = IfgStack {
            data,
            epochs: stack.epochs.clone(),
            pairs: stack.pairs.clone(),
            meta: stack.meta.clone(),
        };
        null_reds.push(corrected_reduction(&null_stack, est.coefficients.clone())?);
    }
    let mean = null_reds.iter().sum::<f64>() / null_reds.len() as f64;
    let sd = (null_reds.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
        / (null_reds.len() - 1) as f64)
        .sqrt();
    println!(
        "[2] nulo empírico ({} corridas, grilla {}²): {:.1}% ± {:.1}%",
        NULL_RUNS, NULL_GRID, mean, sd
    );

    // ── 3. Hold-out espacial: δ̂ del bloque donut, no del píxel ─────────────
    let cfg = PhaseBiasConfig {
        coefficients: Some(est.coefficients.clone()),
        ..Default::default()
    };
    let delta = estimate_bias_terms(&stack, &cfg)?;
    let delta_donut = donut_block_mean(&delta);
    let (before, _) = closure_rms(&stack, cfg.max_span)?;
    let mut held = stack.clone();
    apply_bias_terms(&mut held, &delta_donut, &est.coefficients);
    let (after_held, _) = closure_rms(&held, cfg.max_span)?;
    let r_holdout = reduction(before, after_held);
    println!(
        "[3] hold-out espacial (bloques {BLOCK}×{BLOCK}, donut): {r_holdout:.1}%  \
         (esto es lo genuinamente predictivo)"
    );
    // El mismo hold-out sobre el nulo: cuánto sobrevive el sobreajuste al donut.
    {
        let mut rng = Lcg(0xC0FFEE);
        let mut data =
            Array3::<Complex32>::zeros((stack.pairs.len(), NULL_GRID, NULL_GRID));
        for v in data.iter_mut() {
            let phi = (rng.next_f32() * 2.0 - 1.0) * std::f32::consts::PI;
            *v = Complex32::from_polar(1.0, phi);
        }
        let null_stack = IfgStack {
            data,
            epochs: stack.epochs.clone(),
            pairs: stack.pairs.clone(),
            meta: stack.meta.clone(),
        };
        let nd = estimate_bias_terms(&null_stack, &cfg)?;
        let ndd = donut_block_mean(&nd);
        let (nb, _) = closure_rms(&null_stack, cfg.max_span)?;
        let mut nh = null_stack.clone();
        apply_bias_terms(&mut nh, &ndd, &est.coefficients);
        let (na, _) = closure_rms(&nh, cfg.max_span)?;
        println!("    control: hold-out sobre el nulo: {:.1}%  (≈0 esperado)", reduction(nb, na));
    }

    // ── 4. Sensibilidad a los coeficientes (in-sample, solo referencia) ────
    println!("\n[4] sensibilidad a los aₙ (reducción RMS in-sample — referencia):");
    let sets: [(&str, Vec<f64>); 4] = [
        ("escena (Ñuble)", est.coefficients.clone()),
        ("Turquía 2022 (0.47, 0.31)", vec![0.47, 0.31]),
        ("Campi Flegrei (0.50, 0.36)", vec![0.50, 0.36]),
        ("Azores (0.53, 0.33)", vec![0.53, 0.33]),
    ];
    let mut sens = Vec::new();
    for (name, coefs) in &sets {
        let r = corrected_reduction(&stack, coefs.clone())?;
        println!("    {name:<28} {r:.1}%");
        sens.push((name.to_string(), r));
    }

    // ── 5. Cierre SISTEMÁTICO (medias): la métrica que separa sesgo de ruido ─
    // El sesgo es la componente sistemática del cierre; el ruido de
    // decorrelación domina el RMS por píxel pero se cancela al promediar.
    // Se evalúa con la corrección hold-out (donut): reducción del cierre
    // medio = sesgo genuinamente removido, sin sobreajuste posible.
    use insar_core::io::licsar::read_licsar_coherence;
    let coh = read_licsar_coherence(&dir, &load)?;
    let (_, rows, cols) = stack.data.dim();
    let mut mcoh = ndarray::Array2::<f32>::from_elem((rows, cols), f32::NAN);
    for r in 0..rows {
        for c in 0..cols {
            let col = coh.slice(ndarray::s![.., r, c]);
            let (mut s, mut n) = (0.0f32, 0u32);
            for &v in col {
                if v.is_finite() {
                    s += v;
                    n += 1;
                }
            }
            if n > 0 {
                mcoh[[r, c]] = s / n as f32;
            }
        }
    }

    // Cierre medio temporal por píxel (media circular sobre los cierres del
    // píxel), con signo: es lo que se acumula hacia la velocidad.
    let mean_closure_map = |s: &IfgStack| -> ndarray::Array2<f32> {
        let topo_closures = |s: &IfgStack| -> ndarray::Array2<Complex32> {
            // Reusa closure vía producto complejo por triplete: aquí lo
            // replicamos con la API pública mínima — para el driver basta
            // recomputar los cierres de span 2 y 3 a mano.
            let mut acc = ndarray::Array2::<Complex32>::zeros((rows, cols));
            let index: std::collections::HashMap<(usize, usize), usize> = s
                .pairs
                .iter()
                .enumerate()
                .map(|(k, p)| ((p.reference, p.secondary), k))
                .collect();
            for span in 2..=3usize {
                for start in 0..s.epochs.len().saturating_sub(span) {
                    let Some(&long) = index.get(&(start, start + span)) else { continue };
                    let chain: Option<Vec<usize>> = (start..start + span)
                        .map(|i| index.get(&(i, i + 1)).copied())
                        .collect();
                    let Some(chain) = chain else { continue };
                    for r in 0..rows {
                        for c in 0..cols {
                            let unit = |z: Complex32| -> Option<Complex32> {
                                let n = z.norm();
                                (n.is_finite() && n > 0.0).then(|| z / n)
                            };
                            let Some(mut z) = unit(s.data[[long, r, c]]) else { continue };
                            let mut ok = true;
                            for &k in &chain {
                                match unit(s.data[[k, r, c]]) {
                                    Some(u) => z *= u.conj(),
                                    None => {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                            if ok {
                                acc[[r, c]] += z;
                            }
                        }
                    }
                }
            }
            acc
        };
        topo_closures(s).mapv(|z| if z.norm() > 0.0 { z.arg() } else { f32::NAN })
    };

    let strata: [(&str, f32, f32); 3] =
        [("coh<0.3", 0.0, 0.3), ("0.3-0.5", 0.3, 0.5), ("coh>=0.5", 0.5, f32::INFINITY)];
    let stratum_mean = |m: &ndarray::Array2<f32>, lo: f32, hi: f32| -> (f64, usize) {
        let (mut s, mut n) = (0.0f64, 0usize);
        for ((r, c), v) in m.indexed_iter() {
            let k = mcoh[[r, c]];
            if v.is_finite() && k.is_finite() && k >= lo && k < hi {
                s += *v as f64;
                n += 1;
            }
        }
        (if n > 0 { s / n as f64 } else { f64::NAN }, n)
    };

    let m_before = mean_closure_map(&stack);
    let m_donut = mean_closure_map(&held);
    println!("\n[5] cierre medio (sistemático) por estrato, ANTES → HOLD-OUT (rad):");
    let mut systematic = Vec::new();
    for (name, lo, hi) in strata {
        let (b, n) = stratum_mean(&m_before, lo, hi);
        let (a, _) = stratum_mean(&m_donut, lo, hi);
        println!("    {name:<10} {b:+.4} → {a:+.4}   ({n} px, reducción {:.0}%)",
                 100.0 * (1.0 - a.abs() / b.abs()));
        systematic.push((name.to_string(), b, a, n));
    }
    // Equivalente en velocidad: sesgo unitario medio acumulado sobre la
    // ventana (~1 año) → mm/año LOS espurios que la corrección remueve.
    let wavelength = stack.meta.wavelength_m;
    let years = stack.epochs.last().unwrap().years_since(&stack.epochs[0]);
    println!("\n    equivalente en velocidad del δ̂ donut acumulado (mm/año LOS):");
    let mut vel_equiv = Vec::new();
    for (name, lo, hi) in strata {
        let mut cum = ndarray::Array2::<f32>::from_elem((rows, cols), f32::NAN);
        for r in 0..rows {
            for c in 0..cols {
                let (mut s, mut ok) = (0.0f64, true);
                for slot in 0..delta_donut.dim().0 {
                    let v = delta_donut[[slot, r, c]];
                    if v.is_finite() {
                        s += v as f64;
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    cum[[r, c]] = (-(wavelength / (4.0 * std::f64::consts::PI)) * s / years
                        * 1000.0) as f32;
                }
            }
        }
        let (v, n) = stratum_mean(&cum, lo, hi);
        println!("    {name:<10} {v:+.2} mm/año  ({n} px)");
        vel_equiv.push((name.to_string(), v, n));
    }

    // Sensibilidad de la métrica SISTEMÁTICA a los aₙ (hold-out por set).
    println!("\n    sensibilidad del cierre medio global al set de aₙ (hold-out):");
    let global_mean_abs = |m: &ndarray::Array2<f32>| -> f64 {
        let (mut s, mut n) = (0.0f64, 0usize);
        for v in m.iter() {
            if v.is_finite() {
                s += *v as f64;
                n += 1;
            }
        }
        (s / n as f64).abs()
    };
    let g_before = global_mean_abs(&m_before);
    let mut sens_sys = Vec::new();
    for (name, coefs) in &sets {
        let cfg_s = PhaseBiasConfig { coefficients: Some(coefs.clone()), ..Default::default() };
        let d_s = estimate_bias_terms(&stack, &cfg_s)?;
        let dd_s = donut_block_mean(&d_s);
        let mut h_s = stack.clone();
        apply_bias_terms(&mut h_s, &dd_s, coefs);
        let g_after = global_mean_abs(&mean_closure_map(&h_s));
        let red = 100.0 * (1.0 - g_after / g_before);
        println!("    {name:<28} |media| {g_before:.4} → {g_after:.4} rad  (−{red:.0}%)");
        sens_sys.push((name.to_string(), g_after, red));
    }

    // Export de mapas para la figura nueva (cierre medio antes/después +
    // equivalente en velocidad del sesgo removido).
    let dump = |path: &str, m: &ndarray::Array2<f32>| -> std::io::Result<()> {
        fs::write(path, m.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
    };
    fs::create_dir_all("validation/phase_bias_export")?;
    dump("validation/phase_bias_export/mean_closure_before.f32", &m_before)?;
    dump("validation/phase_bias_export/mean_closure_after.f32", &m_donut)?;
    {
        let mut cum = ndarray::Array2::<f32>::from_elem((rows, cols), f32::NAN);
        for r in 0..rows {
            for c in 0..cols {
                let (mut s, mut ok) = (0.0f64, true);
                for slot in 0..delta_donut.dim().0 {
                    let v = delta_donut[[slot, r, c]];
                    if v.is_finite() {
                        s += v as f64;
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    cum[[r, c]] = (-(wavelength / (4.0 * std::f64::consts::PI)) * s / years
                        * 1000.0) as f32;
                }
            }
        }
        dump("validation/phase_bias_export/vel_bias_removed_mmyr.f32", &cum)?;
    }

    let out = serde_json::json!({
        "systematic_mean_closure": systematic.iter().map(|(n, b, a, px)| serde_json::json!({
            "stratum": n, "before_rad": b, "after_holdout_rad": a, "n_px": px})).collect::<Vec<_>>(),
        "velocity_equivalent_mmyr": vel_equiv.iter().map(|(n, v, px)| serde_json::json!({
            "stratum": n, "mmyr": v, "n_px": px})).collect::<Vec<_>>(),
        "systematic_global_before_rad": g_before,
        "systematic_sensitivity": sens_sys.iter().map(|(n, a, r)| serde_json::json!({
            "set": n, "after_rad": a, "reduction_pct": r})).collect::<Vec<_>>(),
        "n_epochs": n_epochs, "n_obs": n_obs, "n_unknowns": n_unknowns,
        "dof_null_pct": dof_null,
        "real_insample_pct": r_real,
        "null_empirical_pct": { "mean": mean, "sd": sd, "runs": NULL_RUNS },
        "spatial_holdout_pct": r_holdout,
        "block": BLOCK,
        "sensitivity": sens.iter().map(|(n, r)| serde_json::json!({"set": n, "pct": r})).collect::<Vec<_>>(),
        "coefficients_scene": est.coefficients,
    });
    fs::create_dir_all("validation/phase_bias_export")?;
    fs::write(
        "validation/phase_bias_export/validation.json",
        serde_json::to_string_pretty(&out)?,
    )?;
    println!("\n→ validation/phase_bias_export/validation.json");
    Ok(())
}

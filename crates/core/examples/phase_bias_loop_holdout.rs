//! Compuerta B del paper #1 (review blind b2): una métrica que pueda fallar.
//!
//! 1. Hold-out en el ESPACIO DE LOOPS (no espacial): para cada span n ∈ {2,3}
//!    se reparten sus loops en K folds intercalados; en cada fold se quitan
//!    del stack los interferogramas largos de esos loops, se estima δ̂ con el
//!    resto y se predice el cierre retenido como (aₙ − 1)·Σδ̂. Cada loop se
//!    predice sin haber entrado al ajuste, así que la absorción de grados de
//!    libertad por mínimos cuadrados no puede inflar el resultado.
//! 2. Lo mismo con los cuatro juegos de aₙ, más la velocidad SBAS de la
//!    pantalla de corrección por píxel (definición de "velocidad espuria"
//!    como diferencia de inversiones, no Σδ̂).
//! 3. Sintéticos con la red real y ruido realista: un control positivo que
//!    sigue el modelo y dos nulos estructurados que no lo siguen (cierre igual
//!    en ambos spans; signo opuesto entre spans). En los nulos el hold-out
//!    debe fallar.
//!
//! Uso: cargo run --release --example phase_bias_loop_holdout -- <dir_GEOC>
//! Salida: validation/phase_bias_export/loop_holdout/

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use insar_core::inversion::{estimate_velocity, invert_sbas};
use insar_core::io::licsar::{Aoi, LicsarLoadConfig, read_licsar_stack};
use insar_core::phase_bias::{
    PhaseBiasConfig, correct_phase_bias, estimate_bias_terms, estimate_coefficients,
};
use insar_core::types::{IfgStack, UnwrappedStack};
use ndarray::{Array2, Array3, Axis};
use num_complex::Complex32;

const K_FOLDS: usize = 5;

thread_local! {
    static DEBUG_PX: Option<(usize, usize)> = std::env::var("DEBUG_PX").ok().and_then(|v| {
        let mut it = v.split(',').map(|x| x.parse::<usize>().ok());
        Some((it.next()??, it.next()??))
    });
}

/// LCG determinista (Numerical Recipes), sin dependencia de `rand`, con
/// Box-Muller para el ruido gaussiano.
struct Lcg(u64);
impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self, sigma: f64) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
const MAX_SPAN: usize = 3;

/// `OUTLIER_SIGMA=k` activa el enmascarado robusto (Maghsoudi 2025 §2.1).
fn outlier_sigma() -> Option<f64> {
    std::env::var("OUTLIER_SIGMA")
        .ok()
        .and_then(|v| v.parse().ok())
}

/// Acumuladores por píxel del cierre retenido: suma con signo y suma de
/// fasores (media circular), antes y después de restar la predicción.
struct Acc {
    sum_b: Array2<f64>,
    sum_a: Array2<f64>,
    cos_b: Array2<f64>,
    sin_b: Array2<f64>,
    cos_a: Array2<f64>,
    sin_a: Array2<f64>,
    n: Array2<f64>,
}

impl Acc {
    fn new(rows: usize, cols: usize) -> Self {
        let z = || Array2::<f64>::zeros((rows, cols));
        Self {
            sum_b: z(),
            sum_a: z(),
            cos_b: z(),
            sin_b: z(),
            cos_a: z(),
            sin_a: z(),
            n: z(),
        }
    }
    fn push(&mut self, r: usize, c: usize, before: f64, after: f64) {
        self.sum_b[[r, c]] += before;
        self.sum_a[[r, c]] += after;
        self.cos_b[[r, c]] += before.cos();
        self.sin_b[[r, c]] += before.sin();
        self.cos_a[[r, c]] += after.cos();
        self.sin_a[[r, c]] += after.sin();
        self.n[[r, c]] += 1.0;
    }
    /// Media de escena (sobre píxeles con loops evaluados): (aritmética antes,
    /// después, circular antes, después).
    fn scene(&self, mask: Option<&Array2<bool>>) -> (f64, f64, f64, f64) {
        let (mut sb, mut sa, mut cb, mut snb, mut ca, mut sna, mut n) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for ((r, c), &k) in self.n.indexed_iter() {
            if k == 0.0 || mask.is_some_and(|m| !m[[r, c]]) {
                continue;
            }
            sb += self.sum_b[[r, c]];
            sa += self.sum_a[[r, c]];
            cb += self.cos_b[[r, c]];
            snb += self.sin_b[[r, c]];
            ca += self.cos_a[[r, c]];
            sna += self.sin_a[[r, c]];
            n += k;
        }
        (sb / n, sa / n, snb.atan2(cb), sna.atan2(ca))
    }
}

fn wrap(x: f64) -> f64 {
    (x + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

fn unit(z: Complex32) -> Option<Complex32> {
    let n = z.norm();
    (n.is_finite() && n > 0.0).then(|| z / n)
}

/// Loops completos de span `n`: (capa larga, capas unitarias, época inicial).
fn loops(stack: &IfgStack, n: usize) -> Vec<(usize, Vec<usize>, usize)> {
    let idx: std::collections::HashMap<(usize, usize), usize> = stack
        .pairs
        .iter()
        .enumerate()
        .map(|(k, p)| ((p.reference, p.secondary), k))
        .collect();
    let mut out = Vec::new();
    for i in 0..stack.epochs.len().saturating_sub(n) {
        let Some(&long) = idx.get(&(i, i + n)) else {
            continue;
        };
        let chain: Option<Vec<usize>> = (i..i + n).map(|t| idx.get(&(t, t + 1)).copied()).collect();
        if let Some(chain) = chain {
            out.push((long, chain, i));
        }
    }
    out
}

fn without_layers(stack: &IfgStack, drop: &HashSet<usize>) -> IfgStack {
    let keep: Vec<usize> = (0..stack.n_layers())
        .filter(|k| !drop.contains(k))
        .collect();
    IfgStack {
        data: stack.data.select(Axis(0), &keep),
        epochs: stack.epochs.clone(),
        pairs: keep.iter().map(|&k| stack.pairs[k]).collect(),
        meta: stack.meta.clone(),
    }
}

/// Hold-out K-fold en el espacio de loops, spans 2 y 3. Devuelve el
/// acumulador combinado y uno por span.
fn loop_holdout(
    stack: &IfgStack,
    coefs: &[f64],
) -> Result<(Acc, Vec<Acc>), Box<dyn std::error::Error>> {
    let (rows, cols) = stack.dims();
    let mut all = Acc::new(rows, cols);
    let mut per_span = Vec::new();
    for n in 2..=MAX_SPAN {
        let mut acc = Acc::new(rows, cols);
        let ls = loops(stack, n);
        for fold in 0..K_FOLDS {
            let held: Vec<_> = ls
                .iter()
                .enumerate()
                .filter(|(j, _)| j % K_FOLDS == fold)
                .map(|(_, l)| l)
                .collect();
            let drop: HashSet<usize> = held.iter().map(|l| l.0).collect();
            let reduced = without_layers(stack, &drop);
            let cfg = PhaseBiasConfig {
                coefficients: Some(coefs.to_vec()),
                max_span: MAX_SPAN,
                outlier_sigma: outlier_sigma(),
                ..Default::default()
            };
            let delta = estimate_bias_terms(&reduced, &cfg)?;
            let a = coefs[n - 2];
            let dbg = DEBUG_PX.with(|d| *d);
            for (long, chain, i) in &held {
                for r in 0..rows {
                    for c in 0..cols {
                        let Some(mut z) = unit(stack.data[[*long, r, c]]) else {
                            continue;
                        };
                        let mut ok = true;
                        for &k in chain {
                            match unit(stack.data[[k, r, c]]) {
                                Some(u) => z *= u.conj(),
                                None => ok = false,
                            }
                        }
                        let s: f64 = (0..n).map(|t| delta[[i + t, r, c]] as f64).sum();
                        if !ok || !s.is_finite() {
                            continue;
                        }
                        let obs = z.arg() as f64;
                        let after = wrap(obs - (a - 1.0) * s);
                        if dbg == Some((r, c)) {
                            let ds: Vec<String> = (0..n)
                                .map(|t| format!("{:+.3}", delta[[i + t, r, c]]))
                                .collect();
                            println!(
                                "    dbg span{n} i={i:>2} fold{fold} obs {obs:+.3} δ̂ [{}] pred {:+.3} resid {after:+.3}",
                                ds.join(" "),
                                (a - 1.0) * s
                            );
                        }
                        acc.push(r, c, obs, after);
                        all.push(r, c, obs, after);
                    }
                }
            }
        }
        per_span.push(acc);
    }
    Ok((all, per_span))
}

/// Velocidad SBAS (mm/año) de la pantalla de corrección, por píxel, sin
/// referenciar: el efecto puro de la corrección sobre la velocidad.
fn screen_velocity(
    stack: &IfgStack,
    coefs: &[f64],
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let mut c = stack.clone();
    correct_phase_bias(
        &mut c,
        &PhaseBiasConfig {
            coefficients: Some(coefs.to_vec()),
            outlier_sigma: outlier_sigma(),
            ..Default::default()
        },
    )?;
    let (n, rows, cols) = stack.data.dim();
    let mut screen = Array3::<f32>::from_elem((n, rows, cols), f32::NAN);
    for ((k, r, col), v) in screen.indexed_iter_mut() {
        let (zb, zc) = (stack.data[[k, r, col]], c.data[[k, r, col]]);
        if zb.norm() > 0.0 && zc.norm() > 0.0 {
            *v = (zb * zc.conj()).arg();
        }
    }
    let s = UnwrappedStack {
        data: screen,
        epochs: stack.epochs.clone(),
        pairs: stack.pairs.clone(),
        meta: stack.meta.clone(),
    };
    Ok(estimate_velocity(&invert_sbas(&s, None)?)?
        .data
        .mapv(|v| v * 1000.0))
}

fn dump(dir: &Path, name: &str, a: &Array2<f64>) -> std::io::Result<()> {
    std::fs::write(
        dir.join(name),
        a.iter()
            .flat_map(|x| (*x as f32).to_le_bytes())
            .collect::<Vec<u8>>(),
    )
}

/// Stack sintético con la red real (épocas y pares) y ruido gaussiano por
/// interferograma. `closure(n, i)` fija el cierre esperado del loop (i, n):
/// las unitarias llevan δ_t (si hay) y las largas a_n·Σδ o el cierre dado.
fn synthetic(
    template: &IfgStack,
    rows: usize,
    cols: usize,
    noise: f64,
    phase: &dyn Fn(usize, usize) -> f64,
    seed: u64,
) -> IfgStack {
    let mut rng = Lcg(0x9E3779B97F4A7C15 ^ seed.wrapping_mul(0xD1B54A32D192ED03));
    let mut data = Array3::from_elem((template.n_layers(), rows, cols), Complex32::new(0.0, 0.0));
    for (k, p) in template.pairs.iter().enumerate() {
        let base = phase(p.reference, p.secondary);
        for r in 0..rows {
            for c in 0..cols {
                let ph = (base + rng.gauss(noise)) as f32;
                data[[k, r, c]] = Complex32::new(ph.cos(), ph.sin());
            }
        }
    }
    let mut meta = template.meta.clone();
    meta.transform = surtgis_core::GeoTransform::new(0.0, 0.0, 100.0, -100.0);
    IfgStack {
        data,
        epochs: template.epochs.clone(),
        pairs: template.pairs.clone(),
        meta,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "data/licsar_083D_12636/GEOC".into()),
    );
    let out = PathBuf::from(match outlier_sigma() {
        Some(k) => format!("validation/phase_bias_export/loop_holdout_robust{k}"),
        None => "validation/phase_bias_export/loop_holdout".into(),
    });
    std::fs::create_dir_all(&out)?;
    let d = |s: &str| NaiveDate::parse_from_str(s, "%Y%m%d").unwrap();
    let aoi = Aoi {
        min_lon: -72.20,
        max_lon: -71.80,
        min_lat: -36.80,
        max_lat: -36.40,
    };
    let est = estimate_coefficients(
        &read_licsar_stack(
            &dir,
            &LicsarLoadConfig {
                aoi: Some(aoi),
                ..Default::default()
            },
        )?,
        &PhaseBiasConfig {
            anchor_days: 72.0,
            ..Default::default()
        },
    )?;
    let stack = read_licsar_stack(
        &dir,
        &LicsarLoadConfig {
            aoi: Some(aoi),
            date_range: Some((d("20211221"), d("20221228"))),
            ..Default::default()
        },
    )?;
    println!(
        "{} pares, {} épocas | loops span 2: {}, span 3: {} | K = {K_FOLDS}",
        stack.n_layers(),
        stack.epochs.len(),
        loops(&stack, 2).len(),
        loops(&stack, 3).len()
    );

    let sets: [(&str, Vec<f64>); 5] = [
        ("scene", est.coefficients.clone()),
        ("fig11_12d", vec![0.55, 0.30]),
        ("turkey2022", vec![0.47, 0.31]),
        ("campi_flegrei", vec![0.50, 0.36]),
        ("azores", vec![0.53, 0.33]),
    ];
    let mut summary = serde_json::Map::new();
    println!("\n[1–2] datos reales: cierre medio retenido (rad), aritmético y circular");
    for (name, coefs) in &sets {
        let (all, per_span) = loop_holdout(&stack, coefs)?;
        let (b, a, cb, ca) = all.scene(None);
        let spans: Vec<_> = per_span.iter().map(|acc| acc.scene(None)).collect();
        println!(
            "  {name:<14} a={coefs:.3?} | todos: {b:+.4} → {a:+.4} ({:.0} %), circ {cb:+.4} → {ca:+.4} ({:.0} %) | span2 {:+.4}→{:+.4} | span3 {:+.4}→{:+.4}",
            100.0 * (1.0 - a.abs() / b.abs()),
            100.0 * (1.0 - ca.abs() / cb.abs()),
            spans[0].0,
            spans[0].1,
            spans[1].0,
            spans[1].1
        );
        let sub = out.join(name);
        std::fs::create_dir_all(&sub)?;
        for (fname, arr) in [
            ("sum_before.f32", &all.sum_b),
            ("sum_after.f32", &all.sum_a),
            ("cos_before.f32", &all.cos_b),
            ("sin_before.f32", &all.sin_b),
            ("cos_after.f32", &all.cos_a),
            ("sin_after.f32", &all.sin_a),
            ("n.f32", &all.n),
        ] {
            dump(&sub, fname, arr)?;
        }
        for (n_idx, acc) in per_span.iter().enumerate() {
            let span = n_idx + 2;
            dump(&sub, &format!("span{span}_sum_before.f32"), &acc.sum_b)?;
            dump(&sub, &format!("span{span}_sum_after.f32"), &acc.sum_a)?;
            dump(&sub, &format!("span{span}_n.f32"), &acc.n)?;
        }
        if std::env::var("ONLY_SCENE").is_ok() {
            break;
        }
        let v = screen_velocity(&stack, coefs)?;
        std::fs::write(
            sub.join("vel_screen_mmyr.f32"),
            v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
        )?;
        summary.insert(
            name.to_string(),
            serde_json::json!({
                "coefficients": coefs, "mean_before": b, "mean_after": a,
                "circ_before": cb, "circ_after": ca,
                "span2": [spans[0].0, spans[0].1], "span3": [spans[1].0, spans[1].1],
            }),
        );
    }

    if std::env::var("ONLY_SCENE").is_ok() {
        return Ok(());
    }
    // ── 3. Sintéticos con la red real ──────────────────────────────────────
    let (rows, cols, noise) = (40usize, 40usize, 0.7);
    let a_true = est.coefficients.clone();
    let n_ep = stack.epochs.len();
    // δ_t estacional (radianes por intervalo unitario), mismo orden que el
    // sesgo observado: suma anual ≈ −1.3 rad.
    let delta: Vec<f64> = (0..n_ep - 1)
        .map(|t| -0.04 - 0.03 * (2.0 * std::f64::consts::PI * t as f64 / (n_ep - 1) as f64).cos())
        .collect();
    let model = |i: usize, j: usize| -> f64 {
        let s: f64 = delta[i..j].iter().sum();
        if j - i == 1 { s } else { a_true[j - i - 2] * s }
    };
    // Nulo A: cierre igual (−0.05 rad) en todos los loops, sin estructura de aₙ.
    let null_equal = |i: usize, j: usize| -> f64 { if j - i == 1 { 0.0 } else { -0.05 } };
    // Nulo B: cierre de signo opuesto entre spans (−0.05 en span 2, +0.05 en span 3).
    let null_opposite = |i: usize, j: usize| -> f64 {
        match j - i {
            1 => 0.0,
            2 => -0.05,
            _ => 0.05,
        }
    };
    // Nulo C: sin cierre sistemático (ruido puro).
    let pure_noise = |_: usize, _: usize| -> f64 { 0.0 };
    println!("\n[3] sintéticos ({rows}×{cols}, ruido {noise} rad por interferograma, red real):");
    type Scenario<'a> = (&'a str, &'a dyn Fn(usize, usize) -> f64);
    let scen: [Scenario; 4] = [
        ("positive_model", &model),
        ("null_equal", &null_equal),
        ("null_opposite", &null_opposite),
        ("null_noise", &pure_noise),
    ];
    for (k, (name, f)) in scen.iter().enumerate() {
        let syn = synthetic(&stack, rows, cols, noise, *f, 17 + k as u64);
        let (all, per_span) = loop_holdout(&syn, &a_true)?;
        let (b, a, cb, ca) = all.scene(None);
        let spans: Vec<_> = per_span.iter().map(|acc| acc.scene(None)).collect();
        println!(
            "  {name:<15} todos: {b:+.4} → {a:+.4} ({:.0} %), circ {cb:+.4} → {ca:+.4} | span2 {:+.4}→{:+.4} | span3 {:+.4}→{:+.4}",
            100.0 * (1.0 - a.abs() / b.abs()),
            spans[0].0,
            spans[0].1,
            spans[1].0,
            spans[1].1
        );
        summary.insert(
            format!("synthetic_{name}"),
            serde_json::json!({
                "mean_before": b, "mean_after": a, "circ_before": cb, "circ_after": ca,
                "span2": [spans[0].0, spans[0].1], "span3": [spans[1].0, spans[1].1],
            }),
        );
    }
    summary.insert("rows".into(), stack.dims().0.into());
    summary.insert("cols".into(), stack.dims().1.into());
    std::fs::write(
        out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    println!("\nsalida en {}", out.display());
    Ok(())
}

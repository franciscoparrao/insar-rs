//! Datos para la figura F3 del paper #1: el sintético con ground truth que
//! demuestra el mecanismo completo del phase bias — el sesgo de cierre se
//! acumula e imita subsidencia, y la corrección recupera la velocidad exacta.
//!
//! Misma construcción que el test e2e (`tests/phase_bias_e2e.rs`): 10 épocas
//! cada 12 d, red daisy-chain spans 1–3, V_TRUE = −20 mm/año, sesgo unitario
//! 0,15 rad con a₂ = 0,47 y a₃ = 0,31 (los del paper de Maghsoudi 2022). La
//! fase nunca envuelve (|φ| ≪ π), así que el resultado no depende del
//! desenrollado: el arg del complejo ES la fase desenrollada.
//!
//! Salida: validation/phase_bias_export/synthetic.json (series verdad /
//! sesgada / corregida en mm + velocidades + RMS de cierre).
//!
//! Uso: cargo run --release --example phase_bias_synthetic_figs

use std::f64::consts::PI;
use std::fs;

use chrono::NaiveDate;
use insar_core::inversion::{estimate_velocity, invert_sbas};
use insar_core::phase_bias::{PhaseBiasConfig, closure_rms, correct_phase_bias};
use insar_core::types::{
    Epoch, IfgPair, IfgStack, SENTINEL1_WAVELENGTH_M, StackMeta, UnwrappedStack,
};
use ndarray::Array3;
use num_complex::Complex32;
use surtgis_core::GeoTransform;

const N_EPOCHS: usize = 10;
const MAX_SPAN: usize = 3;
const GRID: usize = 8;
const DAYS_STEP: i64 = 12;
const V_TRUE: f64 = -0.02; // m/año
const BIAS_UNIT_RAD: f64 = 0.15;
const A: [f64; MAX_SPAN - 1] = [0.47, 0.31];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start: NaiveDate = "2023-01-01".parse().unwrap();
    let epochs: Vec<Epoch> =
        (0..N_EPOCHS).map(|i| Epoch(start + chrono::Duration::days(DAYS_STEP * i as i64))).collect();
    let years: Vec<f64> = epochs.iter().map(|e| e.years_since(&epochs[0])).collect();

    let mut pairs = Vec::new();
    for span in 1..=MAX_SPAN {
        for i in 0..N_EPOCHS - span {
            pairs.push(IfgPair { reference: i, secondary: i + span, perp_baseline_m: 0.0 });
        }
    }

    let meta = StackMeta {
        transform: GeoTransform::new(0.0, 0.0, 30.0, -30.0),
        crs: None,
        wavelength_m: SENTINEL1_WAVELENGTH_M,
        incidence_deg: 39.0,
        heading_deg: None,
    };

    // Fase de cada par: deformación verdadera + sesgo del modelo (Eqs. 6–7).
    let phi_of = |p: &IfgPair, with_bias: bool| -> f64 {
        let dd = V_TRUE * (years[p.secondary] - years[p.reference]);
        let phi_def = -4.0 * PI / SENTINEL1_WAVELENGTH_M * dd;
        let span = p.secondary - p.reference;
        let bias = if with_bias {
            let coef = if span == 1 { 1.0 } else { A[span - 2] };
            coef * span as f64 * BIAS_UNIT_RAD
        } else {
            0.0
        };
        phi_def + bias
    };

    let make_stack = |with_bias: bool| -> IfgStack {
        let mut data = Array3::<Complex32>::zeros((pairs.len(), GRID, GRID));
        for (k, p) in pairs.iter().enumerate() {
            let phi = phi_of(p, with_bias) as f32;
            data.index_axis_mut(ndarray::Axis(0), k)
                .fill(Complex32::from_polar(1.0, phi));
        }
        IfgStack { data, epochs: epochs.clone(), pairs: pairs.clone(), meta: meta.clone() }
    };

    // |φ| ≪ π → el arg del complejo es directamente la fase desenrollada.
    let to_unwrapped = |s: &IfgStack| -> UnwrappedStack {
        UnwrappedStack {
            data: s.data.mapv(|z| z.arg()),
            epochs: s.epochs.clone(),
            pairs: s.pairs.clone(),
            meta: s.meta.clone(),
        }
    };

    let biased = make_stack(true);
    let (rms_before, _) = closure_rms(&biased, MAX_SPAN)?;

    let mut corrected = biased.clone();
    let cfg = PhaseBiasConfig {
        max_span: MAX_SPAN,
        coefficients: Some(A.to_vec()),
        ..Default::default()
    };
    correct_phase_bias(&mut corrected, &cfg)?;
    let (rms_after, _) = closure_rms(&corrected, MAX_SPAN)?;

    // Serie de desplazamiento (mm, relativa a la primera época) en el centro.
    let c = GRID / 2;
    let series_mm = |s: &IfgStack| -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let ts = invert_sbas(&to_unwrapped(s), None)?;
        Ok((0..N_EPOCHS).map(|e| ts.data[[e, c, c]] as f64 * 1000.0).collect())
    };
    let vel_mm = |s: &IfgStack| -> Result<f64, Box<dyn std::error::Error>> {
        let ts = invert_sbas(&to_unwrapped(s), None)?;
        Ok(estimate_velocity(&ts)?.data[[c, c]] as f64 * 1000.0)
    };

    let out = serde_json::json!({
        "epochs": epochs.iter().map(|e| e.0.to_string()).collect::<Vec<_>>(),
        "years": years,
        "v_true_mmyr": V_TRUE * 1000.0,
        "v_biased_mmyr": vel_mm(&biased)?,
        "v_corrected_mmyr": vel_mm(&corrected)?,
        "series_true_mm": years.iter().map(|y| V_TRUE * 1000.0 * y).collect::<Vec<_>>(),
        "series_biased_mm": series_mm(&biased)?,
        "series_corrected_mm": series_mm(&corrected)?,
        "closure_rms_before": rms_before,
        "closure_rms_after": rms_after,
        "bias_unit_rad": BIAS_UNIT_RAD,
        "coefficients": A,
    });
    fs::create_dir_all("validation/phase_bias_export")?;
    fs::write(
        "validation/phase_bias_export/synthetic.json",
        serde_json::to_string_pretty(&out)?,
    )?;
    println!(
        "v_true {:.2} | v_biased {:.2} | v_corrected {:.2} mm/año | cierre {:.3} → {:.2e} rad",
        V_TRUE * 1000.0,
        vel_mm(&biased)?,
        vel_mm(&corrected)?,
        rms_before,
        rms_after
    );
    println!("→ validation/phase_bias_export/synthetic.json");
    Ok(())
}

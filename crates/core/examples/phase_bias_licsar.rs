//! Corrección de phase bias sobre datos reales LiCSAR (frame 083D_12636,
//! valle agrícola de Ñuble/Chillán). Flujo en dos fases, como en Maghsoudi
//! et al. (2022): (1) estimar los coeficientes aₙ donde la red tiene un par
//! ancla teselable; (2) aplicarlos sobre la sub-ventana temporal densa
//! (hole-free, full-rank) y medir la caída del RMS de cierre.
//!
//! Uso: cargo run --release --example phase_bias_licsar -- <dir_GEOC>

use std::path::PathBuf;

use chrono::NaiveDate;
use insar_core::io::licsar::{Aoi, LicsarLoadConfig, read_licsar_stack};
use insar_core::phase_bias::{
    PhaseBiasConfig, closure_rms, correct_phase_bias, estimate_coefficients,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "data/licsar_083D_12636/GEOC".to_string()),
    );
    let d = |s: &str| NaiveDate::parse_from_str(s, "%Y%m%d").unwrap();

    // AOI: valle agrícola/forestal de Ñuble en torno a Chillán (~36.6°S).
    let aoi = Aoi {
        min_lon: -72.20,
        max_lon: -71.80,
        min_lat: -36.80,
        max_lat: -36.40,
    };

    // ── Fase 1: estimar coeficientes sobre toda la red descargada (tiene un
    //    par ancla teselable de span 12). ────────────────────────────────
    let load_full = LicsarLoadConfig {
        aoi: Some(aoi),
        ..LicsarLoadConfig::default()
    };
    let stack_full = read_licsar_stack(&dir, &load_full)?;
    println!(
        "Fase 1 (estimación): {} pares × {:?} px, {} épocas",
        stack_full.n_layers(),
        stack_full.dims(),
        stack_full.epochs.len()
    );
    let cfg_est = PhaseBiasConfig {
        anchor_days: 72.0,
        ..PhaseBiasConfig::default()
    };
    let est = estimate_coefficients(&stack_full, &cfg_est)?;
    println!(
        "  coeficientes aₙ = {:?}  (ancla span {} = {:.0} d, {} anclas)",
        est.coefficients
            .iter()
            .map(|c| format!("{c:.3}"))
            .collect::<Vec<_>>(),
        est.anchor_span,
        est.anchor_days,
        est.n_anchors,
    );

    // ── Fase 2: aplicar sobre la ventana densa hole-free (Dic 2021–Dic 2022)
    //    con los coeficientes ya estimados (no necesita ancla). ───────────
    let load_win = LicsarLoadConfig {
        aoi: Some(aoi),
        date_range: Some((d("20211221"), d("20221228"))),
        ..LicsarLoadConfig::default()
    };
    let mut stack = read_licsar_stack(&dir, &load_win)?;
    println!(
        "\nFase 2 (corrección): {} pares × {:?} px, {} épocas (ventana hole-free)",
        stack.n_layers(),
        stack.dims(),
        stack.epochs.len()
    );

    let cfg_corr = PhaseBiasConfig {
        coefficients: Some(est.coefficients.clone()),
        ..PhaseBiasConfig::default()
    };
    let (rms_before, n) = closure_rms(&stack, cfg_corr.max_span)?;
    println!("  RMS de cierre ANTES:  {rms_before:.4} rad  ({n} cierres)");
    let report = correct_phase_bias(&mut stack, &cfg_corr)?;
    let (rms_after, _) = closure_rms(&stack, cfg_corr.max_span)?;
    println!("  RMS de cierre DESPUÉS: {rms_after:.4} rad");
    println!(
        "\n→ reducción de cierre: {:.1}%",
        100.0 * (1.0 - rms_after / rms_before)
    );
    println!("  report: {report:?}");
    Ok(())
}

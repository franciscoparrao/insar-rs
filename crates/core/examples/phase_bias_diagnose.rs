//! Diagnóstico de phase bias sobre un directorio de interferogramas ISCE.
//!
//! Responde, sin correr el pipeline completo, las tres preguntas que deciden
//! si un stack sirve como caso de estudio de la corrección:
//!
//! 1. **¿La red admite el método?** Necesita spans 1, 2 y 3 en índice de
//!    época — con solo 1 y 2 el sistema es indeterminado para todo N
//!    (ver doc de [`insar_core::phase_bias`]).
//! 2. **¿Hay ancla para estimar los coeficientes?** Un par cuyo span sea
//!    múltiplo de mcm(1..M) y se pueda teselar con cadenas completas.
//! 3. **¿Hay sesgo que valga la pena corregir?** El RMS de la fase de cierre
//!    sobre datos reales, en radianes.
//!
//! Uso:
//! ```sh
//! cargo run --release --example phase_bias_diagnose -- <dir_interferogramas>
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use insar_core::io::isce::{read_isce_wrapped_stack, IsceLoadConfig};
use insar_core::phase_bias::{closure_rms, estimate_coefficients, PhaseBiasConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .ok_or("uso: phase_bias_diagnose <dir_interferogramas>")?
        .into();

    println!("Leyendo {} ...", dir.display());
    let config = IsceLoadConfig::default();
    let stack = read_isce_wrapped_stack(&dir, &config)?;

    let (n_rows, n_cols) = stack.dims();
    let n_epochs = stack.epochs.len();
    println!(
        "\n== Stack ==\n  {} pares · {} épocas · grilla {}×{}\n  rango: {} → {}",
        stack.pairs.len(),
        n_epochs,
        n_rows,
        n_cols,
        stack.epochs[0].0,
        stack.epochs[n_epochs - 1].0
    );

    // --- 1. Composición de la red por span (en índice de época) ---
    let mut spans: BTreeMap<usize, usize> = BTreeMap::new();
    for p in &stack.pairs {
        *spans.entry(p.secondary - p.reference).or_default() += 1;
    }
    println!("\n== Red (spans en índice de época) ==");
    for (span, n) in &spans {
        let days = stack.epochs[*span.min(&(n_epochs - 1))].days_since(&stack.epochs[0]);
        println!("  span {span:>3} : {n:>4} pares  (~{days} días)");
    }

    let has_3 = spans.contains_key(&3);
    println!(
        "\n  Método aplicable: {}",
        if has_3 {
            "SÍ (hay span 3)"
        } else {
            "NO — con solo spans 1 y 2 el sistema es indeterminado para todo N. \
             Faltan pares de span 3."
        }
    );

    // --- 2. Fracción de píxeles válidos ---
    let valid = stack
        .data
        .iter()
        .filter(|z| z.norm().is_finite() && z.norm() > 0.0)
        .count();
    let total = stack.data.len();
    println!(
        "\n== Validez ==\n  {:.1} % de las muestras tienen fase válida",
        100.0 * valid as f64 / total as f64
    );

    if !has_3 {
        println!("\nSin span 3 no se puede diagnosticar el sesgo. Fin.");
        return Ok(());
    }

    // --- 3. RMS de cierre: cuánto sesgo hay realmente ---
    let (rms, n_obs) = closure_rms(&stack, 3)?;
    println!(
        "\n== Fase de cierre (max_span 3) ==\n  RMS = {rms:.4} rad sobre {n_obs} observaciones"
    );
    println!(
        "  (referencia: en el sintético del test e2e, 0,242 rad de cierre \
         producían 8,6 mm/año de velocidad espuria)"
    );

    // --- 4. ¿Se pueden estimar los coeficientes? ---
    let pb = PhaseBiasConfig { max_span: 3, ..Default::default() };
    println!("\n== Coeficientes ==");
    match estimate_coefficients(&stack, &pb) {
        Ok(est) => {
            println!(
                "  a₁ = {:.3}  a₂ = {:.3}\n  ancla: span {} épocas (~{:.0} días), \
                 {} anclas, {} píxeles\n  cierre acumulado unitario: {:.3} rad",
                est.coefficients[0],
                est.coefficients[1],
                est.anchor_span,
                est.anchor_days,
                est.n_anchors,
                est.n_pixels,
                est.cumulative_unit_closure
            );
            println!("  (paper, Turquía: a₁ = 0,47  a₂ = 0,31)");
        }
        Err(e) => {
            println!("  NO estimables: {e}");
            println!(
                "  → se puede correr igual pasando coefficients explícitos \
                 (p. ej. los 0,47 / 0,31 del paper) y dejar la estimación \
                 como análisis de sensibilidad."
            );
        }
    }

    Ok(())
}

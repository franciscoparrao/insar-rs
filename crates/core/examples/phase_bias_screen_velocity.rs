//! Diagnóstico de la compuerta A: velocidad SBAS de la pantalla de corrección
//! sola (φ_sin_corregir − φ_corregida, envuelta, pequeña), por píxel y SIN
//! referenciar — el efecto puro de la corrección en el dominio de velocidad,
//! libre de desenrollado y de la elección del píxel de referencia. También
//! exporta la fase envuelta del píxel de referencia usado por
//! `phase_bias_velocity_unw` para inspeccionarlo.
//!
//! Uso: cargo run --release --example phase_bias_screen_velocity -- <dir_GEOC>

use std::path::PathBuf;

use chrono::NaiveDate;
use insar_core::inversion::{estimate_velocity, invert_sbas, select_reference_pixel};
use insar_core::io::licsar::{Aoi, LicsarLoadConfig, read_licsar_coherence, read_licsar_stack};
use insar_core::phase_bias::{PhaseBiasConfig, correct_phase_bias, estimate_coefficients};
use insar_core::types::UnwrappedStack;
use ndarray::Array3;

/// `PB_COEFS="a2,a3"` fija los aₙ (p. ej. Maghsoudi 2025 Fig. 11, base 12 d);
/// sin la variable se estiman de la red completa como en el paper original.
fn coefs_from_env() -> Option<Vec<f64>> {
    std::env::var("PB_COEFS").ok().map(|v| {
        v.split(',')
            .map(|x| x.trim().parse().expect("PB_COEFS"))
            .collect()
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "data/licsar_083D_12636/GEOC".into()),
    );
    let d = |s: &str| NaiveDate::parse_from_str(s, "%Y%m%d").unwrap();
    let aoi = Aoi {
        min_lon: -72.20,
        max_lon: -71.80,
        min_lat: -36.80,
        max_lat: -36.40,
    };
    let coefficients = match coefs_from_env() {
        Some(a) => a,
        None => {
            estimate_coefficients(
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
            )?
            .coefficients
        }
    };
    let load = LicsarLoadConfig {
        aoi: Some(aoi),
        date_range: Some((d("20211221"), d("20221228"))),
        ..Default::default()
    };
    let b = read_licsar_stack(&dir, &load)?;
    let coh = read_licsar_coherence(&dir, &load)?;
    let mut c = b.clone();
    correct_phase_bias(
        &mut c,
        &PhaseBiasConfig {
            coefficients: Some(coefficients.clone()),
            ..Default::default()
        },
    )?;

    let (n, rows, cols) = b.data.dim();
    let mut screen = Array3::<f32>::from_elem((n, rows, cols), f32::NAN);
    for k in 0..n {
        for r in 0..rows {
            for col in 0..cols {
                let (zb, zc) = (b.data[[k, r, col]], c.data[[k, r, col]]);
                if zb.norm() > 0.0 && zc.norm() > 0.0 {
                    screen[[k, r, col]] = (zb * zc.conj()).arg();
                }
            }
        }
    }
    let s = UnwrappedStack {
        data: screen,
        epochs: b.epochs.clone(),
        pairs: b.pairs.clone(),
        meta: b.meta.clone(),
    };
    let v = estimate_velocity(&invert_sbas(&s, None)?)?.data;

    let out = PathBuf::from(
        std::env::var("OUT_DIR")
            .unwrap_or_else(|_| "validation/phase_bias_export/unw_snaphu".into()),
    );
    std::fs::create_dir_all(&out)?;
    let bytes: Vec<u8> = v.iter().flat_map(|x| (x * 1000.0).to_le_bytes()).collect();
    std::fs::write(out.join("vel_screen_unref_mmyr.f32"), bytes)?;

    // Píxel de referencia (182, 329): fase envuelta y coherencia por par.
    let rr: usize = std::env::var("REF_R")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(182);
    let rc: usize = std::env::var("REF_C")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(329);
    println!("par        fase_ref   coh_ref   pantalla_ref");
    for k in 0..n {
        let p = &b.pairs[k];
        println!(
            "{:>2}->{:<2}  {:+8.3}  {:8.3}  {:+10.4}",
            p.reference,
            p.secondary,
            b.data[[k, rr, rc]].arg(),
            coh[[k, rr, rc]],
            s.data[[k, rr, rc]]
        );
    }
    println!(
        "velocidad de la pantalla en ref: {:+.2} mm/año",
        v[[rr, rc]] * 1000.0
    );
    let sel = select_reference_pixel(&coh, None);
    if let Some((r, c)) = sel {
        println!(
            "select_reference_pixel (librería, con filtro de saturación) → ({r}, {c}), pantalla {:+.2} mm/año",
            v[[r, c]] * 1000.0
        );
    }
    Ok(())
}

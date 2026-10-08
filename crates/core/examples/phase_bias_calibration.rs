//! Recalibración de aₙ (plan post blind b2, punto 4).
//!
//! 1. Escala absoluta: prueba `estimate_coefficients` con anclas de distinto
//!    largo, sobre la red completa y restringida a la era de 12 días, e
//!    informa qué ancla se usó o por qué no hay ancla teselable.
//! 2. Exporta los cierres envueltos por loop y por píxel de la ventana de
//!    evaluación (spans 2 y 3) para estimar el cociente (1−a₃)/(1−a₂), que es
//!    lo único que los cierres identifican: (aₙ−1)·Σδ es invariante a escalar
//!    δ por k y (aₙ−1) por 1/k.
//!
//! Uso: cargo run --release --example phase_bias_calibration -- <dir_GEOC>
//! Salida: validation/phase_bias_export/calibration/

use std::path::PathBuf;

use chrono::NaiveDate;
use insar_core::io::licsar::{Aoi, LicsarLoadConfig, read_licsar_stack};
use insar_core::phase_bias::{PhaseBiasConfig, estimate_coefficients};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "data/licsar_083D_12636/GEOC".into()),
    );
    let out = PathBuf::from("validation/phase_bias_export/calibration");
    std::fs::create_dir_all(&out)?;
    let d = |s: &str| NaiveDate::parse_from_str(s, "%Y%m%d").unwrap();
    let aoi = Aoi {
        min_lon: -72.20,
        max_lon: -71.80,
        min_lat: -36.80,
        max_lat: -36.40,
    };

    println!("[1] anclas para la escala absoluta");
    let networks = [
        ("red completa 2021–2022", None),
        (
            "era 12 d (desde 2021-12-15)",
            Some((d("20211215"), d("20221228"))),
        ),
        (
            "era 12 d (desde 2021-12-21)",
            Some((d("20211221"), d("20221228"))),
        ),
    ];
    let mut rows = Vec::new();
    for (name, range) in networks {
        let stack = read_licsar_stack(
            &dir,
            &LicsarLoadConfig {
                aoi: Some(aoi),
                date_range: range,
                ..Default::default()
            },
        )?;
        for anchor in [72.0, 120.0, 180.0, 264.0, 300.0, 348.0, 354.0] {
            let cfg = PhaseBiasConfig {
                anchor_days: anchor,
                ..Default::default()
            };
            match estimate_coefficients(&stack, &cfg) {
                Ok(e) => {
                    println!(
                        "  {name:<28} ancla pedida {anchor:>5.0} d → usada {:>5.0} d (span {}), {} anclas, {} px | a = {:.3?}",
                        e.anchor_days, e.anchor_span, e.n_anchors, e.n_pixels, e.coefficients
                    );
                    rows.push(serde_json::json!({"network": name, "requested": anchor,
                        "anchor_days": e.anchor_days, "anchor_span": e.anchor_span,
                        "n_anchors": e.n_anchors, "n_pixels": e.n_pixels, "coefficients": e.coefficients}));
                }
                Err(err) => {
                    let msg = err.to_string();
                    println!(
                        "  {name:<28} ancla pedida {anchor:>5.0} d → sin ancla: {}",
                        &msg[..msg.len().min(110)]
                    );
                }
            }
        }
    }
    std::fs::write(
        out.join("anchors.json"),
        serde_json::to_string_pretty(&rows)?,
    )?;

    println!("\n[2] cierres por loop y píxel de la ventana de evaluación");
    let stack = read_licsar_stack(
        &dir,
        &LicsarLoadConfig {
            aoi: Some(aoi),
            date_range: Some((d("20211221"), d("20221228"))),
            ..Default::default()
        },
    )?;
    let idx: std::collections::HashMap<(usize, usize), usize> = stack
        .pairs
        .iter()
        .enumerate()
        .map(|(k, p)| ((p.reference, p.secondary), k))
        .collect();
    let (rows_px, cols_px) = stack.dims();
    let mut meta = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    for n in 2..=3usize {
        for i in 0..stack.epochs.len() - n {
            let Some(&long) = idx.get(&(i, i + n)) else {
                continue;
            };
            let Some(chain) = (i..i + n)
                .map(|t| idx.get(&(t, t + 1)).copied())
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            for r in 0..rows_px {
                for c in 0..cols_px {
                    let mut z = stack.data[[long, r, c]];
                    for &k in &chain {
                        z *= stack.data[[k, r, c]].conj();
                    }
                    let v = if z.norm() > 0.0 && z.norm().is_finite() {
                        z.arg()
                    } else {
                        f32::NAN
                    };
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            meta.push(serde_json::json!({"span": n, "start": i}));
        }
    }
    std::fs::write(out.join("closures.f32"), buf)?;
    std::fs::write(
        out.join("closures.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "rows": rows_px, "cols": cols_px, "loops": meta,
            "epochs": stack.epochs.iter().map(|e| e.0.to_string()).collect::<Vec<_>>(),
        }))?,
    )?;
    println!(
        "  {} loops × {rows_px}×{cols_px} → {}",
        meta.len(),
        out.display()
    );
    Ok(())
}

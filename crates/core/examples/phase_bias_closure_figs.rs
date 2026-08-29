//! Figuras de CIERRE del paper #1 (narrativa post-pivote): mapas por píxel del
//! RMS de la fase de cierre antes/después de corregir el phase bias sobre el
//! stack real LiCSAR de Ñuble, más estadísticas estratificadas por coherencia
//! media (proxy de cobertura: cultivo/bosque decorrelacionado vs urbano/roca
//! coherente). Todo en dominio *wrapped* — nada de esto pasa por el
//! desenrollado, que es exactamente lo que hace la métrica afirmable.
//!
//! Salidas:
//! - `docs/phase_bias/figs/closure_rms_{before,after}.tif`, `closure_reduction_pct.tif`,
//!   `mean_coherence.tif` (GeoTIFF georreferenciados, para GIS)
//! - `validation/phase_bias_export/{meta.json,closure_before.f32,closure_after.f32,
//!   mean_coh.f32,closure_count.f32}` (crudos para el script de figuras Python)
//!
//! Uso: cargo run --release --example phase_bias_closure_figs -- <dir_GEOC>

use std::fs;
use std::path::PathBuf;

use chrono::NaiveDate;
use insar_core::io::licsar::{Aoi, LicsarLoadConfig, read_licsar_coherence, read_licsar_stack};
use insar_core::phase_bias::{
    PhaseBiasConfig, closure_rms_map, correct_phase_bias, estimate_coefficients,
};
use ndarray::{Array2, Array3};
use surtgis_core::io::write_geotiff;
use surtgis_core::{CRS, GeoTransform, Raster};

/// Media de coherencia por píxel (ignorando NaN).
fn mean_coherence(coh: &Array3<f32>) -> Array2<f32> {
    let (_, rows, cols) = coh.dim();
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let col = coh.slice(ndarray::s![.., r, c]);
        let (mut s, mut n) = (0.0f32, 0u32);
        for &v in col {
            if v.is_finite() {
                s += v;
                n += 1;
            }
        }
        if n > 0 { s / n as f32 } else { f32::NAN }
    })
}

fn write_map(
    v: &Array2<f32>,
    transform: GeoTransform,
    crs: Option<CRS>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (rows, cols) = v.dim();
    let mut r = Raster::from_vec(v.iter().copied().collect(), rows, cols)?;
    r.set_transform(transform);
    r.set_crs(crs);
    r.set_nodata(Some(f32::NAN));
    write_geotiff(&r, path, None)?;
    Ok(())
}

fn write_f32(path: &str, v: &Array2<f32>) -> std::io::Result<()> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write(path, bytes)
}

/// RMS agregado (media cuadrática ponderada por conteo) sobre los píxeles que
/// pasan el filtro.
fn aggregate_rms(
    map: &Array2<f32>,
    count: &Array2<u32>,
    keep: impl Fn(usize, usize) -> bool,
) -> (f64, usize) {
    let (mut sum_sq, mut n_obs, mut n_px) = (0.0f64, 0usize, 0usize);
    for ((r, c), v) in map.indexed_iter() {
        let k = count[[r, c]];
        if k > 0 && v.is_finite() && keep(r, c) {
            sum_sq += (*v as f64).powi(2) * k as f64;
            n_obs += k as usize;
            n_px += 1;
        }
    }
    if n_obs == 0 { (f64::NAN, 0) } else { ((sum_sq / n_obs as f64).sqrt(), n_px) }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(
        std::env::args().nth(1).unwrap_or_else(|| "data/licsar_083D_12636/GEOC".to_string()),
    );
    let d = |s: &str| NaiveDate::parse_from_str(s, "%Y%m%d").unwrap();
    let aoi = Aoi { min_lon: -72.20, max_lon: -71.80, min_lat: -36.80, max_lat: -36.40 };

    // Coeficientes de la red completa (con ancla teselable, span 12 ≈ 72 d).
    let est = estimate_coefficients(
        &read_licsar_stack(&dir, &LicsarLoadConfig { aoi: Some(aoi), ..Default::default() })?,
        &PhaseBiasConfig { anchor_days: 72.0, ..Default::default() },
    )?;
    println!(
        "coeficientes aₙ = {:?}  ({} anclas)",
        est.coefficients.iter().map(|c| format!("{c:.3}")).collect::<Vec<_>>(),
        est.n_anchors
    );

    // Ventana hole-free: stack + coherencia alineada.
    let load = LicsarLoadConfig {
        aoi: Some(aoi),
        date_range: Some((d("20211221"), d("20221228"))),
        ..Default::default()
    };
    let mut stack = read_licsar_stack(&dir, &load)?;
    let coh = read_licsar_coherence(&dir, &load)?;
    println!(
        "ventana: {} pares × {:?} px, {} épocas",
        stack.n_layers(),
        stack.dims(),
        stack.epochs.len()
    );

    let cfg = PhaseBiasConfig {
        coefficients: Some(est.coefficients.clone()),
        ..Default::default()
    };
    let (rms_before, count) = closure_rms_map(&stack, cfg.max_span)?;
    correct_phase_bias(&mut stack, &cfg)?;
    let (rms_after, _) = closure_rms_map(&stack, cfg.max_span)?;
    let mcoh = mean_coherence(&coh);

    // Reducción porcentual por píxel (solo donde hay cierre a ambos lados).
    let (rows, cols) = rms_before.dim();
    let mut reduction = Array2::<f32>::from_elem((rows, cols), f32::NAN);
    for r in 0..rows {
        for c in 0..cols {
            let (b, a) = (rms_before[[r, c]], rms_after[[r, c]]);
            if b.is_finite() && a.is_finite() && b > 0.0 {
                reduction[[r, c]] = 100.0 * (1.0 - a / b);
            }
        }
    }

    // ── Estadísticas: global + estratificado por coherencia media ──────────
    let (g_before, n_px) = aggregate_rms(&rms_before, &count, |_, _| true);
    let (g_after, _) = aggregate_rms(&rms_after, &count, |_, _| true);
    println!(
        "\nGLOBAL ({n_px} px): RMS {g_before:.4} → {g_after:.4} rad  (−{:.1}%)",
        100.0 * (1.0 - g_after / g_before)
    );

    println!("\nEstratificado por coherencia media (proxy de cobertura):");
    println!("{:<28} {:>8} {:>10} {:>10} {:>9}", "estrato", "px", "antes", "después", "reducc.");
    let strata: [(&str, f32, f32); 3] = [
        ("coh < 0.3 (cultivo/bosque)", 0.0, 0.3),
        ("0.3 ≤ coh < 0.5 (mixto)", 0.3, 0.5),
        ("coh ≥ 0.5 (urbano/estable)", 0.5, f32::INFINITY),
    ];
    for (name, lo, hi) in strata {
        let keep = |r: usize, c: usize| {
            let m = mcoh[[r, c]];
            m.is_finite() && m >= lo && m < hi
        };
        let (b, n) = aggregate_rms(&rms_before, &count, keep);
        let (a, _) = aggregate_rms(&rms_after, &count, keep);
        println!(
            "{name:<28} {n:>8} {b:>10.4} {a:>10.4} {:>8.1}%",
            100.0 * (1.0 - a / b)
        );
    }

    // ── GeoTIFF para GIS ───────────────────────────────────────────────────
    fs::create_dir_all("docs/phase_bias/figs")?;
    let (gt, crs) = (stack.meta.transform, stack.meta.crs.clone());
    write_map(&rms_before, gt, crs.clone(), "docs/phase_bias/figs/closure_rms_before.tif")?;
    write_map(&rms_after, gt, crs.clone(), "docs/phase_bias/figs/closure_rms_after.tif")?;
    write_map(&reduction, gt, crs.clone(), "docs/phase_bias/figs/closure_reduction_pct.tif")?;
    write_map(&mcoh, gt, crs, "docs/phase_bias/figs/mean_coherence.tif")?;

    // ── Crudos + meta.json para el script de figuras Python ────────────────
    let out = "validation/phase_bias_export";
    fs::create_dir_all(out)?;
    write_f32(&format!("{out}/closure_before.f32"), &rms_before)?;
    write_f32(&format!("{out}/closure_after.f32"), &rms_after)?;
    write_f32(&format!("{out}/mean_coh.f32"), &mcoh)?;
    let count_f32 = count.mapv(|v| v as f32);
    write_f32(&format!("{out}/closure_count.f32"), &count_f32)?;
    let meta = serde_json::json!({
        "rows": rows,
        "cols": cols,
        "n_pairs": stack.n_layers(),
        "n_epochs": stack.epochs.len(),
        "epochs": stack.epochs.iter().map(|e| e.0.to_string()).collect::<Vec<_>>(),
        "coefficients": est.coefficients,
        "rms_before": g_before,
        "rms_after": g_after,
        "geo": {
            "lon0": gt.origin_x,
            "lat0": gt.origin_y,
            "dlon": gt.pixel_width,
            "dlat": gt.pixel_height,
        },
    });
    fs::write(format!("{out}/meta.json"), serde_json::to_string_pretty(&meta)?)?;
    println!("\nGeoTIFF en docs/phase_bias/figs/ y crudos en {out}/");
    Ok(())
}

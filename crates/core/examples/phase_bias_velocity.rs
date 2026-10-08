//! Figura headline del paper #1: sesgo de VELOCIDAD por phase bias sobre datos
//! reales LiCSAR (valle agrícola de Ñuble). Procesa el mismo stack por dos
//! caminos idénticos salvo la corrección de phase bias, con desenrollado
//! GUIADO POR COHERENCIA + corrección de saltos 2π + referencia en píxel
//! estable, y mapea la diferencia de velocidad LOS enmascarada a píxeles
//! coherentes — el sesgo que imita subsidencia en cultivo/bosque.
//!
//! Uso: cargo run --release --example phase_bias_velocity -- <dir_GEOC>

use std::path::PathBuf;

use chrono::NaiveDate;
use insar_core::inversion::{estimate_velocity, invert_sbas, reference_to_pixel};
use insar_core::io::licsar::{Aoi, LicsarLoadConfig, read_licsar_coherence, read_licsar_stack};
use insar_core::phase_bias::{PhaseBiasConfig, correct_phase_bias, estimate_coefficients};
use insar_core::types::{IfgStack, UnwrappedStack, VelocityMap};
use insar_core::unwrap::unwrap_stack_min_quality;
use insar_core::unwrap_error::correct_unwrap_errors;
use ndarray::{Array2, Array3};
use surtgis_core::io::write_geotiff;
use surtgis_core::{CRS, GeoTransform, Raster};

const MIN_QUALITY: f32 = 0.5; // coherencia mínima para desenrollar / enmascarar

/// Desenrolla (guiado por coherencia) → corrige saltos 2π → referencia →
/// invierte → velocidad (m/año). Cadena idéntica en ambos caminos.
fn to_velocity(
    stack: &IfgStack,
    coh: &Array3<f32>,
    refrc: (usize, usize),
) -> Result<VelocityMap, Box<dyn std::error::Error>> {
    let mut unw: UnwrappedStack = unwrap_stack_min_quality(stack, Some(coh), Some(MIN_QUALITY))?;
    correct_unwrap_errors(&mut unw)?;
    reference_to_pixel(&mut unw, refrc.0, refrc.1)?;
    let series = invert_sbas(&unw, None)?;
    Ok(estimate_velocity(&series)?)
}

/// Media de coherencia por píxel (ignorando NaN).
fn mean_coherence(coh: &Array3<f32>) -> Array2<f32> {
    let (_, rows, cols) = coh.dim();
    let mut m = Array2::<f32>::zeros((rows, cols));
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
            m[[r, c]] = if n > 0 { s / n as f32 } else { f32::NAN };
        }
    }
    m
}

/// Píxel de referencia: el de mayor coherencia MÍNIMA sobre todas las capas
/// (coherente en todos los pares → no pierde pares al referenciar).
fn pick_reference(coh: &Array3<f32>) -> (usize, usize) {
    let (_, rows, cols) = coh.dim();
    let (mut best, mut brc) = (f32::NEG_INFINITY, (rows / 2, cols / 2));
    for r in 0..rows {
        for c in 0..cols {
            let col = coh.slice(ndarray::s![.., r, c]);
            let mut mn = f32::INFINITY;
            for &v in col {
                mn = mn.min(if v.is_finite() { v } else { 0.0 });
            }
            if mn > best {
                best = mn;
                brc = (r, c);
            }
        }
    }
    brc
}

fn write_vel(
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "data/licsar_083D_12636/GEOC".to_string()),
    );
    let d = |s: &str| NaiveDate::parse_from_str(s, "%Y%m%d").unwrap();
    let aoi = Aoi {
        min_lon: -72.20,
        max_lon: -71.80,
        min_lat: -36.80,
        max_lat: -36.40,
    };

    // Coeficientes (red completa con ancla teselable).
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
    println!("coeficientes aₙ = {:?}", est.coefficients);

    // Ventana densa hole-free: stack + coherencia alineada.
    let load = LicsarLoadConfig {
        aoi: Some(aoi),
        date_range: Some((d("20211221"), d("20221228"))),
        ..Default::default()
    };
    let stack_biased = read_licsar_stack(&dir, &load)?;
    let coh = read_licsar_coherence(&dir, &load)?;
    let mut stack_corr = stack_biased.clone();
    correct_phase_bias(
        &mut stack_corr,
        &PhaseBiasConfig {
            coefficients: Some(est.coefficients.clone()),
            ..Default::default()
        },
    )?;
    println!(
        "stacks: {} pares × {:?} px, {} épocas | coherencia {:?}",
        stack_biased.n_layers(),
        stack_biased.dims(),
        stack_biased.epochs.len(),
        coh.dim()
    );

    let refrc = pick_reference(&coh);
    let mcoh = mean_coherence(&coh);
    println!("píxel de referencia: {refrc:?} (coh mín {:.2})", {
        let col = coh.slice(ndarray::s![.., refrc.0, refrc.1]);
        col.iter()
            .cloned()
            .filter(|v| v.is_finite())
            .fold(f32::INFINITY, f32::min)
    });

    let vel_biased = to_velocity(&stack_biased, &coh, refrc)?;
    let vel_corr = to_velocity(&stack_corr, &coh, refrc)?;

    // Diferencia = sesgo de velocidad (mm/año), solo en píxeles coherentes.
    let (rows, cols) = vel_biased.data.dim();
    let mut diff = Array2::<f32>::from_elem((rows, cols), f32::NAN);
    let (mut sum, mut sum2, mut nfin, mut amax) = (0.0f64, 0.0f64, 0usize, 0.0f32);
    for r in 0..rows {
        for c in 0..cols {
            let (vb, vc) = (vel_biased.data[[r, c]], vel_corr.data[[r, c]]);
            if vb.is_finite() && vc.is_finite() && mcoh[[r, c]] >= MIN_QUALITY {
                let dd = (vb - vc) * 1000.0;
                diff[[r, c]] = dd;
                sum += dd as f64;
                sum2 += (dd * dd) as f64;
                nfin += 1;
                if dd.abs() > amax.abs() {
                    amax = dd;
                }
            }
        }
    }
    let mean = sum / nfin as f64;
    let rms = (sum2 / nfin as f64).sqrt();
    println!("\n── Sesgo de velocidad (biased − corregida), mm/año, coh≥{MIN_QUALITY} ──");
    println!("  píxeles válidos: {nfin} | media: {mean:+.3} | RMS: {rms:.3} | |máx|: {amax:.3}");
    let mut vals: Vec<f32> = diff.iter().copied().filter(|v| v.is_finite()).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |q: f64| vals[((vals.len() as f64 - 1.0) * q) as usize];
    println!(
        "  mediana: {:+.3} | P10: {:+.3} | P90: {:+.3}",
        pct(0.5),
        pct(0.10),
        pct(0.90)
    );

    std::fs::create_dir_all("docs/phase_bias/figs")?;
    let (gt, crs) = (vel_biased.meta.transform, vel_biased.meta.crs.clone());
    write_vel(
        &vel_biased.data,
        gt,
        crs.clone(),
        "docs/phase_bias/figs/vel_biased_mpyr.tif",
    )?;
    write_vel(
        &vel_corr.data,
        gt,
        crs.clone(),
        "docs/phase_bias/figs/vel_corrected_mpyr.tif",
    )?;
    write_vel(&diff, gt, crs, "docs/phase_bias/figs/vel_bias_mmpyr.tif")?;
    println!("\nGeoTIFF escritos en docs/phase_bias/figs/");
    Ok(())
}

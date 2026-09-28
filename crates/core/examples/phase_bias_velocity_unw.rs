//! Compuerta A del paper #1 (review blind b2): velocidad LOS real sobre Ñuble,
//! sin corregir vs corregida por phase bias, con desenrollador seleccionable
//! (SNAPHU o guiado por coherencia). Además de las velocidades, mide cuántos
//! píxeles cambian de solución entera de desenrollado al aplicar la corrección
//! — la hipótesis detrás del salto de ~64 mm/año en CLL1 con flood-fill.
//!
//! Uso:
//!   SNAPHU_BIN=/ruta/a/snaphu cargo run --release --example phase_bias_velocity_unw -- \
//!       <dir_GEOC> <snaphu|quality>
//!
//! Salida en `validation/phase_bias_export/unw_<método>/`: velocidades (m/año)
//! y cambios de ciclo por píxel como f32 crudo + `summary.json`.

use std::f32::consts::PI;
use std::path::PathBuf;

use chrono::NaiveDate;
use insar_core::inversion::{estimate_velocity, invert_sbas, reference_to_pixel, select_reference_pixel};
use insar_core::io::licsar::{Aoi, LicsarLoadConfig, read_licsar_coherence, read_licsar_stack};
use insar_core::phase_bias::{PhaseBiasConfig, correct_phase_bias, estimate_coefficients};
use insar_core::types::{IfgStack, UnwrappedStack};
use insar_core::unwrap::snaphu::{SnaphuConfig, unwrap_stack_snaphu};
use insar_core::unwrap::unwrap_stack_min_quality;
use insar_core::unwrap_error::correct_unwrap_errors;
use ndarray::{Array2, Array3};

const MIN_QUALITY: f32 = 0.5;

fn unwrap(
    stack: &IfgStack,
    coh: &Array3<f32>,
    method: &str,
) -> Result<UnwrappedStack, Box<dyn std::error::Error>> {
    Ok(match method {
        "snaphu" => {
            let binary = std::env::var("SNAPHU_BIN").unwrap_or_else(|_| "snaphu".into());
            unwrap_stack_snaphu(stack, Some(coh), &SnaphuConfig { binary: PathBuf::from(binary) })?
        }
        "quality" => unwrap_stack_min_quality(stack, Some(coh), Some(MIN_QUALITY))?,
        m => return Err(format!("método desconocido: {m} (snaphu|quality)").into()),
    })
}

/// Cadena idéntica en ambos caminos: corrección de saltos 2π → referencia →
/// inversión SBAS → velocidad (m/año).
fn to_velocity(
    mut unw: UnwrappedStack,
    refrc: (usize, usize),
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    correct_unwrap_errors(&mut unw)?;
    reference_to_pixel(&mut unw, refrc.0, refrc.1)?;
    Ok(estimate_velocity(&invert_sbas(&unw, None)?)?.data)
}

/// Cambios de solución entera entre los desenrollados crudos de ambos
/// caminos. La corrección solo resta una pantalla envuelta pequeña, así que
/// `unw_b − unw_c − wrap(φ_b − φ_c)` debe ser un múltiplo de 2π constante por
/// capa (el offset global arbitrario del desenrollador); todo píxel que se
/// aparta de la moda de su capa cambió de ciclo por culpa de la corrección.
/// Devuelve (capas cambiadas por píxel, fracción de celdas cambiadas por capa).
fn cycle_changes(
    b: &IfgStack,
    c: &IfgStack,
    ub: &UnwrappedStack,
    uc: &UnwrappedStack,
) -> (Array2<f32>, Vec<f64>) {
    let (n, rows, cols) = ub.data.dim();
    let mut per_pixel = Array2::<f32>::zeros((rows, cols));
    let mut per_layer = Vec::with_capacity(n);
    for k in 0..n {
        let mut cyc = Array2::<i32>::from_elem((rows, cols), i32::MIN);
        let mut hist = std::collections::HashMap::<i32, usize>::new();
        for r in 0..rows {
            for col in 0..cols {
                let (xb, xc) = (ub.data[[k, r, col]], uc.data[[k, r, col]]);
                let (zb, zc) = (b.data[[k, r, col]], c.data[[k, r, col]]);
                if !(xb.is_finite() && xc.is_finite() && zb.norm() > 0.0 && zc.norm() > 0.0) {
                    continue;
                }
                let screen = (zb * zc.conj()).arg();
                let m = ((xb - xc - screen) / (2.0 * PI)).round() as i32;
                cyc[[r, col]] = m;
                *hist.entry(m).or_default() += 1;
            }
        }
        let Some((&mode, _)) = hist.iter().max_by_key(|(_, n)| **n) else {
            per_layer.push(f64::NAN);
            continue;
        };
        let (mut changed, mut valid) = (0usize, 0usize);
        for r in 0..rows {
            for col in 0..cols {
                let m = cyc[[r, col]];
                if m == i32::MIN {
                    continue;
                }
                valid += 1;
                if m != mode {
                    changed += 1;
                    per_pixel[[r, col]] += 1.0;
                }
            }
        }
        per_layer.push(changed as f64 / valid.max(1) as f64);
    }
    (per_pixel, per_layer)
}

fn write_f32(a: &Array2<f32>, path: &std::path::Path) -> std::io::Result<()> {
    let bytes: Vec<u8> = a.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().unwrap_or_else(|| "data/licsar_083D_12636/GEOC".into()));
    let method = args.next().unwrap_or_else(|| "snaphu".into());
    let d = |s: &str| NaiveDate::parse_from_str(s, "%Y%m%d").unwrap();
    let aoi = Aoi { min_lon: -72.20, max_lon: -71.80, min_lat: -36.80, max_lat: -36.40 };

    // Coeficientes: mismos que el paper (red completa, ancla de 72 d).
    let est = estimate_coefficients(
        &read_licsar_stack(&dir, &LicsarLoadConfig { aoi: Some(aoi), ..Default::default() })?,
        &PhaseBiasConfig { anchor_days: 72.0, ..Default::default() },
    )?;
    println!("coeficientes aₙ = {:?}", est.coefficients);

    let load = LicsarLoadConfig {
        aoi: Some(aoi),
        date_range: Some((d("20211221"), d("20221228"))),
        ..Default::default()
    };
    let stack_b = read_licsar_stack(&dir, &load)?;
    let coh = read_licsar_coherence(&dir, &load)?;
    let mut stack_c = stack_b.clone();
    correct_phase_bias(
        &mut stack_c,
        &PhaseBiasConfig { coefficients: Some(est.coefficients.clone()), ..Default::default() },
    )?;
    let refrc = select_reference_pixel(&coh, None).ok_or("sin píxel de referencia válido")?;
    let gt = stack_b.meta.transform;
    let ref_lon = gt.origin_x + (refrc.1 as f64 + 0.5) * gt.pixel_width;
    let ref_lat = gt.origin_y + (refrc.0 as f64 + 0.5) * gt.pixel_height;
    println!(
        "{} pares × {:?} px, {} épocas | desenrollador: {method} | referencia {refrc:?} ({ref_lon:.5}, {ref_lat:.5})",
        stack_b.n_layers(),
        stack_b.dims(),
        stack_b.epochs.len()
    );

    let t0 = std::time::Instant::now();
    let unw_b = unwrap(&stack_b, &coh, &method)?;
    let unw_c = unwrap(&stack_c, &coh, &method)?;
    println!("desenrollado ({method}, 2 caminos): {:.1} s", t0.elapsed().as_secs_f64());

    let (changed_px, changed_layer) = cycle_changes(&stack_b, &stack_c, &unw_b, &unw_c);
    let frac_mean = changed_layer.iter().filter(|v| v.is_finite()).sum::<f64>()
        / changed_layer.iter().filter(|v| v.is_finite()).count().max(1) as f64;
    let frac_max = changed_layer.iter().cloned().filter(|v| v.is_finite()).fold(0.0, f64::max);
    println!(
        "cambio de ciclo por la corrección: media {:.2} % de celdas por capa, máx {:.2} %",
        100.0 * frac_mean,
        100.0 * frac_max
    );

    // Tercer camino: corrección DESPUÉS del desenrollado. Resta la misma
    // pantalla envuelta (pequeña, |aₙΣδ̂| ≪ π) al desenrollado sin corregir,
    // sin re-desenrollar: aísla el efecto de la corrección del de los cambios
    // de ciclo que provoca aplicarla antes del desenrollado.
    let mut unw_p = unw_b.clone();
    let (n, rows, cols) = unw_p.data.dim();
    for k in 0..n {
        for r in 0..rows {
            for col in 0..cols {
                let (zb, zc) = (stack_b.data[[k, r, col]], stack_c.data[[k, r, col]]);
                let x = &mut unw_p.data[[k, r, col]];
                if x.is_finite() && zb.norm() > 0.0 && zc.norm() > 0.0 {
                    *x -= (zb * zc.conj()).arg();
                }
            }
        }
    }

    let out = PathBuf::from(format!("validation/phase_bias_export/unw_{method}"));
    std::fs::create_dir_all(&out)?;
    for (name, u) in [("unw_biased", &unw_b), ("unw_corrected", &unw_c)] {
        let bytes: Vec<u8> = u.data.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(out.join(format!("{name}.f32")), bytes)?;
    }

    let vel_b = to_velocity(unw_b, refrc)?;
    let vel_c = to_velocity(unw_c, refrc)?;
    let vel_p = to_velocity(unw_p, refrc)?;

    write_f32(&vel_b, &out.join("vel_biased_mpyr.f32"))?;
    write_f32(&vel_c, &out.join("vel_corrected_mpyr.f32"))?;
    write_f32(&vel_p, &out.join("vel_postcorr_mpyr.f32"))?;
    write_f32(&changed_px, &out.join("cycle_changes_layers.f32"))?;
    let summary = serde_json::json!({
        "method": method,
        "coefficients": est.coefficients,
        "rows": vel_b.nrows(), "cols": vel_b.ncols(),
        "geo": {"lon0": gt.origin_x, "lat0": gt.origin_y,
                "dlon": gt.pixel_width, "dlat": gt.pixel_height},
        "reference_rc": [refrc.0, refrc.1],
        "reference_lonlat": [ref_lon, ref_lat],
        "n_pairs": stack_b.n_layers(),
        "cycle_change_frac_per_layer": changed_layer,
        "pairs": stack_b.pairs.iter().map(|p| format!("{}_{}", p.reference, p.secondary)).collect::<Vec<_>>(),
    });
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    println!("salida en {}", out.display());
    Ok(())
}

//! Genera un stack InSAR sintético consumible por el CLI / `io::read_ifg_stack`.
//!
//! Crea un directorio con `stack.json` + GeoTIFF (pares re/im y amplitudes)
//! que reproduce una deformación gaussiana lineal en el tiempo sobre una
//! grilla 24×24 (6 épocas Sentinel-1, EPSG:32719, pixel 30 m). La fase de
//! cada par se mantiene por debajo de π en toda la grilla, de modo que el
//! pipeline SBAS la recupera sin depender del desenrollado.
//!
//! Uso:
//!
//! ```text
//! cargo run -p insar-core --example synthetic_stack -- /tmp/stack
//! ```
//!
//! Sin argumento, escribe en `./synthetic_stack`.

use std::f64::consts::PI;
use std::fs;
use std::path::Path;

use insar_core::types::SENTINEL1_WAVELENGTH_M;
use surtgis_core::io::write_geotiff;
use surtgis_core::{Raster, CRS, GeoTransform};

const N_EPOCHS: usize = 6;
const GRID: usize = 24;
const PIXEL_M: f64 = 30.0;
const ORIGIN_X: f64 = 500_000.0;
const ORIGIN_Y: f64 = 7_000_000.0;
const EPSG: u32 = 32719;
const CENTER: f64 = 12.0;
const SIGMA_PX: f64 = 5.0;
const V_OBJETIVO: f64 = -0.06; // m/año en el centro.

fn epoch_dates() -> Vec<chrono::NaiveDate> {
    let start: chrono::NaiveDate = "2023-01-01".parse().expect("fecha base válida");
    (0..N_EPOCHS)
        .map(|i| start + chrono::Duration::days(12 * i as i64))
        .collect()
}

fn years_since_start(dates: &[chrono::NaiveDate], i: usize) -> f64 {
    (dates[i] - dates[0]).num_days() as f64 / 365.25
}

fn gaussian(r: usize, c: usize) -> f64 {
    let dr = r as f64 - CENTER;
    let dc = c as f64 - CENTER;
    (-(dr * dr + dc * dc) / (2.0 * SIGMA_PX * SIGMA_PX)).exp()
}

fn displacement(dates: &[chrono::NaiveDate], r: usize, c: usize, i: usize) -> f64 {
    V_OBJETIVO * gaussian(r, c) * years_since_start(dates, i)
}

/// Red SBAS conexa: consecutivos (i, i+1) y saltos de 2 (i, i+2).
fn build_pairs() -> Vec<(usize, usize, f64)> {
    let baseline = |i: usize| 20.0 * i as f64;
    let mut pairs = Vec::new();
    for i in 0..N_EPOCHS - 1 {
        pairs.push((i, i + 1, baseline(i + 1) - baseline(i)));
    }
    for i in 0..N_EPOCHS - 2 {
        pairs.push((i, i + 2, baseline(i + 2) - baseline(i)));
    }
    pairs
}

fn write_tif(path: &Path, data: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let mut raster = Raster::from_vec(data, GRID, GRID)?;
    raster.set_transform(GeoTransform::new(ORIGIN_X, ORIGIN_Y, PIXEL_M, -PIXEL_M));
    raster.set_crs(Some(CRS::from_epsg(EPSG)));
    raster.set_nodata(Some(f32::NAN));
    write_geotiff(&raster, path, None)?;
    Ok(())
}

fn generate(dir: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    fs::create_dir_all(dir)?;
    let dates = epoch_dates();
    let pairs = build_pairs();

    let mut ifg_entries = String::new();
    for (idx, &(reference, secondary, perp)) in pairs.iter().enumerate() {
        let mut re = vec![0.0f32; GRID * GRID];
        let mut im = vec![0.0f32; GRID * GRID];
        for r in 0..GRID {
            for c in 0..GRID {
                let dd = displacement(&dates, r, c, secondary)
                    - displacement(&dates, r, c, reference);
                let phi = -4.0 * PI / SENTINEL1_WAVELENGTH_M * dd;
                let k = r * GRID + c;
                re[k] = phi.cos() as f32;
                im[k] = phi.sin() as f32;
            }
        }
        write_tif(&dir.join(format!("ifg_{idx}_re.tif")), re)?;
        write_tif(&dir.join(format!("ifg_{idx}_im.tif")), im)?;

        if idx > 0 {
            ifg_entries.push_str(",\n");
        }
        ifg_entries.push_str(&format!(
            "    {{\"reference\": {reference}, \"secondary\": {secondary}, \
             \"perp_baseline_m\": {perp}, \"file\": \"ifg_{idx}.tif\"}}"
        ));
    }

    let mut amp_list = String::new();
    for i in 0..N_EPOCHS {
        let mut amp = vec![0.0f32; GRID * GRID];
        for r in 0..GRID {
            for c in 0..GRID {
                let jitter = ((i * 7 + r * 13 + c * 17) % 11) as f32 * 1e-5;
                amp[r * GRID + c] = 1.0 + jitter;
            }
        }
        write_tif(&dir.join(format!("amp_{i}.tif")), amp)?;
        if i > 0 {
            amp_list.push_str(", ");
        }
        amp_list.push_str(&format!("\"amp_{i}.tif\""));
    }

    let epochs_iso = dates
        .iter()
        .map(|d| format!("\"{}\"", d.format("%Y-%m-%d")))
        .collect::<Vec<_>>()
        .join(", ");

    let manifest = format!(
        "{{\n  \"wavelength_m\": {SENTINEL1_WAVELENGTH_M},\n  \
         \"incidence_deg\": 39.0,\n  \"heading_deg\": null,\n  \
         \"epochs\": [{epochs_iso}],\n  \"ifgs\": [\n{ifg_entries}\n  ],\n  \
         \"amplitudes\": [{amp_list}]\n}}"
    );
    fs::write(dir.join("stack.json"), manifest)?;

    Ok(pairs.len())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./synthetic_stack".to_string());
    let dir = Path::new(&out);

    let n_pairs = generate(dir)?;

    println!("Stack InSAR sintético generado:");
    println!("  Ruta:    {}", dir.display());
    println!("  Épocas:  {N_EPOCHS} (cada 12 días desde 2023-01-01)");
    println!("  Pares:   {n_pairs} (consecutivos + saltos de 2, red conexa)");
    println!("  Grilla:  {GRID}×{GRID} px, pixel {PIXEL_M} m, EPSG:{EPSG}");
    println!("  Señal:   deformación gaussiana lineal, v central = {V_OBJETIVO} m/año");
    println!();
    println!("Procesar con: run_sbas apuntando input_dir a esta ruta.");

    Ok(())
}

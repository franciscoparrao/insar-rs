//! Test de integración end-to-end del pipeline SBAS.
//!
//! Genera un stack sintético de Sentinel-1 (6 épocas, grilla 24×24) con un
//! campo de deformación gaussiano lineal en el tiempo, lo escribe a disco en
//! el formato `stack.json` + GeoTIFF del módulo `io`, corre [`run_sbas`] y
//! verifica que la velocidad LOS recuperada coincide con la sintética en el
//! centro (deformación) y en las esquinas (sin deformación).
//!
//! ## Diseño del stack sintético
//!
//! - 6 épocas cada 12 días desde 2023-01-01; grilla 24×24, EPSG:32719,
//!   pixel 30 m.
//! - Red conexa: pares consecutivos (i, i+1) MÁS saltos de 2 (i, i+2).
//! - Campo LOS: `d(r,c,t) = pico·exp(-((r-rc)²+(c-cc)²)/(2σ²))·t_años`,
//!   con `pico` elegido para que la velocidad central sea `V_OBJETIVO`.
//! - Fase por par: `φ = -4π/λ·(d_sec − d_ref)`.
//!
//! ## No-wrapping (justificación)
//!
//! El salto temporal máximo de un par es 24 días (saltos de 2 épocas). Como
//! la deformación es lineal, el incremento máximo de `d` entre las dos épocas
//! de un par ocurre en el centro y vale `|V_OBJETIVO|·(24/365.25)` años.
//! Con V_OBJETIVO = -0.06 m/año y λ de Sentinel-1:
//!
//!   |φ_max| = 4π/λ · 0.06 · (24/365.25) ≈ 0.89 rad < π
//!
//! Por lo tanto la fase NUNCA envuelve en toda la grilla y el resultado no
//! depende del desenrollado en los bordes. V_OBJETIVO = -0.06 m/año se usa
//! tal cual (sin reducir pico ni sigma).

use std::f64::consts::PI;
use std::fs;
use std::path::Path;

use insar_core::atmosphere::ApsConfig;
use insar_core::network::SbasConfig;
use insar_core::pipeline::{run_sbas, SbasPipelineConfig};
use insar_core::types::SENTINEL1_WAVELENGTH_M;

use surtgis_core::io::{read_geotiff, write_geotiff};
use surtgis_core::{Raster, CRS, GeoTransform};

// --- Parámetros del stack sintético (compartidos con el ejemplo) ---

const N_EPOCHS: usize = 6;
const GRID: usize = 24;
const PIXEL_M: f64 = 30.0;
const ORIGIN_X: f64 = 500_000.0;
const ORIGIN_Y: f64 = 7_000_000.0;
const EPSG: u32 = 32719;
const CENTER: f64 = 12.0; // (rc, cc)
const SIGMA_PX: f64 = 5.0;
const V_OBJETIVO: f64 = -0.06; // m/año en el centro de la deformación.

/// Épocas cada 12 días desde 2023-01-01, en formato ISO.
fn epoch_dates() -> Vec<chrono::NaiveDate> {
    let start: chrono::NaiveDate = "2023-01-01".parse().unwrap();
    (0..N_EPOCHS)
        .map(|i| start + chrono::Duration::days(12 * i as i64))
        .collect()
}

/// Años decimales de la época `i` respecto a la época 0.
fn years_since_start(dates: &[chrono::NaiveDate], i: usize) -> f64 {
    (dates[i] - dates[0]).num_days() as f64 / 365.25
}

/// Factor gaussiano espacial (1 en el centro, →0 en los bordes).
fn gaussian(r: usize, c: usize) -> f64 {
    let dr = r as f64 - CENTER;
    let dc = c as f64 - CENTER;
    (-(dr * dr + dc * dc) / (2.0 * SIGMA_PX * SIGMA_PX)).exp()
}

/// Desplazamiento LOS sintético en (r, c) a la época `i` (relativo a época 0).
/// `pico` se calibra para que la velocidad central sea V_OBJETIVO: como el
/// factor gaussiano vale 1 en el centro, `pico = V_OBJETIVO`.
fn displacement(dates: &[chrono::NaiveDate], r: usize, c: usize, i: usize) -> f64 {
    let pico = V_OBJETIVO;
    pico * gaussian(r, c) * years_since_start(dates, i)
}

/// Red SBAS conexa: consecutivos (i, i+1) y saltos de 2 (i, i+2).
/// `perp_baseline_m` del par = b[sec] − b[ref] con b[i] = 20·i metros.
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

/// Escribe un GeoTIFF f32 de 1 banda con la georef del stack sintético.
fn write_tif(path: &Path, data: Vec<f32>) {
    let mut raster = Raster::from_vec(data, GRID, GRID).expect("raster from_vec");
    raster.set_transform(GeoTransform::new(ORIGIN_X, ORIGIN_Y, PIXEL_M, -PIXEL_M));
    raster.set_crs(Some(CRS::from_epsg(EPSG)));
    raster.set_nodata(Some(f32::NAN));
    write_geotiff(&raster, path, None).expect("escribir GeoTIFF de prueba");
}

/// Genera el stack sintético completo en `dir` (manifest + tifs re/im + amps).
fn generate_synthetic_stack(dir: &Path) {
    fs::create_dir_all(dir).expect("crear dir de entrada");
    let dates = epoch_dates();
    let pairs = build_pairs();

    // Interferogramas: re = cos(φ), im = sin(φ).
    let mut ifg_entries = String::new();
    for (idx, &(reference, secondary, perp)) in pairs.iter().enumerate() {
        let mut re = vec![0.0f32; GRID * GRID];
        let mut im = vec![0.0f32; GRID * GRID];
        for r in 0..GRID {
            for c in 0..GRID {
                let dd = displacement(&dates, r, c, secondary) - displacement(&dates, r, c, reference);
                let phi = -4.0 * PI / SENTINEL1_WAVELENGTH_M * dd;
                let k = r * GRID + c;
                re[k] = phi.cos() as f32;
                im[k] = phi.sin() as f32;
            }
        }
        write_tif(&dir.join(format!("ifg_{idx}_re.tif")), re);
        write_tif(&dir.join(format!("ifg_{idx}_im.tif")), im);

        if idx > 0 {
            ifg_entries.push_str(",\n");
        }
        ifg_entries.push_str(&format!(
            "    {{\"reference\": {reference}, \"secondary\": {secondary}, \
             \"perp_baseline_m\": {perp}, \"file\": \"ifg_{idx}.tif\"}}"
        ));
    }

    // Amplitudes: estables (≈1.0) con ruido determinista minúsculo para
    // dispersión baja → toda la grilla queda como PS con umbral 0.5.
    let mut amp_list = String::new();
    for i in 0..N_EPOCHS {
        let mut amp = vec![0.0f32; GRID * GRID];
        for r in 0..GRID {
            for c in 0..GRID {
                // Ruido determinista pequeño (~1e-4) dependiente de (i, r, c).
                let jitter = ((i * 7 + r * 13 + c * 17) % 11) as f32 * 1e-5;
                amp[r * GRID + c] = 1.0 + jitter;
            }
        }
        write_tif(&dir.join(format!("amp_{i}.tif")), amp);
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
    fs::write(dir.join("stack.json"), manifest).expect("escribir stack.json");
}

/// Lee `velocity.tif` de vuelta como `Raster<f32>`.
fn read_velocity(path: &Path) -> Raster<f32> {
    read_geotiff::<f32, _>(path, None).expect("re-leer velocity.tif")
}

fn pipeline_config(input: &Path, output: &Path, ps_threshold: Option<f32>) -> SbasPipelineConfig {
    SbasPipelineConfig {
        input_dir: input.to_path_buf(),
        output_dir: output.to_path_buf(),
        ps_threshold,
        network: SbasConfig::default(),
        aps: ApsConfig::default(),
    }
}

#[test]
fn pipeline_sbas_e2e_recupera_velocidad_sintetica() {
    let base = std::env::temp_dir().join(format!("insar_e2e_{}", std::process::id()));
    let _ = fs::remove_dir_all(&base);

    let result = std::panic::catch_unwind(|| run_e2e(&base));

    // Limpieza del dir temporal pase lo que pase.
    let _ = fs::remove_dir_all(&base);

    if let Err(payload) = result {
        std::panic::resume_unwind(payload);
    }
}

/// Cuerpo del test; aislado para garantizar la limpieza del dir temporal.
fn run_e2e(base: &Path) {
    let input = base.join("input");
    generate_synthetic_stack(&input);

    let tol = 0.005_f64; // m/año

    // --- Caso 1: ps_threshold = None (invertir toda la grilla) ---
    let output_full = base.join("output_full");
    let products = run_sbas(&pipeline_config(&input, &output_full, None))
        .expect("run_sbas (ps=None) debe tener éxito");

    let vel_path = output_full.join("velocity.tif");
    let series_dir = output_full.join("series");
    assert!(vel_path.exists(), "velocity.tif debe existir");
    assert!(series_dir.is_dir(), "series/ debe existir");
    assert_eq!(
        fs::read_dir(&series_dir).unwrap().count(),
        N_EPOCHS,
        "series/ debe tener un disp_*.tif por época"
    );

    // El producto en memoria y el archivo deben coincidir.
    let velocity = read_velocity(&vel_path);
    let center = CENTER as usize; // (12, 12)
    let v_center = velocity.data()[[center, center]] as f64;
    assert!(
        (v_center - V_OBJETIVO).abs() < tol,
        "velocidad central {v_center} vs objetivo {V_OBJETIVO} (tol {tol})"
    );
    assert!(
        (products.velocity.data[[center, center]] as f64 - V_OBJETIVO).abs() < tol,
        "el producto en memoria debe coincidir con el archivo en el centro"
    );

    // Las 4 esquinas: sin deformación (gaussiana ≈ 0) → |v| < tol.
    for &(r, c) in &[(0, 0), (0, GRID - 1), (GRID - 1, 0), (GRID - 1, GRID - 1)] {
        let v = velocity.data()[[r, c]] as f64;
        assert!(
            v.abs() < tol,
            "esquina ({r},{c}): |v| = {} debe ser < {tol}",
            v.abs()
        );
    }

    // --- Caso 2: ps_threshold = Some(0.5) ---
    // Las amplitudes son ≈uniformes (ruido ~1e-4) → dispersión muy baja, por
    // lo que toda la grilla resulta PS con umbral 0.5 y la velocidad central
    // se recupera igual. Documentado: con esta calibración no hay píxeles
    // no-PS; el assert relevante es la recuperación del centro.
    let output_ps = base.join("output_ps");
    let products_ps = run_sbas(&pipeline_config(&input, &output_ps, Some(0.5)))
        .expect("run_sbas (ps=Some(0.5)) debe tener éxito");

    let vel_ps = read_velocity(&output_ps.join("velocity.tif"));
    let v_center_ps = vel_ps.data()[[center, center]] as f64;
    assert!(
        (v_center_ps - V_OBJETIVO).abs() < tol,
        "PS: velocidad central {v_center_ps} vs objetivo {V_OBJETIVO} (tol {tol})"
    );
    assert!(
        (products_ps.velocity.data[[center, center]] as f64 - V_OBJETIVO).abs() < tol,
        "PS: producto en memoria debe coincidir en el centro"
    );
}

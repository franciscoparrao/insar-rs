//! Test de integración end-to-end de la corrección de **phase bias** dentro
//! del pipeline SBAS.
//!
//! Es la prueba del argumento central: que el sesgo de no-cierre, si no se
//! corrige, **sesga la velocidad** que sale del pipeline, y que la corrección
//! nativa la recupera.
//!
//! ## Diseño
//!
//! Stack sintético de 10 épocas cada 12 días, grilla 8×8, con red daisy-chain
//! de spans 1, 2 y 3 (la red LiCSAR que asume Maghsoudi et al. 2022, y el
//! mínimo con el que el sistema de cierres queda determinado).
//!
//! La fase de cada par es `φ_{i,j} = φ_def + aₙ · Σ δ`, o sea deformación
//! verdadera **más** el sesgo del modelo del paper (Eqs. 6–7) con
//! a₁ = 0,47 y a₂ = 0,31. El sesgo unitario es constante y **espacialmente
//! uniforme**: como este stack no declara coherencia, el pipeline no referencia
//! espacialmente, así que el sesgo sobrevive como un offset de velocidad — que
//! es exactamente el modo de falla que el paper describe ("the bias for
//! cropland mimics subsidence throughout the year").
//!
//! ## No-wrapping
//!
//! Span máximo 3 = 36 días. Con V = −0,02 m/año la fase de deformación máxima
//! por par es 4π/λ · 0,02 · 36/365,25 ≈ 0,45 rad, y el sesgo de span 3 vale
//! a₂·3·δ = 0,31·0,45 ≈ 0,14 rad. El total queda muy por debajo de π: la fase
//! nunca envuelve y el resultado no depende del desenrollado.

use std::f64::consts::PI;
use std::fs;
use std::path::Path;

use insar_core::phase_bias::PhaseBiasConfig;
use insar_core::pipeline::{run_sbas, SbasPipelineConfig};
use insar_core::types::SENTINEL1_WAVELENGTH_M;

use surtgis_core::io::write_geotiff;
use surtgis_core::{Raster, CRS, GeoTransform};

const N_EPOCHS: usize = 10;
const MAX_SPAN: usize = 3;
const GRID: usize = 8;
const PIXEL_M: f64 = 30.0;
const EPSG: u32 = 32719;
const DAYS_STEP: i64 = 12;

/// Velocidad LOS verdadera, uniforme en la grilla (m/año).
const V_TRUE: f64 = -0.02;
/// Sesgo verdadero de cada par de span 1, en radianes.
const BIAS_UNIT_RAD: f64 = 0.15;
/// Coeficientes del modelo (a₂, a₃ en la nomenclatura de span; a₁ ≡ 1).
const A: [f64; MAX_SPAN - 1] = [0.47, 0.31];

fn epoch_dates() -> Vec<chrono::NaiveDate> {
    let start: chrono::NaiveDate = "2023-01-01".parse().unwrap();
    (0..N_EPOCHS)
        .map(|i| start + chrono::Duration::days(DAYS_STEP * i as i64))
        .collect()
}

fn years_since_start(dates: &[chrono::NaiveDate], i: usize) -> f64 {
    (dates[i] - dates[0]).num_days() as f64 / 365.25
}

/// Red daisy-chain de spans 1..=MAX_SPAN.
fn build_pairs() -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for span in 1..=MAX_SPAN {
        for i in 0..N_EPOCHS - span {
            pairs.push((i, i + span));
        }
    }
    pairs
}

fn write_tif(path: &Path, data: Vec<f32>) {
    let mut raster = Raster::from_vec(data, GRID, GRID).expect("raster from_vec");
    raster.set_transform(GeoTransform::new(500_000.0, 7_000_000.0, PIXEL_M, -PIXEL_M));
    raster.set_crs(Some(CRS::from_epsg(EPSG)));
    raster.set_nodata(Some(f32::NAN));
    write_geotiff(&raster, path, None).expect("escribir GeoTIFF de prueba");
}

/// Genera el stack. Con `with_bias = false` escribe solo la deformación (la
/// verdad de referencia); con `true`, deformación + sesgo del modelo.
fn generate_stack(dir: &Path, with_bias: bool) {
    fs::create_dir_all(dir).expect("crear dir de entrada");
    let dates = epoch_dates();
    let pairs = build_pairs();

    let mut ifg_entries = String::new();
    for (idx, &(reference, secondary)) in pairs.iter().enumerate() {
        let span = secondary - reference;
        let dd = V_TRUE * (years_since_start(&dates, secondary) - years_since_start(&dates, reference));
        let phi_def = -4.0 * PI / SENTINEL1_WAVELENGTH_M * dd;
        let bias = if with_bias {
            let coef = if span == 1 { 1.0 } else { A[span - 2] };
            coef * span as f64 * BIAS_UNIT_RAD
        } else {
            0.0
        };
        let phi = phi_def + bias;

        let re = vec![phi.cos() as f32; GRID * GRID];
        let im = vec![phi.sin() as f32; GRID * GRID];
        write_tif(&dir.join(format!("ifg_{idx}_re.tif")), re);
        write_tif(&dir.join(format!("ifg_{idx}_im.tif")), im);

        if idx > 0 {
            ifg_entries.push_str(",\n");
        }
        ifg_entries.push_str(&format!(
            "    {{\"reference\": {reference}, \"secondary\": {secondary}, \
             \"perp_baseline_m\": 0.0, \"file\": \"ifg_{idx}.tif\"}}"
        ));
    }

    let epochs_iso = dates
        .iter()
        .map(|d| format!("\"{}\"", d.format("%Y-%m-%d")))
        .collect::<Vec<_>>()
        .join(", ");
    let manifest = format!(
        "{{\n  \"wavelength_m\": {SENTINEL1_WAVELENGTH_M},\n  \
         \"incidence_deg\": 39.0,\n  \"heading_deg\": null,\n  \
         \"epochs\": [{epochs_iso}],\n  \"ifgs\": [\n{ifg_entries}\n  ]\n}}"
    );
    fs::write(dir.join("stack.json"), manifest).expect("escribir stack.json");
}

fn config(input: &Path, output: &Path, phase_bias: Option<PhaseBiasConfig>) -> SbasPipelineConfig {
    SbasPipelineConfig {
        phase_bias,
        ..SbasPipelineConfig::new(input.to_path_buf(), output.to_path_buf())
    }
}

fn pb_config() -> PhaseBiasConfig {
    PhaseBiasConfig {
        max_span: MAX_SPAN,
        coefficients: Some(A.to_vec()),
        ..Default::default()
    }
}

#[test]
fn phase_bias_e2e_recupera_la_velocidad_sesgada() {
    let base = std::env::temp_dir().join(format!("insar_pb_e2e_{}", std::process::id()));
    let _ = fs::remove_dir_all(&base);
    let result = std::panic::catch_unwind(|| run_e2e(&base));
    let _ = fs::remove_dir_all(&base);
    if let Err(payload) = result {
        std::panic::resume_unwind(payload);
    }
}

fn run_e2e(base: &Path) {
    let center = GRID / 2;

    // --- Verdad de referencia: stack sin sesgo, pipeline sin corrección ---
    let clean_in = base.join("clean");
    generate_stack(&clean_in, false);
    let clean = run_sbas(&config(&clean_in, &base.join("out_clean"), None))
        .expect("pipeline sobre el stack limpio");
    let v_true = clean.velocity.data[[center, center]] as f64;
    assert!(
        (v_true - V_TRUE).abs() < 1e-3,
        "el caso limpio debe recuperar V_TRUE: {v_true} vs {V_TRUE}"
    );
    assert!(clean.phase_bias_report.is_none(), "sin config, sin reporte");

    // --- Stack sesgado, SIN corregir: la velocidad queda sesgada ---
    let biased_in = base.join("biased");
    generate_stack(&biased_in, true);
    let biased = run_sbas(&config(&biased_in, &base.join("out_biased"), None))
        .expect("pipeline sobre el stack sesgado");
    let v_biased = biased.velocity.data[[center, center]] as f64;
    let err_biased = (v_biased - V_TRUE).abs();
    assert!(
        err_biased > 5e-3,
        "el sesgo debe contaminar la velocidad de forma medible: \
         v_sesgada = {v_biased}, v_true = {V_TRUE}, error = {err_biased}"
    );

    // --- Stack sesgado, CON corrección: la velocidad se recupera ---
    let corrected = run_sbas(&config(
        &biased_in,
        &base.join("out_corrected"),
        Some(pb_config()),
    ))
    .expect("pipeline con corrección de phase bias");
    let v_corrected = corrected.velocity.data[[center, center]] as f64;
    let err_corrected = (v_corrected - V_TRUE).abs();

    let report = corrected
        .phase_bias_report
        .expect("con config, el reporte debe estar presente");

    // Visible con `cargo test -- --nocapture`: las magnitudes del caso
    // sintético, útiles como referencia al interpretar los casos reales.
    println!(
        "v_true = {:.4} m/año | v_sesgada = {:.4} (error {:.1} mm/año) | \
         v_corregida = {:.4} (error {:.2} mm/año) | cierre RMS {:.4} → {:.2e} rad",
        V_TRUE,
        v_biased,
        err_biased * 1000.0,
        v_corrected,
        err_corrected * 1000.0,
        report.closure_rms_before,
        report.closure_rms_after
    );
    assert_eq!(report.pixels_corrected, GRID * GRID, "toda la grilla es válida");
    assert_eq!(report.pixels_skipped, 0);
    assert_eq!(report.coefficients, A.to_vec());
    assert!(
        report.closure_rms_after < report.closure_rms_before,
        "el RMS de cierre debe bajar: {} → {}",
        report.closure_rms_before,
        report.closure_rms_after
    );

    assert!(
        err_corrected < 1e-3,
        "tras corregir, la velocidad debe volver a V_TRUE: \
         v_corregida = {v_corrected}, v_true = {V_TRUE}, error = {err_corrected}"
    );
    assert!(
        err_corrected < err_biased / 10.0,
        "la corrección debe reducir el error al menos un orden de magnitud: \
         {err_biased} → {err_corrected}"
    );
}

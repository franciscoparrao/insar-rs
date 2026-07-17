//! Smoke test de la CLI: verifica el wiring binario→core sobre un stack
//! sintético mínimo (parseo clap, lectura del stack, salida legible).
//! Usa el binario compilado vía `CARGO_BIN_EXE_insar` (sin dependencias
//! de testing externas).

use std::fs;
use std::path::Path;
use std::process::Command;

use insar_core::types::{DisplacementSeries, Epoch, StackMeta, VelocityMap};
use ndarray::{Array2, Array3};
use surtgis_core::{CRS, GeoTransform};

const ROWS: usize = 4;
const COLS: usize = 5;

/// Escribe un GeoTIFF f32 1-banda con el writer del core (vía VelocityMap).
fn write_tif(path: &Path, value: f32) {
    let meta = StackMeta {
        transform: GeoTransform::new(500_000.0, 7_000_000.0, 30.0, -30.0),
        crs: Some(CRS::from_epsg(32719)),
        wavelength_m: insar_core::types::SENTINEL1_WAVELENGTH_M,
        incidence_deg: 39.0,
        heading_deg: None,
    };
    let map = VelocityMap { data: Array2::from_elem((ROWS, COLS), value), meta };
    insar_core::io::write_velocity(&map, path).expect("escribir tif de prueba");
}

/// Stack sintético mínimo: 3 épocas, 2 ifgs (re/im por par).
fn build_stack(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    let manifest = r#"{
        "wavelength_m": 0.05546576,
        "incidence_deg": 39.0,
        "epochs": ["2023-01-01", "2023-01-13", "2023-01-25"],
        "ifgs": [
            {"reference": 0, "secondary": 1, "perp_baseline_m": 50.0, "file": "ifg_a.tif"},
            {"reference": 1, "secondary": 2, "perp_baseline_m": -30.0, "file": "ifg_b.tif"}
        ]
    }"#;
    fs::write(dir.join("stack.json"), manifest).unwrap();
    for name in ["ifg_a", "ifg_b"] {
        write_tif(&dir.join(format!("{name}_re.tif")), 1.0);
        write_tif(&dir.join(format!("{name}_im.tif")), 0.1);
    }
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_insar"))
        .args(args)
        .output()
        .expect("ejecutar binario insar")
}

#[test]
fn info_y_network_sobre_stack_sintetico() {
    let base = std::env::temp_dir().join(format!("insar_cli_smoke_{}", std::process::id()));
    let _ = fs::remove_dir_all(&base);
    build_stack(&base);

    // `insar info`: dimensiones y metadata legibles.
    let out = run(&["info", base.to_str().unwrap()]);
    assert!(out.status.success(), "info falló: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("épocas"), "stdout: {stdout}");
    assert!(stdout.contains("3"), "3 épocas esperadas: {stdout}");
    assert!(stdout.contains(&format!("{ROWS} × {COLS}")), "grilla: {stdout}");

    // `insar network`: la red consecutiva es conexa.
    let out = run(&["network", base.to_str().unwrap()]);
    assert!(out.status.success(), "network falló: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("conexa:   sí"), "stdout: {stdout}");

    // Directorio inexistente: error limpio (exit != 0), no panic.
    let out = run(&["info", "/directorio/que/no/existe"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("panicked"), "no debe haber panic: {stderr}");

    let _ = fs::remove_dir_all(&base);
}

fn test_meta() -> StackMeta {
    StackMeta {
        transform: GeoTransform::new(500_000.0, 7_000_000.0, 30.0, -30.0),
        crs: Some(CRS::from_epsg(32719)),
        wavelength_m: insar_core::types::SENTINEL1_WAVELENGTH_M,
        incidence_deg: 39.0,
        heading_deg: None,
    }
}

#[test]
fn decompose_features_deramp_sobre_datos_sinteticos() {
    let base = std::env::temp_dir().join(format!("insar_cli_smoke_g17_{}", std::process::id()));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    // Dos velocity.tif (asc/desc) para `decompose`.
    let asc = base.join("asc.tif");
    let desc = base.join("desc.tif");
    write_tif(&asc, -0.03);
    write_tif(&desc, -0.02);
    let decompose_out = base.join("decompose");
    let out = run(&[
        "decompose",
        asc.to_str().unwrap(),
        desc.to_str().unwrap(),
        decompose_out.to_str().unwrap(),
        "--asc-incidence-deg",
        "39",
        "--asc-heading-deg",
        "349",
        "--desc-incidence-deg",
        "39",
        "--desc-heading-deg",
        "191",
    ]);
    assert!(out.status.success(), "decompose falló: {}", String::from_utf8_lossy(&out.stderr));
    assert!(decompose_out.join("up.tif").exists());
    assert!(decompose_out.join("east.tif").exists());

    // Serie sintética (5 épocas: mínimo por defecto de `extract_features`)
    // para `features`/`deramp`.
    let series_dir = base.join("series");
    let epochs = [
        "2023-01-01",
        "2023-01-13",
        "2023-01-25",
        "2023-02-06",
        "2023-02-18",
    ]
    .iter()
    .map(|s| Epoch(s.parse().unwrap()))
    .collect::<Vec<_>>();
    let n = epochs.len();
    let data = Array3::from_shape_fn((n, ROWS, COLS), |(k, r, c)| {
        -0.01 * k as f32 + (r * COLS + c) as f32 * 1e-4
    });
    let series = DisplacementSeries { data, epochs, meta: test_meta() };
    insar_core::io::write_series(&series, &series_dir).expect("escribir serie de prueba");

    let features_out = base.join("features");
    let out = run(&["features", series_dir.to_str().unwrap(), features_out.to_str().unwrap(), "--csv"]);
    assert!(out.status.success(), "features falló: {}", String::from_utf8_lossy(&out.stderr));
    assert!(features_out.join("velocity.tif").exists());
    assert!(features_out.join("features.csv").exists());

    let deramp_out = base.join("deramp");
    let out = run(&["deramp", series_dir.to_str().unwrap(), deramp_out.to_str().unwrap(), "linear"]);
    assert!(out.status.success(), "deramp falló: {}", String::from_utf8_lossy(&out.stderr));
    assert!(deramp_out.join("disp_20230101.tif").exists());

    // `tropo-era5`: cubo de retardo sintético con las MISMAS fechas que
    // `series_dir`, creciente en el tiempo (simula humedad creciente).
    let delay_epochs: Vec<Epoch> = [
        "2023-01-01",
        "2023-01-13",
        "2023-01-25",
        "2023-02-06",
        "2023-02-18",
    ]
    .iter()
    .map(|s| Epoch(s.parse().unwrap()))
    .collect();
    let delay_dir = base.join("era5_delay");
    let delay_data =
        Array3::from_shape_fn((delay_epochs.len(), ROWS, COLS), |(k, _, _)| 0.001 * k as f32);
    let delay_series =
        DisplacementSeries { data: delay_data, epochs: delay_epochs, meta: test_meta() };
    insar_core::io::write_series(&delay_series, &delay_dir).expect("escribir cubo ERA5 de prueba");

    let era5_out = base.join("era5_out");
    let out = run(&[
        "tropo-era5",
        series_dir.to_str().unwrap(),
        delay_dir.to_str().unwrap(),
        era5_out.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "tropo-era5 falló: {}", String::from_utf8_lossy(&out.stderr));
    let corrected = insar_core::io::read_series(&era5_out, test_meta()).expect("leer serie corregida");
    // Época 0 (referencia): sin cambio. Época k: += delay[k] - delay[0].
    for k in 0..n {
        let expected_delta = 0.001 * k as f32;
        for r in 0..ROWS {
            for c in 0..COLS {
                let before = -0.01 * k as f32 + (r * COLS + c) as f32 * 1e-4;
                let got = corrected.data[[k, r, c]];
                assert!(
                    (got - (before + expected_delta)).abs() < 1e-6,
                    "época {k} ({r},{c}): {got} vs {}",
                    before + expected_delta
                );
            }
        }
    }

    // Cubo con fechas distintas → error explícito, no panic.
    let bad_delay_dir = base.join("era5_delay_bad");
    let bad_epochs: Vec<Epoch> = ["2023-01-01", "2023-01-13"].iter().map(|s| Epoch(s.parse().unwrap())).collect();
    let bad_delay = DisplacementSeries {
        data: Array3::from_elem((2, ROWS, COLS), 0.0_f32),
        epochs: bad_epochs,
        meta: test_meta(),
    };
    insar_core::io::write_series(&bad_delay, &bad_delay_dir).expect("escribir cubo ERA5 inválido");
    let out = run(&[
        "tropo-era5",
        series_dir.to_str().unwrap(),
        bad_delay_dir.to_str().unwrap(),
        base.join("era5_out_bad").to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("panicked"), "no debe haber panic: {stderr}");

    let _ = fs::remove_dir_all(&base);
}

/// Regresión: `--ref-region` repetido causaba panic (`unreachable!`) porque
/// `Option<Vec<usize>>` infiere `ArgAction::Append` por defecto en clap, y dos
/// ocurrencias acumulaban 8 valores en vez de reemplazar. También cubre la
/// validación de rango inválido y el conflicto con `--ref-row`/`--ref-col`,
/// ninguno cubierto antes de este fix. El directorio de entrada no necesita
/// existir: toda esta validación corre antes de leer el stack.
#[test]
fn ref_region_no_hace_panic_con_flags_invalidos() {
    // Doble ocurrencia: clap debe rechazarla con error, no acumular 8 valores.
    let out = run(&[
        "isce",
        "/no/existe",
        "/tmp/no_importa",
        "--ref-region",
        "0",
        "0",
        "1",
        "1",
        "--ref-region",
        "2",
        "2",
        "3",
        "3",
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("panicked"), "no debe haber panic: {stderr}");
    assert!(!stderr.contains("unreachable"), "no debe haber unreachable: {stderr}");

    // Rango inválido (mín > máx): error limpio, no panic.
    let out = run(&[
        "isce",
        "/no/existe",
        "/tmp/no_importa",
        "--ref-region",
        "5",
        "5",
        "1",
        "1",
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("panicked"), "no debe haber panic: {stderr}");
    assert!(stderr.contains("inválido"), "stderr: {stderr}");

    // --ref-row/--ref-col junto a --ref-region: conflicto explícito.
    let out = run(&[
        "isce",
        "/no/existe",
        "/tmp/no_importa",
        "--ref-row",
        "1",
        "--ref-col",
        "1",
        "--ref-region",
        "0",
        "0",
        "1",
        "1",
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("panicked"), "no debe haber panic: {stderr}");
    assert!(stderr.contains("excluyentes"), "stderr: {stderr}");
}

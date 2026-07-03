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

    let _ = fs::remove_dir_all(&base);
}

//! Smoke test de la CLI: verifica el wiring binario→core sobre un stack
//! sintético mínimo (parseo clap, lectura del stack, salida legible).
//! Usa el binario compilado vía `CARGO_BIN_EXE_insar` (sin dependencias
//! de testing externas).

use std::fs;
use std::path::Path;
use std::process::Command;

use insar_core::types::{StackMeta, VelocityMap};
use ndarray::Array2;
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

//! Validación vs MintPy: ingiere las fases desenrolladas exportadas desde un
//! `ifgramStack.h5` (validation/export/{meta.json,phase.f32}), corre la
//! inversión SBAS de insar-rs y escribe la serie y la velocidad en binario
//! crudo f32 para que el comparador Python las contraste con MintPy.
//!
//! Uso: cargo run --release -p insar-core --example validate_fernandina -- <export_dir>

use std::fs;
use std::io::Read;
use std::path::Path;

use ndarray::Array3;
use serde::Deserialize;
use surtgis_core::GeoTransform;

use insar_core::inversion::{estimate_velocity, invert_sbas};
use insar_core::types::{Epoch, IfgPair, StackMeta, UnwrappedStack};

#[derive(Deserialize)]
struct PairJson {
    reference: usize,
    secondary: usize,
    perp_baseline_m: f64,
}

#[derive(Deserialize)]
struct Meta {
    wavelength_m: f64,
    incidence_deg: f64,
    n_epochs: usize,
    n_pairs: usize,
    rows: usize,
    cols: usize,
    epochs: Vec<String>,
    pairs: Vec<PairJson>,
}

fn read_f32_le(path: &Path, n: usize) -> Vec<f32> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .unwrap_or_else(|e| panic!("abrir {}: {e}", path.display()))
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(bytes.len(), n * 4, "tamaño inesperado de {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn write_f32_le(path: &Path, data: &[f32]) {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).unwrap_or_else(|e| panic!("escribir {}: {e}", path.display()));
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "validation/export".into());
    let dir = Path::new(&dir);

    let meta: Meta = serde_json::from_str(&fs::read_to_string(dir.join("meta.json")).unwrap())
        .expect("meta.json");
    println!(
        "Cargando: {} épocas, {} pares, grilla {}×{}",
        meta.n_epochs, meta.n_pairs, meta.rows, meta.cols
    );

    let n = meta.n_pairs * meta.rows * meta.cols;
    let phase = read_f32_le(&dir.join("phase.f32"), n);
    let data = Array3::from_shape_vec((meta.n_pairs, meta.rows, meta.cols), phase)
        .expect("reshape phase");

    let epochs: Vec<Epoch> = meta
        .epochs
        .iter()
        .map(|s| Epoch(s.parse().unwrap_or_else(|e| panic!("fecha {s}: {e}"))))
        .collect();
    let pairs: Vec<IfgPair> = meta
        .pairs
        .iter()
        .map(|p| IfgPair {
            reference: p.reference,
            secondary: p.secondary,
            perp_baseline_m: p.perp_baseline_m,
        })
        .collect();

    let stack = UnwrappedStack {
        data,
        epochs,
        pairs,
        meta: StackMeta {
            // Georreferencia irrelevante para la comparación numérica (ambos
            // lados trabajan sobre la misma grilla en coordenadas radar).
            transform: GeoTransform::new(0.0, 0.0, 1.0, -1.0),
            crs: None,
            wavelength_m: meta.wavelength_m,
            incidence_deg: meta.incidence_deg,
            heading_deg: None,
        },
    };

    let t0 = std::time::Instant::now();
    let series = invert_sbas(&stack, None).expect("invert_sbas");
    let velocity = estimate_velocity(&series).expect("estimate_velocity");
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "Inversión SBAS + velocidad: {:.2}s ({} píxeles)",
        secs,
        meta.rows * meta.cols
    );

    // Serie: (épocas, filas, cols) C-order; velocidad: (filas, cols) C-order.
    write_f32_le(&dir.join("insar_timeseries.f32"), series.data.as_slice().unwrap());
    write_f32_le(&dir.join("insar_velocity.f32"), velocity.data.as_slice().unwrap());
    println!("OK → insar_timeseries.f32 + insar_velocity.f32 en {}", dir.display());
}

//! Validación end-to-end del camino NATIVO: lee el stack de interferogramas
//! ISCE directamente (sin MintPy ni h5), invierte y escribe la velocidad LOS
//! para compararla con el velocity.h5 de MintPy.
//!
//! Uso: cargo run --release -p insar-core --example validate_fernandina_isce -- \
//!        data/FernandinaSenDT128/merged/interferograms \
//!        data/FernandinaSenDT128/baselines \
//!        validation/export/insar_velocity_isce.f32

use std::fs;
use std::path::{Path, PathBuf};

use insar_core::inversion::{estimate_velocity, invert_sbas};
use insar_core::io::isce::{IsceLoadConfig, read_isce_unwrapped_stack};

fn main() {
    let mut args = std::env::args().skip(1);
    let ifg_dir = args.next().expect("arg1: dir de interferogramas ISCE");
    let baselines = args.next();
    let out = args.next().unwrap_or_else(|| "validation/export/insar_velocity_isce.f32".into());

    let config = IsceLoadConfig {
        baselines_dir: baselines.map(PathBuf::from),
        ..Default::default()
    };

    let t0 = std::time::Instant::now();
    let stack = read_isce_unwrapped_stack(Path::new(&ifg_dir), &config).expect("lectura ISCE");
    let read_s = t0.elapsed().as_secs_f64();
    let (rows, cols) = stack.dims();
    println!(
        "Leído (ISCE nativo): {} épocas, {} pares, {}×{} en {:.2}s",
        stack.epochs.len(), stack.pairs.len(), rows, cols, read_s
    );

    let t1 = std::time::Instant::now();
    let series = invert_sbas(&stack, None).expect("invert_sbas");
    let velocity = estimate_velocity(&series).expect("estimate_velocity");
    println!("Inversión + velocidad: {:.2}s", t1.elapsed().as_secs_f64());

    let mut bytes = Vec::with_capacity(rows * cols * 4);
    for &v in velocity.data.as_slice().unwrap() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(&out, bytes).unwrap();
    println!("OK → {out} ({rows}×{cols} f32)");
}

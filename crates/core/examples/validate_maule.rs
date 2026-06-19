//! Caso chileno: Laguna del Maule (ARIA S1-GUNW, track 83 desc, 2017–2018).
//! Lee el stack exportado (meta.json + phase.f32 + coherence.f32), referencia
//! al píxel de máxima coherencia, invierte SBAS y estima velocidad + coherencia
//! temporal. Escribe velocity.f32, tcoh.f32 y series.f32 para visualizar.
//!
//! Uso: cargo run --release -p insar-core --example validate_maule -- validation/maule_export

use std::fs;
use std::io::Read;
use std::path::Path;

use ndarray::Array3;
use serde::Deserialize;
use surtgis_core::GeoTransform;

use insar_core::inversion::{
    estimate_velocity, invert_sbas, reference_to_pixel, temporal_coherence,
};
use insar_core::types::{Epoch, IfgPair, StackMeta, UnwrappedStack};

#[derive(Deserialize)]
struct PairJson { reference: usize, secondary: usize, perp_baseline_m: f64 }
#[derive(Deserialize)]
struct Meta {
    wavelength_m: f64, incidence_deg: f64,
    n_epochs: usize, n_pairs: usize, rows: usize, cols: usize,
    epochs: Vec<String>, pairs: Vec<PairJson>,
}

fn read_f32(path: &Path, n: usize) -> Vec<f32> {
    let mut b = Vec::new();
    fs::File::open(path).unwrap().read_to_end(&mut b).unwrap();
    assert_eq!(b.len(), n * 4, "tamaño {}", path.display());
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn write_f32(path: &Path, d: &[f32]) {
    let mut b = Vec::with_capacity(d.len() * 4);
    for &v in d { b.extend_from_slice(&v.to_le_bytes()); }
    fs::write(path, b).unwrap();
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "validation/maule_export".into());
    let dir = Path::new(&dir);
    let meta: Meta = serde_json::from_str(&fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let (np, nr, nc) = (meta.n_pairs, meta.rows, meta.cols);
    println!("Maule: {} épocas, {np} pares, {nr}×{nc}", meta.n_epochs);

    let phase = read_f32(&dir.join("phase.f32"), np * nr * nc);
    let coh = read_f32(&dir.join("coherence.f32"), np * nr * nc);
    let data = Array3::from_shape_vec((np, nr, nc), phase).unwrap();
    let coh = Array3::from_shape_vec((np, nr, nc), coh).unwrap();

    let mut stack = UnwrappedStack {
        data,
        epochs: meta.epochs.iter().map(|s| Epoch(s.parse().unwrap())).collect(),
        pairs: meta.pairs.iter().map(|p| IfgPair {
            reference: p.reference, secondary: p.secondary, perp_baseline_m: p.perp_baseline_m,
        }).collect(),
        meta: StackMeta {
            transform: GeoTransform::new(0.0, 0.0, 1.0, -1.0), crs: None,
            wavelength_m: meta.wavelength_m, incidence_deg: meta.incidence_deg, heading_deg: None,
        },
    };

    // Corrección de errores de desenrollado (saltos 2π entre componentes
    // conexas de cada GUNW) por cierre de fase de la red SBAS.
    let n_corr = insar_core::unwrap_error::correct_unwrap_errors(&mut stack).unwrap();
    println!("errores de desenrollado corregidos: {n_corr} píxeles");

    // Píxel de referencia: el de mayor COBERTURA (presente en más pares, para
    // no anular interferogramas al referenciar) y, a igualdad, mayor coherencia.
    let (mut best, mut best_score) = ((nr / 2, nc / 2), -1.0f64);
    for r in 0..nr {
        for c in 0..nc {
            let (mut s, mut k, mut cov) = (0.0f64, 0u32, 0u32);
            for p in 0..np {
                if stack.data[[p, r, c]].is_finite() { cov += 1; }
                let v = coh[[p, r, c]];
                if v.is_finite() { s += v as f64; k += 1; }
            }
            let mean_coh = if k > 0 { s / k as f64 } else { 0.0 };
            let score = cov as f64 * 2.0 + mean_coh; // cobertura domina
            if score > best_score { best_score = score; best = (r, c); }
        }
    }
    let cov = (0..np).filter(|&p| stack.data[[p, best.0, best.1]].is_finite()).count();
    println!("referencia: {:?} cobertura {}/{} pares", best, cov, np);
    reference_to_pixel(&mut stack, best.0, best.1).unwrap();

    let t = std::time::Instant::now();
    let series = invert_sbas(&stack, None).unwrap();
    let vel = estimate_velocity(&series).unwrap();
    let tcoh = temporal_coherence(&stack, &series).unwrap();
    println!("inversión + velocidad + coherencia: {:.2}s", t.elapsed().as_secs_f64());

    // Velocidad de deformación: extremo entre píxeles coherentes (cm/año).
    let mut peak = 0.0f32;
    for r in 0..nr { for c in 0..nc {
        let v = vel.data[[r, c]];
        if v.is_finite() && tcoh[[r, c]] > 0.7 && v.abs() > peak.abs() { peak = v; }
    }}
    let med = {
        let mut g: Vec<f32> = tcoh.iter().copied().filter(|x| x.is_finite()).collect();
        g.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if g.is_empty() { f32::NAN } else { g[g.len() / 2] }
    };
    println!("velocidad LOS máx (γ>0.7): {:.1} cm/año", peak * 100.0);
    println!("coherencia temporal mediana: {:.3}", med);

    write_f32(&dir.join("velocity.f32"), vel.data.as_slice().unwrap());
    write_f32(&dir.join("tcoh.f32"), tcoh.as_slice().unwrap());
    write_f32(&dir.join("series.f32"), series.data.as_slice().unwrap());
    println!("OK → velocity.f32 + tcoh.f32 + series.f32");
}

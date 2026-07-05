//! Caso chileno: Laguna del Maule (ARIA S1-GUNW, track 83 desc, red estacional
//! 2016-2020) con el motor AUDITADO (Sprint 1-4): WLS por coherencia, error de
//! DEM, IRLS robusto — no solo OLS como `validate_maule.rs` original.
//!
//! Uso: cargo run --release -p insar-core --example validate_maule_audited -- \
//!        validation/maule_summer_export

use std::fs;
use std::io::Read;
use std::path::Path;

use ndarray::Array3;
use serde::Deserialize;
use surtgis_core::GeoTransform;

use insar_core::inversion::{
    DemErrorConfig, IrlsConfig, SbasSolverConfig, WeightScheme, estimate_velocity,
    estimate_velocity_uncertainty, invert_sbas_ext, reference_to_pixel,
};
use insar_core::types::{Epoch, IfgPair, StackMeta, UnwrappedStack};
use insar_core::unwrap_error::nonzero_closure_count;

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

/// Rango oblicuo medio S1 IW (m), para la columna de error de DEM.
const S1_IW_SLANT_RANGE_M: f64 = 850_000.0;

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "validation/maule_summer_export".into());
    let dir = Path::new(&dir);
    let meta: Meta = serde_json::from_str(&fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let (np, nr, nc) = (meta.n_pairs, meta.rows, meta.cols);
    println!("Maule: {} épocas, {np} pares, {nr}×{nc}", meta.n_epochs);

    let phase = read_f32(&dir.join("phase.f32"), np * nr * nc);
    let coh_vec = read_f32(&dir.join("coherence.f32"), np * nr * nc);
    let data = Array3::from_shape_vec((np, nr, nc), phase).unwrap();
    let coh = Array3::from_shape_vec((np, nr, nc), coh_vec).unwrap();

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
    // conexas de cada GUNW) por cierre de fase de la red SBAS + QC.
    let rep = insar_core::unwrap_error::correct_unwrap_errors(&mut stack).unwrap();
    println!(
        "errores de desenrollado: {} píxeles corregidos, {} detectados sin corregir",
        rep.corrected, rep.detected_uncorrected
    );
    let closure_qc = nonzero_closure_count(&stack).unwrap();
    let mean_closure: f64 = closure_qc.iter().filter(|v| v.is_finite())
        .map(|&v| v as f64).sum::<f64>() / closure_qc.iter().filter(|v| v.is_finite()).count() as f64;
    println!("cierre de fase no-cero (QC), media: {mean_closure:.2}");

    // Píxel de referencia: mayor cobertura (presente en más pares) y, a
    // igualdad, mayor coherencia — igual que el original (funciona bien
    // porque el GUNW ya está recortado a la escena, no hay AOI-vs-footprint
    // que resolver como en el caso topsStack/El Canelo).
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
            let score = cov as f64 * 2.0 + mean_coh;
            if score > best_score { best_score = score; best = (r, c); }
        }
    }
    let cov = (0..np).filter(|&p| stack.data[[p, best.0, best.1]].is_finite()).count();
    println!("referencia: {:?} cobertura {}/{} pares", best, cov, np);
    reference_to_pixel(&mut stack, best.0, best.1).unwrap();

    // Inversión AUDITADA: WLS por coherencia + IRLS robusto + error de DEM
    // (en vez del OLS simple del validate_maule.rs original).
    let solver = SbasSolverConfig {
        weighting: WeightScheme::InversePhaseVariance,
        dem_error: Some(DemErrorConfig { slant_range_m: S1_IW_SLANT_RANGE_M }),
        robust: Some(IrlsConfig::default()),
    };
    let t = std::time::Instant::now();
    let solution = invert_sbas_ext(&stack, None, Some(&coh), &solver).unwrap();
    let mut series = solution.series;
    let tcoh = insar_core::postprocess::temporal_coherence(&stack, &series).unwrap();
    println!("inversión WLS+IRLS+DEM-error: {:.2}s", t.elapsed().as_secs_f64());

    if let Some(dem_err) = &solution.dem_error_m {
        let vals: Vec<f64> = dem_err.iter().filter(|v| v.is_finite()).map(|&v| v as f64).collect();
        let mean = vals.iter().sum::<f64>() / vals.len().max(1) as f64;
        let mx = vals.iter().cloned().fold(0.0f64, f64::max);
        println!("error de DEM estimado: media {mean:.1} m, máx |.| {mx:.1} m");
        write_f32(&dir.join("dem_error_audited.f32"), dem_err.as_slice().unwrap());
    }

    // Máscara de coherencia para los ajustes de post-proceso.
    let mut mask = ndarray::Array2::from_elem((nr, nc), false);
    for r in 0..nr {
        for c in 0..nc {
            mask[[r, c]] = tcoh[[r, c]].is_finite() && tcoh[[r, c]] > 0.7;
        }
    }

    // Corrección troposférica topo-correlacionada (si hay dem.f32 exportado).
    let dem_path = dir.join("dem.f32");
    if dem_path.exists() {
        let dem_vec = read_f32(&dem_path, nr * nc);
        let dem = Array3::from_shape_vec((1, nr, nc), dem_vec).unwrap();
        let dem = dem.index_axis(ndarray::Axis(0), 0).to_owned();
        insar_core::troposphere::correct_topo_series(&mut series, &dem, Some(&mask), 1, true).unwrap();
        println!("corrección troposférica topo-correlacionada aplicada");
    }
    // Deramp por época sobre píxeles coherentes.
    insar_core::postprocess::deramp_series(
        &mut series, insar_core::postprocess::RampKind::Linear, Some(&mask),
    ).unwrap();
    println!("deramp por época aplicado (máscara coherencia>0.7)");

    let vel = estimate_velocity(&series).unwrap();
    let vel_std = estimate_velocity_uncertainty(&series).unwrap();

    // Velocidad de deformación: extremo entre píxeles coherentes (cm/año).
    let mut peak = 0.0f32;
    let mut peak_rc = (0usize, 0usize);
    for r in 0..nr { for c in 0..nc {
        let v = vel.data[[r, c]];
        if v.is_finite() && tcoh[[r, c]] > 0.7 && v.abs() > peak.abs() { peak = v; peak_rc = (r, c); }
    }}
    let med = {
        let mut g: Vec<f32> = tcoh.iter().copied().filter(|x| x.is_finite()).collect();
        g.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if g.is_empty() { f32::NAN } else { g[g.len() / 2] }
    };
    let coverage = tcoh.iter().filter(|v| v.is_finite() && **v > 0.7).count() as f64 / (nr * nc) as f64;
    println!("velocidad LOS máx (γ>0.7): {:.1} cm/año en {:?}, σ={:.2} cm/año",
        peak * 100.0, peak_rc, vel_std[peak_rc] * 100.0);
    println!("coherencia temporal mediana: {med:.3}, cobertura γ>0.7: {:.0}%", coverage * 100.0);

    write_f32(&dir.join("velocity_audited.f32"), vel.data.as_slice().unwrap());
    write_f32(&dir.join("velocity_std_audited.f32"), vel_std.as_slice().unwrap());
    write_f32(&dir.join("tcoh_audited.f32"), tcoh.as_slice().unwrap());
    write_f32(&dir.join("series_audited.f32"), series.data.as_slice().unwrap());
    println!("OK → velocity_audited.f32 + velocity_std_audited.f32 + tcoh_audited.f32 + series_audited.f32");
}

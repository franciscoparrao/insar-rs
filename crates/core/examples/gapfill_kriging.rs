//! Relleno de huecos (decorrelación) del campo de velocidad por **kriging
//! ordinario**, usando geostat-core. Demuestra el enganche insar-rs ↔ geostat-rs:
//! los píxeles coherentes condicionan un variograma + kriging que estima la
//! velocidad en los huecos, con su **varianza de kriging** como incertidumbre.
//!
//! Uso: cargo run --release -p insar-core --example gapfill_kriging -- <export_dir>

use std::fs;
use std::io::Read;
use std::path::Path;

use geostat_core::{
    Kriging, KrigingConfig, KrigingMethod, ModelKind, PointSet, VariogramConfig,
    experimental_variogram, fit_best,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Meta { rows: usize, cols: usize }

fn read_f32(p: &Path, n: usize) -> Vec<f32> {
    let mut b = Vec::new();
    fs::File::open(p).unwrap().read_to_end(&mut b).unwrap();
    assert_eq!(b.len(), n * 4);
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let dir = std::env::args().nth(1).expect("export_dir");
    let dir = Path::new(&dir);
    let meta: Meta = serde_json::from_str(&fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let (nr, nc) = (meta.rows, meta.cols);
    let vel = read_f32(&dir.join("velocity.f32"), nr * nc);
    let tcoh = read_f32(&dir.join("tcoh.f32"), nr * nc);

    // Píxeles coherentes = muestras; huecos (NaN o baja coherencia) = objetivos.
    let coherent = |i: usize| tcoh[i].is_finite() && tcoh[i] > 0.7 && vel[i].is_finite();
    let (mut cx, mut cy, mut cv) = (Vec::new(), Vec::new(), Vec::new());
    let mut gaps: Vec<(usize, usize)> = Vec::new();
    for r in 0..nr {
        for c in 0..nc {
            let i = r * nc + c;
            if coherent(i) {
                cx.push(c as f64);
                cy.push(r as f64);
                cv.push(vel[i] as f64 * 100.0); // cm/año
            } else {
                gaps.push((r, c));
            }
        }
    }
    println!("coherentes (muestras): {}  huecos a rellenar: {}  ({:.0}% de la grilla)",
             cv.len(), gaps.len(), 100.0 * gaps.len() as f64 / (nr * nc) as f64);

    // Submuestreo de condicionamiento por velocidad (kriging usa vecindarios kd-tree).
    let stride = (cv.len() / 12000).max(1);
    let (sx, sy, sv): (Vec<f64>, Vec<f64>, Vec<f64>) = (0..cv.len())
        .step_by(stride)
        .map(|i| (cx[i], cy[i], cv[i]))
        .fold((vec![], vec![], vec![]), |(mut a, mut b, mut c), (x, y, v)| {
            a.push(x); b.push(y); c.push(v); (a, b, c)
        });
    let data = PointSet::<2>::from_xyz(&sx, &sy, &sv).unwrap();
    println!("condicionamiento: {} puntos (stride {stride})", sv.len());

    // Variograma experimental + ajuste automático (mejor familia).
    let vcfg = VariogramConfig { n_lags: 15, max_dist: (nr.min(nc) as f64) / 3.0, direction: None };
    let exp = experimental_variogram(&data, &vcfg).unwrap();
    let fit = fit_best(&exp, &[ModelKind::Spherical, ModelKind::Exponential, ModelKind::Gaussian]).unwrap();
    println!("variograma ajustado (wsse={:.3})", fit.wsse);

    // Kriging ordinario, vecindario de 40 puntos.
    let cfg = KrigingConfig { method: KrigingMethod::Ordinary, max_neighbors: Some(40), ..Default::default() };
    let kr = Kriging::new(&data, &fit.model, cfg).unwrap();

    let t = std::time::Instant::now();
    let targets: Vec<[f64; 2]> = gaps.iter().map(|&(r, c)| [c as f64, r as f64]).collect();
    let est = kr.predict_many(&targets);
    println!("kriging de {} huecos: {:.1}s", targets.len(), t.elapsed().as_secs_f64());

    // Campo relleno (cm/año) + mapa de desviación estándar de kriging (cm/año).
    let mut filled = vec![f32::NAN; nr * nc];
    let mut kstd = vec![0.0f32; nr * nc];
    for i in 0..nr * nc {
        if coherent(i) { filled[i] = vel[i] * 100.0; }
    }
    for (k, &(r, c)) in gaps.iter().enumerate() {
        filled[r * nc + c] = est[k].value as f32;
        kstd[r * nc + c] = est[k].variance.max(0.0).sqrt() as f32;
    }
    let write = |name: &str, d: &[f32]| {
        let mut b = Vec::with_capacity(d.len() * 4);
        for &v in d { b.extend_from_slice(&v.to_le_bytes()); }
        fs::write(dir.join(name), b).unwrap();
    };
    write("velocity_filled.f32", &filled);
    write("kriging_std.f32", &kstd);
    let gap_std: f64 = gaps.iter().map(|&(r, c)| kstd[r * nc + c] as f64).sum::<f64>() / gaps.len() as f64;
    println!("OK → velocity_filled.f32 + kriging_std.f32  (σ_kriging media en huecos = {gap_std:.2} cm/año)");
}

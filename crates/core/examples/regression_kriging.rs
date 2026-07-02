//! Regression kriging (insar-rs + Smelt + geostat-rs): el método estándar-oro
//! de predicción espacial, uniendo los tres motores.
//!
//! En los huecos de decorrelación las features InSAR son NaN, así que la
//! TENDENCIA se modela con covariables de **terreno** (elevación + pendiente
//! del DEM, disponibles en todas partes) vía un RandomForest de **Smelt**; los
//! **residuos** (lo que el terreno no explica) se krigean con **geostat-rs**.
//! Predicción final = tendencia(terreno) + residuo kriged, con incertidumbre.
//!
//! Uso: cargo run --release -p insar-core --example regression_kriging -- <export_dir>

use std::fs;
use std::io::Read;
use std::path::Path;

use ndarray::{Array2, s};
use serde::Deserialize;

use geostat_core::{
    KrigingConfig, KrigingMethod, ModelKind, PointSet, RegressionKriging, VariogramConfig,
    experimental_variogram, fit_best,
};
use smelt_ml::prelude::*;

#[derive(Deserialize)]
struct Meta { rows: usize, cols: usize }

fn read_f32(p: &Path, n: usize) -> Vec<f32> {
    let mut b = Vec::new();
    fs::File::open(p).unwrap().read_to_end(&mut b).unwrap();
    assert_eq!(b.len(), n * 4);
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn predicted(p: &Prediction) -> Vec<f64> {
    match p {
        Prediction::Regression { predicted, .. } => predicted.clone(),
        _ => panic!("se esperaba una predicción de regresión"),
    }
}

fn main() {
    let dir = std::env::args().nth(1).expect("export_dir");
    let dir = Path::new(&dir);
    let m: Meta = serde_json::from_str(&fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let (nr, nc) = (m.rows, m.cols);
    let vel = read_f32(&dir.join("velocity.f32"), nr * nc);
    let tcoh = read_f32(&dir.join("tcoh.f32"), nr * nc);
    let dem = read_f32(&dir.join("dem.f32"), nr * nc);

    // Pendiente: magnitud del gradiente del DEM (covariable de terreno).
    let slope = |r: usize, c: usize| -> f32 {
        let g = |rr: usize, cc: usize| dem[rr.min(nr - 1) * nc + cc.min(nc - 1)];
        let dzdx = (g(r, c + 1) - g(r, c.saturating_sub(1))) / 2.0;
        let dzdy = (g(r + 1, c) - g(r.saturating_sub(1), c)) / 2.0;
        (dzdx * dzdx + dzdy * dzdy).sqrt()
    };

    // Píxeles coherentes = datos; huecos = objetivos. Covariables = [elev, slope].
    let coherent = |i: usize| tcoh[i].is_finite() && tcoh[i] > 0.7 && vel[i].is_finite();
    let (mut cx, mut cy, mut cval) = (Vec::new(), Vec::new(), Vec::new());
    let mut cov_data: Vec<[f64; 2]> = Vec::new();
    let mut gaps: Vec<(usize, usize)> = Vec::new();
    for r in 0..nr {
        for c in 0..nc {
            let i = r * nc + c;
            if dem[i].is_nan() { continue; }
            if coherent(i) {
                cx.push(c as f64); cy.push(r as f64); cval.push(vel[i] as f64 * 100.0);
                cov_data.push([dem[i] as f64, slope(r, c) as f64]);
            } else {
                gaps.push((r, c));
            }
        }
    }
    println!("datos coherentes: {}  huecos: {}", cval.len(), gaps.len());

    // Submuestreo del condicionamiento.
    let stride = (cval.len() / 8000).max(1);
    let idx: Vec<usize> = (0..cval.len()).step_by(stride).collect();
    let sx: Vec<f64> = idx.iter().map(|&i| cx[i]).collect();
    let sy: Vec<f64> = idx.iter().map(|&i| cy[i]).collect();
    let sval: Vec<f64> = idx.iter().map(|&i| cval[i]).collect();
    let scov = Array2::from_shape_vec((idx.len(), 2), idx.iter().flat_map(|&i| cov_data[i]).collect()).unwrap();

    // --- TENDENCIA: RandomForest de Smelt, terreno → velocidad ---
    let task = RegressionTask::new("trend", scov.clone(), sval.clone()).unwrap();
    let mut rf = RandomForest::new().with_n_estimators(200);
    let model = rf.train_regress(&task).unwrap();
    let trend_at_data = predicted(&model.predict(&scov).unwrap());
    // R² HONESTO (held-out): el RF sobreajusta in-sample, así que la skill real
    // del terreno se reporta sobre un 30% retenido (entrenando en el otro 70%).
    let n = sval.len();
    let cut = n * 7 / 10;
    let tr = RegressionTask::new("tr", scov.slice(s![..cut, ..]).to_owned(), sval[..cut].to_vec()).unwrap();
    let mut rf_cv = RandomForest::new().with_n_estimators(200);
    let pred_te = predicted(&rf_cv.train_regress(&tr).unwrap().predict(&scov.slice(s![cut.., ..]).to_owned()).unwrap());
    let te = &sval[cut..];
    let mean = te.iter().sum::<f64>() / te.len() as f64;
    let r2 = 1.0
        - te.iter().zip(&pred_te).map(|(v, p)| (v - p).powi(2)).sum::<f64>()
            / te.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    println!("tendencia terreno→velocidad: R² held-out = {:.3} (in-sample optimista)", r2);

    // --- RESIDUOS: regression kriging con geostat-rs ---
    let data = PointSet::<2>::from_xyz(&sx, &sy, &sval).unwrap();
    let rk = RegressionKriging::new(&data, &trend_at_data).unwrap();
    let vcfg = VariogramConfig { n_lags: 15, max_dist: (nr.min(nc) as f64) / 3.0, direction: None };
    let exp = experimental_variogram(rk.residuals(), &vcfg).unwrap();
    let fit = fit_best(&exp, &[ModelKind::Spherical, ModelKind::Exponential, ModelKind::Gaussian]).unwrap();
    println!("variograma de residuos ajustado (wsse={:.3})", fit.wsse);

    // Predicción en los huecos: covariables de terreno allí + kriging de residuos.
    let targets: Vec<[f64; 2]> = gaps.iter().map(|&(r, c)| [c as f64, r as f64]).collect();
    let tcov = Array2::from_shape_vec(
        (gaps.len(), 2),
        gaps.iter().flat_map(|&(r, c)| [dem[r * nc + c] as f64, slope(r, c) as f64]).collect(),
    ).unwrap();
    let trend_at_targets = predicted(&model.predict(&tcov).unwrap());
    let cfg = KrigingConfig { method: KrigingMethod::Ordinary, max_neighbors: Some(40), ..Default::default() };
    let t = std::time::Instant::now();
    let est = rk.predict(&targets, &trend_at_targets, &fit.model, &cfg).unwrap();
    println!("regression kriging de {} huecos: {:.1}s", targets.len(), t.elapsed().as_secs_f64());

    // Campo predicho + incertidumbre.
    let mut filled = vec![f32::NAN; nr * nc];
    let mut rkstd = vec![0.0f32; nr * nc];
    for i in 0..nr * nc { if coherent(i) { filled[i] = vel[i] * 100.0; } }
    for (k, &(r, c)) in gaps.iter().enumerate() {
        filled[r * nc + c] = est[k].value as f32;
        rkstd[r * nc + c] = est[k].variance.max(0.0).sqrt() as f32;
    }
    let write = |name: &str, d: &[f32]| {
        let mut b = Vec::with_capacity(d.len() * 4);
        for &v in d { b.extend_from_slice(&v.to_le_bytes()); }
        fs::write(dir.join(name), b).unwrap();
    };
    write("velocity_rk.f32", &filled);
    write("rk_std.f32", &rkstd);
    let gstd: f64 = gaps.iter().map(|&(r, c)| rkstd[r * nc + c] as f64).sum::<f64>() / gaps.len().max(1) as f64;
    println!("OK → velocity_rk.f32 + rk_std.f32  (σ media en huecos = {gstd:.2} cm/año)");
}

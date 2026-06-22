//! Enganche insar-rs → Smelt: de features de deformación a un clasificador con
//! **CV espacial** y **predicción conforme**, todo nativo en Rust.
//!
//! Toma un export procesado (serie + coherencia), extrae descriptores por píxel
//! con `features::extract_features`, los pasa por `to_table` a la matriz que
//! consume Smelt, y entrena un RandomForest.
//!
//! NOTA: la etiqueta de este ejemplo es ILUSTRATIVA — se deriva de la velocidad
//! (|v| > umbral = "deformación significativa") solo para demostrar la mecánica
//! end-to-end. En uso real la etiqueta viene de un inventario externo
//! (deslizamientos SERNAGEOMIN, daño, etc.) y las features de deformación se
//! cruzan con las de terreno de SurtGIS. Para no hacer la tarea trivial, las
//! columnas derivadas de la velocidad se EXCLUYEN de los predictores.
//!
//! Uso: cargo run --release -p insar-core --example landslide_smelt -- <export_dir>

use std::fs;
use std::io::Read;
use std::path::Path;

use ndarray::{Array2, Array3, Axis};
use serde::Deserialize;
use surtgis_core::GeoTransform;

use insar_core::features::{FeatureConfig, extract_features};
use insar_core::types::{DisplacementSeries, Epoch, StackMeta, SENTINEL1_WAVELENGTH_M};

use smelt_ml::conformal::ConformalClassifier;
use smelt_ml::measure::{Accuracy, F1Score, Measure};
use smelt_ml::prelude::*;
use smelt_ml::resample::{CrossValidation, SpatialBlockCV};

#[derive(Deserialize)]
struct Geo { lon0: f64, lat0: f64, dlon: f64, dlat: f64 }
#[derive(Deserialize)]
struct Meta { n_epochs: usize, rows: usize, cols: usize, epochs: Vec<String>, geo: Geo }

fn read_f32(p: &Path, n: usize) -> Vec<f32> {
    let mut b = Vec::new();
    fs::File::open(p).unwrap().read_to_end(&mut b).unwrap();
    assert_eq!(b.len(), n * 4);
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let dir = std::env::args().nth(1).expect("export_dir");
    let dir = Path::new(&dir);
    let m: Meta = serde_json::from_str(&fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let (ne, nr, nc) = (m.n_epochs, m.rows, m.cols);

    // --- Reconstruir la serie + coherencia y extraer features ---
    let series_data = Array3::from_shape_vec((ne, nr, nc), read_f32(&dir.join("series.f32"), ne * nr * nc)).unwrap();
    let tcoh = Array2::from_shape_vec((nr, nc), read_f32(&dir.join("tcoh.f32"), nr * nc)).unwrap();
    let series = DisplacementSeries {
        data: series_data,
        epochs: m.epochs.iter().map(|s| Epoch(s.parse().unwrap())).collect(),
        meta: StackMeta {
            transform: GeoTransform::new(m.geo.lon0, m.geo.lat0, m.geo.dlon, m.geo.dlat),
            crs: None, wavelength_m: SENTINEL1_WAVELENGTH_M, incidence_deg: 39.0, heading_deg: None,
        },
    };
    let feats = extract_features(&series, Some(&tcoh), &FeatureConfig::default()).unwrap();

    // Máscara de coherencia → tabla de features.
    let mask = tcoh.mapv(|v| v.is_finite() && v > 0.7);
    let (x_all, coords_all, names) = feats.to_table(Some(&mask));
    println!("tabla de features: {} puntos × {} features {:?}", x_all.nrows(), x_all.ncols(), names);

    // --- Etiqueta ILUSTRATIVA: |velocidad| > 5 cm/año = "deformación significativa" ---
    let vel_col = names.iter().position(|&n| n == "velocity").unwrap();
    let labels_all: Vec<usize> = (0..x_all.nrows())
        .map(|i| usize::from((x_all[[i, vel_col]] * 100.0).abs() > 5.0))
        .collect();

    // Predictores = todo MENOS las columnas derivadas de la velocidad (anti-fuga).
    let drop = ["velocity", "velocity_std", "cumulative"];
    let keep: Vec<usize> = (0..names.len()).filter(|&j| !drop.contains(&names[j])).collect();
    let keep_names: Vec<&str> = keep.iter().map(|&j| names[j]).collect();

    // Submuestreo determinista para una demo liviana (~6000 puntos).
    let stride = (x_all.nrows() / 6000).max(1);
    let rows: Vec<usize> = (0..x_all.nrows()).step_by(stride).collect();
    let x = x_all.select(Axis(0), &rows).select(Axis(1), &keep);
    let labels: Vec<usize> = rows.iter().map(|&i| labels_all[i]).collect();
    let coords: Vec<(f64, f64)> = rows.iter().map(|&i| coords_all[i]).collect();
    let pos = labels.iter().filter(|&&l| l == 1).count();
    println!("muestras: {} ({} positivas), predictores: {:?}", x.nrows(), pos, keep_names);

    let task = ClassificationTask::new("maule_deform", x.clone(), labels.clone()).unwrap();
    let measures: Vec<&dyn Measure> = vec![&Accuracy, &F1Score];

    // --- CV ALEATORIA vs CV ESPACIAL (la espacial es la honesta bajo autocorrelación) ---
    let mut rf = RandomForest::new().with_n_estimators(200);
    let random_cv = CrossValidation::new(5);
    let r_rand = benchmark::resample_classif(&mut rf, &task, &random_cv, &measures).unwrap();
    let mut rf2 = RandomForest::new().with_n_estimators(200);
    let spatial_cv = SpatialBlockCV::new(5, coords.clone());
    let r_spat = benchmark::resample_classif(&mut rf2, &task, &spatial_cv, &measures).unwrap();
    let s_rand = r_rand.mean_scores();
    let s_spat = r_spat.mean_scores();
    println!("\nCV aleatoria (optimista): Accuracy={:.3}  F1={:.3}", s_rand[0], s_rand[1]);
    println!("CV espacial  (honesta):   Accuracy={:.3}  F1={:.3}", s_spat[0], s_spat[1]);

    // --- Predicción conforme: incertidumbre calibrada por punto ---
    let n = x.nrows();
    let cut = n * 7 / 10;
    let tr_idx: Vec<usize> = (0..cut).collect();
    let cal_idx: Vec<usize> = (cut..n).collect();
    let tr = ClassificationTask::new(
        "tr",
        x.select(Axis(0), &tr_idx),
        tr_idx.iter().map(|&i| labels[i]).collect(),
    ).unwrap();
    let mut rf3 = RandomForest::new().with_n_estimators(200);
    let model = rf3.train_classif(&tr).unwrap();
    let cal_x = x.select(Axis(0), &cal_idx);
    let cal_y: Vec<usize> = cal_idx.iter().map(|&i| labels[i]).collect();
    let conf = ConformalClassifier::calibrate(model.as_ref(), &cal_x, &cal_y, 0.1).unwrap();
    let sets = conf.predict(&cal_x).unwrap();
    let avg_set: f64 = sets.iter().map(|s| s.prediction_set.len() as f64).sum::<f64>() / sets.len() as f64;
    let singletons = sets.iter().filter(|s| s.prediction_set.len() == 1).count();
    println!(
        "\nConformal (α=0.1): tamaño medio del conjunto = {:.2}; {:.0}% predicciones únicas (alta confianza)",
        avg_set, 100.0 * singletons as f64 / sets.len() as f64
    );
}

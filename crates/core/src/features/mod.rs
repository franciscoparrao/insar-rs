//! Extracción de **descriptores por píxel** de una serie temporal de
//! desplazamiento, para alimentar modelos de clasificación o regresión
//! (susceptibilidad de deslizamientos, unrest volcánico, nowcasting, etc.).
//!
//! La idea: una [`DisplacementSeries`] (épocas × filas × cols) se resume, por
//! píxel, en un vector de features interpretables. El núcleo es un único ajuste
//! por mínimos cuadrados que descompone la serie temporal `d(t)` en
//!
//! ```text
//!   d(t) = c0 + c1·t + c2·t²  +  A·sin(2π t) + B·cos(2π t)  + residuo
//!          └constante┘ └tendencia┘ └acel.┘   └─── ciclo anual ───┘
//! ```
//!
//! de donde salen: velocidad (`c1`), aceleración (`2·c2`), amplitud/fase
//! estacional (`√(A²+B²)`, `atan2`), bondad de ajuste (`R²`, RMS del residuo) y
//! detectores de evento (mayor salto entre épocas). La coherencia temporal se
//! adjunta como feature de calidad.
//!
//! Las salidas se entregan como **mapas** (un `Array2` por feature, exportables
//! a GeoTIFF) y como **tabla** (`n_puntos × n_features`) lista para ML.
//!
//! ## Integración con Smelt (ML nativo en Rust)
//!
//! La tabla se devuelve como `Array2<f64>` para entrar directo al motor ML de
//! la familia, **Smelt** (`smelt-ml`, mismo `ndarray 0.16`), sin Python ni
//! copias:
//!
//! ```ignore
//! use smelt_ml::prelude::*;
//! let (x, coords, _names) = feats.to_table(Some(&coherent_mask));
//! let task = ClassificationTask::new("deslizamientos", x, labels)?;   // labels: inventario
//! let model = RandomForest::new().with_n_estimators(300).train_classif(&task)?;
//! // `coords` (x,y geográficos) alimentan la CV ESPACIAL de Smelt (sin fuga por
//! // autocorrelación) y la predicción conforme da incertidumbre calibrada por píxel.
//! ```
//!
//! Para regresión (nowcast / tasa) es análogo con `RegressionTask`. Cruzando
//! estas columnas con las de terreno de **SurtGIS** (pendiente, aspecto, TWI)
//! se arma la matriz de features completa, toda en Rust.

use std::f64::consts::PI;
use std::path::Path;

use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, Axis};
use rayon::prelude::*;

use crate::error::{InsarError, Result};
use crate::types::{DisplacementSeries, StackMeta};

/// Qué componentes ajustar / qué features calcular.
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Ajustar el ciclo anual (`A·sin + B·cos`) → amplitud y fase estacional.
    pub seasonal: bool,
    /// Ajustar el término cuadrático → aceleración.
    pub acceleration: bool,
    /// Mínimo de épocas finitas para computar features (si no, NaN).
    pub min_valid_epochs: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self { seasonal: true, acceleration: true, min_valid_epochs: 5 }
    }
}

/// Conjunto de mapas de features (uno por descriptor). Cada `Array2` es
/// `filas × cols`; NaN donde el píxel no se pudo describir.
#[derive(Debug, Clone)]
pub struct FeatureMaps {
    /// Velocidad LOS media (m/año), pendiente lineal.
    pub velocity: Array2<f32>,
    /// Error estándar de la velocidad (m/año).
    pub velocity_std: Array2<f32>,
    /// Aceleración LOS (m/año²); `NaN` si `!config.acceleration`.
    pub acceleration: Array2<f32>,
    /// Bondad del ajuste lineal+modelo: R² en [0, 1] (1 = serie bien explicada).
    pub linearity_r2: Array2<f32>,
    /// RMS del residuo tras el ajuste (m) — ruido / dinámica no modelada.
    pub residual_rms: Array2<f32>,
    /// Desplazamiento acumulado total (m): `d(t_final) − d(t_0)`.
    pub cumulative: Array2<f32>,
    /// Amplitud del ciclo anual (m); `NaN` si `!config.seasonal`.
    pub seasonal_amplitude: Array2<f32>,
    /// Fase del ciclo anual (rad, fecha del máximo); `NaN` si `!config.seasonal`.
    pub seasonal_phase: Array2<f32>,
    /// Mayor salto absoluto entre épocas consecutivas (m) — detector de evento.
    pub max_step: Array2<f32>,
    /// Coherencia temporal adjunta como feature de calidad (si se pasó).
    pub temporal_coherence: Option<Array2<f32>>,
    /// Georreferencia compartida.
    pub meta: StackMeta,
}

/// Extrae los mapas de features de la serie. `quality` (coherencia temporal,
/// p. ej. de [`crate::inversion::temporal_coherence`]) se adjunta como feature
/// y puede usarse luego para enmascarar la tabla. Error si la serie tiene menos
/// de `config.min_valid_epochs` épocas.
pub fn extract_features(
    series: &DisplacementSeries,
    quality: Option<&Array2<f32>>,
    config: &FeatureConfig,
) -> Result<FeatureMaps> {
    let n_epochs = series.n_layers();
    let (n_rows, n_cols) = series.dims();

    if series.epochs.len() != n_epochs {
        return Err(InsarError::DimensionMismatch(format!(
            "{} épocas declaradas vs {n_epochs} capas en la serie",
            series.epochs.len()
        )));
    }
    if n_epochs < config.min_valid_epochs {
        return Err(InsarError::DimensionMismatch(format!(
            "se requieren al menos {} épocas para extraer features ({n_epochs} recibidas)",
            config.min_valid_epochs
        )));
    }

    // Tiempo en años decimales relativo a la primera época.
    let t: Vec<f64> = series
        .epochs
        .iter()
        .map(|e| e.years_since(&series.epochs[0]))
        .collect();

    // ----- Matriz de diseño (idéntica para todos los píxeles) -----
    // Columnas, en orden: [1, t, (t² si accel), (sin 2πt, cos 2πt si seasonal)].
    let col_t = 1usize; // la columna de velocidad es siempre la 1.
    let col_t2 = if config.acceleration { Some(2usize) } else { None };
    let n_lin = 2 + usize::from(config.acceleration); // constante + lineal (+ cuadrático)
    let (col_sin, col_cos) = if config.seasonal {
        (Some(n_lin), Some(n_lin + 1))
    } else {
        (None, None)
    };
    let n_coef = n_lin + if config.seasonal { 2 } else { 0 };

    if n_epochs < n_coef {
        return Err(InsarError::DimensionMismatch(format!(
            "el modelo tiene {n_coef} coeficientes pero solo hay {n_epochs} épocas"
        )));
    }

    // A: n_epochs × n_coef.
    let a = DMatrix::<f64>::from_fn(n_epochs, n_coef, |i, j| {
        let ti = t[i];
        if j == 0 {
            1.0
        } else if j == col_t {
            ti
        } else if Some(j) == col_t2 {
            ti * ti
        } else if Some(j) == col_sin {
            (2.0 * PI * ti).sin()
        } else if Some(j) == col_cos {
            (2.0 * PI * ti).cos()
        } else {
            0.0
        }
    });

    // Pseudoinversa cacheada (n_coef × n_epochs) con tolerancia rcond estilo
    // numpy/LAPACK: σ_max · max(m,n) · ε_f64 (igual que en inversion).
    let svd = a.clone().svd(true, true);
    let s_max = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let eps = s_max * (n_epochs.max(n_coef) as f64) * f64::EPSILON;
    let pinv = svd
        .pseudo_inverse(eps)
        .map_err(|e| InsarError::Inversion(format!("pseudoinversa SVD de la matriz de diseño: {e}")))?;

    // g = Σ_k pinv[col_t, k]²  → factor de varianza de la velocidad (una vez).
    let g_vel: f64 = (0..n_epochs).map(|k| pinv[(col_t, k)].powi(2)).sum();

    // ----- Mapas de salida -----
    let mut velocity = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut velocity_std = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut acceleration = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut linearity_r2 = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut residual_rms = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut cumulative = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut seasonal_amplitude = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut seasonal_phase = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut max_step = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);

    let data = series.data.view();

    // Vista mutable por columnas espaciales = filas de la grilla (eje 1 del
    // Array3 = filas); paralelizamos por filas como en inversion.
    let mut vel_r: Vec<_> = velocity.axis_iter_mut(Axis(0)).collect();
    let mut vstd_r: Vec<_> = velocity_std.axis_iter_mut(Axis(0)).collect();
    let mut acc_r: Vec<_> = acceleration.axis_iter_mut(Axis(0)).collect();
    let mut r2_r: Vec<_> = linearity_r2.axis_iter_mut(Axis(0)).collect();
    let mut rms_r: Vec<_> = residual_rms.axis_iter_mut(Axis(0)).collect();
    let mut cum_r: Vec<_> = cumulative.axis_iter_mut(Axis(0)).collect();
    let mut samp_r: Vec<_> = seasonal_amplitude.axis_iter_mut(Axis(0)).collect();
    let mut sph_r: Vec<_> = seasonal_phase.axis_iter_mut(Axis(0)).collect();
    let mut step_r: Vec<_> = max_step.axis_iter_mut(Axis(0)).collect();

    // Empaquetamos los mutables por fila en tuplas para un único `par_iter`.
    let rows_iter = vel_r
        .par_iter_mut()
        .zip(vstd_r.par_iter_mut())
        .zip(acc_r.par_iter_mut())
        .zip(r2_r.par_iter_mut())
        .zip(rms_r.par_iter_mut())
        .zip(cum_r.par_iter_mut())
        .zip(samp_r.par_iter_mut())
        .zip(sph_r.par_iter_mut())
        .zip(step_r.par_iter_mut())
        .enumerate();

    rows_iter.for_each(
        |(r, ((((((((vel, vstd), acc), r2), rms), cum), samp), sph), step))| {
            let mut d = DVector::<f64>::zeros(n_epochs);
            for c in 0..n_cols {
                // Validez: TODAS las épocas finitas, si no → todas las features NaN.
                let mut valid = true;
                for e in 0..n_epochs {
                    let v = data[[e, r, c]];
                    if !v.is_finite() {
                        valid = false;
                        break;
                    }
                    d[e] = v as f64;
                }
                if !valid {
                    continue;
                }

                // coef = pinv · d.
                let coef = &pinv * &d;

                // Ajuste y residuo.
                let fitted = &a * &coef;
                let mut ss_res = 0.0_f64;
                let mut d_sum = 0.0_f64;
                for e in 0..n_epochs {
                    let resid = d[e] - fitted[e];
                    ss_res += resid * resid;
                    d_sum += d[e];
                }
                let d_mean = d_sum / n_epochs as f64;
                let mut ss_tot = 0.0_f64;
                for e in 0..n_epochs {
                    let dev = d[e] - d_mean;
                    ss_tot += dev * dev;
                }

                // Velocidad (m/año).
                vel[c] = coef[col_t] as f32;

                // Aceleración (m/año²) = 2·c2.
                if let Some(ct2) = col_t2 {
                    acc[c] = (2.0 * coef[ct2]) as f32;
                }

                // Estacional: amplitud √(A²+B²), fase atan2(B, A).
                if let (Some(cs), Some(cc)) = (col_sin, col_cos) {
                    let amp_a = coef[cs];
                    let amp_b = coef[cc];
                    samp[c] = (amp_a * amp_a + amp_b * amp_b).sqrt() as f32;
                    sph[c] = amp_b.atan2(amp_a) as f32;
                }

                // R² del ajuste (clamp [0,1]; SS_tot≈0 → 1).
                let r2_val = if ss_tot <= f64::EPSILON {
                    1.0
                } else {
                    (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
                };
                r2[c] = r2_val as f32;

                // RMS del residuo (m).
                rms[c] = (ss_res / n_epochs as f64).sqrt() as f32;

                // Error estándar de la velocidad: sqrt(σ²·g), σ²=SS_res/(n−n_coef).
                if n_epochs > n_coef {
                    let sigma2 = ss_res / (n_epochs - n_coef) as f64;
                    vstd[c] = (sigma2 * g_vel).sqrt() as f32;
                }

                // Desplazamiento acumulado total (m).
                cum[c] = (d[n_epochs - 1] - d[0]) as f32;

                // Mayor salto absoluto entre épocas consecutivas (m).
                let mut mstep = 0.0_f64;
                for e in 1..n_epochs {
                    let s = (d[e] - d[e - 1]).abs();
                    if s > mstep {
                        mstep = s;
                    }
                }
                step[c] = mstep as f32;
            }
        },
    );

    // Coherencia temporal adjunta como feature de calidad (clon si se pasó).
    let temporal_coherence = quality.cloned();

    Ok(FeatureMaps {
        velocity,
        velocity_std,
        acceleration,
        linearity_r2,
        residual_rms,
        cumulative,
        seasonal_amplitude,
        seasonal_phase,
        max_step,
        temporal_coherence,
        meta: series.meta.clone(),
    })
}

impl FeatureMaps {
    /// Nombres de las features, en el mismo orden que las columnas de
    /// [`Self::to_table`]. Excluye las desactivadas (todo-NaN) en la config.
    pub fn feature_names(&self) -> Vec<&'static str> {
        // Una feature opcional está "activa" si su mapa tiene algún valor finito.
        let active = |m: &Array2<f32>| m.iter().any(|v| v.is_finite());
        let accel_on = active(&self.acceleration);
        let seasonal_on = active(&self.seasonal_amplitude);

        let mut names = Vec::new();
        names.push("velocity");
        names.push("velocity_std");
        if accel_on {
            names.push("acceleration");
        }
        names.push("linearity_r2");
        names.push("residual_rms");
        names.push("cumulative");
        if seasonal_on {
            names.push("seasonal_amplitude");
            names.push("seasonal_phase");
        }
        names.push("max_step");
        if self.temporal_coherence.is_some() {
            names.push("temporal_coherence");
        }
        names
    }

    /// Mapas activos en el mismo orden que [`Self::feature_names`].
    fn active_maps(&self) -> Vec<&Array2<f32>> {
        let active = |m: &Array2<f32>| m.iter().any(|v| v.is_finite());
        let accel_on = active(&self.acceleration);
        let seasonal_on = active(&self.seasonal_amplitude);

        let mut maps: Vec<&Array2<f32>> = Vec::new();
        maps.push(&self.velocity);
        maps.push(&self.velocity_std);
        if accel_on {
            maps.push(&self.acceleration);
        }
        maps.push(&self.linearity_r2);
        maps.push(&self.residual_rms);
        maps.push(&self.cumulative);
        if seasonal_on {
            maps.push(&self.seasonal_amplitude);
            maps.push(&self.seasonal_phase);
        }
        maps.push(&self.max_step);
        if let Some(tc) = &self.temporal_coherence {
            maps.push(tc);
        }
        maps
    }

    /// Matriz tabular `(n_puntos × n_features)` en `f64` (lista para
    /// `smelt_ml::ClassificationTask`/`RegressionTask`), más las **coordenadas
    /// geográficas** `(x, y)` de cada punto (derivadas del `GeoTransform`, para
    /// la CV espacial de Smelt) y los nombres de columna. Incluye solo los
    /// píxeles que pasan `mask` (p. ej. coherencia > umbral) y sin NaN.
    pub fn to_table(
        &self,
        mask: Option<&Array2<bool>>,
    ) -> (Array2<f64>, Vec<(f64, f64)>, Vec<&'static str>) {
        let names = self.feature_names();
        let maps = self.active_maps();
        let n_feat = maps.len();
        let (n_rows, n_cols) = self.velocity.dim();

        let mut rows: Vec<f64> = Vec::new();
        let mut coords: Vec<(f64, f64)> = Vec::new();

        for r in 0..n_rows {
            for c in 0..n_cols {
                if let Some(m) = mask
                    && !m[[r, c]]
                {
                    continue;
                }
                // Solo píxeles con TODAS sus features activas finitas.
                let mut all_finite = true;
                for map in &maps {
                    if !map[[r, c]].is_finite() {
                        all_finite = false;
                        break;
                    }
                }
                if !all_finite {
                    continue;
                }
                for map in &maps {
                    rows.push(map[[r, c]] as f64);
                }
                // pixel_to_geo toma (col, row).
                coords.push(self.meta.transform.pixel_to_geo(c, r));
            }
        }

        let n_points = coords.len();
        // Invariante por construcción: se acumulan exactamente `n_feat` valores
        // por cada punto aceptado, así que `rows.len() == n_points · n_feat`. Se
        // maneja el `Result` sin `unwrap`/`expect` para no tener panics en la
        // ruta pública; el `unwrap_or_else` nunca se ejecuta en la práctica.
        let table = Array2::<f64>::from_shape_vec((n_points, n_feat), rows)
            .unwrap_or_else(|_| Array2::<f64>::zeros((0, n_feat)));
        (table, coords, names)
    }

    /// Escribe cada mapa de feature como un GeoTIFF Float32 en `dir`
    /// (`velocity.tif`, `acceleration.tif`, …), vía el writer de [`crate::io`].
    pub fn write_geotiffs(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let names = self.feature_names();
        let maps = self.active_maps();
        for (name, map) in names.iter().zip(maps.iter()) {
            let vm = crate::types::VelocityMap {
                data: (*map).clone(),
                meta: self.meta.clone(),
            };
            crate::io::write_velocity(&vm, &dir.join(format!("{name}.tif")))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // En los tests `e` indexa la serie 3D y el vector de verdad d[e].
    #![allow(clippy::needless_range_loop)]
    use super::*;
    use crate::types::{Epoch, SENTINEL1_WAVELENGTH_M};
    use ndarray::Array3;
    use surtgis_core::GeoTransform;

    fn meta() -> StackMeta {
        StackMeta {
            transform: GeoTransform::new(0.0, 0.0, 1.0, -1.0),
            crs: None,
            wavelength_m: SENTINEL1_WAVELENGTH_M,
            incidence_deg: 39.0,
            heading_deg: None,
        }
    }

    /// `n` épocas separadas `days` días desde 2023-01-01.
    fn epochs_n(n: usize, days: i64) -> Vec<Epoch> {
        let start: chrono::NaiveDate = "2023-01-01".parse().unwrap();
        (0..n)
            .map(|i| Epoch(start + chrono::Duration::days(days * i as i64)))
            .collect()
    }

    /// Serie de un solo píxel (1×1) a partir de los desplazamientos por época.
    fn series_1px(epochs: &[Epoch], d: &[f64]) -> DisplacementSeries {
        let n = epochs.len();
        let mut data = Array3::<f32>::zeros((n, 1, 1));
        for e in 0..n {
            data[[e, 0, 0]] = d[e] as f32;
        }
        DisplacementSeries { data, epochs: epochs.to_vec(), meta: meta() }
    }

    fn cfg() -> FeatureConfig {
        FeatureConfig { seasonal: true, acceleration: true, min_valid_epochs: 5 }
    }

    #[test]
    fn lineal_puro_recupera_velocidad() {
        let v_true = -0.05_f64;
        let epochs = epochs_n(8, 24); // 8 épocas, ~0.46 años de span
        let t: Vec<f64> = epochs.iter().map(|e| e.years_since(&epochs[0])).collect();
        let d: Vec<f64> = t.iter().map(|&ti| v_true * ti).collect();
        let series = series_1px(&epochs, &d);

        let f = extract_features(&series, None, &cfg()).unwrap();
        assert!((f.velocity[[0, 0]] as f64 - v_true).abs() < 1e-5, "v = {}", f.velocity[[0, 0]]);
        assert!(f.acceleration[[0, 0]].abs() < 1e-4, "acc = {}", f.acceleration[[0, 0]]);
        assert!((f.linearity_r2[[0, 0]] - 1.0).abs() < 1e-4, "r2 = {}", f.linearity_r2[[0, 0]]);
        assert!(f.residual_rms[[0, 0]] < 1e-5, "rms = {}", f.residual_rms[[0, 0]]);
        // Acumulado = v·t_final.
        assert!((f.cumulative[[0, 0]] as f64 - v_true * t[t.len() - 1]).abs() < 1e-5);
    }

    #[test]
    fn cuadratico_detecta_aceleracion() {
        // d = 0.5·a·t²  con a > 0  → aceleración recuperada = a > 0.
        let a_true = 0.08_f64;
        let epochs = epochs_n(10, 30);
        let t: Vec<f64> = epochs.iter().map(|e| e.years_since(&epochs[0])).collect();
        let d: Vec<f64> = t.iter().map(|&ti| 0.5 * a_true * ti * ti).collect();
        let series = series_1px(&epochs, &d);

        let f = extract_features(&series, None, &cfg()).unwrap();
        let acc = f.acceleration[[0, 0]] as f64;
        assert!(acc > 0.0, "aceleración debería ser positiva: {acc}");
        assert!((acc - a_true).abs() < 1e-3, "acc = {acc} vs {a_true}");
    }

    #[test]
    fn estacional_recupera_amplitud() {
        // d = A·sin(2π t), ≥3 años de épocas mensuales.
        let amp_true = 0.03_f64;
        let epochs = epochs_n(40, 30); // ~3.2 años
        let t: Vec<f64> = epochs.iter().map(|e| e.years_since(&epochs[0])).collect();
        let d: Vec<f64> = t.iter().map(|&ti| amp_true * (2.0 * PI * ti).sin()).collect();
        let series = series_1px(&epochs, &d);

        let f = extract_features(&series, None, &cfg()).unwrap();
        let amp = f.seasonal_amplitude[[0, 0]] as f64;
        assert!((amp - amp_true).abs() < 1e-3, "amplitud = {amp} vs {amp_true}");
    }

    #[test]
    fn nan_en_serie_propaga_a_todas_las_features() {
        let epochs = epochs_n(8, 24);
        let t: Vec<f64> = epochs.iter().map(|e| e.years_since(&epochs[0])).collect();
        let d: Vec<f64> = t.iter().map(|&ti| -0.05 * ti).collect();
        let mut series = series_1px(&epochs, &d);
        series.data[[3, 0, 0]] = f32::NAN;

        let f = extract_features(&series, None, &cfg()).unwrap();
        assert!(f.velocity[[0, 0]].is_nan());
        assert!(f.velocity_std[[0, 0]].is_nan());
        assert!(f.acceleration[[0, 0]].is_nan());
        assert!(f.linearity_r2[[0, 0]].is_nan());
        assert!(f.residual_rms[[0, 0]].is_nan());
        assert!(f.cumulative[[0, 0]].is_nan());
        assert!(f.seasonal_amplitude[[0, 0]].is_nan());
        assert!(f.seasonal_phase[[0, 0]].is_nan());
        assert!(f.max_step[[0, 0]].is_nan());
    }

    #[test]
    fn max_step_detecta_salto() {
        // Serie casi plana con un salto inyectado de 0.2 m entre épocas 4 y 5.
        let epochs = epochs_n(8, 24);
        let mut d = vec![0.0; 8];
        for e in 5..8 {
            d[e] = 0.2;
        }
        let series = series_1px(&epochs, &d);

        let f = extract_features(&series, None, &cfg()).unwrap();
        assert!((f.max_step[[0, 0]] as f64 - 0.2).abs() < 1e-5, "max_step = {}", f.max_step[[0, 0]]);
    }

    #[test]
    fn to_table_excluye_nan_y_aplica_mascara() {
        // Grilla 1×2: píxel (0,0) válido lineal, píxel (0,1) con un NaN.
        let epochs = epochs_n(8, 24);
        let t: Vec<f64> = epochs.iter().map(|e| e.years_since(&epochs[0])).collect();
        let n = epochs.len();
        let mut data = Array3::<f32>::zeros((n, 1, 2));
        for e in 0..n {
            data[[e, 0, 0]] = (-0.05 * t[e]) as f32;
            data[[e, 0, 1]] = (0.02 * t[e]) as f32;
        }
        data[[3, 0, 1]] = f32::NAN; // píxel (0,1) inválido
        let series = DisplacementSeries { data, epochs, meta: meta() };

        let f = extract_features(&series, None, &cfg()).unwrap();

        // Máscara que admite ambos píxeles; el NaN del (0,1) debe excluirlo igual.
        let mask = Array2::<bool>::from_elem((1, 2), true);
        let (table, coords, names) = f.to_table(Some(&mask));

        assert_eq!(table.ncols(), names.len());
        assert_eq!(table.ncols(), f.feature_names().len());
        assert_eq!(table.nrows(), 1, "solo el píxel válido debe quedar");
        assert_eq!(coords.len(), 1);
        // Coord del píxel (0,0) vía pixel_to_geo(col=0, row=0).
        let expected = meta().transform.pixel_to_geo(0, 0);
        assert!((coords[0].0 - expected.0).abs() < 1e-9);
        assert!((coords[0].1 - expected.1).abs() < 1e-9);
        // La primera columna es velocity ≈ -0.05.
        assert!((table[[0, 0]] - (-0.05)).abs() < 1e-4, "velocity tabla = {}", table[[0, 0]]);
    }

    #[test]
    fn feature_names_excluye_inactivas() {
        let epochs = epochs_n(8, 24);
        let t: Vec<f64> = epochs.iter().map(|e| e.years_since(&epochs[0])).collect();
        let d: Vec<f64> = t.iter().map(|&ti| -0.05 * ti).collect();
        let series = series_1px(&epochs, &d);

        // Sin seasonal ni acceleration ni coherencia.
        let config = FeatureConfig { seasonal: false, acceleration: false, min_valid_epochs: 5 };
        let f = extract_features(&series, None, &config).unwrap();
        let names = f.feature_names();
        assert!(!names.contains(&"acceleration"));
        assert!(!names.contains(&"seasonal_amplitude"));
        assert!(!names.contains(&"seasonal_phase"));
        assert!(!names.contains(&"temporal_coherence"));
        assert!(names.contains(&"velocity"));
        assert!(names.contains(&"max_step"));

        // Con coherencia: aparece la columna.
        let coh = Array2::<f32>::from_elem((1, 1), 0.9);
        let f2 = extract_features(&series, Some(&coh), &config).unwrap();
        assert!(f2.feature_names().contains(&"temporal_coherence"));
        assert!(f2.temporal_coherence.is_some());
    }

    #[test]
    fn pocas_epocas_es_error() {
        let epochs = epochs_n(4, 24);
        let d = vec![0.0; 4];
        let series = series_1px(&epochs, &d);
        let err = extract_features(&series, None, &cfg()).unwrap_err();
        assert!(matches!(err, InsarError::DimensionMismatch(_)));
    }
}

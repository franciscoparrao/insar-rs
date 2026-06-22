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

use std::path::Path;

use ndarray::Array2;

use crate::error::Result;
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
    let _ = (series, quality, config);
    todo!("features — ver PLAN.md: ajuste LSQ por píxel (constante+lineal+cuadrático+anual)")
}

impl FeatureMaps {
    /// Nombres de las features, en el mismo orden que las columnas de
    /// [`Self::to_table`]. Excluye las desactivadas (todo-NaN) en la config.
    pub fn feature_names(&self) -> Vec<&'static str> {
        todo!("features — orden de columnas estable")
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
        let _ = mask;
        todo!("features — apilar features por píxel válido; coords vía meta.transform")
    }

    /// Escribe cada mapa de feature como un GeoTIFF Float32 en `dir`
    /// (`velocity.tif`, `acceleration.tif`, …), vía el writer de [`crate::io`].
    pub fn write_geotiffs(&self, dir: &Path) -> Result<()> {
        let _ = dir;
        todo!("features — un GeoTIFF por feature")
    }
}

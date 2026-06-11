//! Corrección atmosférica simple (alcance MVP): el APS se estima como la
//! componente pasa-bajo espacial / pasa-alto temporal de la serie
//! (esquema SBAS clásico) y se resta. APS avanzado (modelos meteorológicos,
//! GACOS) es v0.2.

use crate::error::Result;
use crate::types::DisplacementSeries;

/// Parámetros del filtro espacio-temporal.
#[derive(Debug, Clone)]
pub struct ApsConfig {
    /// Sigma del filtro gaussiano espacial, en píxeles.
    pub spatial_sigma_px: f32,
    /// Ventana de la media móvil temporal, en épocas (impar).
    pub temporal_window_epochs: usize,
}

impl Default for ApsConfig {
    fn default() -> Self {
        Self { spatial_sigma_px: 8.0, temporal_window_epochs: 5 }
    }
}

/// Estima y remueve la señal atmosférica de la serie, in place.
/// NaN se propaga sin contaminar vecinos (filtro con normalización por
/// pesos válidos).
pub fn correct_aps(series: &mut DisplacementSeries, config: &ApsConfig) -> Result<()> {
    let _ = (series, config);
    todo!("Fase 2, módulo atmosphere — ver PLAN.md")
}

//! Pipeline SBAS end-to-end: lectura → (PS opcional) → desenrollado →
//! inversión → corrección APS → velocidad. Nivel 1 del plan — se implementa
//! cuando los módulos de Nivel 0 estén verdes.

use std::path::PathBuf;

use crate::atmosphere::ApsConfig;
use crate::error::Result;
use crate::network::SbasConfig;
use crate::types::{DisplacementSeries, VelocityMap};

/// Configuración del pipeline completo.
#[derive(Debug, Clone)]
pub struct SbasPipelineConfig {
    /// Directorio del stack de entrada (formato del módulo `io`).
    pub input_dir: PathBuf,
    /// Directorio de salida para velocity.tif y la serie por época.
    pub output_dir: PathBuf,
    /// Umbral de amplitude dispersion; `None` = invertir toda la grilla.
    pub ps_threshold: Option<f32>,
    pub network: SbasConfig,
    pub aps: ApsConfig,
}

/// Productos del pipeline.
#[derive(Debug, Clone)]
pub struct SbasProducts {
    pub velocity: VelocityMap,
    pub series: DisplacementSeries,
}

/// Ejecuta el pipeline SBAS completo y escribe los productos en
/// `config.output_dir`.
pub fn run_sbas(config: &SbasPipelineConfig) -> Result<SbasProducts> {
    let _ = config;
    todo!("Fase 3, módulo pipeline — ver PLAN.md")
}

//! Pipeline SBAS end-to-end: lectura → (PS opcional) → desenrollado →
//! corrección de cierre → referenciado → inversión (OLS/WLS ± error de DEM) →
//! corrección APS → deramp → velocidad → escritura de productos.
//!
//! ## Flujo de [`run_sbas`] (orden físico del procesamiento)
//!
//! 1. Lee el stack de interferogramas (`io::read_ifg_stack`) y, si el
//!    manifiesto la declara, la coherencia (`io::read_coherence_stack`).
//! 2. Si hay `ps_threshold`, lee amplitudes del mismo directorio, calcula
//!    amplitude dispersion y selecciona PS; sin umbral se invierte toda la
//!    grilla (modo SBAS clásico).
//! 3. Desenrolla la fase de cada par con el backend configurado
//!    ([`SbasPipelineConfig::unwrap_backend`]): flood-fill propio
//!    (`unwrap::unwrap_stack_min_quality`, default, con el umbral opcional
//!    [`SbasPipelineConfig::unwrap_min_quality`]) o SNAPHU externo
//!    (`unwrap::snaphu::unwrap_stack_snaphu`, requiere el binario `snaphu`).
//!    En ambos casos la coherencia se usa como mapa de calidad.
//! 4. Corrige errores de desenrollado por cierre de fase
//!    (`unwrap_error::correct_unwrap_errors`, si `correct_unwrap`) y computa
//!    el QC de cierres residuales (`unwrap_error::nonzero_closure_count`).
//! 5. Referencia espacialmente el stack (`inversion::reference_to_pixel`):
//!    al píxel configurado, o al de máxima coherencia media si hay
//!    coherencia; sin coherencia ni configuración, no se referencia.
//! 6. Invierte la serie LOS (`inversion::invert_sbas_ext`): pesos WLS por
//!    coherencia y/o estimación de error de DEM según
//!    [`SbasPipelineConfig::solver`].
//! 7. Corrige APS turbulento (`atmosphere::correct_aps`) — ver regla de
//!    salto abajo.
//! 8. Deramp por época (`postprocess::deramp_series`, si `deramp`).
//! 9. Estima la velocidad media LOS y la coherencia temporal; escribe los
//!    productos en `output_dir`.
//!
//! ## Orden troposfera estratificada vs APS turbulento
//!
//! La componente troposférica **estratificada** (fase-elevación,
//! [`crate::troposphere`]) es estacional — correlada en el tiempo — así que
//! el filtro pasa-alto temporal del APS **no** la remueve: debe corregirse
//! ANTES del filtro turbulento. Como requiere un DEM y el formato de stack
//! v0.1 no lo incluye, este pipeline no la aplica: para escenas con
//! topografía use la API ([`crate::troposphere::correct_topo_series`] sobre
//! `SbasProducts::series` re-derivando velocidad) o espere al formato v0.2.
//!
//! ## Limitaciones y decisiones
//!
//! - **Salto de APS**: la corrección atmosférica se **salta** si la serie
//!   tiene menos de 3 épocas o si `aps.temporal_window_epochs <= 1` — en
//!   ambos casos el filtro pasa-alto temporal degenera (no hay ventana
//!   centrada con información) y no puede separar atmósfera de deformación.
//! - **`network` ([`SbasConfig`]) NO se usa en v0.1**: los pares vienen
//!   dados en el stack de entrada (campo `ifgs` del manifiesto).
//!   `network::build_network` se usará cuando se generen interferogramas
//!   desde SLC (v0.2). El campo se mantiene en la config como parte del
//!   contrato del pipeline.

use std::fs;
use std::path::PathBuf;

use ndarray::Array2;

use crate::atmosphere::{self, ApsConfig};
use crate::error::{InsarError, IoResultExt, Result};
use crate::inversion::SbasSolverConfig;
use crate::network::SbasConfig;
use crate::postprocess::RampKind;
use crate::types::{DisplacementSeries, PsCandidate, VelocityMap};
use crate::unwrap_error::UnwrapCorrectionReport;
use crate::{inversion, io, postprocess, ps, unwrap, unwrap_error};

/// Backend de desenrollado 2D usado por [`run_sbas`] (paso 3).
#[derive(Debug, Clone, Default)]
pub enum UnwrapBackend {
    /// Flood-fill quality-guided propio (ver [`unwrap`]) — default, sin
    /// dependencias externas.
    #[default]
    FloodFill,
    /// Shell-out a un binario `snaphu` instalado por separado (ver
    /// [`unwrap::snaphu`], gap G-3 de la auditoría). Requiere el binario en
    /// PATH o en [`unwrap::snaphu::SnaphuConfig::binary`].
    Snaphu(unwrap::snaphu::SnaphuConfig),
}

/// Configuración del pipeline completo. Construir con
/// [`SbasPipelineConfig::new`] y ajustar campos según necesidad.
#[derive(Debug, Clone)]
pub struct SbasPipelineConfig {
    /// Directorio del stack de entrada (formato del módulo `io`).
    pub input_dir: PathBuf,
    /// Directorio de salida para los productos.
    pub output_dir: PathBuf,
    /// Umbral de amplitude dispersion; `None` = invertir toda la grilla.
    pub ps_threshold: Option<f32>,
    pub network: SbasConfig,
    pub aps: ApsConfig,
    /// Umbral mínimo de coherencia para desenrollar un píxel (requiere
    /// coherencia declarada en el manifiesto). `None` = sin umbral. Solo
    /// aplica al backend [`UnwrapBackend::FloodFill`] (default).
    pub unwrap_min_quality: Option<f32>,
    /// Backend de desenrollado 2D (default [`UnwrapBackend::FloodFill`]).
    pub unwrap_backend: UnwrapBackend,
    /// Corregir errores de desenrollado por cierre de fase antes de invertir
    /// (default `true`; es no-op verificado en stacks sin cierres ≠ 0).
    pub correct_unwrap: bool,
    /// Píxel de referencia (fila, col). `None` = automático: máxima
    /// coherencia media si hay coherencia; sin referenciar si no la hay.
    pub reference: Option<(usize, usize)>,
    /// Restringe la auto-selección (cuando `reference` es `None`) a este
    /// rectángulo `(fila_min, col_min, fila_max, col_max)`, inclusivo.
    /// Necesario cuando el stack cubre un área mucho más grande que el AOI
    /// real (p. ej. el bbox de `stackSentinel.py` solo filtra bursts, no
    /// recorta el producto final) — sin esto la auto-selección puede caer
    /// arbitrariamente lejos del área de interés. Ignorado si `reference`
    /// ya viene fijado manualmente.
    pub reference_region: Option<(usize, usize, usize, usize)>,
    /// Solver de la inversión: pesos WLS y/o error de DEM.
    pub solver: SbasSolverConfig,
    /// Deramp de cada época de la serie tras las correcciones (`None` = no).
    pub deramp: Option<RampKind>,
}

impl SbasPipelineConfig {
    /// Config con los defaults del pipeline: sin PS, corrección de cierre
    /// activada, referencia automática, OLS sin error de DEM, sin deramp.
    pub fn new(input_dir: PathBuf, output_dir: PathBuf) -> Self {
        Self {
            input_dir,
            output_dir,
            ps_threshold: None,
            network: SbasConfig::default(),
            aps: ApsConfig::default(),
            unwrap_min_quality: None,
            unwrap_backend: UnwrapBackend::default(),
            correct_unwrap: true,
            reference: None,
            reference_region: None,
            solver: SbasSolverConfig::default(),
            deramp: None,
        }
    }
}

/// Productos del pipeline.
#[derive(Debug, Clone)]
pub struct SbasProducts {
    pub velocity: VelocityMap,
    pub series: DisplacementSeries,
    /// Coherencia temporal γ (Pepe & Lanari 2006) del ajuste.
    pub temporal_coherence: Array2<f32>,
    /// Error de DEM Δz (m), si `solver.dem_error` se configuró.
    pub dem_error_m: Option<Array2<f32>>,
    /// Reporte de la corrección de cierre, si `correct_unwrap`.
    pub unwrap_report: Option<UnwrapCorrectionReport>,
}

/// ¿Corresponde aplicar la corrección APS? Ver doc del módulo: con menos de
/// 3 épocas o ventana temporal ≤ 1 el filtro degenera y se salta.
fn should_apply_aps(n_epochs: usize, temporal_window_epochs: usize) -> bool {
    n_epochs >= 3 && temporal_window_epochs > 1
}

/// Selecciona PS leyendo amplitudes desde el directorio del stack.
/// Error [`InsarError::Metadata`] claro si el manifiesto no trae amplitudes
/// o si la selección queda vacía.
fn select_ps_from_dir(input_dir: &std::path::Path, threshold: f32) -> Result<Vec<PsCandidate>> {
    let amps = io::read_amplitude_stack(input_dir).map_err(|e| match e {
        InsarError::Metadata(msg) if msg.contains("no tiene campo 'amplitudes'") => {
            InsarError::Metadata(
                "ps_threshold requiere amplitudes en stack.json (el manifiesto no \
                 declara el campo 'amplitudes')"
                    .to_string(),
            )
        }
        other => other,
    })?;
    let dispersion = ps::amplitude_dispersion(&amps)?;
    let candidates = ps::select_ps(&dispersion, threshold);
    if candidates.is_empty() {
        return Err(InsarError::Metadata(format!(
            "selección de PS vacía: ningún píxel con amplitude dispersion <= {threshold} \
             (sube ps_threshold o usa None para invertir toda la grilla)"
        )));
    }
    Ok(candidates)
}

/// Ejecuta el pipeline SBAS completo y escribe los productos en
/// `config.output_dir`: `velocity.tif`, `series/disp_YYYYMMDD.tif`,
/// `temporal_coherence.tif`, y si corresponde `dem_error.tif` y
/// `closure_qc.tif`.
///
/// Ver doc del módulo para el flujo paso a paso, el orden físico de las
/// correcciones y las limitaciones (troposfera estratificada requiere DEM y
/// se aplica vía API).
pub fn run_sbas(config: &SbasPipelineConfig) -> Result<SbasProducts> {
    // 1) Stack de interferogramas envueltos + coherencia opcional.
    let stack = io::read_ifg_stack(&config.input_dir)?;
    let coherence = io::read_coherence_stack(&config.input_dir)?;
    if let Some(coh) = &coherence
        && coh.dim() != stack.data.dim()
    {
        return Err(InsarError::DimensionMismatch(format!(
            "coherencia {:?} vs stack {:?}",
            coh.dim(),
            stack.data.dim()
        )));
    }

    // 2) Selección de PS (opcional).
    let ps_candidates = match config.ps_threshold {
        Some(threshold) => Some(select_ps_from_dir(&config.input_dir, threshold)?),
        None => None,
    };

    // 3) Desenrollado con la coherencia como calidad (+ umbral opcional).
    let mut unwrapped = match &config.unwrap_backend {
        UnwrapBackend::FloodFill => {
            unwrap::unwrap_stack_min_quality(&stack, coherence.as_ref(), config.unwrap_min_quality)?
        }
        UnwrapBackend::Snaphu(snaphu_config) => {
            unwrap::snaphu::unwrap_stack_snaphu(&stack, coherence.as_ref(), snaphu_config)?
        }
    };

    // 4) Corrección de errores de desenrollado por cierre de fase + QC.
    let (unwrap_report, closure_qc) = if config.correct_unwrap {
        let report = unwrap_error::correct_unwrap_errors(&mut unwrapped)?;
        let qc = unwrap_error::nonzero_closure_count(&unwrapped)?;
        (Some(report), Some(qc))
    } else {
        (None, None)
    };

    // 5) Referenciado espacial (configurado > automático por coherencia,
    //    opcionalmente restringido a `reference_region`).
    let reference = config.reference.or_else(|| {
        let (n_rows, n_cols) = unwrapped.dims();
        let region_mask = config.reference_region.map(|(r0, c0, r1, c1)| {
            let mut mask = Array2::from_elem((n_rows, n_cols), false);
            for r in r0..=r1.min(n_rows.saturating_sub(1)) {
                for c in c0..=c1.min(n_cols.saturating_sub(1)) {
                    mask[[r, c]] = true;
                }
            }
            mask
        });
        coherence
            .as_ref()
            .and_then(|coh| inversion::select_reference_pixel(coh, region_mask.as_ref()))
    });
    if let Some((r, c)) = reference {
        inversion::reference_to_pixel(&mut unwrapped, r, c)?;
    }

    // 6) Inversión SBAS de la serie LOS (OLS/WLS ± error de DEM).
    let solution = inversion::invert_sbas_ext(
        &unwrapped,
        ps_candidates.as_deref(),
        coherence.as_ref(),
        &config.solver,
    )?;
    let mut series = solution.series;

    // 7) Corrección APS turbulento (se salta si la serie es demasiado corta
    //    o la ventana temporal degenera; ver doc del módulo).
    if should_apply_aps(series.epochs.len(), config.aps.temporal_window_epochs) {
        atmosphere::correct_aps(&mut series, &config.aps)?;
    }

    // 8) Deramp por época (opcional).
    if let Some(kind) = config.deramp {
        postprocess::deramp_series(&mut series, kind, None)?;
    }

    // 9) Velocidad media LOS + coherencia temporal del ajuste.
    let velocity = inversion::estimate_velocity(&series)?;
    let gamma = postprocess::temporal_coherence(&unwrapped, &series)?;

    // 10) Productos a disco.
    fs::create_dir_all(&config.output_dir).with_path(&config.output_dir)?;
    io::write_velocity(&velocity, &config.output_dir.join("velocity.tif"))?;
    io::write_series(&series, &config.output_dir.join("series"))?;
    let as_map = |data: Array2<f32>| VelocityMap { data, meta: series.meta.clone() };
    io::write_velocity(
        &as_map(gamma.clone()),
        &config.output_dir.join("temporal_coherence.tif"),
    )?;
    if let Some(dem) = &solution.dem_error_m {
        io::write_velocity(&as_map(dem.clone()), &config.output_dir.join("dem_error.tif"))?;
    }
    if let Some(qc) = &closure_qc {
        io::write_velocity(&as_map(qc.clone()), &config.output_dir.join("closure_qc.tif"))?;
    }

    Ok(SbasProducts {
        velocity,
        series,
        temporal_coherence: gamma,
        dem_error_m: solution.dem_error_m,
        unwrap_report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aps_se_salta_con_pocas_epocas_o_ventana_degenerada() {
        // Menos de 3 épocas → saltar siempre.
        assert!(!should_apply_aps(2, 5));
        assert!(!should_apply_aps(0, 5));
        // Ventana ≤ 1 → saltar.
        assert!(!should_apply_aps(6, 1));
        assert!(!should_apply_aps(6, 0));
        // Caso normal → aplicar.
        assert!(should_apply_aps(3, 3));
        assert!(should_apply_aps(6, 5));
    }

    #[test]
    fn config_new_tiene_defaults_del_pipeline() {
        let c = SbasPipelineConfig::new("in".into(), "out".into());
        assert!(c.correct_unwrap);
        assert!(c.ps_threshold.is_none());
        assert!(c.reference.is_none());
        assert!(c.deramp.is_none());
        assert!(c.unwrap_min_quality.is_none());
        assert!(matches!(c.unwrap_backend, UnwrapBackend::FloodFill));
    }
}

//! Pipeline SBAS end-to-end: lectura → (PS opcional) → desenrollado →
//! inversión → corrección APS → velocidad → escritura de productos.
//!
//! ## Flujo de [`run_sbas`]
//!
//! 1. Lee el stack de interferogramas (`io::read_ifg_stack`).
//! 2. Si hay `ps_threshold`, lee amplitudes del mismo directorio, calcula
//!    amplitude dispersion y selecciona PS; sin umbral se invierte toda la
//!    grilla (modo SBAS clásico).
//! 3. Desenrolla la fase de cada par (`unwrap::unwrap_stack`).
//! 4. Invierte la serie de desplazamiento LOS (`inversion::invert_sbas`).
//! 5. Corrige APS (`atmosphere::correct_aps`) — ver regla de salto abajo.
//! 6. Estima la velocidad media LOS (`inversion::estimate_velocity`).
//! 7. Escribe `velocity.tif` y `series/disp_*.tif` en `output_dir`.
//!
//! ## Limitaciones y decisiones v0.1
//!
//! - **Coherencia**: el formato de stack v0.1 (`stack.json`, ver módulo
//!   [`crate::io`]) no incluye mapas de coherencia, por lo que el
//!   desenrollado corre **sin mapa de calidad** (`unwrap_stack(&stack,
//!   None)`: calidad uniforme, semilla en el centro). Cuando el formato
//!   incorpore coherencia, debe pasarse aquí como mapa de calidad.
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

use crate::atmosphere::{self, ApsConfig};
use crate::error::{InsarError, Result};
use crate::network::SbasConfig;
use crate::types::{DisplacementSeries, PsCandidate, VelocityMap};
use crate::{inversion, io, ps, unwrap};

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
/// `config.output_dir` (`velocity.tif` + `series/disp_YYYYMMDD.tif`).
///
/// Ver doc del módulo para el flujo paso a paso y las limitaciones v0.1
/// (desenrollado sin coherencia, regla de salto de APS, `config.network`
/// sin uso hasta que se generen redes desde SLC).
pub fn run_sbas(config: &SbasPipelineConfig) -> Result<SbasProducts> {
    // 1) Stack de interferogramas envueltos.
    let stack = io::read_ifg_stack(&config.input_dir)?;

    // 2) Selección de PS (opcional).
    let ps_candidates = match config.ps_threshold {
        Some(threshold) => Some(select_ps_from_dir(&config.input_dir, threshold)?),
        None => None,
    };

    // 3) Desenrollado. Sin coherencia en el formato v0.1 → sin mapa de
    //    calidad (limitación documentada en el doc del módulo).
    let unwrapped = unwrap::unwrap_stack(&stack, None)?;

    // 4) Inversión SBAS de la serie LOS.
    let mut series = inversion::invert_sbas(&unwrapped, ps_candidates.as_deref())?;

    // 5) Corrección APS (se salta si la serie es demasiado corta o la
    //    ventana temporal degenera; ver doc del módulo).
    if should_apply_aps(series.epochs.len(), config.aps.temporal_window_epochs) {
        atmosphere::correct_aps(&mut series, &config.aps)?;
    }

    // 6) Velocidad media LOS.
    let velocity = inversion::estimate_velocity(&series)?;

    // 7) Productos a disco.
    fs::create_dir_all(&config.output_dir)?;
    io::write_velocity(&velocity, &config.output_dir.join("velocity.tif"))?;
    io::write_series(&series, &config.output_dir.join("series"))?;

    Ok(SbasProducts { velocity, series })
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
}

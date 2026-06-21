//! # insar-core
//!
//! Motor de análisis de series temporales InSAR: selección de Persistent
//! Scatterers (amplitude dispersion), red Small-Baseline Subset (SBAS),
//! inversión de la serie temporal de desplazamiento LOS y corrección
//! atmosférica simple.
//!
//! Convenciones transversales (ver PLAN.md):
//! - Layout `Array3`: eje 0 = tiempo/par, eje 1 = filas, eje 2 = columnas.
//! - NoData = `f32::NAN` (complejos: ambas partes NaN).
//! - Fase → desplazamiento LOS: `d = -λ/(4π)·φ`.
//! - API pública devuelve [`Result`]; sin `panic!` en rutas públicas.

pub mod atmosphere;
pub mod error;
pub mod inversion;
pub mod io;
pub mod network;
pub mod pipeline;
pub mod postprocess;
pub mod ps;
pub mod types;
pub mod unwrap;
pub mod unwrap_error;

pub use error::{InsarError, Result};

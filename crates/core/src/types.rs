//! Contratos del dominio: stacks, pares interferométricos, metadata y
//! productos de salida. Este archivo es el contrato entre módulos —
//! NO modificar firmas durante la implementación de Fase 2 (ver PLAN.md).

use chrono::NaiveDate;
use ndarray::{Array2, Array3};
use num_complex::Complex32;
use surtgis_core::{CRS, GeoTransform};

use crate::error::{InsarError, Result};

/// Longitud de onda C-band de Sentinel-1, en metros.
pub const SENTINEL1_WAVELENGTH_M: f64 = 0.05546576;

/// Fecha de adquisición de una escena SAR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Epoch(pub NaiveDate);

impl Epoch {
    /// Días transcurridos desde `other` (positivo si `self` es posterior).
    pub fn days_since(&self, other: &Epoch) -> i64 {
        (self.0 - other.0).num_days()
    }

    /// Años decimales transcurridos desde `other`.
    pub fn years_since(&self, other: &Epoch) -> f64 {
        self.days_since(other) as f64 / 365.25
    }
}

/// Par interferométrico: índices dentro del vector de épocas del stack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IfgPair {
    /// Índice de la época de referencia (la más antigua del par).
    pub reference: usize,
    /// Índice de la época secundaria.
    pub secondary: usize,
    /// Baseline perpendicular en metros.
    pub perp_baseline_m: f64,
}

impl IfgPair {
    /// Baseline temporal en días (secundaria − referencia).
    pub fn temporal_baseline_days(&self, epochs: &[Epoch]) -> i64 {
        epochs[self.secondary].days_since(&epochs[self.reference])
    }
}

/// Geometría de adquisición y georreferencia compartida por todos los
/// productos de un stack.
#[derive(Debug, Clone)]
pub struct StackMeta {
    pub transform: GeoTransform,
    pub crs: Option<CRS>,
    /// Longitud de onda radar en metros (ver [`SENTINEL1_WAVELENGTH_M`]).
    pub wavelength_m: f64,
    /// Ángulo de incidencia medio en grados.
    pub incidence_deg: f64,
    /// Heading de la plataforma en grados (opcional en MVP).
    pub heading_deg: Option<f64>,
}

/// Stack de interferogramas complejos envueltos. `data`: pares × filas × cols.
#[derive(Debug, Clone)]
pub struct IfgStack {
    pub data: Array3<Complex32>,
    pub epochs: Vec<Epoch>,
    pub pairs: Vec<IfgPair>,
    pub meta: StackMeta,
}

/// Stack de amplitudes SLC coregistradas (para selección de PS).
/// `data`: épocas × filas × cols.
#[derive(Debug, Clone)]
pub struct AmplitudeStack {
    pub data: Array3<f32>,
    pub epochs: Vec<Epoch>,
    pub meta: StackMeta,
}

/// Stack de fase desenrollada en radianes. `data`: pares × filas × cols.
#[derive(Debug, Clone)]
pub struct UnwrappedStack {
    pub data: Array3<f32>,
    pub epochs: Vec<Epoch>,
    pub pairs: Vec<IfgPair>,
    pub meta: StackMeta,
}

/// Candidato a Persistent Scatterer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsCandidate {
    pub row: usize,
    pub col: usize,
    /// Amplitude dispersion D_A = σ_A / μ_A (menor = más estable).
    pub amp_dispersion: f32,
}

/// Serie temporal de desplazamiento LOS en metros, relativa a la primera
/// época. `data`: épocas × filas × cols.
#[derive(Debug, Clone)]
pub struct DisplacementSeries {
    pub data: Array3<f32>,
    pub epochs: Vec<Epoch>,
    pub meta: StackMeta,
}

/// Mapa de velocidad media LOS en m/año.
#[derive(Debug, Clone)]
pub struct VelocityMap {
    pub data: Array2<f32>,
    pub meta: StackMeta,
}

macro_rules! impl_stack_dims {
    ($t:ty) => {
        impl $t {
            /// Número de capas en el eje temporal (eje 0).
            pub fn n_layers(&self) -> usize {
                self.data.shape()[0]
            }
            /// (filas, columnas) de la grilla espacial.
            pub fn dims(&self) -> (usize, usize) {
                (self.data.shape()[1], self.data.shape()[2])
            }
        }
    };
}

impl_stack_dims!(IfgStack);
impl_stack_dims!(AmplitudeStack);
impl_stack_dims!(UnwrappedStack);
impl_stack_dims!(DisplacementSeries);

impl IfgStack {
    /// Consistencia interna: nº de pares vs eje 0, índices dentro de rango.
    pub fn validate(&self) -> Result<()> {
        validate_pairs(self.n_layers(), &self.pairs, self.epochs.len())
    }
}

impl UnwrappedStack {
    /// Consistencia interna: nº de pares vs eje 0, índices dentro de rango.
    pub fn validate(&self) -> Result<()> {
        validate_pairs(self.n_layers(), &self.pairs, self.epochs.len())
    }
}

impl DisplacementSeries {
    /// Consistencia interna: nº de épocas declaradas vs capas en `data` (eje
    /// 0). Compartido por todos los ajustes temporales de `inversion` y
    /// `features` — antes triplicado idéntico en cada uno.
    pub fn validate(&self) -> Result<()> {
        let n_layers = self.n_layers();
        if self.epochs.len() != n_layers {
            return Err(InsarError::DimensionMismatch(format!(
                "{} épocas declaradas vs {n_layers} capas en la serie",
                self.epochs.len()
            )));
        }
        Ok(())
    }

    /// Tiempo en años decimales de cada época, relativo a la primera. Mismo
    /// cálculo (`Epoch::years_since` contra `epochs[0]`) que antes se repetía
    /// en cada función de ajuste temporal.
    pub fn epoch_years(&self) -> Vec<f64> {
        self.epochs.iter().map(|e| e.years_since(&self.epochs[0])).collect()
    }
}

fn validate_pairs(n_layers: usize, pairs: &[IfgPair], n_epochs: usize) -> Result<()> {
    if pairs.len() != n_layers {
        return Err(InsarError::DimensionMismatch(format!(
            "{} pares declarados vs {} capas en el stack",
            pairs.len(),
            n_layers
        )));
    }
    for (i, p) in pairs.iter().enumerate() {
        if p.reference >= n_epochs || p.secondary >= n_epochs {
            return Err(InsarError::InvalidNetwork(format!(
                "par {i}: índices ({}, {}) fuera de rango para {n_epochs} épocas",
                p.reference, p.secondary
            )));
        }
        if p.reference == p.secondary {
            return Err(InsarError::InvalidNetwork(format!(
                "par {i}: referencia y secundaria son la misma época"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn meta() -> StackMeta {
        StackMeta {
            transform: GeoTransform::new(0.0, 0.0, 30.0, -30.0),
            crs: None,
            wavelength_m: SENTINEL1_WAVELENGTH_M,
            incidence_deg: 39.0,
            heading_deg: None,
        }
    }

    fn epochs() -> Vec<Epoch> {
        ["2023-01-01", "2023-01-13", "2023-01-25"]
            .iter()
            .map(|s| Epoch(s.parse().unwrap()))
            .collect()
    }

    #[test]
    fn baseline_temporal() {
        let e = epochs();
        let pair = IfgPair { reference: 0, secondary: 1, perp_baseline_m: 50.0 };
        assert_eq!(pair.temporal_baseline_days(&e), 12);
    }

    #[test]
    fn validate_detecta_par_fuera_de_rango() {
        let stack = UnwrappedStack {
            data: Array3::zeros((1, 4, 4)),
            epochs: epochs(),
            pairs: vec![IfgPair { reference: 0, secondary: 9, perp_baseline_m: 0.0 }],
            meta: meta(),
        };
        assert!(stack.validate().is_err());
    }

    #[test]
    fn validate_detecta_conteo_inconsistente() {
        let stack = UnwrappedStack {
            data: Array3::zeros((2, 4, 4)),
            epochs: epochs(),
            pairs: vec![IfgPair { reference: 0, secondary: 1, perp_baseline_m: 0.0 }],
            meta: meta(),
        };
        assert!(stack.validate().is_err());
    }
}

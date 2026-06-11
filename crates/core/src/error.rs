//! Tipo de error compartido por todo el crate.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, InsarError>;

#[derive(Debug, Error)]
pub enum InsarError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    /// Errores del backend raster (surtgis-core), mapeados por mensaje para
    /// no acoplar el enum público a su tipo de error.
    #[error("raster: {0}")]
    Raster(String),

    #[error("dimensiones inconsistentes: {0}")]
    DimensionMismatch(String),

    #[error("red de interferogramas inválida: {0}")]
    InvalidNetwork(String),

    #[error("inversión falló: {0}")]
    Inversion(String),

    #[error("metadata inválida: {0}")]
    Metadata(String),

    #[error("formato no soportado: {0}")]
    UnsupportedFormat(String),
}

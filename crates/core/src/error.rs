//! Tipo de error compartido por todo el crate.

use std::path::{Path, PathBuf};

use thiserror::Error;

pub type Result<T> = std::result::Result<T, InsarError>;

#[derive(Debug, Error)]
pub enum InsarError {
    /// Sin `#[from]` a propósito (M-6): una conversión implícita de
    /// `std::io::Error` vía `?` no tiene forma de saber qué archivo falló.
    /// Se exige pasar por [`InsarError::io`] / [`IoResultExt::with_path`] en
    /// cada call site, lo que el compilador verifica por nosotros — sin esto
    /// bisecar un stack de cientos de pares por un solo error "I/O: ..." sin
    /// ruta es adivinar a ciegas cuál falló.
    #[error("I/O en {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

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

impl InsarError {
    /// Construye un [`InsarError::Io`] con el path que causó el error.
    pub fn io(path: impl AsRef<Path>, source: std::io::Error) -> Self {
        InsarError::Io { path: path.as_ref().to_path_buf(), source }
    }
}

/// Adjunta `path` a un `Result<_, std::io::Error>` al convertirlo en
/// [`InsarError::Io`] — ver la nota de esa variante para el motivo.
pub trait IoResultExt<T> {
    fn with_path<P: AsRef<Path>>(self, path: P) -> Result<T>;
}

impl<T> IoResultExt<T> for std::result::Result<T, std::io::Error> {
    fn with_path<P: AsRef<Path>>(self, path: P) -> Result<T> {
        self.map_err(|source| InsarError::io(path, source))
    }
}

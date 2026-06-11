//! Lectura/escritura de stacks y productos.
//!
//! Formato de stack de entrada (definido aquí, documentar al implementar):
//! un directorio con un GeoTIFF por interferograma (2 bandas: re, im — o fase
//! y coherencia) más un `stack.json` con épocas, pares, baselines y metadata
//! de adquisición (wavelength, incidencia). Lectura/escritura GeoTIFF vía
//! `surtgis_core::io::{read_geotiff, write_geotiff}` (sin GDAL).
//!
//! El lector de formato ISCE binario plano (.int/.unw + XML) es fase tardía
//! v0.1 — ver PLAN.md.

use std::path::Path;

use crate::error::Result;
use crate::types::{AmplitudeStack, DisplacementSeries, IfgStack, VelocityMap};

/// Lee un stack de interferogramas complejos desde `dir` (formato del módulo).
pub fn read_ifg_stack(dir: &Path) -> Result<IfgStack> {
    let _ = dir;
    todo!("Fase 2, módulo io — ver PLAN.md")
}

/// Lee un stack de amplitudes SLC coregistradas desde `dir`.
pub fn read_amplitude_stack(dir: &Path) -> Result<AmplitudeStack> {
    let _ = dir;
    todo!("Fase 2, módulo io — ver PLAN.md")
}

/// Escribe el mapa de velocidad LOS (m/año) como GeoTIFF Float32.
pub fn write_velocity(map: &VelocityMap, path: &Path) -> Result<()> {
    let _ = (map, path);
    todo!("Fase 2, módulo io — ver PLAN.md")
}

/// Escribe la serie de desplazamiento como un GeoTIFF por época en `dir`,
/// nombrados `disp_YYYYMMDD.tif`.
pub fn write_series(series: &DisplacementSeries, dir: &Path) -> Result<()> {
    let _ = (series, dir);
    todo!("Fase 2, módulo io — ver PLAN.md")
}

//! Lector de productos LiCSAR (COMET Sentinel-1 InSAR portal) → [`IfgStack`].
//!
//! LiCSAR entrega interferogramas **geocodificados**: por par, un subdirectorio
//! `<YYYYMMDD_YYYYMMDD>/` con la fase envuelta
//! `<pair>.geo.diff_unfiltered_pha.tif` (o la filtrada `.geo.diff_pha.tif`) —
//! Float32 en radianes, EPSG:4326, NoData = 0 — y la coherencia
//! `<pair>.geo.cc.tif` (Byte 0–255, NoData = 0; coherencia = valor/255).
//!
//! Este lector arma un [`IfgStack`] complejo de **amplitud unitaria**
//! (`exp(iφ)`), pensado para [`crate::phase_bias`], que estima el sesgo de
//! no-cierre sobre fase **envuelta**. El reader GeoTIFF de `surtgis-core` lee
//! estos `.tif` (DEFLATE + PREDICTOR=3) directamente — sin GDAL.
//!
//! La red LiCSAR conecta cada época con sus vecinas a spans 1–3 (más pares
//! largos), justo la topología que exige [`crate::phase_bias::estimate_coefficients`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use num_complex::Complex32;
use surtgis_core::io::read_geotiff;

use super::{accumulate_layers, check_dims, read_f32};
use crate::error::{InsarError, Result};
use crate::types::{Epoch, IfgPair, IfgStack, SENTINEL1_WAVELENGTH_M, StackMeta};

/// Producto de fase envuelta a leer de cada par.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LicsarProduct {
    /// `geo.diff_unfiltered_pha.tif` — **sin** filtro ADF. Preferido para
    /// phase bias: el filtro Goldstein altera el cierre de fase.
    Unfiltered,
    /// `geo.diff_pha.tif` — filtrado ADF (menos ruido, cierre alterado).
    Filtered,
}

impl LicsarProduct {
    fn suffix(self) -> &'static str {
        match self {
            LicsarProduct::Unfiltered => "geo.diff_unfiltered_pha.tif",
            LicsarProduct::Filtered => "geo.diff_pha.tif",
        }
    }
}

/// Configuración de [`read_licsar_stack`].
#[derive(Debug, Clone)]
pub struct LicsarLoadConfig {
    /// Qué producto de fase leer (default: [`LicsarProduct::Unfiltered`]).
    pub product: LicsarProduct,
    /// Umbral de coherencia en [0, 1]. Si es `> 0`, se lee `<pair>.geo.cc.tif`
    /// (Byte) y se enmascaran a NaN los píxeles con `cc/255 < umbral`. Si es
    /// `<= 0` (default), NO se lee coherencia y solo se enmascara el NoData de
    /// la fase (`φ == 0`).
    pub coherence_threshold: f32,
    /// Longitud de onda radar (default [`SENTINEL1_WAVELENGTH_M`]).
    pub wavelength_m: f64,
    /// Incidencia media en grados (default 39.0, típico de Sentinel-1 IW).
    pub incidence_deg: f64,
    /// Heading de la plataforma en grados (opcional).
    pub heading_deg: Option<f64>,
}

impl Default for LicsarLoadConfig {
    fn default() -> Self {
        Self {
            product: LicsarProduct::Unfiltered,
            coherence_threshold: 0.0,
            wavelength_m: SENTINEL1_WAVELENGTH_M,
            incidence_deg: 39.0,
            heading_deg: None,
        }
    }
}

/// Lee un directorio de productos LiCSAR como un [`IfgStack`] complejo envuelto.
///
/// `dir` debe contener subdirectorios `YYYYMMDD_YYYYMMDD/`, cada uno con el
/// producto de fase pedido. Se descubren automáticamente los pares presentes
/// (una red incompleta es válida: [`crate::phase_bias`] la maneja). La
/// georreferencia se toma del primer archivo; todos deben compartir
/// dimensiones. Los pares se ordenan por (referencia, secundaria).
pub fn read_licsar_stack(dir: &Path, config: &LicsarLoadConfig) -> Result<IfgStack> {
    // 1. Descubrir pares con el producto pedido.
    let mut found: Vec<(NaiveDate, NaiveDate, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| InsarError::io(dir, e))? {
        let entry = entry.map_err(|e| InsarError::io(dir, e))?;
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some((a, b)) = parse_pair_name(&name) else {
            continue;
        };
        let tif = entry.path().join(format!("{name}.{}", config.product.suffix()));
        if tif.is_file() {
            found.push((a, b, tif));
        }
    }
    if found.is_empty() {
        return Err(InsarError::Metadata(format!(
            "{}: sin pares LiCSAR con {} (se esperan subdirs YYYYMMDD_YYYYMMDD/)",
            dir.display(),
            config.product.suffix()
        )));
    }

    // 2. Orden estable por (referencia, secundaria) e índice de épocas.
    found.sort_by_key(|(a, b, _)| (*a, *b));
    let mut dates: Vec<NaiveDate> = found.iter().flat_map(|(a, b, _)| [*a, *b]).collect();
    dates.sort_unstable();
    dates.dedup();
    let idx: HashMap<NaiveDate, usize> = dates.iter().enumerate().map(|(i, d)| (*d, i)).collect();
    let epochs: Vec<Epoch> = dates.iter().map(|d| Epoch(*d)).collect();

    // 3. Leer capas: φ → exp(iφ), enmascarando NoData (φ==0) y coherencia baja.
    let thr = config.coherence_threshold;
    let mut pairs = Vec::with_capacity(found.len());
    let mut geo: Option<(surtgis_core::GeoTransform, Option<surtgis_core::CRS>)> = None;
    let data = accumulate_layers(found.len(), |i, expected| {
        let (a, b, tif) = &found[i];
        let pha = read_f32(tif)?;
        let shape = pha.shape();
        check_dims(tif, shape, expected.unwrap_or(shape))?;
        geo.get_or_insert_with(|| (*pha.transform(), pha.crs().cloned()));

        // Coherencia opcional (Byte 0–255) → máscara.
        let coh: Option<Vec<u8>> = if thr > 0.0 {
            let cc_path = tif.with_file_name(format!(
                "{}_{}.geo.cc.tif",
                a.format("%Y%m%d"),
                b.format("%Y%m%d")
            ));
            let cc = read_geotiff::<u8, _>(&cc_path, None)
                .map_err(|e| InsarError::Raster(format!("{}: {e}", cc_path.display())))?;
            check_dims(&cc_path, cc.shape(), shape)?;
            Some(cc.data().iter().copied().collect())
        } else {
            None
        };
        let thr_byte = (thr * 255.0).round() as i32;

        let values: Vec<Complex32> = pha
            .data()
            .iter()
            .enumerate()
            .map(|(j, &phi)| {
                let coh_ok = coh
                    .as_ref()
                    .map(|c| c[j] as i32 >= thr_byte && c[j] != 0)
                    .unwrap_or(true);
                if !phi.is_finite() || phi == 0.0 || !coh_ok {
                    Complex32::new(f32::NAN, f32::NAN)
                } else {
                    Complex32::new(phi.cos(), phi.sin())
                }
            })
            .collect();

        pairs.push(IfgPair {
            reference: idx[a],
            secondary: idx[b],
            perp_baseline_m: 0.0,
        });
        Ok((shape, values))
    })?;

    let (transform, crs) = geo.expect("geo definida: found no vacío");
    let meta = StackMeta {
        transform,
        crs,
        wavelength_m: config.wavelength_m,
        incidence_deg: config.incidence_deg,
        heading_deg: config.heading_deg,
    };
    let stack = IfgStack {
        data,
        epochs,
        pairs,
        meta,
    };
    stack.validate()?;
    Ok(stack)
}

/// Parsea `YYYYMMDD_YYYYMMDD` → (referencia, secundaria). `None` si el nombre
/// no calza (así se ignoran subdirectorios ajenos como `metadata/`).
fn parse_pair_name(name: &str) -> Option<(NaiveDate, NaiveDate)> {
    let (a, b) = name.split_once('_')?;
    if a.len() != 8 || b.len() != 8 {
        return None;
    }
    let da = NaiveDate::parse_from_str(a, "%Y%m%d").ok()?;
    let db = NaiveDate::parse_from_str(b, "%Y%m%d").ok()?;
    Some((da, db))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;
    use surtgis_core::io::write_geotiff;
    use surtgis_core::{GeoTransform, Raster};

    /// Escribe un GeoTIFF float32 de 1 banda (fase) en `<dir>/<pair>/<pair>.<suffix>`.
    fn write_pair_pha(root: &Path, pair: &str, phase: &[f32], rows: usize, cols: usize) {
        let d = root.join(pair);
        std::fs::create_dir_all(&d).unwrap();
        let mut r = Raster::from_vec(phase.to_vec(), rows, cols).unwrap();
        r.set_transform(GeoTransform::new(-72.0, -34.0, 0.001, -0.001));
        write_geotiff(&r, &d.join(format!("{pair}.geo.diff_unfiltered_pha.tif")), None).unwrap();
    }

    #[test]
    fn lee_red_minima_y_convierte_a_exp_i_phi() {
        let tmp = std::env::temp_dir().join(format!("licsar_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        // 3 épocas, 2 píxeles: pares span-1 (0-1, 1-2) y span-2 (0-2).
        let (rows, cols) = (1, 2);
        write_pair_pha(&tmp, "20220101_20220107", &[0.5, 1.0], rows, cols);
        write_pair_pha(&tmp, "20220107_20220113", &[0.3, 0.0], rows, cols); // 0.0 = NoData
        write_pair_pha(&tmp, "20220101_20220113", &[0.8, -1.2], rows, cols);

        let stack = read_licsar_stack(&tmp, &LicsarLoadConfig::default()).unwrap();

        assert_eq!(stack.epochs.len(), 3, "3 épocas únicas");
        assert_eq!(stack.pairs.len(), 3, "3 pares");
        assert_eq!(stack.dims(), (rows, cols));
        // Orden (ref, sec): (0,1), (0,2), (1,2).
        assert_eq!(
            stack.pairs.iter().map(|p| (p.reference, p.secondary)).collect::<Vec<_>>(),
            vec![(0, 1), (0, 2), (1, 2)]
        );
        // Primer par, píxel 0: exp(i·0.5).
        let c = stack.data[[0, 0, 0]];
        assert!((c.re - 0.5_f32.cos()).abs() < 1e-6 && (c.im - 0.5_f32.sin()).abs() < 1e-6);
        // NoData (φ==0) → NaN: par (1,2)=layer 2, píxel 1.
        assert!(stack.data[[2, 0, 1]].re.is_nan());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn subdirs_ajenos_se_ignoran() {
        let tmp = std::env::temp_dir().join(format!("licsar_test2_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        write_pair_pha(&tmp, "20220101_20220107", &[PI, -PI], 1, 2);
        std::fs::create_dir_all(tmp.join("metadata")).unwrap(); // no calza el patrón
        let stack = read_licsar_stack(&tmp, &LicsarLoadConfig::default()).unwrap();
        assert_eq!(stack.pairs.len(), 1);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}

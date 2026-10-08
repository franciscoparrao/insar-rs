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
use ndarray::Array3;
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

/// Ventana geográfica de interés (grados, EPSG:4326) para recortar el stack
/// y no cargar el frame completo en RAM (un frame LiCSAR full-res × cientos de
/// pares son decenas de GB).
#[derive(Debug, Clone, Copy)]
pub struct Aoi {
    pub min_lon: f64,
    pub min_lat: f64,
    pub max_lon: f64,
    pub max_lat: f64,
}

/// Configuración de [`read_licsar_stack`].
#[derive(Debug, Clone)]
pub struct LicsarLoadConfig {
    /// Qué producto de fase leer (default: [`LicsarProduct::Unfiltered`]).
    pub product: LicsarProduct,
    /// Recorte geográfico opcional. `None` (default) carga el frame completo
    /// — inviable en memoria para full-res × cientos de pares; en la práctica
    /// siempre se pasa un AOI para datos reales.
    pub aoi: Option<Aoi>,
    /// Rango de fechas `[inicio, fin]` inclusivo: solo se cargan los pares con
    /// AMBAS épocas dentro del rango. `None` (default) = todos. Sirve para
    /// acotar a una sub-ventana temporal densa (sin huecos en spans 1-3), que
    /// es lo que exige la corrección de phase bias.
    pub date_range: Option<(NaiveDate, NaiveDate)>,
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
            aoi: None,
            date_range: None,
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
    let found = discover_pairs(dir, config)?;

    // 2. Índice de épocas (found ya viene ordenado por (ref, sec)).
    let mut dates: Vec<NaiveDate> = found.iter().flat_map(|(a, b, _)| [*a, *b]).collect();
    dates.sort_unstable();
    dates.dedup();
    let idx: HashMap<NaiveDate, usize> = dates.iter().enumerate().map(|(i, d)| (*d, i)).collect();
    let epochs: Vec<Epoch> = dates.iter().map(|d| Epoch(*d)).collect();

    // 3. Leer capas: φ → exp(iφ), enmascarando NoData (φ==0) y coherencia baja,
    //    recortando al AOI (ventana calculada una vez desde el 1er raster).
    let thr = config.coherence_threshold;
    let thr_byte = (thr * 255.0).round() as i32;
    let mut pairs = Vec::with_capacity(found.len());
    let mut geo: Option<(surtgis_core::GeoTransform, Option<surtgis_core::CRS>)> = None;
    let mut window: Option<(usize, usize, usize, usize)> = None; // (r0, r1, c0, c1)
    let data = accumulate_layers(found.len(), |i, expected| {
        let (a, b, tif) = &found[i];
        let pha = read_f32(tif)?;
        let (full_rows, full_cols) = pha.shape();

        // Ventana de recorte + transform recortado, una sola vez.
        if window.is_none() {
            let w = match config.aoi {
                Some(aoi) => crop_window(pha.transform(), &aoi, full_rows, full_cols)?,
                None => (0, full_rows, 0, full_cols),
            };
            let (ox, oy) = pha.transform().pixel_to_geo_corner(w.2, w.0);
            let gt = surtgis_core::GeoTransform::new(
                ox,
                oy,
                pha.transform().pixel_width,
                pha.transform().pixel_height,
            );
            geo = Some((gt, pha.crs().cloned()));
            window = Some(w);
        }
        let (r0, r1, c0, c1) = window.unwrap();
        if r1 > full_rows || c1 > full_cols {
            return Err(InsarError::DimensionMismatch(format!(
                "{}: {full_rows}x{full_cols} menor que la ventana AOI {r1}x{c1}",
                tif.display()
            )));
        }
        let (out_rows, out_cols) = (r1 - r0, c1 - c0);
        check_dims(
            tif,
            (out_rows, out_cols),
            expected.unwrap_or((out_rows, out_cols)),
        )?;

        // Coherencia opcional (Byte 0–255).
        let cc = if thr > 0.0 {
            let cc_path = tif.with_file_name(format!(
                "{}_{}.geo.cc.tif",
                a.format("%Y%m%d"),
                b.format("%Y%m%d")
            ));
            let r = read_geotiff::<u8, _>(&cc_path, None)
                .map_err(|e| InsarError::Raster(format!("{}: {e}", cc_path.display())))?;
            check_dims(&cc_path, r.shape(), (full_rows, full_cols))?;
            Some(r)
        } else {
            None
        };

        let pha_arr = pha.data();
        let mut values = Vec::with_capacity(out_rows * out_cols);
        for r in r0..r1 {
            for c in c0..c1 {
                let phi = pha_arr[[r, c]];
                let coh_ok = cc
                    .as_ref()
                    .map(|m| {
                        let v = m.data()[[r, c]];
                        v as i32 >= thr_byte && v != 0
                    })
                    .unwrap_or(true);
                let val = if !phi.is_finite() || phi == 0.0 || !coh_ok {
                    Complex32::new(f32::NAN, f32::NAN)
                } else {
                    Complex32::new(phi.cos(), phi.sin())
                };
                values.push(val);
            }
        }

        pairs.push(IfgPair {
            reference: idx[a],
            secondary: idx[b],
            perp_baseline_m: 0.0,
        });
        Ok(((out_rows, out_cols), values))
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

/// Descubre los pares (subdirs `YYYYMMDD_YYYYMMDD/`) con el producto de fase
/// pedido, aplica el filtro de fechas y los ordena por (referencia, secundaria).
fn discover_pairs(
    dir: &Path,
    config: &LicsarLoadConfig,
) -> Result<Vec<(NaiveDate, NaiveDate, PathBuf)>> {
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
        if let Some((start, end)) = config.date_range
            && (a < start || b > end)
        {
            continue;
        }
        let tif = entry
            .path()
            .join(format!("{name}.{}", config.product.suffix()));
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
    found.sort_by_key(|(a, b, _)| (*a, *b));
    Ok(found)
}

/// Lee la coherencia (`geo.cc.tif`, Byte 0–255) de los mismos pares que
/// [`read_licsar_stack`], en el MISMO orden y con el MISMO recorte AOI, como
/// `Array3<f32>` (coherencia = valor/255; NoData 0 → NaN). Queda alineada capa
/// a capa con el `IfgStack` — pensada para pasarla como calidad a
/// [`crate::unwrap::unwrap_stack_min_quality`].
pub fn read_licsar_coherence(dir: &Path, config: &LicsarLoadConfig) -> Result<Array3<f32>> {
    let found = discover_pairs(dir, config)?;
    let mut window: Option<(usize, usize, usize, usize)> = None;
    accumulate_layers(found.len(), |i, expected| {
        let (a, b, tif) = &found[i];
        let cc_path = tif.with_file_name(format!(
            "{}_{}.geo.cc.tif",
            a.format("%Y%m%d"),
            b.format("%Y%m%d")
        ));
        let cc = read_geotiff::<u8, _>(&cc_path, None)
            .map_err(|e| InsarError::Raster(format!("{}: {e}", cc_path.display())))?;
        let (full_rows, full_cols) = cc.shape();
        if window.is_none() {
            window = Some(match config.aoi {
                Some(aoi) => crop_window(cc.transform(), &aoi, full_rows, full_cols)?,
                None => (0, full_rows, 0, full_cols),
            });
        }
        let (r0, r1, c0, c1) = window.unwrap();
        let (out_rows, out_cols) = (r1 - r0, c1 - c0);
        check_dims(
            &cc_path,
            (out_rows, out_cols),
            expected.unwrap_or((out_rows, out_cols)),
        )?;
        let arr = cc.data();
        let mut vals = Vec::with_capacity(out_rows * out_cols);
        for r in r0..r1 {
            for c in c0..c1 {
                let v = arr[[r, c]];
                vals.push(if v == 0 { f32::NAN } else { v as f32 / 255.0 });
            }
        }
        Ok(((out_rows, out_cols), vals))
    })
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

/// Traduce un [`Aoi`] geográfico a una ventana de píxeles `(r0, r1, c0, c1)`
/// (rangos semiabiertos), recortada a la grilla. Error si el AOI no
/// intersecta el frame.
fn crop_window(
    gt: &surtgis_core::GeoTransform,
    aoi: &Aoi,
    rows: usize,
    cols: usize,
) -> Result<(usize, usize, usize, usize)> {
    // Esquinas: (min_lon, max_lat) = arriba-izquierda; (max_lon, min_lat) = abajo-derecha.
    let (col_a, row_a) = gt.geo_to_pixel(aoi.min_lon, aoi.max_lat);
    let (col_b, row_b) = gt.geo_to_pixel(aoi.max_lon, aoi.min_lat);
    let c0 = col_a.floor().clamp(0.0, cols as f64) as usize;
    let r0 = row_a.floor().clamp(0.0, rows as f64) as usize;
    let c1 = (col_b.ceil().clamp(0.0, cols as f64) as usize).max(c0);
    let r1 = (row_b.ceil().clamp(0.0, rows as f64) as usize).max(r0);
    if c1 <= c0 || r1 <= r0 {
        return Err(InsarError::Metadata(format!(
            "AOI ({}, {})–({}, {}) no intersecta el frame ({rows}x{cols})",
            aoi.min_lon, aoi.min_lat, aoi.max_lon, aoi.max_lat
        )));
    }
    Ok((r0, r1, c0, c1))
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
        write_geotiff(
            &r,
            d.join(format!("{pair}.geo.diff_unfiltered_pha.tif")),
            None,
        )
        .unwrap();
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
            stack
                .pairs
                .iter()
                .map(|p| (p.reference, p.secondary))
                .collect::<Vec<_>>(),
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
    fn recorta_al_aoi() {
        let tmp = std::env::temp_dir().join(format!("licsar_aoi_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        // raster 4x4, origin (-72.0,-34.0), pixel 0.001/-0.001; fase = (j+1)*0.1.
        let phase: Vec<f32> = (0..16).map(|j| (j as f32 + 1.0) * 0.1).collect();
        write_pair_pha(&tmp, "20220101_20220107", &phase, 4, 4);

        // AOI que selecciona filas 1-2, columnas 1-2 (ventana 2x2).
        let cfg = LicsarLoadConfig {
            aoi: Some(Aoi {
                min_lon: -71.9985,
                max_lon: -71.9975,
                min_lat: -34.0025,
                max_lat: -34.0015,
            }),
            ..LicsarLoadConfig::default()
        };
        let stack = read_licsar_stack(&tmp, &cfg).unwrap();
        assert_eq!(stack.dims(), (2, 2), "recorte 2x2");
        // Píxel (0,0) del recorte = raster (fila 1, col 1) = j=5 → fase 0.6.
        let c = stack.data[[0, 0, 0]];
        assert!((c.re - 0.6_f32.cos()).abs() < 1e-6 && (c.im - 0.6_f32.sin()).abs() < 1e-6);
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

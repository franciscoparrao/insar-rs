//! Lector nativo de productos ISCE (sin GDAL ni Python).
//!
//! Consume un directorio de interferogramas desenrollados producido por ISCE
//! (p. ej. `merged/interferograms/`), con un subdirectorio por par nombrado
//! `YYYYMMDD_YYYYMMDD` (referencia_secundaria). Cada subdirectorio contiene:
//!
//! - `filt_fine.unw`: raster BIL de 2 bandas Float32 (LSB). **Banda 1 =
//!   amplitud, banda 2 = fase desenrollada en radianes.** El layout exacto
//!   (offsets por banda) se lee del sidecar `filt_fine.unw.vrt` (formato GDAL
//!   `VRTRawRasterBand`: `rasterXSize`/`rasterYSize`, y por banda `dataType`,
//!   `ByteOrder`, `ImageOffset`, `PixelOffset`, `LineOffset`).
//! - `filt_fine.cor`: raster de 1 banda Float32 con la coherencia (opcional,
//!   sirve como mapa de calidad para el desenrollado o como máscara).
//!
//! ## NoData de ISCE (enmascarado por amplitud)
//!
//! ISCE escribe `0.0` en la banda de fase donde el desenrollado falló o el
//! píxel está enmascarado (agua, layover): esos ceros NO son observaciones de
//! fase válidas. Como la convención del crate es NoData = NaN, por defecto
//! ([`IsceLoadConfig::mask_zero_amplitude`]) la fase se enmascara a NaN donde
//! la banda 1 (amplitud) es 0 — el marcador de NoData de ISCE. Sin este
//! enmascarado, los ceros entrarían a la inversión SBAS como fase legítima y
//! sesgarían las velocidades alrededor de las zonas enmascaradas.
//!
//! Las baselines perpendiculares se leen, si se indica `baselines_dir`, de
//! `baselines/<par>/...` (texto con líneas `Bperp (average): <valor>` por
//! swath; se promedian los swaths). Si el directorio configurado no existe, o
//! ningún par tiene baseline legible, es error ([`InsarError::Metadata`]):
//! degradar todas las Bperp a 0 en silencio invalidaría la futura corrección
//! de error de DEM. Pares individuales sin entrada (p. ej. layouts de
//! topsStack donde `baselines/` solo cubre pares con la fecha de referencia
//! del stack) quedan en 0.0.
//!
//! Como ISCE entrega la fase ya desenrollada, la salida natural es
//! [`UnwrappedStack`] (no `IfgStack`): puede alimentarse directamente a
//! [`crate::inversion::invert_sbas`].
//!
//! Otros productos ISCE soportados:
//! - `filt_fine.int` (CFloat32, 1 banda): interferograma **envuelto** →
//!   [`read_isce_wrapped_stack`] entrega un [`IfgStack`] para el
//!   desenrollador propio.
//! - `filt_fine.unw.conncomp` (Byte, 1 banda): componentes conexas del
//!   desenrollado → [`read_isce_conncomp`].
//! - `merged/geom_reference/los.rdr` (2 bandas): incidencia y azimut por
//!   píxel → [`read_isce_los`] (para descomposición con geometría por píxel).
//! - Bandas `Float64` se leen casteadas a f32 (lat/lon/hgt de la geometría).

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use ndarray::{Array2, Array3};
use num_complex::Complex32;
use surtgis_core::GeoTransform;

use crate::error::{InsarError, Result};
use crate::types::{Epoch, IfgPair, IfgStack, StackMeta, UnwrappedStack};

/// Configuración de carga de un stack ISCE.
#[derive(Debug, Clone)]
pub struct IsceLoadConfig {
    /// Nombre del archivo de fase desenrollada en cada par (default `filt_fine.unw`).
    pub unw_filename: String,
    /// Nombre del archivo de coherencia (default `filt_fine.cor`).
    pub cor_filename: String,
    /// Nombre del interferograma envuelto complejo, CFloat32 de 1 banda
    /// (default `filt_fine.int`) — para [`read_isce_wrapped_stack`].
    pub int_filename: String,
    /// Nombre del archivo de componentes conexas del desenrollado, Byte de
    /// 1 banda (default `filt_fine.unw.conncomp`) — para [`read_isce_conncomp`].
    pub conncomp_filename: String,
    /// Directorio de baselines (`baselines/<par>/`); `None` → baselines = 0.
    pub baselines_dir: Option<PathBuf>,
    /// Longitud de onda radar en metros (ISCE no la guarda en el .vrt).
    pub wavelength_m: f64,
    /// Ángulo de incidencia medio en grados.
    pub incidence_deg: f64,
    /// Enmascarar a NaN la fase donde la amplitud (banda 1) es 0 — el
    /// marcador NoData de ISCE (agua, máscara, unwrap fallido). Default
    /// `true`; ver doc del módulo. Con `false` la fase se lee cruda.
    pub mask_zero_amplitude: bool,
}

impl Default for IsceLoadConfig {
    fn default() -> Self {
        Self {
            unw_filename: "filt_fine.unw".to_string(),
            cor_filename: "filt_fine.cor".to_string(),
            int_filename: "filt_fine.int".to_string(),
            conncomp_filename: "filt_fine.unw.conncomp".to_string(),
            baselines_dir: None,
            wavelength_m: crate::types::SENTINEL1_WAVELENGTH_M,
            incidence_deg: 39.0,
            mask_zero_amplitude: true,
        }
    }
}

/// Layout de un raster crudo descrito por un `.vrt` (GDAL `VRTRawRasterBand`).
#[derive(Debug, Clone)]
pub(crate) struct VrtBand {
    pub data_type: String,
    pub byte_order_lsb: bool,
    pub image_offset: u64,
    pub pixel_offset: u64,
    pub line_offset: u64,
    pub source: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct VrtLayout {
    pub width: usize,
    pub height: usize,
    pub bands: Vec<VrtBand>,
}

/// Parsea un `.vrt` de GDAL (VRTRawRasterBand) a [`VrtLayout`].
/// `source` de cada banda se resuelve relativo al directorio del `.vrt`.
pub(crate) fn parse_vrt(path: &Path) -> Result<VrtLayout> {
    let text = fs::read_to_string(path)?;
    let doc = roxmltree::Document::parse(&text).map_err(|e| {
        InsarError::UnsupportedFormat(format!("{}: XML inválido: {e}", path.display()))
    })?;

    let root = doc.root_element();
    if root.tag_name().name() != "VRTDataset" {
        return Err(InsarError::UnsupportedFormat(format!(
            "{}: raíz no es VRTDataset",
            path.display()
        )));
    }

    let width = parse_attr_usize(&root, "rasterXSize", path)?;
    let height = parse_attr_usize(&root, "rasterYSize", path)?;

    // Directorio del .vrt para resolver SourceFilename relativos.
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));

    let mut bands: Vec<VrtBand> = Vec::new();
    for band_node in root
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "VRTRasterBand")
    {
        let sub_class = band_node.attribute("subClass").unwrap_or("");
        if sub_class != "VRTRawRasterBand" {
            return Err(InsarError::UnsupportedFormat(format!(
                "{}: subClass '{sub_class}' no soportado (se requiere VRTRawRasterBand)",
                path.display()
            )));
        }

        let data_type = band_node.attribute("dataType").ok_or_else(|| {
            InsarError::UnsupportedFormat(format!("{}: banda sin dataType", path.display()))
        })?;

        let byte_order = child_text(&band_node, "ByteOrder").unwrap_or("LSB");
        let byte_order_lsb = match byte_order {
            "LSB" => true,
            "MSB" => false,
            other => {
                return Err(InsarError::UnsupportedFormat(format!(
                    "{}: ByteOrder '{other}' no soportado",
                    path.display()
                )));
            }
        };

        let image_offset = parse_child_u64(&band_node, "ImageOffset", path)?;
        let pixel_offset = parse_child_u64(&band_node, "PixelOffset", path)?;
        let line_offset = parse_child_u64(&band_node, "LineOffset", path)?;

        let source_rel = child_text(&band_node, "SourceFilename").ok_or_else(|| {
            InsarError::UnsupportedFormat(format!("{}: banda sin SourceFilename", path.display()))
        })?;
        let source = base_dir.join(source_rel.trim());

        bands.push(VrtBand {
            data_type: data_type.to_string(),
            byte_order_lsb,
            image_offset,
            pixel_offset,
            line_offset,
            source,
        });
    }

    if bands.is_empty() {
        return Err(InsarError::UnsupportedFormat(format!(
            "{}: ninguna VRTRasterBand",
            path.display()
        )));
    }

    Ok(VrtLayout {
        width,
        height,
        bands,
    })
}

/// Lee un atributo entero del elemento (error UnsupportedFormat si falta/inválido).
fn parse_attr_usize(node: &roxmltree::Node, attr: &str, path: &Path) -> Result<usize> {
    node.attribute(attr)
        .ok_or_else(|| {
            InsarError::UnsupportedFormat(format!("{}: falta atributo {attr}", path.display()))
        })?
        .trim()
        .parse::<usize>()
        .map_err(|e| {
            InsarError::UnsupportedFormat(format!("{}: {attr} inválido: {e}", path.display()))
        })
}

/// Texto de un nodo hijo directo con el tag dado.
fn child_text<'a>(node: &'a roxmltree::Node, tag: &str) -> Option<&'a str> {
    node.children()
        .find(|n| n.is_element() && n.tag_name().name() == tag)
        .and_then(|n| n.text())
}

/// Parsea el texto de un nodo hijo como u64 (error UnsupportedFormat si falta/inválido).
fn parse_child_u64(node: &roxmltree::Node, tag: &str, path: &Path) -> Result<u64> {
    child_text(node, tag)
        .ok_or_else(|| {
            InsarError::UnsupportedFormat(format!("{}: falta {tag}", path.display()))
        })?
        .trim()
        .parse::<u64>()
        .map_err(|e| {
            InsarError::UnsupportedFormat(format!("{}: {tag} inválido: {e}", path.display()))
        })
}

/// Lee una banda (1-based) de un raster crudo como `Array2<f32>`
/// (filas = height, cols = width). Soporta `Float32` y `Float64` (este
/// último casteado a f32: precisión relativa ~1e-7, sub-métrica para
/// lat/lon/geometría — documentado).
pub(crate) fn read_raw_band(layout: &VrtLayout, band_1based: usize) -> Result<Array2<f32>> {
    let band = get_band(layout, band_1based, &["Float32", "Float64"])?;
    let raw = fs::read(&band.source)?;
    decode_raw_band(&raw, layout, band_1based)
}

/// Lee una banda CFloat32 (complejo intercalado `re,im` float32) — los
/// interferogramas envueltos `.int` y SLC `.slc` de ISCE.
pub(crate) fn read_raw_band_complex(
    layout: &VrtLayout,
    band_1based: usize,
) -> Result<Array2<Complex32>> {
    let band = get_band(layout, band_1based, &["CFloat32"])?;
    let raw = fs::read(&band.source)?;
    decode_raw_with(&raw, layout, band, band_1based, 8, |b, lsb| {
        let (re, im) = if lsb {
            (le_f32(&b[0..4]), le_f32(&b[4..8]))
        } else {
            (be_f32(&b[0..4]), be_f32(&b[4..8]))
        };
        Complex32::new(re, im)
    })
}

/// Lee una banda Byte (u8) — p. ej. `filt_fine.unw.conncomp` (componentes
/// conexas del desenrollado).
pub(crate) fn read_raw_band_byte(layout: &VrtLayout, band_1based: usize) -> Result<Array2<u8>> {
    let band = get_band(layout, band_1based, &["Byte"])?;
    let raw = fs::read(&band.source)?;
    decode_raw_with(&raw, layout, band, band_1based, 1, |b, _| b[0])
}

/// Valida índice y dtype (contra `allowed`) de la banda (1-based).
fn get_band<'a>(
    layout: &'a VrtLayout,
    band_1based: usize,
    allowed: &[&str],
) -> Result<&'a VrtBand> {
    if band_1based == 0 || band_1based > layout.bands.len() {
        return Err(InsarError::UnsupportedFormat(format!(
            "banda {band_1based} fuera de rango (1..={})",
            layout.bands.len()
        )));
    }
    let band = &layout.bands[band_1based - 1];
    if !allowed.contains(&band.data_type.as_str()) {
        return Err(InsarError::UnsupportedFormat(format!(
            "dataType '{}' no soportado (se esperaba uno de {allowed:?})",
            band.data_type
        )));
    }
    Ok(band)
}

#[inline]
fn le_f32(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
#[inline]
fn be_f32(b: &[u8]) -> f32 {
    f32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

/// Decodifica una banda f32 (1-based) desde los bytes crudos ya leídos
/// (permite decodificar varias bandas del mismo archivo con una sola
/// lectura). Float64 se castea a f32.
fn decode_raw_band(raw: &[u8], layout: &VrtLayout, band_1based: usize) -> Result<Array2<f32>> {
    let band = get_band(layout, band_1based, &["Float32", "Float64"])?;
    match band.data_type.as_str() {
        "Float32" => decode_raw_with(raw, layout, band, band_1based, 4, |b, lsb| {
            if lsb { le_f32(b) } else { be_f32(b) }
        }),
        _ => decode_raw_with(raw, layout, band, band_1based, 8, |b, lsb| {
            let v = if lsb {
                f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
            } else {
                f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
            };
            v as f32
        }),
    }
}

/// Núcleo genérico de decodificación: recorre la grilla con los offsets de la
/// banda y convierte cada elemento de `elem` bytes con `convert(bytes, lsb)`.
/// La cota superior se verifica con aritmética checked ANTES de alocar.
fn decode_raw_with<T: Copy + Default>(
    raw: &[u8],
    layout: &VrtLayout,
    band: &VrtBand,
    band_1based: usize,
    elem: usize,
    convert: impl Fn(&[u8], bool) -> T,
) -> Result<Array2<T>> {
    let rows = layout.height;
    let cols = layout.width;
    if rows == 0 || cols == 0 {
        return Ok(Array2::from_elem((rows, cols), T::default()));
    }

    let image_offset = band.image_offset as usize;
    let pixel_offset = band.pixel_offset as usize;
    let line_offset = band.line_offset as usize;

    // Verificación de cota superior (último píxel + elem bytes), con
    // multiplicaciones checked: un VRT hostil no puede hacer wrap-around.
    let last_byte = (rows - 1)
        .checked_mul(line_offset)
        .and_then(|v| image_offset.checked_add(v))
        .and_then(|v| (cols - 1).checked_mul(pixel_offset).and_then(|w| v.checked_add(w)))
        .and_then(|v| v.checked_add(elem));
    match last_byte {
        Some(needed) if needed <= raw.len() => {}
        _ => {
            return Err(InsarError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "{}: archivo más corto de lo que exigen los offsets de la banda {band_1based}",
                    band.source.display()
                ),
            )));
        }
    }

    let mut out = Array2::from_elem((rows, cols), T::default());
    let lsb = band.byte_order_lsb;
    for r in 0..rows {
        let row_base = image_offset + r * line_offset;
        for c in 0..cols {
            let pos = row_base + c * pixel_offset;
            out[[r, c]] = convert(&raw[pos..pos + elem], lsb);
        }
    }

    Ok(out)
}

/// Lista los subdirectorios que matchean exactamente `YYYYMMDD_YYYYMMDD`,
/// devolviendo `(NaiveDate ref, NaiveDate sec, nombre, ruta)` ordenados por
/// `(ref, sec)` ascendente para determinismo.
fn list_pair_dirs(dir: &Path) -> Result<Vec<(NaiveDate, NaiveDate, String, PathBuf)>> {
    let mut pairs: Vec<(NaiveDate, NaiveDate, String, PathBuf)> = Vec::new();

    let entries = fs::read_dir(dir)?;
    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = match entry.file_name().into_string() {
            Ok(n) => n,
            Err(_) => continue,
        };
        if let Some((d1, d2)) = parse_pair_name(&name) {
            if d1 >= d2 {
                return Err(InsarError::Metadata(format!(
                    "par '{name}': la fecha de referencia no es anterior a la secundaria"
                )));
            }
            pairs.push((d1, d2, name, entry.path()));
        }
    }

    if pairs.is_empty() {
        return Err(InsarError::Metadata(format!(
            "{}: ningún subdirectorio con formato YYYYMMDD_YYYYMMDD",
            dir.display()
        )));
    }

    pairs.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    Ok(pairs)
}

/// Parsea exactamente `YYYYMMDD_YYYYMMDD` (8 dígitos, '_', 8 dígitos).
fn parse_pair_name(name: &str) -> Option<(NaiveDate, NaiveDate)> {
    let (a, b) = name.split_once('_')?;
    if a.len() != 8 || b.len() != 8 || !a.bytes().all(|c| c.is_ascii_digit())
        || !b.bytes().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    let d1 = NaiveDate::parse_from_str(a, "%Y%m%d").ok()?;
    let d2 = NaiveDate::parse_from_str(b, "%Y%m%d").ok()?;
    Some((d1, d2))
}

/// Promedia todos los valores `Bperp (average): <float>` hallados en
/// `baselines_dir/<par>/*`. `None` si el par no tiene subdirectorio o ningún
/// archivo con el patrón (el caller decide si eso es error o Bperp = 0).
fn read_baseline(baselines_dir: &Path, pair_name: &str) -> Option<f64> {
    let pair_dir = baselines_dir.join(pair_name);
    let entries = fs::read_dir(&pair_dir).ok()?;

    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for entry in entries.flatten() {
        if !entry.path().is_file() {
            continue;
        }
        let text = match fs::read_to_string(entry.path()) {
            Ok(t) => t,
            Err(_) => continue,
        };
        for line in text.lines() {
            if let Some(rest) = line.trim().strip_prefix("Bperp (average):")
                && let Ok(v) = rest.trim().parse::<f64>()
            {
                sum += v;
                count += 1;
            }
        }
    }

    if count == 0 { None } else { Some(sum / count as f64) }
}

/// Lee la banda de fase (2) de un `.unw` enmascarando el NoData de ISCE: los
/// píxeles con amplitud (banda 1) == 0 son agua/máscara/desenrollado fallido
/// y su fase 0.0 no es una observación válida → NaN (ver doc del módulo).
/// Cuando ambas bandas comparten archivo fuente (el caso ISCE) se lee una
/// sola vez del disco.
fn read_unw_phase_masked(layout: &VrtLayout) -> Result<Array2<f32>> {
    if layout.bands.len() < 2 {
        return Err(InsarError::UnsupportedFormat(format!(
            "se esperaban ≥2 bandas (amplitud, fase) para enmascarar NoData; hay {} \
             (usa mask_zero_amplitude = false para leer la fase cruda)",
            layout.bands.len()
        )));
    }
    let b1 = &layout.bands[0];
    let b2 = get_band(layout, 2, &["Float32", "Float64"])?;

    let raw = fs::read(&b2.source)?;
    let mut phase = decode_raw_band(&raw, layout, 2)?;
    let amp = if b1.source == b2.source {
        decode_raw_band(&raw, layout, 1)?
    } else {
        read_raw_band(layout, 1)?
    };

    ndarray::Zip::from(&mut phase).and(&amp).for_each(|p, &a| {
        if a == 0.0 {
            *p = f32::NAN;
        }
    });
    Ok(phase)
}

/// Lee un directorio de interferogramas ISCE en un [`UnwrappedStack`].
///
/// Descubre los subdirectorios `YYYYMMDD_YYYYMMDD`, deriva las épocas (unión
/// ordenada de fechas) y los pares, lee la banda 2 (fase) de cada `.unw`
/// (enmascarando el NoData de ISCE por defecto, ver doc del módulo) y las
/// baselines si se configuran. La georreferencia es nominal (coordenadas
/// radar): se usa una transform identidad. Llama `validate()` antes de devolver.
///
/// Errores de baselines (solo si `baselines_dir` está configurado): el
/// directorio no existe, o ningún par tiene `Bperp (average)` legible —
/// ambos casos indican ruta equivocada o formato distinto, y degradar en
/// silencio a Bperp = 0 invalidaría los usos aguas abajo.
pub fn read_isce_unwrapped_stack(dir: &Path, config: &IsceLoadConfig) -> Result<UnwrappedStack> {
    let pair_dirs = list_pair_dirs(dir)?;
    let (epochs, pairs) = epochs_pairs_baselines(&pair_dirs, config)?;

    let data = stack_pair_layers(&pair_dirs, &config.unw_filename, |layout| {
        if config.mask_zero_amplitude {
            read_unw_phase_masked(layout)
        } else {
            read_raw_band(layout, 2)
        }
    })?;

    let stack = UnwrappedStack {
        data,
        epochs,
        pairs,
        meta: nominal_meta(config),
    };
    stack.validate()?;
    Ok(stack)
}

/// Lee un directorio de interferogramas ISCE **envueltos** (`filt_fine.int`,
/// CFloat32 de 1 banda) en un [`IfgStack`] — la entrada del desenrollador
/// propio ([`crate::unwrap::unwrap_stack`]). Con
/// [`IsceLoadConfig::mask_zero_amplitude`], los píxeles con módulo 0 (el
/// NoData de ISCE en `.int`) quedan `NaN + NaN·i`. Mismo descubrimiento de
/// pares, épocas y baselines que [`read_isce_unwrapped_stack`].
pub fn read_isce_wrapped_stack(dir: &Path, config: &IsceLoadConfig) -> Result<IfgStack> {
    let pair_dirs = list_pair_dirs(dir)?;
    let (epochs, pairs) = epochs_pairs_baselines(&pair_dirs, config)?;

    let data = stack_pair_layers(&pair_dirs, &config.int_filename, |layout| {
        let mut band = read_raw_band_complex(layout, 1)?;
        if config.mask_zero_amplitude {
            band.mapv_inplace(|z| {
                if z.re == 0.0 && z.im == 0.0 {
                    Complex32::new(f32::NAN, f32::NAN)
                } else {
                    z
                }
            });
        }
        Ok(band)
    })?;

    let stack = IfgStack {
        data,
        epochs,
        pairs,
        meta: nominal_meta(config),
    };
    stack.validate()?;
    Ok(stack)
}

/// Lee la coherencia (banda 1 de cada `.cor`) alineada con los pares del
/// stack, en el mismo orden que `read_isce_unwrapped_stack`.
/// `data`: pares × filas × cols.
pub fn read_isce_coherence(dir: &Path, config: &IsceLoadConfig) -> Result<Array3<f32>> {
    // Mismo descubrimiento y ORDEN que read_isce_unwrapped_stack.
    let pair_dirs = list_pair_dirs(dir)?;
    stack_pair_layers(&pair_dirs, &config.cor_filename, |layout| read_raw_band(layout, 1))
}

/// Lee las componentes conexas del desenrollado (`filt_fine.unw.conncomp`,
/// Byte de 1 banda), alineadas con los pares de `read_isce_unwrapped_stack`
/// (mismo orden). Convención ISCE/SNAPHU: `0` = píxel no desenrollado o
/// enmascarado; cada componente `k ≥ 1` tiene su propia referencia de fase
/// 2πn. Útil como máscara (`comp == 0 → NaN`) o para corrección de saltos
/// por componente. `data`: pares × filas × cols.
pub fn read_isce_conncomp(dir: &Path, config: &IsceLoadConfig) -> Result<Array3<u8>> {
    let pair_dirs = list_pair_dirs(dir)?;
    stack_pair_layers(&pair_dirs, &config.conncomp_filename, |layout| {
        read_raw_band_byte(layout, 1)
    })
}

/// Lee la geometría LOS por píxel de ISCE (`los.rdr` + sidecar
/// `<nombre>.vrt`): **banda 1 = ángulo de incidencia** (grados desde la
/// vertical), **banda 2 = azimut del vector LOS suelo→satélite** (grados
/// desde el Norte, antihorario positivo — convención ISCE). Píxeles con
/// `(0, 0)` exacto (NoData de ISCE) quedan NaN en ambos mapas.
///
/// Para convertir el azimut al heading que usa
/// [`crate::decompose::LosVector::from_incidence_heading`], ver
/// [`crate::decompose::isce_azimuth_to_heading`] (`heading = 90° − azimut`).
pub fn read_isce_los(los_path: &Path) -> Result<(Array2<f32>, Array2<f32>)> {
    let vrt_path = PathBuf::from(format!("{}.vrt", los_path.display()));
    let layout = parse_vrt(&vrt_path)?;
    if layout.bands.len() < 2 {
        return Err(InsarError::UnsupportedFormat(format!(
            "{}: se esperaban 2 bandas (incidencia, azimut); hay {}",
            vrt_path.display(),
            layout.bands.len()
        )));
    }
    let mut incidence = read_raw_band(&layout, 1)?;
    let mut azimuth = read_raw_band(&layout, 2)?;
    // NoData de ISCE: (0, 0) exacto en ambas bandas.
    ndarray::Zip::from(&mut incidence).and(&mut azimuth).for_each(|inc, az| {
        if *inc == 0.0 && *az == 0.0 {
            *inc = f32::NAN;
            *az = f32::NAN;
        }
    });
    Ok((incidence, azimuth))
}

/// Metadata nominal de un stack ISCE en coordenadas radar (transform
/// identidad) con la geometría escalar de la config.
fn nominal_meta(config: &IsceLoadConfig) -> StackMeta {
    StackMeta {
        transform: GeoTransform::new(0.0, 0.0, 1.0, -1.0),
        crs: None,
        wavelength_m: config.wavelength_m,
        incidence_deg: config.incidence_deg,
        heading_deg: None,
    }
}

/// Épocas (unión ordenada de fechas → índice) y pares con baselines desde
/// los subdirectorios descubiertos. Aplica la política de baselines del
/// módulo: directorio configurado inexistente o sin ningún `Bperp (average)`
/// legible = error; pares individuales sin entrada quedan en 0.0.
fn epochs_pairs_baselines(
    pair_dirs: &[(NaiveDate, NaiveDate, String, PathBuf)],
    config: &IsceLoadConfig,
) -> Result<(Vec<Epoch>, Vec<IfgPair>)> {
    // baselines_dir configurado pero inexistente: error temprano y claro
    // (típicamente un typo de ruta) en vez de Bperp = 0 silencioso.
    if let Some(bdir) = &config.baselines_dir
        && !bdir.is_dir()
    {
        return Err(InsarError::Metadata(format!(
            "baselines_dir no existe o no es un directorio: {}",
            bdir.display()
        )));
    }

    // Épocas = unión ordenada ascendente de todas las fechas → índice.
    let mut date_set: BTreeSet<NaiveDate> = BTreeSet::new();
    for (d1, d2, _, _) in pair_dirs {
        date_set.insert(*d1);
        date_set.insert(*d2);
    }
    let epochs: Vec<Epoch> = date_set.iter().map(|d| Epoch(*d)).collect();
    let date_to_idx = |d: &NaiveDate| -> usize {
        // BTreeSet ordenado → posición = índice.
        date_set.iter().position(|x| x == d).expect("fecha presente en el set")
    };

    let mut pairs: Vec<IfgPair> = Vec::with_capacity(pair_dirs.len());
    let mut baselines_missing = 0usize;
    for (d1, d2, name, _) in pair_dirs {
        let perp_baseline_m = match &config.baselines_dir {
            Some(bdir) => read_baseline(bdir, name).unwrap_or_else(|| {
                // Par sin entrada de baseline (normal en layouts topsStack
                // donde baselines/ solo cubre pares con la fecha de
                // referencia): Bperp = 0, pero se contabiliza — si NINGÚN
                // par tiene baseline, es error (ver abajo).
                baselines_missing += 1;
                0.0
            }),
            None => 0.0,
        };
        pairs.push(IfgPair {
            reference: date_to_idx(d1),
            secondary: date_to_idx(d2),
            perp_baseline_m,
        });
    }

    if let Some(bdir) = &config.baselines_dir
        && baselines_missing == pair_dirs.len()
    {
        return Err(InsarError::Metadata(format!(
            "ningún par tiene 'Bperp (average)' legible en {} — ¿ruta equivocada \
             o formato de baselines distinto? (omite baselines_dir para Bperp = 0)",
            bdir.display()
        )));
    }

    Ok((epochs, pairs))
}

/// Lee una capa 2D por par (con `read_layer` sobre el layout del `.vrt` de
/// `filename`) y las apila en (pares × filas × cols), validando que todas
/// compartan dimensiones.
fn stack_pair_layers<T: Copy>(
    pair_dirs: &[(NaiveDate, NaiveDate, String, PathBuf)],
    filename: &str,
    mut read_layer: impl FnMut(&VrtLayout) -> Result<Array2<T>>,
) -> Result<Array3<T>> {
    let mut values: Vec<T> = Vec::new();
    let mut dims: Option<(usize, usize)> = None;

    for (_, _, _, path) in pair_dirs {
        let vrt_path = path.join(format!("{filename}.vrt"));
        let layout = parse_vrt(&vrt_path)?;
        let band = read_layer(&layout)?;

        let (rows, cols) = band.dim();
        match dims {
            None => {
                dims = Some((rows, cols));
                values.reserve(pair_dirs.len() * rows * cols);
            }
            Some(exp) if exp != (rows, cols) => {
                return Err(InsarError::DimensionMismatch(format!(
                    "{}: {rows}x{cols} difiere de {}x{} del primer par",
                    vrt_path.display(),
                    exp.0,
                    exp.1
                )));
            }
            _ => {}
        }
        values.extend(band.iter().copied());
    }

    let (rows, cols) = dims.expect("pair_dirs no vacío → dims definidas");
    Array3::from_shape_vec((pair_dirs.len(), rows, cols), values)
        .map_err(|e| InsarError::DimensionMismatch(e.to_string()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Directorio temporal único por test (sin crate tempfile).
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("insar_isce_test_{}_{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("crear dir temporal de test");
        dir
    }

    /// .vrt sintético de 2 bandas Float32 LSB para un raw de `rows`x`cols`.
    /// Banda 1: image_offset 0; banda 2: image_offset = 4 (intercaladas BIP)
    /// → pixel_offset 8, line_offset 8*cols. Devuelve el texto del .vrt.
    fn synthetic_vrt_2band(src: &str, rows: usize, cols: usize) -> String {
        let pixel_offset = 8;
        let line_offset = 8 * cols;
        format!(
            "<VRTDataset rasterXSize=\"{cols}\" rasterYSize=\"{rows}\">\n\
             <VRTRasterBand band=\"1\" dataType=\"Float32\" subClass=\"VRTRawRasterBand\">\n\
               <SourceFilename relativeToVRT=\"1\">{src}</SourceFilename>\n\
               <ByteOrder>LSB</ByteOrder>\n\
               <ImageOffset>0</ImageOffset><PixelOffset>{pixel_offset}</PixelOffset>\
               <LineOffset>{line_offset}</LineOffset>\n\
             </VRTRasterBand>\n\
             <VRTRasterBand band=\"2\" dataType=\"Float32\" subClass=\"VRTRawRasterBand\">\n\
               <SourceFilename relativeToVRT=\"1\">{src}</SourceFilename>\n\
               <ByteOrder>LSB</ByteOrder>\n\
               <ImageOffset>4</ImageOffset><PixelOffset>{pixel_offset}</PixelOffset>\
               <LineOffset>{line_offset}</LineOffset>\n\
             </VRTRasterBand>\n\
             </VRTDataset>\n"
        )
    }

    /// Patrón verificable para banda 1 y banda 2.
    fn band1_val(r: usize, c: usize) -> f32 {
        (r * 10 + c) as f32
    }
    fn band2_val(r: usize, c: usize) -> f32 {
        (r * 10 + c) as f32 + 0.5
    }

    /// Escribe el binario raw intercalado (BIP): por píxel, b1 (4 bytes) luego
    /// b2 (4 bytes), recorriendo filas y columnas.
    fn write_raw_2band(path: &Path, rows: usize, cols: usize) {
        let mut buf: Vec<u8> = Vec::with_capacity(rows * cols * 8);
        for r in 0..rows {
            for c in 0..cols {
                buf.extend_from_slice(&band1_val(r, c).to_le_bytes());
                buf.extend_from_slice(&band2_val(r, c).to_le_bytes());
            }
        }
        fs::write(path, buf).unwrap();
    }

    #[test]
    fn parse_y_lee_banda2() {
        let dir = temp_dir("parse_band2");
        let rows = 3;
        let cols = 4;
        let src = "raw.bin";
        write_raw_2band(&dir.join(src), rows, cols);
        fs::write(dir.join("raw.vrt"), synthetic_vrt_2band(src, rows, cols)).unwrap();

        let layout = parse_vrt(&dir.join("raw.vrt")).unwrap();
        assert_eq!(layout.width, cols);
        assert_eq!(layout.height, rows);
        assert_eq!(layout.bands.len(), 2);
        assert!(layout.bands[1].byte_order_lsb);
        assert_eq!(layout.bands[1].image_offset, 4);
        assert_eq!(layout.bands[1].source, dir.join(src));

        let b2 = read_raw_band(&layout, 2).unwrap();
        assert_eq!(b2.dim(), (rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(b2[[r, c]], band2_val(r, c), "b2[{r},{c}]");
            }
        }
        let b1 = read_raw_band(&layout, 1).unwrap();
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(b1[[r, c]], band1_val(r, c), "b1[{r},{c}]");
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn archivo_corto_da_error() {
        let dir = temp_dir("short_file");
        let rows = 3;
        let cols = 4;
        let src = "raw.bin";
        // Escribir un raw demasiado corto (solo 2 bytes).
        fs::write(dir.join(src), [0u8, 0u8]).unwrap();
        fs::write(dir.join("raw.vrt"), synthetic_vrt_2band(src, rows, cols)).unwrap();

        let layout = parse_vrt(&dir.join("raw.vrt")).unwrap();
        let err = read_raw_band(&layout, 2).unwrap_err();
        assert!(matches!(err, InsarError::Io(_)), "got: {err:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn subclass_invalido_da_unsupported() {
        let dir = temp_dir("bad_subclass");
        let vrt = "<VRTDataset rasterXSize=\"2\" rasterYSize=\"2\">\n\
            <VRTRasterBand band=\"1\" dataType=\"Float32\" subClass=\"VRTDerivedRasterBand\">\n\
            <SourceFilename relativeToVRT=\"1\">x.bin</SourceFilename>\n\
            <ImageOffset>0</ImageOffset><PixelOffset>4</PixelOffset><LineOffset>8</LineOffset>\n\
            </VRTRasterBand></VRTDataset>";
        fs::write(dir.join("bad.vrt"), vrt).unwrap();
        let err = parse_vrt(&dir.join("bad.vrt")).unwrap_err();
        assert!(matches!(err, InsarError::UnsupportedFormat(_)), "got: {err:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fase_enmascarada_donde_amplitud_cero() {
        // band1_val(0,0) == 0 → NoData de ISCE: la fase (0,0) debe salir NaN;
        // el resto de la banda 2 intacto.
        let dir = temp_dir("mask_amp0");
        let (rows, cols) = (3, 4);
        let src = "raw.bin";
        write_raw_2band(&dir.join(src), rows, cols);
        fs::write(dir.join("raw.vrt"), synthetic_vrt_2band(src, rows, cols)).unwrap();

        let layout = parse_vrt(&dir.join("raw.vrt")).unwrap();
        let phase = read_unw_phase_masked(&layout).unwrap();
        assert!(phase[[0, 0]].is_nan(), "amp==0 debía enmascarar la fase");
        for r in 0..rows {
            for c in 0..cols {
                if (r, c) != (0, 0) {
                    assert_eq!(phase[[r, c]], band2_val(r, c), "fase[{r},{c}]");
                }
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// Arma un directorio de stack ISCE mínimo: un par 20230101_20230113 con
    /// filt_fine.unw (+.vrt). Devuelve la raíz del stack.
    fn minimal_isce_stack(name: &str) -> PathBuf {
        let root = temp_dir(name);
        let pair = root.join("20230101_20230113");
        fs::create_dir_all(&pair).unwrap();
        write_raw_2band(&pair.join("filt_fine.unw"), 3, 4);
        fs::write(
            pair.join("filt_fine.unw.vrt"),
            synthetic_vrt_2band("filt_fine.unw", 3, 4),
        )
        .unwrap();
        root
    }

    #[test]
    fn stack_respeta_flag_de_enmascarado() {
        let root = minimal_isce_stack("stack_mask_flag");

        // Default: mask_zero_amplitude = true → (0,0) NaN.
        let stack = read_isce_unwrapped_stack(&root, &IsceLoadConfig::default()).unwrap();
        assert!(stack.data[[0, 0, 0]].is_nan());
        assert_eq!(stack.data[[0, 1, 2]], band2_val(1, 2));

        // Flag apagado → fase cruda (0.5 en (0,0)).
        let config = IsceLoadConfig { mask_zero_amplitude: false, ..Default::default() };
        let stack = read_isce_unwrapped_stack(&root, &config).unwrap();
        assert_eq!(stack.data[[0, 0, 0]], band2_val(0, 0));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn baselines_dir_inexistente_es_error() {
        let root = minimal_isce_stack("bl_typo");
        let config = IsceLoadConfig {
            baselines_dir: Some(root.join("baselineZ")), // typo
            ..Default::default()
        };
        let err = read_isce_unwrapped_stack(&root, &config).unwrap_err();
        assert!(matches!(err, InsarError::Metadata(_)), "got: {err:?}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn baselines_sin_ningun_bperp_es_error() {
        // El directorio existe pero no contiene 'Bperp (average)' para ningún
        // par (formato distinto / dir vacío) → error, no Bperp=0 silencioso.
        let root = minimal_isce_stack("bl_empty");
        let bdir = root.join("baselines");
        fs::create_dir_all(&bdir).unwrap();
        let config =
            IsceLoadConfig { baselines_dir: Some(bdir), ..Default::default() };
        let err = read_isce_unwrapped_stack(&root, &config).unwrap_err();
        assert!(matches!(err, InsarError::Metadata(_)), "got: {err:?}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn baselines_parciales_promedian_y_los_ausentes_quedan_cero() {
        let root = minimal_isce_stack("bl_partial");
        // Segundo par sin entrada de baseline.
        let pair2 = root.join("20230113_20230125");
        fs::create_dir_all(&pair2).unwrap();
        write_raw_2band(&pair2.join("filt_fine.unw"), 3, 4);
        fs::write(
            pair2.join("filt_fine.unw.vrt"),
            synthetic_vrt_2band("filt_fine.unw", 3, 4),
        )
        .unwrap();

        let bdir = root.join("baselines");
        let bpair = bdir.join("20230101_20230113");
        fs::create_dir_all(&bpair).unwrap();
        fs::write(
            bpair.join("20230101_20230113.txt"),
            "Bperp (average): 40.0\nBperp (average): 60.0\n",
        )
        .unwrap();

        let config =
            IsceLoadConfig { baselines_dir: Some(bdir), ..Default::default() };
        let stack = read_isce_unwrapped_stack(&root, &config).unwrap();
        assert_eq!(stack.pairs[0].perp_baseline_m, 50.0, "promedio de swaths");
        assert_eq!(stack.pairs[1].perp_baseline_m, 0.0, "par sin entrada → 0");
        let _ = fs::remove_dir_all(&root);
    }

    /// .vrt sintético de 1 banda con el dtype dado (raw plano BSQ).
    fn synthetic_vrt_1band(src: &str, rows: usize, cols: usize, dtype: &str, elem: usize) -> String {
        let pixel_offset = elem;
        let line_offset = elem * cols;
        format!(
            "<VRTDataset rasterXSize=\"{cols}\" rasterYSize=\"{rows}\">\n\
             <VRTRasterBand band=\"1\" dataType=\"{dtype}\" subClass=\"VRTRawRasterBand\">\n\
               <SourceFilename relativeToVRT=\"1\">{src}</SourceFilename>\n\
               <ByteOrder>LSB</ByteOrder>\n\
               <ImageOffset>0</ImageOffset><PixelOffset>{pixel_offset}</PixelOffset>\
               <LineOffset>{line_offset}</LineOffset>\n\
             </VRTRasterBand>\n\
             </VRTDataset>\n"
        )
    }

    #[test]
    fn lee_banda_cfloat32() {
        let dir = temp_dir("cfloat32");
        let (rows, cols) = (2, 3);
        let mut buf: Vec<u8> = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                buf.extend_from_slice(&((r * 10 + c) as f32).to_le_bytes()); // re
                buf.extend_from_slice(&(-((r * 10 + c) as f32) - 1.0).to_le_bytes()); // im
            }
        }
        fs::write(dir.join("ifg.int"), buf).unwrap();
        fs::write(
            dir.join("ifg.int.vrt"),
            synthetic_vrt_1band("ifg.int", rows, cols, "CFloat32", 8),
        )
        .unwrap();

        let layout = parse_vrt(&dir.join("ifg.int.vrt")).unwrap();
        let z = read_raw_band_complex(&layout, 1).unwrap();
        assert_eq!(z.dim(), (rows, cols));
        assert_eq!(z[[1, 2]].re, 12.0);
        assert_eq!(z[[1, 2]].im, -13.0);
        // f32 sobre CFloat32 debe rechazarse con error claro.
        assert!(matches!(
            read_raw_band(&layout, 1).unwrap_err(),
            InsarError::UnsupportedFormat(_)
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn lee_banda_byte_y_float64() {
        let dir = temp_dir("byte_f64");
        let (rows, cols) = (2, 2);
        // Byte.
        fs::write(dir.join("cc.bin"), [0u8, 1, 2, 255]).unwrap();
        fs::write(
            dir.join("cc.bin.vrt"),
            synthetic_vrt_1band("cc.bin", rows, cols, "Byte", 1),
        )
        .unwrap();
        let layout = parse_vrt(&dir.join("cc.bin.vrt")).unwrap();
        let b = read_raw_band_byte(&layout, 1).unwrap();
        assert_eq!(b[[0, 0]], 0);
        assert_eq!(b[[1, 1]], 255);

        // Float64 → casteado a f32.
        let mut buf: Vec<u8> = Vec::new();
        for v in [1.5_f64, -2.25, 1e6, 0.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        fs::write(dir.join("lat.bin"), buf).unwrap();
        fs::write(
            dir.join("lat.bin.vrt"),
            synthetic_vrt_1band("lat.bin", rows, cols, "Float64", 8),
        )
        .unwrap();
        let layout = parse_vrt(&dir.join("lat.bin.vrt")).unwrap();
        let f = read_raw_band(&layout, 1).unwrap();
        assert_eq!(f[[0, 0]], 1.5);
        assert_eq!(f[[0, 1]], -2.25);
        assert_eq!(f[[1, 0]], 1e6);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn stack_envuelto_desde_int_con_enmascarado() {
        let root = temp_dir("wrapped_stack");
        let pair = root.join("20230101_20230113");
        fs::create_dir_all(&pair).unwrap();
        let (rows, cols) = (2, 3);
        let mut buf: Vec<u8> = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                // Píxel (0,0) = 0+0i → NoData de ISCE.
                let re: f32 = if (r, c) == (0, 0) { 0.0 } else { (r * 10 + c) as f32 };
                let im: f32 = if (r, c) == (0, 0) { 0.0 } else { 0.5 };
                buf.extend_from_slice(&re.to_le_bytes());
                buf.extend_from_slice(&im.to_le_bytes());
            }
        }
        fs::write(pair.join("filt_fine.int"), buf).unwrap();
        fs::write(
            pair.join("filt_fine.int.vrt"),
            synthetic_vrt_1band("filt_fine.int", rows, cols, "CFloat32", 8),
        )
        .unwrap();

        let stack = read_isce_wrapped_stack(&root, &IsceLoadConfig::default()).unwrap();
        assert_eq!(stack.epochs.len(), 2);
        assert_eq!(stack.pairs.len(), 1);
        assert!(stack.data[[0, 0, 0]].re.is_nan() && stack.data[[0, 0, 0]].im.is_nan());
        assert_eq!(stack.data[[0, 1, 2]].re, 12.0);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn conncomp_desde_byte() {
        let root = temp_dir("conncomp");
        let pair = root.join("20230101_20230113");
        fs::create_dir_all(&pair).unwrap();
        fs::write(pair.join("filt_fine.unw.conncomp"), [0u8, 1, 1, 2]).unwrap();
        fs::write(
            pair.join("filt_fine.unw.conncomp.vrt"),
            synthetic_vrt_1band("filt_fine.unw.conncomp", 2, 2, "Byte", 1),
        )
        .unwrap();
        let cc = read_isce_conncomp(&root, &IsceLoadConfig::default()).unwrap();
        assert_eq!(cc.shape(), &[1, 2, 2]);
        assert_eq!(cc[[0, 0, 0]], 0);
        assert_eq!(cc[[0, 1, 1]], 2);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn los_rdr_enmascara_cero_cero() {
        let dir = temp_dir("los_rdr");
        let (rows, cols) = (2, 2);
        // BIP 2 bandas f32: (incidencia, azimut) por píxel; (1,1) = (0,0).
        let vals: [(f32, f32); 4] = [(30.0, 102.0), (46.0, 102.5), (39.0, 101.8), (0.0, 0.0)];
        let mut buf: Vec<u8> = Vec::new();
        for (inc, az) in vals {
            buf.extend_from_slice(&inc.to_le_bytes());
            buf.extend_from_slice(&az.to_le_bytes());
        }
        fs::write(dir.join("los.rdr"), buf).unwrap();
        fs::write(
            dir.join("los.rdr.vrt"),
            synthetic_vrt_2band("los.rdr", rows, cols),
        )
        .unwrap();

        let (inc, az) = read_isce_los(&dir.join("los.rdr")).unwrap();
        assert_eq!(inc[[0, 0]], 30.0);
        assert_eq!(az[[0, 1]], 102.5);
        assert!(inc[[1, 1]].is_nan() && az[[1, 1]].is_nan(), "(0,0) es NoData");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_pair_name_estricto() {
        assert!(parse_pair_name("20141213_20141225").is_some());
        assert!(parse_pair_name("2014121_20141225").is_none());
        assert!(parse_pair_name("20141213-20141225").is_none());
        assert!(parse_pair_name("foo_bar").is_none());
        assert!(parse_pair_name("20141213_2014122x").is_none());
    }

    // -----------------------------------------------------------------------
    // Test de integración con datos reales (Fernandina Sentinel-1 DT128).
    // Marcado #[ignore] para no depender de los datos en CI.
    // Correr con: cargo test -p insar-core --ignored isce_real
    // -----------------------------------------------------------------------
    #[test]
    #[ignore = "requiere datos reales en data/FernandinaSenDT128 (cargo test --ignored isce_real)"]
    fn isce_real_fernandina_stack() {
        use crate::network;

        let base = Path::new(
            "/home/franciscoparrao/proyectos/insar-rs/data/FernandinaSenDT128",
        );
        let ifg_dir = base.join("merged/interferograms");
        if !ifg_dir.exists() {
            eprintln!("datos reales ausentes; saltando");
            return;
        }

        let config = IsceLoadConfig {
            baselines_dir: Some(base.join("baselines")),
            ..IsceLoadConfig::default()
        };

        let stack = read_isce_unwrapped_stack(&ifg_dir, &config).unwrap();

        assert_eq!(stack.epochs.len(), 98, "épocas");
        assert_eq!(stack.pairs.len(), 288, "pares");
        assert_eq!(stack.dims(), (450, 600), "dims (filas, cols)");
        stack.validate().unwrap();
        assert!(
            network::is_connected(&stack.pairs, stack.epochs.len()),
            "la red SBAS debe ser conexa"
        );

        // La fase no es toda cero ni toda NaN.
        let mut any_finite_nonzero = false;
        let mut any_baseline_nonzero = false;
        for &v in stack.data.iter() {
            if v.is_finite() && v != 0.0 {
                any_finite_nonzero = true;
                break;
            }
        }
        assert!(any_finite_nonzero, "la fase es toda cero/NaN");
        for p in &stack.pairs {
            if p.perp_baseline_m != 0.0 {
                any_baseline_nonzero = true;
                break;
            }
        }
        assert!(any_baseline_nonzero, "ninguna baseline distinta de cero");

        // Coherencia alineada con los mismos pares.
        let coh = read_isce_coherence(&ifg_dir, &config).unwrap();
        assert_eq!(coh.shape(), &[288, 450, 600], "shape coherencia");
    }
}

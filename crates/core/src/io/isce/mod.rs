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
//! Las baselines perpendiculares se leen, si se indica `baselines_dir`, de
//! `baselines/<par>/...` (texto con líneas `Bperp (average): <valor>` por
//! swath; se promedian los swaths).
//!
//! Como ISCE entrega la fase ya desenrollada, la salida natural es
//! [`UnwrappedStack`] (no `IfgStack`): puede alimentarse directamente a
//! [`crate::inversion::invert_sbas`].

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use ndarray::{Array2, Array3};
use surtgis_core::GeoTransform;

use crate::error::{InsarError, Result};
use crate::types::{Epoch, IfgPair, StackMeta, UnwrappedStack};

/// Configuración de carga de un stack ISCE.
#[derive(Debug, Clone)]
pub struct IsceLoadConfig {
    /// Nombre del archivo de fase desenrollada en cada par (default `filt_fine.unw`).
    pub unw_filename: String,
    /// Nombre del archivo de coherencia (default `filt_fine.cor`).
    pub cor_filename: String,
    /// Directorio de baselines (`baselines/<par>/`); `None` → baselines = 0.
    pub baselines_dir: Option<PathBuf>,
    /// Longitud de onda radar en metros (ISCE no la guarda en el .vrt).
    pub wavelength_m: f64,
    /// Ángulo de incidencia medio en grados.
    pub incidence_deg: f64,
}

impl Default for IsceLoadConfig {
    fn default() -> Self {
        Self {
            unw_filename: "filt_fine.unw".to_string(),
            cor_filename: "filt_fine.cor".to_string(),
            baselines_dir: None,
            wavelength_m: crate::types::SENTINEL1_WAVELENGTH_M,
            incidence_deg: 39.0,
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
/// (filas = height, cols = width). Solo soporta `Float32`.
pub(crate) fn read_raw_band(layout: &VrtLayout, band_1based: usize) -> Result<Array2<f32>> {
    if band_1based == 0 || band_1based > layout.bands.len() {
        return Err(InsarError::UnsupportedFormat(format!(
            "banda {band_1based} fuera de rango (1..={})",
            layout.bands.len()
        )));
    }
    let band = &layout.bands[band_1based - 1];

    if band.data_type != "Float32" {
        return Err(InsarError::UnsupportedFormat(format!(
            "dataType '{}' no soportado (solo Float32)",
            band.data_type
        )));
    }

    let rows = layout.height;
    let cols = layout.width;

    let raw = fs::read(&band.source)?;

    // Byte máximo requerido por el píxel (rows-1, cols-1) de esta banda.
    let mut out = Array2::<f32>::zeros((rows, cols));
    if rows == 0 || cols == 0 {
        return Ok(out);
    }

    let image_offset = band.image_offset as usize;
    let pixel_offset = band.pixel_offset as usize;
    let line_offset = band.line_offset as usize;

    // Verificación de cota superior (último píxel + 4 bytes).
    let last_byte = image_offset
        .checked_add((rows - 1) * line_offset)
        .and_then(|v| v.checked_add((cols - 1) * pixel_offset))
        .and_then(|v| v.checked_add(4));
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

    for r in 0..rows {
        let row_base = image_offset + r * line_offset;
        for c in 0..cols {
            let pos = row_base + c * pixel_offset;
            let bytes = [raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]];
            // ByteOrder LSB = little-endian; MSB = big-endian.
            let v = if band.byte_order_lsb {
                f32::from_le_bytes(bytes)
            } else {
                f32::from_be_bytes(bytes)
            };
            out[[r, c]] = v;
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
/// `baselines_dir/<par>/*`. Si el directorio/archivo no existe → 0.0.
fn read_baseline(baselines_dir: &Path, pair_name: &str) -> f64 {
    let pair_dir = baselines_dir.join(pair_name);
    let entries = match fs::read_dir(&pair_dir) {
        Ok(e) => e,
        Err(_) => return 0.0,
    };

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

    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Lee un directorio de interferogramas ISCE en un [`UnwrappedStack`].
///
/// Descubre los subdirectorios `YYYYMMDD_YYYYMMDD`, deriva las épocas (unión
/// ordenada de fechas) y los pares, lee la banda 2 (fase) de cada `.unw` y
/// las baselines si se configuran. La georreferencia es nominal (coordenadas
/// radar): se usa una transform identidad. Llama `validate()` antes de devolver.
pub fn read_isce_unwrapped_stack(dir: &Path, config: &IsceLoadConfig) -> Result<UnwrappedStack> {
    let pair_dirs = list_pair_dirs(dir)?;

    // Épocas = unión ordenada ascendente de todas las fechas → índice.
    let mut date_set: BTreeSet<NaiveDate> = BTreeSet::new();
    for (d1, d2, _, _) in &pair_dirs {
        date_set.insert(*d1);
        date_set.insert(*d2);
    }
    let epochs: Vec<Epoch> = date_set.iter().map(|d| Epoch(*d)).collect();
    let date_to_idx = |d: &NaiveDate| -> usize {
        // BTreeSet ordenado → posición = índice.
        date_set.iter().position(|x| x == d).expect("fecha presente en el set")
    };

    let mut pairs: Vec<IfgPair> = Vec::with_capacity(pair_dirs.len());
    let mut values: Vec<f32> = Vec::new();
    let mut dims: Option<(usize, usize)> = None;

    for (d1, d2, name, path) in &pair_dirs {
        let reference = date_to_idx(d1);
        let secondary = date_to_idx(d2);

        let vrt_path = path.join(format!("{}.vrt", config.unw_filename));
        let layout = parse_vrt(&vrt_path)?;
        let band = read_raw_band(&layout, 2)?;

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

        let perp_baseline_m = match &config.baselines_dir {
            Some(bdir) => read_baseline(bdir, name),
            None => 0.0,
        };

        pairs.push(IfgPair {
            reference,
            secondary,
            perp_baseline_m,
        });
    }

    let (rows, cols) = dims.expect("pair_dirs no vacío → dims definidas");
    let meta = StackMeta {
        transform: GeoTransform::new(0.0, 0.0, 1.0, -1.0),
        crs: None,
        wavelength_m: config.wavelength_m,
        incidence_deg: config.incidence_deg,
        heading_deg: None,
    };
    let data = Array3::from_shape_vec((pair_dirs.len(), rows, cols), values)
        .map_err(|e| InsarError::DimensionMismatch(e.to_string()))?;

    let stack = UnwrappedStack {
        data,
        epochs,
        pairs,
        meta,
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

    let mut values: Vec<f32> = Vec::new();
    let mut dims: Option<(usize, usize)> = None;

    for (_, _, _, path) in &pair_dirs {
        let vrt_path = path.join(format!("{}.vrt", config.cor_filename));
        let layout = parse_vrt(&vrt_path)?;
        let band = read_raw_band(&layout, 1)?;

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

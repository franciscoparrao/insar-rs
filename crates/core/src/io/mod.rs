//! Lectura/escritura de stacks y productos.
//!
//! # Formato de stack de entrada
//!
//! Un stack es un **directorio** que contiene un manifiesto `stack.json` más
//! los GeoTIFF referenciados por él. El manifiesto (JSON, fechas ISO-8601):
//!
//! ```json
//! {
//!   "wavelength_m": 0.05546576,
//!   "incidence_deg": 39.0,
//!   "heading_deg": null,
//!   "epochs": ["2023-01-01", "2023-01-13"],
//!   "ifgs": [
//!     {"reference": 0, "secondary": 1, "perp_baseline_m": 50.0,
//!      "file": "ifg_20230101_20230113.tif"}
//!   ],
//!   "amplitudes": ["amp_20230101.tif", "amp_20230113.tif"],
//!   "coherence": ["coh_20230101_20230113.tif"]
//! }
//! ```
//!
//! - `wavelength_m`, `incidence_deg`, `epochs`: obligatorios.
//! - `heading_deg`: opcional (puede faltar o ser `null`).
//! - `ifgs`: requerido por [`read_ifg_stack`]; `reference`/`secondary` son
//!   índices dentro de `epochs`. Puede faltar si solo se leen amplitudes.
//! - `amplitudes`: requerido por [`read_amplitude_stack`]; lista alineada
//!   1:1 con `epochs` (una amplitud por época). Puede faltar si solo se
//!   leen interferogramas.
//! - `coherence`: opcional; lista alineada 1:1 con `ifgs` (un GeoTIFF float32
//!   de 1 banda por interferograma). La lee [`read_coherence_stack`]; el
//!   pipeline la usa como calidad del desenrollado, para el píxel de
//!   referencia automático y para los pesos WLS de la inversión.
//!
//! # Convención de archivos complejos (fallback re/im)
//!
//! El reader GeoTIFF nativo de `surtgis-core` (sin GDAL) **no soporta
//! selección de banda** en archivos multibanda: el parámetro `band` de
//! `read_geotiff` se ignora y un TIFF de 2 bandas falla la verificación de
//! dimensiones. Por eso cada interferograma complejo se persiste como **dos
//! GeoTIFF float32 de 1 banda**, con nombres derivados del campo `file` del
//! manifiesto: para `"file": "ifg_X.tif"` se leen `ifg_X_re.tif` (parte
//! real) e `ifg_X_im.tif` (parte imaginaria). Cada GeoTIFF de amplitud es
//! un único archivo float32 de 1 banda con el nombre literal del manifiesto.
//!
//! - Georreferencia ([`StackMeta::transform`]/`crs`) se toma del **primer**
//!   archivo leído; todos los archivos deben compartir dimensiones
//!   (error [`InsarError::DimensionMismatch`] si difieren).
//! - NoData = `f32::NAN` (en complejos: ambas partes NaN, convención surtgis).
//!
//! El lector de formato ISCE binario plano (.int/.unw + XML) es fase tardía
//! v0.1 — ver PLAN.md.

pub mod isce;

use std::fs;
use std::path::{Path, PathBuf};

use ndarray::{Array2, Array3, Axis};
use num_complex::Complex32;
use serde::Deserialize;
use surtgis_core::Raster;
use surtgis_core::io::{read_geotiff, write_geotiff};

use crate::error::{InsarError, Result};
use crate::types::{
    AmplitudeStack, DisplacementSeries, Epoch, IfgPair, IfgStack, StackMeta, VelocityMap,
};

/// Prefijo de archivo usado por [`write_series`]/[`read_series`]
/// (`disp_YYYYMMDD.tif`).
const SERIES_FILE_PREFIX: &str = "disp_";
/// Formato de fecha del nombre de archivo (`chrono::NaiveDate::parse_from_str`).
const SERIES_DATE_FORMAT: &str = "%Y%m%d";

/// Nombre del manifiesto dentro del directorio del stack.
const MANIFEST_NAME: &str = "stack.json";

// ---------------------------------------------------------------------------
// Manifiesto (stack.json)
// ---------------------------------------------------------------------------

/// Entrada de interferograma en `stack.json`.
#[derive(Debug, Clone, Deserialize)]
struct IfgEntry {
    reference: usize,
    secondary: usize,
    perp_baseline_m: f64,
    file: String,
}

/// Estructura del manifiesto `stack.json` (ver doc del módulo).
#[derive(Debug, Clone, Deserialize)]
struct StackManifest {
    wavelength_m: f64,
    incidence_deg: f64,
    #[serde(default)]
    heading_deg: Option<f64>,
    epochs: Vec<String>,
    #[serde(default)]
    ifgs: Option<Vec<IfgEntry>>,
    #[serde(default)]
    amplitudes: Option<Vec<String>>,
    #[serde(default)]
    coherence: Option<Vec<String>>,
}

/// Lee y valida el manifiesto; devuelve manifiesto + épocas parseadas.
fn load_manifest(dir: &Path) -> Result<(StackManifest, Vec<Epoch>)> {
    let path = dir.join(MANIFEST_NAME);
    let text = fs::read_to_string(&path)?;
    let manifest: StackManifest = serde_json::from_str(&text).map_err(|e| {
        InsarError::Metadata(format!("{} malformado: {e}", path.display()))
    })?;

    if !(manifest.wavelength_m.is_finite() && manifest.wavelength_m > 0.0) {
        return Err(InsarError::Metadata(format!(
            "wavelength_m inválido: {}",
            manifest.wavelength_m
        )));
    }
    if manifest.epochs.is_empty() {
        return Err(InsarError::Metadata("lista de épocas vacía".into()));
    }

    let mut epochs = Vec::with_capacity(manifest.epochs.len());
    for s in &manifest.epochs {
        let date = s.parse().map_err(|e| {
            InsarError::Metadata(format!("época '{s}' no es fecha ISO-8601: {e}"))
        })?;
        epochs.push(Epoch(date));
    }

    // Épocas estrictamente crecientes: una duplicada haría que write_series
    // sobrescriba disp_YYYYMMDD.tif en silencio y que un par entre duplicadas
    // (baseline temporal 0) degenere la matriz de diseño SBAS con un error
    // críptico aguas abajo; el desorden rompe la parametrización por
    // incrementos (reference < secondary en índice = en tiempo).
    if let Some(w) = epochs.windows(2).find(|w| w[0] >= w[1]) {
        return Err(InsarError::Metadata(format!(
            "épocas deben ser estrictamente crecientes y sin duplicados: \
             '{}' no es anterior a '{}'",
            w[0].0, w[1].0
        )));
    }

    Ok((manifest, epochs))
}

// ---------------------------------------------------------------------------
// Helpers de lectura
// ---------------------------------------------------------------------------

/// Lee un GeoTIFF float32 de 1 banda con el reader nativo de surtgis.
fn read_f32(path: &Path) -> Result<Raster<f32>> {
    read_geotiff::<f32, _>(path, None)
        .map_err(|e| InsarError::Raster(format!("{}: {e}", path.display())))
}

/// Deriva las rutas `*_re.tif` / `*_im.tif` desde el campo `file` del
/// manifiesto (ver doc del módulo, "fallback re/im").
fn re_im_paths(dir: &Path, file: &str) -> (PathBuf, PathBuf) {
    let stem = file.strip_suffix(".tif").unwrap_or(file);
    (
        dir.join(format!("{stem}_re.tif")),
        dir.join(format!("{stem}_im.tif")),
    )
}

/// Verifica que `shape` coincida con las dimensiones de referencia.
fn check_dims(path: &Path, shape: (usize, usize), expected: (usize, usize)) -> Result<()> {
    if shape != expected {
        return Err(InsarError::DimensionMismatch(format!(
            "{}: {}x{} difiere de {}x{} del primer archivo del stack",
            path.display(),
            shape.0,
            shape.1,
            expected.0,
            expected.1
        )));
    }
    Ok(())
}

/// Construye el `StackMeta` combinando manifiesto + georreferencia del
/// primer raster leído.
fn meta_from(first: &Raster<f32>, manifest: &StackManifest) -> StackMeta {
    StackMeta {
        transform: *first.transform(),
        crs: first.crs().cloned(),
        wavelength_m: manifest.wavelength_m,
        incidence_deg: manifest.incidence_deg,
        heading_deg: manifest.heading_deg,
    }
}

// ---------------------------------------------------------------------------
// API pública
// ---------------------------------------------------------------------------

/// Lee un stack de interferogramas complejos desde `dir` (formato del módulo).
///
/// Requiere el campo `ifgs` en `stack.json`. Cada interferograma se lee desde
/// el par de archivos `*_re.tif`/`*_im.tif` derivado de su campo `file`
/// (ver doc del módulo). Georreferencia del primer archivo; error
/// [`InsarError::DimensionMismatch`] si algún archivo difiere en dimensiones.
pub fn read_ifg_stack(dir: &Path) -> Result<IfgStack> {
    let (manifest, epochs) = load_manifest(dir)?;
    let entries = manifest.ifgs.as_deref().ok_or_else(|| {
        InsarError::Metadata("stack.json no tiene campo 'ifgs' (requerido para read_ifg_stack)".into())
    })?;
    if entries.is_empty() {
        return Err(InsarError::Metadata("campo 'ifgs' vacío".into()));
    }

    let mut meta: Option<StackMeta> = None;
    let mut dims: Option<(usize, usize)> = None;
    let mut values: Vec<Complex32> = Vec::new();
    let mut pairs = Vec::with_capacity(entries.len());

    for entry in entries {
        let (re_path, im_path) = re_im_paths(dir, &entry.file);
        let re = read_f32(&re_path)?;
        let im = read_f32(&im_path)?;

        let expected = *dims.get_or_insert_with(|| re.shape());
        check_dims(&re_path, re.shape(), expected)?;
        check_dims(&im_path, im.shape(), expected)?;

        if meta.is_none() {
            meta = Some(meta_from(&re, &manifest));
            values.reserve(entries.len() * expected.0 * expected.1);
        }

        values.extend(
            re.data()
                .iter()
                .zip(im.data().iter())
                .map(|(&r, &i)| Complex32::new(r, i)),
        );
        pairs.push(IfgPair {
            reference: entry.reference,
            secondary: entry.secondary,
            perp_baseline_m: entry.perp_baseline_m,
        });
    }

    // entries no está vacío → dims/meta están definidos.
    let (rows, cols) = dims.expect("dims definidas: entries no vacío");
    let meta = meta.expect("meta definida: entries no vacío");
    let data = Array3::from_shape_vec((entries.len(), rows, cols), values)
        .map_err(|e| InsarError::DimensionMismatch(e.to_string()))?;

    let stack = IfgStack { data, epochs, pairs, meta };
    stack.validate()?;
    Ok(stack)
}

/// Lee un stack de amplitudes SLC coregistradas desde `dir`.
///
/// Requiere el campo `amplitudes` en `stack.json`, alineado 1:1 con `epochs`
/// (una amplitud por época). Cada amplitud es un GeoTIFF float32 de 1 banda.
pub fn read_amplitude_stack(dir: &Path) -> Result<AmplitudeStack> {
    let (manifest, epochs) = load_manifest(dir)?;
    let files = manifest.amplitudes.as_deref().ok_or_else(|| {
        InsarError::Metadata(
            "stack.json no tiene campo 'amplitudes' (requerido para read_amplitude_stack)".into(),
        )
    })?;
    if files.len() != epochs.len() {
        return Err(InsarError::Metadata(format!(
            "{} amplitudes declaradas vs {} épocas (deben estar alineadas 1:1)",
            files.len(),
            epochs.len()
        )));
    }

    let mut meta: Option<StackMeta> = None;
    let mut dims: Option<(usize, usize)> = None;
    let mut values: Vec<f32> = Vec::new();

    for file in files {
        let path = dir.join(file);
        let raster = read_f32(&path)?;

        let expected = *dims.get_or_insert_with(|| raster.shape());
        check_dims(&path, raster.shape(), expected)?;

        if meta.is_none() {
            meta = Some(meta_from(&raster, &manifest));
            values.reserve(files.len() * expected.0 * expected.1);
        }
        values.extend(raster.data().iter().copied());
    }

    // files no vacío (len == epochs.len() > 0) → dims/meta definidos.
    let (rows, cols) = dims.expect("dims definidas: files no vacío");
    let meta = meta.expect("meta definida: files no vacío");
    let data = Array3::from_shape_vec((files.len(), rows, cols), values)
        .map_err(|e| InsarError::DimensionMismatch(e.to_string()))?;

    Ok(AmplitudeStack { data, epochs, meta })
}

/// Lee el stack de coherencia declarado en `stack.json` (campo opcional
/// `coherence`, alineado 1:1 con `ifgs`): un GeoTIFF float32 de 1 banda por
/// interferograma. `data`: pares × filas × cols, mismo orden que `ifgs`.
///
/// Devuelve `Ok(None)` si el manifiesto no declara coherencia (es opcional);
/// error si la lista está desalineada con `ifgs`, falta el campo `ifgs`, o
/// las dimensiones difieren entre archivos.
pub fn read_coherence_stack(dir: &Path) -> Result<Option<Array3<f32>>> {
    let (manifest, _epochs) = load_manifest(dir)?;
    let Some(files) = manifest.coherence.as_deref() else {
        return Ok(None);
    };
    let ifgs = manifest.ifgs.as_deref().ok_or_else(|| {
        InsarError::Metadata(
            "stack.json declara 'coherence' pero no 'ifgs' (la coherencia se alinea 1:1 con los interferogramas)"
                .into(),
        )
    })?;
    if files.len() != ifgs.len() {
        return Err(InsarError::Metadata(format!(
            "{} archivos de coherencia declarados vs {} ifgs (deben estar alineados 1:1)",
            files.len(),
            ifgs.len()
        )));
    }
    if files.is_empty() {
        return Err(InsarError::Metadata("campo 'coherence' vacío".into()));
    }

    let mut dims: Option<(usize, usize)> = None;
    let mut values: Vec<f32> = Vec::new();
    for file in files {
        let path = dir.join(file);
        let raster = read_f32(&path)?;
        let expected = *dims.get_or_insert_with(|| raster.shape());
        check_dims(&path, raster.shape(), expected)?;
        if values.is_empty() {
            values.reserve(files.len() * expected.0 * expected.1);
        }
        values.extend(raster.data().iter().copied());
    }

    let (rows, cols) = dims.expect("files no vacío: len == ifgs.len() > 0 validado arriba");
    let data = Array3::from_shape_vec((files.len(), rows, cols), values)
        .map_err(|e| InsarError::DimensionMismatch(e.to_string()))?;
    Ok(Some(data))
}

/// Convierte una capa 2D `f32` + metadata del stack en un `Raster` surtgis
/// listo para escribir (nodata = NaN).
fn raster_from_layer(
    layer: ndarray::ArrayView2<'_, f32>,
    meta: &StackMeta,
) -> Result<Raster<f32>> {
    let (rows, cols) = layer.dim();
    let data: Vec<f32> = layer.iter().copied().collect();
    let mut raster = Raster::from_vec(data, rows, cols)
        .map_err(|e| InsarError::Raster(e.to_string()))?;
    raster.set_transform(meta.transform);
    raster.set_crs(meta.crs.clone());
    raster.set_nodata(Some(f32::NAN));
    Ok(raster)
}

/// Escribe el mapa de velocidad LOS (m/año) como GeoTIFF Float32 de 1 banda,
/// con transform/CRS del meta y nodata = NaN.
pub fn write_velocity(map: &VelocityMap, path: &Path) -> Result<()> {
    let raster = raster_from_layer(map.data.view(), &map.meta)?;
    write_geotiff(&raster, path, None)
        .map_err(|e| InsarError::Raster(format!("{}: {e}", path.display())))
}

/// Escribe la serie de desplazamiento como un GeoTIFF Float32 por época en
/// `dir` (creándolo si no existe), nombrados `disp_YYYYMMDD.tif`.
pub fn write_series(series: &DisplacementSeries, dir: &Path) -> Result<()> {
    if series.data.shape()[0] != series.epochs.len() {
        return Err(InsarError::DimensionMismatch(format!(
            "{} capas en la serie vs {} épocas",
            series.data.shape()[0],
            series.epochs.len()
        )));
    }
    fs::create_dir_all(dir)?;
    for (i, epoch) in series.epochs.iter().enumerate() {
        let layer = series.data.index_axis(Axis(0), i);
        let raster = raster_from_layer(layer, &series.meta)?;
        let path = dir.join(format!(
            "{SERIES_FILE_PREFIX}{}.tif",
            epoch.0.format(SERIES_DATE_FORMAT)
        ));
        write_geotiff(&raster, &path, None)
            .map_err(|e| InsarError::Raster(format!("{}: {e}", path.display())))?;
    }
    Ok(())
}

/// Lee un mapa de velocidad LOS (m/año) desde un GeoTIFF Float32 de 1 banda —
/// inverso de [`write_velocity`].
///
/// El GeoTIFF no guarda `wavelength_m`/`incidence_deg`/`heading_deg` (no son
/// geográficos): se toman de `meta`. `transform`/`crs` se toman del archivo
/// (georreferencia real), descartando los de `meta`.
pub fn read_velocity(path: &Path, meta: StackMeta) -> Result<VelocityMap> {
    let raster = read_f32(path)?;
    let (rows, cols) = raster.shape();
    let data = Array2::from_shape_vec((rows, cols), raster.data().iter().copied().collect())
        .map_err(|e| InsarError::DimensionMismatch(e.to_string()))?;
    let meta = StackMeta { transform: *raster.transform(), crs: raster.crs().cloned(), ..meta };
    Ok(VelocityMap { data, meta })
}

/// Lee una serie de desplazamiento desde `dir` — inverso de [`write_series`]:
/// un GeoTIFF Float32 por época nombrado `disp_YYYYMMDD.tif`. Las épocas se
/// derivan de los nombres de archivo (orden ascendente por fecha, no por
/// orden del directorio); `wavelength_m`/`incidence_deg`/`heading_deg` de
/// `meta` (no georreferenciados, no se guardan en el GeoTIFF); `transform`/
/// `crs` del primer archivo (por fecha).
pub fn read_series(dir: &Path, meta: StackMeta) -> Result<DisplacementSeries> {
    let mut entries: Vec<(Epoch, PathBuf)> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
            let stem = path.file_stem()?.to_str()?;
            let day = stem.strip_prefix(SERIES_FILE_PREFIX)?;
            let date = chrono::NaiveDate::parse_from_str(day, SERIES_DATE_FORMAT).ok()?;
            Some((Epoch(date), path))
        })
        .collect();
    if entries.is_empty() {
        return Err(InsarError::Metadata(format!(
            "{}: no se encontraron archivos {SERIES_FILE_PREFIX}YYYYMMDD.tif",
            dir.display()
        )));
    }
    entries.sort_by_key(|(epoch, _)| *epoch);

    let mut dims: Option<(usize, usize)> = None;
    let mut geo: Option<(surtgis_core::GeoTransform, Option<surtgis_core::CRS>)> = None;
    let mut values: Vec<f32> = Vec::with_capacity(entries.len());
    let mut epochs = Vec::with_capacity(entries.len());
    for (epoch, path) in &entries {
        let raster = read_f32(path)?;
        let expected = *dims.get_or_insert_with(|| raster.shape());
        check_dims(path, raster.shape(), expected)?;
        geo.get_or_insert_with(|| (*raster.transform(), raster.crs().cloned()));
        values.extend(raster.data().iter().copied());
        epochs.push(*epoch);
    }

    let (rows, cols) = dims.expect("dims definidas: entries no vacío");
    let (transform, crs) = geo.expect("geo definida: entries no vacío");
    let data = Array3::from_shape_vec((entries.len(), rows, cols), values)
        .map_err(|e| InsarError::DimensionMismatch(e.to_string()))?;
    let meta = StackMeta { transform, crs, ..meta };
    Ok(DisplacementSeries { data, epochs, meta })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};
    use surtgis_core::{CRS, GeoTransform};

    /// Directorio temporal único por test (sin crate tempfile). Se limpia
    /// al inicio por si quedó basura de una corrida anterior.
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "insar_io_test_{}_{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("crear dir temporal de test");
        dir
    }

    fn test_transform() -> GeoTransform {
        GeoTransform::new(500_000.0, 7_000_000.0, 30.0, -30.0)
    }

    fn test_crs() -> CRS {
        CRS::from_epsg(32719)
    }

    /// Escribe un GeoTIFF f32 1 banda con georef de prueba (writer surtgis).
    fn write_test_tif(path: &Path, data: Vec<f32>, rows: usize, cols: usize) {
        let mut raster = Raster::from_vec(data, rows, cols).unwrap();
        raster.set_transform(test_transform());
        raster.set_crs(Some(test_crs()));
        raster.set_nodata(Some(f32::NAN));
        write_geotiff(&raster, path, None).unwrap();
    }

    fn assert_f32_eq_nan(a: f32, b: f32, ctx: &str) {
        assert!(
            (a.is_nan() && b.is_nan()) || a == b,
            "{ctx}: {a} != {b}"
        );
    }

    const ROWS: usize = 4;
    const COLS: usize = 5;

    /// Valor sintético reproducible por (capa, fila, col); NaN en (0,1,2).
    fn synth(layer: usize, r: usize, c: usize) -> f32 {
        if (layer, r, c) == (0, 1, 2) {
            f32::NAN
        } else {
            (layer * 100 + r * 10 + c) as f32
        }
    }

    /// Manifiesto completo de prueba: 3 épocas, 2 ifgs, 3 amplitudes.
    const MANIFEST: &str = r#"{
        "wavelength_m": 0.05546576,
        "incidence_deg": 39.0,
        "heading_deg": null,
        "epochs": ["2023-01-01", "2023-01-13", "2023-01-25"],
        "ifgs": [
            {"reference": 0, "secondary": 1, "perp_baseline_m": 50.0,
             "file": "ifg_20230101_20230113.tif"},
            {"reference": 1, "secondary": 2, "perp_baseline_m": -30.5,
             "file": "ifg_20230113_20230125.tif"}
        ],
        "amplitudes": ["amp_20230101.tif", "amp_20230113.tif", "amp_20230125.tif"]
    }"#;

    /// Construye en `dir` el stack sintético completo (manifest + tifs).
    fn build_synthetic_stack(dir: &Path) {
        fs::write(dir.join("stack.json"), MANIFEST).unwrap();
        for (k, name) in ["ifg_20230101_20230113", "ifg_20230113_20230125"]
            .iter()
            .enumerate()
        {
            // re = synth(k,..); im = -synth(k,..) - 1 (distinto de re)
            let re: Vec<f32> = (0..ROWS * COLS)
                .map(|i| synth(k, i / COLS, i % COLS))
                .collect();
            let im: Vec<f32> = re.iter().map(|v| -v - 1.0).collect();
            write_test_tif(&dir.join(format!("{name}_re.tif")), re, ROWS, COLS);
            write_test_tif(&dir.join(format!("{name}_im.tif")), im, ROWS, COLS);
        }
        for (k, name) in ["amp_20230101", "amp_20230113", "amp_20230125"]
            .iter()
            .enumerate()
        {
            let amp: Vec<f32> = (0..ROWS * COLS)
                .map(|i| synth(k, i / COLS, i % COLS).abs())
                .collect();
            write_test_tif(&dir.join(format!("{name}.tif")), amp, ROWS, COLS);
        }
    }

    fn expected_epochs() -> Vec<Epoch> {
        ["2023-01-01", "2023-01-13", "2023-01-25"]
            .iter()
            .map(|s| Epoch(s.parse().unwrap()))
            .collect()
    }

    #[test]
    fn round_trip_ifg_stack() {
        let dir = temp_dir("ifg_rt");
        build_synthetic_stack(&dir);

        let stack = read_ifg_stack(&dir).unwrap();

        // Dimensiones y estructura
        assert_eq!(stack.n_layers(), 2);
        assert_eq!(stack.dims(), (ROWS, COLS));
        assert_eq!(stack.epochs, expected_epochs());
        assert_eq!(stack.pairs.len(), 2);
        assert_eq!(stack.pairs[0].reference, 0);
        assert_eq!(stack.pairs[0].secondary, 1);
        assert_eq!(stack.pairs[0].perp_baseline_m, 50.0);
        assert_eq!(stack.pairs[1].reference, 1);
        assert_eq!(stack.pairs[1].secondary, 2);
        assert_eq!(stack.pairs[1].perp_baseline_m, -30.5);

        // Metadata
        assert_eq!(stack.meta.transform, test_transform());
        assert_eq!(stack.meta.crs, Some(test_crs()));
        assert_eq!(stack.meta.wavelength_m, 0.05546576);
        assert_eq!(stack.meta.incidence_deg, 39.0);
        assert_eq!(stack.meta.heading_deg, None);

        // Datos: re/im exactos, NaN preservado en ambas partes
        for k in 0..2 {
            for r in 0..ROWS {
                for c in 0..COLS {
                    let v = stack.data[[k, r, c]];
                    let re = synth(k, r, c);
                    let im = -re - 1.0;
                    let ctx = format!("ifg[{k},{r},{c}]");
                    assert_f32_eq_nan(v.re, re, &ctx);
                    assert_f32_eq_nan(v.im, im, &ctx);
                }
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn round_trip_amplitude_stack() {
        let dir = temp_dir("amp_rt");
        build_synthetic_stack(&dir);

        let stack = read_amplitude_stack(&dir).unwrap();

        assert_eq!(stack.n_layers(), 3);
        assert_eq!(stack.dims(), (ROWS, COLS));
        assert_eq!(stack.epochs, expected_epochs());
        assert_eq!(stack.meta.transform, test_transform());
        assert_eq!(stack.meta.crs, Some(test_crs()));
        assert_eq!(stack.meta.wavelength_m, 0.05546576);

        for k in 0..3 {
            for r in 0..ROWS {
                for c in 0..COLS {
                    let exp = synth(k, r, c).abs();
                    assert_f32_eq_nan(
                        stack.data[[k, r, c]],
                        exp,
                        &format!("amp[{k},{r},{c}]"),
                    );
                }
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn round_trip_velocity() {
        let dir = temp_dir("vel_rt");
        let data = Array2::from_shape_fn((ROWS, COLS), |(r, c)| synth(0, r, c));
        let map = VelocityMap {
            data: data.clone(),
            meta: StackMeta {
                transform: test_transform(),
                crs: Some(test_crs()),
                wavelength_m: crate::types::SENTINEL1_WAVELENGTH_M,
                incidence_deg: 39.0,
                heading_deg: Some(190.0),
            },
        };
        let path = dir.join("velocity.tif");
        write_velocity(&map, &path).unwrap();

        let back: Raster<f32> = read_geotiff(&path, None).unwrap();
        assert_eq!(back.shape(), (ROWS, COLS));
        assert_eq!(*back.transform(), test_transform());
        assert_eq!(back.crs(), Some(&test_crs()));
        for r in 0..ROWS {
            for c in 0..COLS {
                assert_f32_eq_nan(back.data()[[r, c]], data[[r, c]], &format!("vel[{r},{c}]"));
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn round_trip_series() {
        let dir = temp_dir("ser_rt").join("salida"); // subdir inexistente: write_series debe crearlo
        let epochs = expected_epochs();
        let data = Array3::from_shape_fn((3, ROWS, COLS), |(k, r, c)| synth(k, r, c));
        let series = DisplacementSeries {
            data: data.clone(),
            epochs: epochs.clone(),
            meta: StackMeta {
                transform: test_transform(),
                crs: Some(test_crs()),
                wavelength_m: crate::types::SENTINEL1_WAVELENGTH_M,
                incidence_deg: 39.0,
                heading_deg: None,
            },
        };
        write_series(&series, &dir).unwrap();

        for (k, name) in ["disp_20230101.tif", "disp_20230113.tif", "disp_20230125.tif"]
            .iter()
            .enumerate()
        {
            let back: Raster<f32> = read_geotiff(dir.join(name), None).unwrap();
            assert_eq!(back.shape(), (ROWS, COLS), "{name}");
            assert_eq!(*back.transform(), test_transform(), "{name}");
            for r in 0..ROWS {
                for c in 0..COLS {
                    assert_f32_eq_nan(
                        back.data()[[r, c]],
                        data[[k, r, c]],
                        &format!("{name}[{r},{c}]"),
                    );
                }
            }
        }
        let _ = fs::remove_dir_all(dir.parent().unwrap());
    }

    #[test]
    fn round_trip_velocity_read() {
        let dir = temp_dir("vel_rt_read");
        let data = Array2::from_shape_fn((ROWS, COLS), |(r, c)| synth(0, r, c));
        let written = VelocityMap {
            data: data.clone(),
            meta: StackMeta {
                transform: test_transform(),
                crs: Some(test_crs()),
                wavelength_m: crate::types::SENTINEL1_WAVELENGTH_M,
                incidence_deg: 39.0,
                heading_deg: Some(190.0),
            },
        };
        let path = dir.join("velocity.tif");
        write_velocity(&written, &path).unwrap();

        // meta pasada al reader: solo importan wavelength/incidence/heading
        // (transform/crs se sobrescriben desde el archivo).
        let caller_meta = StackMeta {
            transform: surtgis_core::GeoTransform::default(),
            crs: None,
            wavelength_m: crate::types::SENTINEL1_WAVELENGTH_M,
            incidence_deg: 39.0,
            heading_deg: Some(190.0),
        };
        let back = read_velocity(&path, caller_meta).unwrap();
        assert_eq!(back.data.dim(), (ROWS, COLS));
        assert_eq!(back.meta.transform, test_transform());
        assert_eq!(back.meta.crs, Some(test_crs()));
        assert_eq!(back.meta.wavelength_m, written.meta.wavelength_m);
        assert_eq!(back.meta.heading_deg, written.meta.heading_deg);
        for r in 0..ROWS {
            for c in 0..COLS {
                assert_f32_eq_nan(back.data[[r, c]], data[[r, c]], &format!("vel[{r},{c}]"));
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn round_trip_series_read() {
        let dir = temp_dir("ser_rt_read");
        let epochs = expected_epochs();
        let data = Array3::from_shape_fn((3, ROWS, COLS), |(k, r, c)| synth(k, r, c));
        let written = DisplacementSeries {
            data: data.clone(),
            epochs: epochs.clone(),
            meta: StackMeta {
                transform: test_transform(),
                crs: Some(test_crs()),
                wavelength_m: crate::types::SENTINEL1_WAVELENGTH_M,
                incidence_deg: 39.0,
                heading_deg: None,
            },
        };
        write_series(&written, &dir).unwrap();

        let caller_meta = StackMeta {
            transform: surtgis_core::GeoTransform::default(),
            crs: None,
            wavelength_m: crate::types::SENTINEL1_WAVELENGTH_M,
            incidence_deg: 39.0,
            heading_deg: None,
        };
        let back = read_series(&dir, caller_meta).unwrap();
        assert_eq!(back.epochs, epochs);
        assert_eq!(back.meta.transform, test_transform());
        assert_eq!(back.meta.crs, Some(test_crs()));
        for k in 0..3 {
            for r in 0..ROWS {
                for c in 0..COLS {
                    assert_f32_eq_nan(
                        back.data[[k, r, c]],
                        data[[k, r, c]],
                        &format!("ser[{k},{r},{c}]"),
                    );
                }
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_series_dir_vacio_es_error() {
        let dir = temp_dir("ser_rt_empty");
        fs::create_dir_all(&dir).unwrap();
        let err = read_series(
            &dir,
            StackMeta {
                transform: surtgis_core::GeoTransform::default(),
                crs: None,
                wavelength_m: crate::types::SENTINEL1_WAVELENGTH_M,
                incidence_deg: 39.0,
                heading_deg: None,
            },
        )
        .unwrap_err();
        assert!(matches!(err, InsarError::Metadata(_)), "got: {err:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_malformado_da_error_metadata() {
        let dir = temp_dir("bad_json");
        fs::write(dir.join("stack.json"), "{esto no es json válido").unwrap();
        let err = read_ifg_stack(&dir).unwrap_err();
        assert!(matches!(err, InsarError::Metadata(_)), "got: {err:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fecha_invalida_da_error_metadata() {
        let dir = temp_dir("bad_date");
        let manifest = MANIFEST.replace("2023-01-13", "13/01/2023");
        fs::write(dir.join("stack.json"), manifest).unwrap();
        let err = read_ifg_stack(&dir).unwrap_err();
        assert!(matches!(err, InsarError::Metadata(_)), "got: {err:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn coherencia_opcional_ausente_y_presente() {
        let dir = temp_dir("coh_rt");
        build_synthetic_stack(&dir);

        // El MANIFEST base no declara coherencia → Ok(None).
        assert!(read_coherence_stack(&dir).unwrap().is_none());

        // Manifiesto con coherencia alineada a los 2 ifgs.
        let manifest = MANIFEST.replace(
            "\"amplitudes\":",
            "\"coherence\": [\"coh_a.tif\", \"coh_b.tif\"],\n        \"amplitudes\":",
        );
        fs::write(dir.join("stack.json"), manifest).unwrap();
        for (name, val) in [("coh_a", 0.9_f32), ("coh_b", 0.6)] {
            write_test_tif(
                &dir.join(format!("{name}.tif")),
                vec![val; ROWS * COLS],
                ROWS,
                COLS,
            );
        }
        let coh = read_coherence_stack(&dir).unwrap().expect("coherencia presente");
        assert_eq!(coh.shape(), &[2, ROWS, COLS]);
        assert_eq!(coh[[0, 1, 1]], 0.9);
        assert_eq!(coh[[1, 2, 3]], 0.6);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn coherencia_desalineada_da_error_metadata() {
        let dir = temp_dir("coh_misaligned");
        build_synthetic_stack(&dir);
        // 2 ifgs pero 1 archivo de coherencia.
        let manifest = MANIFEST.replace(
            "\"amplitudes\":",
            "\"coherence\": [\"coh_a.tif\"],\n        \"amplitudes\":",
        );
        fs::write(dir.join("stack.json"), manifest).unwrap();
        let err = read_coherence_stack(&dir).unwrap_err();
        assert!(matches!(err, InsarError::Metadata(_)), "got: {err:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn epoca_duplicada_da_error_metadata() {
        let dir = temp_dir("dup_epoch");
        let manifest = MANIFEST.replace("2023-01-13", "2023-01-01");
        fs::write(dir.join("stack.json"), manifest).unwrap();
        let err = read_ifg_stack(&dir).unwrap_err();
        assert!(matches!(err, InsarError::Metadata(_)), "got: {err:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn epocas_desordenadas_da_error_metadata() {
        let dir = temp_dir("unordered_epoch");
        // La última época pasa a ser anterior a las demás → desorden.
        let manifest = MANIFEST.replace("2023-01-25", "2022-01-25");
        fs::write(dir.join("stack.json"), manifest).unwrap();
        let err = read_ifg_stack(&dir).unwrap_err();
        assert!(matches!(err, InsarError::Metadata(_)), "got: {err:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn falta_campo_ifgs_da_error_metadata() {
        let dir = temp_dir("no_ifgs");
        fs::write(
            dir.join("stack.json"),
            r#"{"wavelength_m": 0.055, "incidence_deg": 39.0,
                "epochs": ["2023-01-01"], "amplitudes": ["a.tif"]}"#,
        )
        .unwrap();
        let err = read_ifg_stack(&dir).unwrap_err();
        assert!(matches!(err, InsarError::Metadata(_)), "got: {err:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn amplitudes_desalineadas_da_error_metadata() {
        let dir = temp_dir("amp_misaligned");
        // 3 épocas pero 2 amplitudes
        let manifest = MANIFEST.replace(", \"amp_20230125.tif\"", "");
        fs::write(dir.join("stack.json"), manifest).unwrap();
        let err = read_amplitude_stack(&dir).unwrap_err();
        assert!(matches!(err, InsarError::Metadata(_)), "got: {err:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn dims_distintas_da_error_dimension_mismatch_ifg() {
        let dir = temp_dir("ifg_dims");
        build_synthetic_stack(&dir);
        // Sobrescribir la parte imaginaria del segundo ifg con otras dims
        write_test_tif(
            &dir.join("ifg_20230113_20230125_im.tif"),
            vec![0.0; 3 * 3],
            3,
            3,
        );
        let err = read_ifg_stack(&dir).unwrap_err();
        assert!(matches!(err, InsarError::DimensionMismatch(_)), "got: {err:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn dims_distintas_da_error_dimension_mismatch_amp() {
        let dir = temp_dir("amp_dims");
        build_synthetic_stack(&dir);
        write_test_tif(&dir.join("amp_20230113.tif"), vec![1.0; 2 * 7], 2, 7);
        let err = read_amplitude_stack(&dir).unwrap_err();
        assert!(matches!(err, InsarError::DimensionMismatch(_)), "got: {err:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn archivo_faltante_da_error_raster() {
        let dir = temp_dir("missing_tif");
        fs::write(dir.join("stack.json"), MANIFEST).unwrap();
        // No se escriben los .tif → debe fallar la lectura, no entrar en pánico
        assert!(read_ifg_stack(&dir).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn par_fuera_de_rango_falla_en_validate() {
        let dir = temp_dir("bad_pair");
        build_synthetic_stack(&dir);
        // secondary 9 fuera de rango para 3 épocas
        let manifest = MANIFEST.replace(r#""secondary": 2"#, r#""secondary": 9"#);
        fs::write(dir.join("stack.json"), manifest).unwrap();
        let err = read_ifg_stack(&dir).unwrap_err();
        assert!(matches!(err, InsarError::InvalidNetwork(_)), "got: {err:?}");
        let _ = fs::remove_dir_all(&dir);
    }
}

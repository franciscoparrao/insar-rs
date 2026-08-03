//! Corrección troposférica por mapas **GACOS** (P0.2 del roadmap de datos
//! abiertos).
//!
//! GACOS (Generic Atmospheric Correction Online Service, Newcastle University;
//! Yu et al. 2018a,b) sirve mapas de **retardo cenital total** (ZTD) por fecha
//! y AOI, generados con el modelo ITD (Iterative Tropospheric Decomposition)
//! sobre ECMWF HRES + topografía de alta resolución.
//!
//! ## Por qué este módulo SÍ es operable end-to-end (y [`super::era5`] no)
//!
//! [`super::era5`] implementa el kernel físico (Saastamoinen + integración del
//! húmedo) pero declara explícitamente que no descarga ni remuestrea: hay que
//! resolver el perfil atmosférico por columna fuera del motor. GACOS evita todo
//! eso porque **entrega el ZTD ya modelado y ya grillado** en una malla
//! geográfica regular. Aquí solo hace falta leer, remuestrear a la grilla del
//! stack, proyectar a LOS y ensamblar el diferencial — todo implementado en
//! este archivo, sin dependencias externas ni credenciales.
//!
//! Además, para AOIs pequeños GACOS es **preferible a ERA5 por resolución**:
//! ERA5 tiene celdas de ~31 km, de modo que una cuenca de ~10 km cae dentro de
//! una sola celda y el reanálisis no puede resolver ningún gradiente de retardo
//! topo-correlacionado *dentro* del AOI. GACOS downscalea con la topografía
//! SRTM y entrega ZTD a **~90 m** (0,000833°, su salida por defecto; Yu et
//! al. 2018), que resuelve de sobra ese gradiente. El paso real viene siempre
//! en el `.rsc` — el lector no asume ninguno.
//!
//! ## Formato de entrada
//!
//! Por cada fecha, GACOS entrega dos archivos hermanos:
//!
//! - `YYYYMMDD.ztd` — binario `f32` little-endian, `WIDTH × FILE_LENGTH`,
//!   row-major, north-up, ZTD en **metros**.
//! - `YYYYMMDD.ztd.rsc` — cabecera estilo ROI_PAC (`CLAVE  VALOR` por línea)
//!   con al menos `WIDTH`, `FILE_LENGTH`, `X_FIRST`, `Y_FIRST`, `X_STEP`,
//!   `Y_STEP`.
//!
//! `X_FIRST`/`Y_FIRST` son la **esquina** del primer píxel (convenio ROI_PAC),
//! que es justo lo que espera [`GeoTransform`] como origen.
//!
//! ## Signo
//!
//! Idéntico a [`super::era5::correct_era5_series`], y por la misma razón: un
//! retardo positivo alarga el camino óptico ⇒ mayor rango aparente ⇒ con la
//! convención `d = -λ/(4π)·φ` del motor se propaga como desplazamiento
//! NEGATIVO espurio, así que recuperar el desplazamiento real exige **sumar**
//! la diferencia de retardos, no restarla.
//!
//! ## Referenciado espacial (diferencia con `era5`)
//!
//! Una serie InSAR está referenciada **en tiempo** (a una época) y **en
//! espacio** (a un píxel). La corrección debe estarlo igual, o se introduce un
//! offset espurio uniforme. Por eso [`correct_gacos_series`] acepta
//! `reference_pixel` y aplica el **doble diferencial**
//! `(D[e,p] − D[e₀,p]) − (D[e,p_ref] − D[e₀,p_ref])`.
//! Pasar `None` reproduce el comportamiento de `era5` (solo diferencial
//! temporal), que es correcto únicamente si la serie no fue referenciada a un
//! píxel.

use std::fs;
use std::path::{Path, PathBuf};

use ndarray::{Array2, Array3, Axis};
use surtgis_core::GeoTransform;

use crate::error::{InsarError, Result};
use crate::types::{DisplacementSeries, Epoch};

/// Cabecera `.ztd.rsc` (subconjunto ROI_PAC que necesita el lector).
#[derive(Debug, Clone, PartialEq)]
pub struct RscHeader {
    pub width: usize,
    pub file_length: usize,
    pub x_first: f64,
    pub y_first: f64,
    pub x_step: f64,
    pub y_step: f64,
}

impl RscHeader {
    /// `GeoTransform` equivalente. `X_FIRST`/`Y_FIRST` son la esquina del
    /// primer píxel, igual que `origin_x`/`origin_y`.
    pub fn transform(&self) -> GeoTransform {
        GeoTransform::new(self.x_first, self.y_first, self.x_step, self.y_step)
    }
}

/// Parsea un `.ztd.rsc`. Tolera espacios/tabs múltiples y claves desconocidas.
pub fn read_rsc(path: &Path) -> Result<RscHeader> {
    let text = fs::read_to_string(path)
        .map_err(|e| InsarError::Raster(format!("no se pudo leer {}: {e}", path.display())))?;

    let mut width = None;
    let mut file_length = None;
    let mut x_first = None;
    let mut y_first = None;
    let mut x_step = None;
    let mut y_step = None;

    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(key), Some(val)) = (it.next(), it.next()) else {
            continue;
        };
        match key.to_ascii_uppercase().as_str() {
            "WIDTH" => width = val.parse::<usize>().ok(),
            "FILE_LENGTH" | "LENGTH" => file_length = val.parse::<usize>().ok(),
            "X_FIRST" => x_first = val.parse::<f64>().ok(),
            "Y_FIRST" => y_first = val.parse::<f64>().ok(),
            "X_STEP" => x_step = val.parse::<f64>().ok(),
            "Y_STEP" => y_step = val.parse::<f64>().ok(),
            _ => {}
        }
    }

    let missing = |n: &str| InsarError::Metadata(format!("{} sin {n}", path.display()));
    let h = RscHeader {
        width: width.ok_or_else(|| missing("WIDTH"))?,
        file_length: file_length.ok_or_else(|| missing("FILE_LENGTH"))?,
        x_first: x_first.ok_or_else(|| missing("X_FIRST"))?,
        y_first: y_first.ok_or_else(|| missing("Y_FIRST"))?,
        x_step: x_step.ok_or_else(|| missing("X_STEP"))?,
        y_step: y_step.ok_or_else(|| missing("Y_STEP"))?,
    };
    if h.width == 0 || h.file_length == 0 {
        return Err(InsarError::Metadata(format!(
            "{}: dimensiones nulas {}×{}",
            path.display(),
            h.file_length,
            h.width
        )));
    }
    if h.x_step == 0.0 || h.y_step == 0.0 {
        return Err(InsarError::Metadata(format!(
            "{}: paso nulo (X_STEP {}, Y_STEP {})",
            path.display(),
            h.x_step,
            h.y_step
        )));
    }
    Ok(h)
}

/// Mapa GACOS de retardo cenital total en metros, georreferenciado.
#[derive(Debug, Clone)]
pub struct GacosGrid {
    /// ZTD en metros, `file_length × width`.
    pub data: Array2<f32>,
    pub transform: GeoTransform,
}

impl GacosGrid {
    /// Lee un par `.ztd` + `.ztd.rsc`. `ztd_path` apunta al binario; la
    /// cabecera se busca como `<ztd_path>.rsc`.
    pub fn read(ztd_path: &Path) -> Result<Self> {
        let rsc_path = PathBuf::from(format!("{}.rsc", ztd_path.display()));
        let header = read_rsc(&rsc_path)?;
        Self::read_with_header(ztd_path, &header)
    }

    /// Igual que [`Self::read`] con una cabecera ya parseada.
    pub fn read_with_header(ztd_path: &Path, header: &RscHeader) -> Result<Self> {
        let bytes = fs::read(ztd_path)
            .map_err(|e| InsarError::Raster(format!("no se pudo leer {}: {e}", ztd_path.display())))?;
        let expected = header.width * header.file_length * 4;
        if bytes.len() != expected {
            return Err(InsarError::Raster(format!(
                "{}: {} bytes, se esperaban {} ({}×{} f32)",
                ztd_path.display(),
                bytes.len(),
                expected,
                header.file_length,
                header.width
            )));
        }
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let data = Array2::from_shape_vec((header.file_length, header.width), values)
            .map_err(|e| InsarError::Raster(format!("forma inválida: {e}")))?;
        Ok(Self {
            data,
            transform: header.transform(),
        })
    }

    /// Remuestrea bilinealmente el ZTD a una grilla destino. Los píxeles fuera
    /// de la cobertura GACOS, o con algún vecino no finito, quedan `NaN` — no
    /// se extrapola.
    pub fn resample_to(&self, target: &GeoTransform, rows: usize, cols: usize) -> Array2<f32> {
        let (h, w) = self.data.dim();
        let mut out = Array2::from_elem((rows, cols), f32::NAN);
        if w < 2 || h < 2 {
            return out;
        }
        // Los centros de píxel del borde caen exactamente en 0 y en w-1; el
        // redondeo de `geo_to_pixel` (que divide) los deja en ±1e-16 del
        // entero, así que un `floor` directo perdería la primera fila/columna.
        // Se admite esa tolerancia y luego se acota al dominio interpolable.
        const EPS: f64 = 1e-9;
        let (fmax_c, fmax_r) = ((w - 1) as f64, (h - 1) as f64);
        for r in 0..rows {
            for c in 0..cols {
                let (x, y) = target.pixel_to_geo(c, r);
                // `geo_to_pixel` mide desde la ESQUINA: el centro del píxel
                // (0,0) cae en (0.5, 0.5). Restar 0.5 lleva a coordenadas
                // medidas entre CENTROS, que es lo que interpola bilineal.
                let (fc, fr) = self.transform.geo_to_pixel(x, y);
                let (fc, fr) = (fc - 0.5, fr - 0.5);
                if !fc.is_finite() || !fr.is_finite() {
                    continue;
                }
                if fc < -EPS || fr < -EPS || fc > fmax_c + EPS || fr > fmax_r + EPS {
                    continue; // fuera de cobertura: no se extrapola
                }
                let (fc, fr) = (fc.clamp(0.0, fmax_c), fr.clamp(0.0, fmax_r));
                let c0 = (fc.floor() as usize).min(w - 2);
                let r0 = (fr.floor() as usize).min(h - 2);
                let (tx, ty) = ((fc - c0 as f64) as f32, (fr - r0 as f64) as f32);
                let (v00, v01) = (self.data[[r0, c0]], self.data[[r0, c0 + 1]]);
                let (v10, v11) = (self.data[[r0 + 1, c0]], self.data[[r0 + 1, c0 + 1]]);
                if !(v00.is_finite() && v01.is_finite() && v10.is_finite() && v11.is_finite()) {
                    continue;
                }
                let top = v00 * (1.0 - tx) + v01 * tx;
                let bot = v10 * (1.0 - tx) + v11 * tx;
                out[[r, c]] = top * (1.0 - ty) + bot * ty;
            }
        }
        out
    }
}

/// Resultado de la corrección, para QC.
#[derive(Debug, Clone)]
pub struct GacosReport {
    /// Épocas con mapa GACOS encontrado y leído.
    pub epochs_found: usize,
    /// Fechas sin mapa (formato `YYYYMMDD`); esas épocas quedan sin corregir.
    pub epochs_missing: Vec<String>,
    /// Fechas CON mapa cuyo retardo no es finito en el píxel de referencia
    /// (p. ej. cae fuera de la cobertura de ese mapa): el doble diferencial no
    /// está definido y la época queda sin corregir. Siempre vacío si no se
    /// pasó `reference_pixel`.
    pub epochs_skipped: Vec<String>,
    /// Fracción de píxeles con corrección finita sobre las épocas
    /// **no-referencia** (la referencia se excluye: su corrección es 0 por
    /// definición y contarla solo diluiría la métrica).
    pub coverage: f64,
    /// Magnitud media de la corrección aplicada (épocas no-referencia), en
    /// metros.
    pub mean_abs_correction_m: f64,
}

/// Busca en `dir` el mapa GACOS de cada época (`YYYYMMDD.ztd`) y lo carga.
/// Devuelve un vector alineado con `epochs`: `None` donde no hay archivo.
///
/// Solo el par nativo `.ztd`/`.ztd.rsc` está soportado; si el servicio entregó
/// GeoTIFF, leerlo con SurtGIS y construir [`GacosGrid`] a mano.
///
/// ⚠️ **Memoria**: materializa TODOS los mapas a la vez — a 90 m, un frame
/// continental son cientos de MB por época. Para corregir una serie usar
/// [`correct_gacos_series`], que hace streaming (un mapa en memoria a la vez);
/// esta función queda como building block para composiciones que de verdad
/// necesiten todos los grids simultáneamente.
pub fn load_for_epochs(dir: &Path, epochs: &[Epoch]) -> Result<Vec<Option<GacosGrid>>> {
    if !dir.is_dir() {
        return Err(InsarError::Raster(format!(
            "{} no es un directorio",
            dir.display()
        )));
    }
    let mut out = Vec::with_capacity(epochs.len());
    for e in epochs {
        let name = e.0.format("%Y%m%d").to_string();
        let p = dir.join(format!("{name}.ztd"));
        if p.is_file() {
            out.push(Some(GacosGrid::read(&p)?));
        } else {
            out.push(None);
        }
    }
    Ok(out)
}

/// Ensambla el cubo de retardo **en LOS** (épocas × filas × columnas) a partir
/// de los mapas GACOS, remuestreados a `transform`. Las épocas sin mapa quedan
/// enteras en `NaN`.
pub fn build_los_delay(
    grids: &[Option<GacosGrid>],
    transform: &GeoTransform,
    rows: usize,
    cols: usize,
    incidence_deg: f64,
) -> Result<Array3<f32>> {
    if !(0.0..90.0).contains(&incidence_deg) {
        return Err(InsarError::Metadata(format!(
            "incidencia {incidence_deg}° fuera de [0, 90)"
        )));
    }
    let sec = 1.0 / incidence_deg.to_radians().cos();
    let mut cube = Array3::from_elem((grids.len(), rows, cols), f32::NAN);
    for (e, g) in grids.iter().enumerate() {
        let Some(g) = g else { continue };
        let mut layer = g.resample_to(transform, rows, cols);
        layer.mapv_inplace(|v| (v as f64 * sec) as f32);
        cube.index_axis_mut(Axis(0), e).assign(&layer);
    }
    Ok(cube)
}

/// Corrige una serie de desplazamiento con mapas GACOS de `dir`.
///
/// - `reference_epoch`: época a la que está referenciada la serie.
/// - `reference_pixel`: `(fila, columna)` del punto de referencia espacial;
///   `None` omite el referenciado espacial (ver nota del módulo).
///
/// Las épocas sin mapa GACOS —y las que no tengan retardo finito en la época
/// o en el píxel de referencia— quedan **sin corregir** en vez de contaminarse
/// con `NaN`, y se listan en [`GacosReport`].
///
/// ## Errores (además de los de dimensión/lectura)
///
/// Los estados que convertirían la corrección en un no-op silencioso son
/// **errores duros**, no reportes:
///
/// - la **época de referencia** no tiene mapa GACOS — sin su retardo el
///   diferencial temporal no está definido para ninguna época;
/// - el **píxel de referencia** no tiene retardo finito en la época de
///   referencia (típicamente: cae fuera de la cobertura del mapa) — sin él el
///   doble diferencial no está definido para ninguna época;
/// - al terminar, **ningún píxel** recibió corrección.
pub fn correct_gacos_series(
    series: &mut DisplacementSeries,
    dir: &Path,
    reference_epoch: usize,
    reference_pixel: Option<(usize, usize)>,
) -> Result<GacosReport> {
    let n_epochs = series.n_layers();
    let (rows, cols) = series.dims();
    if reference_epoch >= n_epochs {
        return Err(InsarError::Metadata(format!(
            "reference_epoch {reference_epoch} fuera de rango (0..{n_epochs})"
        )));
    }
    if let Some((r, c)) = reference_pixel
        && (r >= rows || c >= cols)
    {
        return Err(InsarError::DimensionMismatch(format!(
            "reference_pixel ({r},{c}) fuera de la grilla ({rows},{cols})"
        )));
    }

    if !dir.is_dir() {
        return Err(InsarError::Raster(format!(
            "{} no es un directorio",
            dir.display()
        )));
    }
    let incidence_deg = series.meta.incidence_deg;
    if !(0.0..90.0).contains(&incidence_deg) {
        return Err(InsarError::Metadata(format!(
            "incidencia {incidence_deg}° fuera de [0, 90)"
        )));
    }
    let sec = 1.0 / incidence_deg.to_radians().cos();
    // `GeoTransform` es Copy: se saca del meta para que el closure de carga no
    // retenga un préstamo de `series` mientras el bucle lo muta.
    let transform = series.meta.transform;

    let epochs = series.epochs.clone();
    // Solo las RUTAS por época; los mapas se cargan de a uno dentro del bucle
    // (streaming) — a 90 m un frame continental son cientos de MB por mapa, y
    // materializar los N a la vez (como hace [`load_for_epochs`]) escala a GBs.
    let paths: Vec<Option<PathBuf>> = epochs
        .iter()
        .map(|e| {
            let p = dir.join(format!("{}.ztd", e.0.format("%Y%m%d")));
            p.is_file().then_some(p)
        })
        .collect();
    let epochs_missing: Vec<String> = paths
        .iter()
        .zip(&epochs)
        .filter(|(p, _)| p.is_none())
        .map(|(_, e)| e.0.format("%Y%m%d").to_string())
        .collect();
    let epochs_found = paths.len() - epochs_missing.len();
    if epochs_found == 0 {
        return Err(InsarError::Raster(format!(
            "ningún mapa GACOS (YYYYMMDD.ztd) en {} para las {} épocas de la serie",
            dir.display(),
            epochs.len()
        )));
    }
    // Sin el mapa de la época de referencia, `base` sería todo NaN y NINGUNA
    // época se corregiría — pero el comando "terminaría bien". Error duro.
    let Some(reference_path) = paths[reference_epoch].clone() else {
        return Err(InsarError::Metadata(format!(
            "la época de referencia {} no tiene mapa GACOS en {} — sin su retardo \
             el diferencial temporal no está definido y ninguna época puede corregirse",
            epochs[reference_epoch].0.format("%Y%m%d"),
            dir.display()
        )));
    };

    // Carga → remuestreo → proyección a LOS de UN mapa; el grid crudo se
    // libera al salir, solo persiste la capa en la grilla de la serie.
    let load_los_layer = |p: &Path| -> Result<Array2<f32>> {
        let grid = GacosGrid::read(p)?;
        let mut layer = grid.resample_to(&transform, rows, cols);
        layer.mapv_inplace(|v| (v as f64 * sec) as f32);
        Ok(layer)
    };

    let base = load_los_layer(&reference_path)?;
    let base_ref = reference_pixel.map(|(r, c)| base[[r, c]]);
    // Mismo razonamiento: sin retardo finito en el píxel de referencia de la
    // época de referencia, el doble diferencial no existe para ninguna época.
    if let (Some((r, c)), Some(b)) = (reference_pixel, base_ref)
        && !b.is_finite()
    {
        return Err(InsarError::Metadata(format!(
            "el píxel de referencia ({r},{c}) no tiene retardo GACOS finito en la \
             época de referencia {} — probablemente cae fuera de la cobertura del \
             mapa; elegir otro píxel o pedir un AOI GACOS más amplio",
            epochs[reference_epoch].0.format("%Y%m%d")
        )));
    }

    let (mut applied, mut sum_abs) = (0u64, 0f64);
    let mut epochs_skipped: Vec<String> = Vec::new();
    for e in 0..n_epochs {
        // La corrección de la época de referencia es idénticamente 0 (de ≡ dr):
        // se salta sin cargar su mapa de nuevo, y las métricas del reporte
        // quedan definidas sobre las épocas no-referencia.
        if e == reference_epoch {
            continue;
        }
        let Some(path) = &paths[e] else {
            continue; // sin mapa: ya está en `epochs_missing`
        };
        let delay_e = load_los_layer(path)?;
        // Referenciado espacial: la corrección debe anularse en el píxel de
        // referencia, igual que la propia serie.
        let pixel_offset = match (reference_pixel, base_ref) {
            (Some((r, c)), Some(b)) => {
                let d = delay_e[[r, c]];
                if d.is_finite() && b.is_finite() {
                    Some(d - b)
                } else {
                    // Sin referencia válida en esta época no se puede aplicar
                    // el doble diferencial sin sesgar: se salta la época y se
                    // REPORTA (las sin mapa ya están en `epochs_missing`).
                    epochs_skipped.push(epochs[e].0.format("%Y%m%d").to_string());
                    continue;
                }
            }
            _ => None,
        };
        let mut layer = series.data.index_axis(Axis(0), e).to_owned();
        ndarray::Zip::from(&mut layer)
            .and(&delay_e)
            .and(&base)
            .for_each(|d, &de, &dr| {
                if de.is_finite() && dr.is_finite() && d.is_finite() {
                    let corr = (de - dr) - pixel_offset.unwrap_or(0.0);
                    *d += corr;
                    applied += 1;
                    sum_abs += corr.abs() as f64;
                }
            });
        series.data.index_axis_mut(Axis(0), e).assign(&layer);
    }

    // Belt-and-braces: si por cualquier combinación no contemplada arriba no
    // se corrigió ni un píxel, el resultado sería la entrada disfrazada de
    // salida corregida. Eso es un error, no un reporte.
    if applied == 0 {
        return Err(InsarError::Metadata(format!(
            "0 píxeles corregidos: {} de {} épocas con mapa, {} saltadas por \
             píxel de referencia sin retardo finito — la serie de salida sería \
             idéntica a la de entrada",
            epochs_found,
            n_epochs,
            epochs_skipped.len()
        )));
    }

    // Denominador sin la época de referencia: su corrección es 0 por
    // definición, contarla solo diluiría cobertura y |corrección| media.
    let total = ((n_epochs - 1) * rows * cols) as f64;
    Ok(GacosReport {
        epochs_found,
        epochs_missing,
        epochs_skipped,
        coverage: if total > 0.0 { applied as f64 / total } else { 0.0 },
        mean_abs_correction_m: sum_abs / applied as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::StackMeta;
    use chrono::NaiveDate;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("insar_gacos_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// Escribe un par .ztd/.ztd.rsc sintético con ZTD = f(fila, columna).
    fn write_ztd(dir: &Path, date: &str, w: usize, h: usize, f: impl Fn(usize, usize) -> f32) {
        let rsc = format!(
            "WIDTH  {w}\nFILE_LENGTH  {h}\nX_FIRST  -71.5\nY_FIRST  -34.0\n\
             X_STEP  0.0125\nY_STEP  -0.0125\nX_UNIT  degrees\nY_UNIT  degrees\n"
        );
        fs::write(dir.join(format!("{date}.ztd.rsc")), rsc).unwrap();
        let mut bytes = Vec::with_capacity(w * h * 4);
        for r in 0..h {
            for c in 0..w {
                bytes.extend_from_slice(&f(r, c).to_le_bytes());
            }
        }
        fs::write(dir.join(format!("{date}.ztd")), bytes).unwrap();
    }

    fn serie(n_epochs: usize, rows: usize, cols: usize, dates: &[&str]) -> DisplacementSeries {
        DisplacementSeries {
            data: Array3::zeros((n_epochs, rows, cols)),
            epochs: dates
                .iter()
                .map(|d| Epoch(NaiveDate::parse_from_str(d, "%Y%m%d").unwrap()))
                .collect(),
            meta: StackMeta {
                transform: GeoTransform::new(-71.5, -34.0, 0.0125, -0.0125),
                crs: None,
                wavelength_m: crate::types::SENTINEL1_WAVELENGTH_M,
                // incidencia 0 => sec = 1, aísla la física del remuestreo
                incidence_deg: 0.0,
                heading_deg: None,
            },
        }
    }

    #[test]
    fn rsc_se_parsea_y_da_geotransform_coherente() {
        let d = tmpdir("rsc");
        write_ztd(&d, "20240101", 4, 3, |_, _| 2.4);
        let h = read_rsc(&d.join("20240101.ztd.rsc")).unwrap();
        assert_eq!(h.width, 4);
        assert_eq!(h.file_length, 3);
        let t = h.transform();
        assert!((t.origin_x - (-71.5)).abs() < 1e-12);
        assert!((t.pixel_height - (-0.0125)).abs() < 1e-12);
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn ztd_con_tamano_inconsistente_falla() {
        let d = tmpdir("badsize");
        write_ztd(&d, "20240101", 4, 3, |_, _| 2.4);
        // Truncar el binario deja 11 valores donde la cabecera declara 12.
        let p = d.join("20240101.ztd");
        let mut b = fs::read(&p).unwrap();
        b.truncate(b.len() - 4);
        fs::write(&p, b).unwrap();
        assert!(GacosGrid::read(&p).is_err());
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn remuestreo_identidad_preserva_valores() {
        // Grilla destino idéntica a la GACOS: el bilineal debe devolver el
        // mismo campo (salvo el borde, que no tiene 4 vecinos).
        let d = tmpdir("ident");
        write_ztd(&d, "20240101", 8, 6, |r, c| (r * 10 + c) as f32);
        let g = GacosGrid::read(&d.join("20240101.ztd")).unwrap();
        let out = g.resample_to(&g.transform.clone(), 6, 8);
        // Incluye bordes: con la tolerancia de EPS no debe perderse ni la
        // primera ni la última fila/columna.
        for r in 0..6 {
            for c in 0..8 {
                let want = (r * 10 + c) as f32;
                assert!(
                    (out[[r, c]] - want).abs() < 1e-3,
                    "({r},{c}) {} vs {want}",
                    out[[r, c]]
                );
            }
        }
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn remuestreo_grueso_a_fino_es_exacto_en_campo_lineal() {
        // Camino de producción: mapa GACOS (paso grueso) → grilla InSAR más
        // fina. Bilineal sobre un campo lineal debe ser exacto (a redondeo
        // f32) en todo el interior de la cobertura.
        let d = tmpdir("crossres");
        // ZTD lineal en longitud, evaluado en el CENTRO de cada celda gruesa:
        // ztd(lon) = 2.4 + 8.0·(lon + 71.5), con X_FIRST = -71.5 y paso 0.0125.
        write_ztd(&d, "20240101", 20, 20, |_, c| {
            2.4 + 8.0 * ((c as f32 + 0.5) * 0.0125)
        });
        let g = GacosGrid::read(&d.join("20240101.ztd")).unwrap();
        // Grilla fina 5× (paso 0.0025) contenida en el interior de la gruesa.
        let fine = GeoTransform::new(-71.45, -34.05, 0.0025, -0.0025);
        let out = g.resample_to(&fine, 40, 40);
        let mut checked = 0;
        for r in 0..40 {
            for c in 0..40 {
                let v = out[[r, c]];
                assert!(v.is_finite(), "({r},{c}) fuera de cobertura y no debería");
                let lon = -71.45 + (c as f64 + 0.5) * 0.0025;
                let want = 2.4 + 8.0 * (lon + 71.5);
                assert!(
                    (v as f64 - want).abs() < 1e-4,
                    "({r},{c}) {v} vs {want}"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 1600);
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn fuera_de_cobertura_queda_nan_sin_extrapolar() {
        let d = tmpdir("oob");
        write_ztd(&d, "20240101", 4, 4, |_, _| 2.4);
        let g = GacosGrid::read(&d.join("20240101.ztd")).unwrap();
        // Grilla destino desplazada 10° al este: sin solape.
        let lejos = GeoTransform::new(-61.5, -34.0, 0.0125, -0.0125);
        let out = g.resample_to(&lejos, 4, 4);
        assert!(out.iter().all(|v| v.is_nan()), "no debe extrapolar");
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn correccion_es_diferencial_y_nula_en_la_epoca_de_referencia() {
        let d = tmpdir("diff");
        // ZTD uniforme distinto por época: 2.40 y 2.45 m (+50 mm).
        write_ztd(&d, "20240101", 8, 8, |_, _| 2.40);
        write_ztd(&d, "20240113", 8, 8, |_, _| 2.45);
        let mut s = serie(2, 6, 6, &["20240101", "20240113"]);
        let rep = correct_gacos_series(&mut s, &d, 0, None).unwrap();
        assert_eq!(rep.epochs_found, 2);
        assert!(rep.epochs_missing.is_empty());
        // Época de referencia: corrección nula.
        assert!(s.data.index_axis(Axis(0), 0).iter().all(|&v| v.abs() < 1e-6));
        // Segunda época: +50 mm (signo positivo, ver nota de módulo).
        let v = s.data[[1, 3, 3]];
        assert!((v - 0.05).abs() < 1e-4, "corrección {v}, se esperaba +0.05 m");
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn referenciado_espacial_anula_la_correccion_en_el_pixel_de_referencia() {
        let d = tmpdir("refpx");
        // Rampa en columna: el retardo varía en el espacio dentro de la época.
        write_ztd(&d, "20240101", 8, 8, |_, _| 2.40);
        write_ztd(&d, "20240113", 8, 8, |_, c| 2.40 + 0.01 * c as f32);
        let mut s = serie(2, 6, 6, &["20240101", "20240113"]);
        let rep = correct_gacos_series(&mut s, &d, 0, Some((2, 2))).unwrap();
        assert_eq!(rep.epochs_found, 2);
        // En el píxel de referencia la corrección debe ser exactamente 0.
        assert!(
            s.data[[1, 2, 2]].abs() < 1e-6,
            "en la referencia quedó {}",
            s.data[[1, 2, 2]]
        );
        // Y en otra columna debe quedar solo la DIFERENCIA respecto de ella.
        let v = s.data[[1, 2, 4]];
        assert!((v - 0.02).abs() < 1e-4, "corrección {v}, se esperaba +0.02 m");
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn epoca_sin_mapa_se_reporta_y_queda_sin_corregir() {
        let d = tmpdir("missing");
        write_ztd(&d, "20240101", 8, 8, |_, _| 2.40);
        // 20240113 no existe; 20240125 sí.
        write_ztd(&d, "20240125", 8, 8, |_, _| 2.45);
        let mut s = serie(3, 6, 6, &["20240101", "20240113", "20240125"]);
        let rep = correct_gacos_series(&mut s, &d, 0, None).unwrap();
        assert_eq!(rep.epochs_found, 2);
        assert_eq!(rep.epochs_missing, vec!["20240113".to_string()]);
        // La faltante queda intacta; la sana se corrige (+50 mm).
        assert!(s.data.index_axis(Axis(0), 1).iter().all(|&v| v == 0.0));
        assert!((s.data[[2, 3, 3]] - 0.05).abs() < 1e-4);
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn unica_epoca_no_referencia_sin_mapa_es_error_no_noop() {
        // Si la ÚNICA época corregible no tiene mapa, el resultado sería un
        // passthrough: bajo el hardening eso es error, no un reporte.
        let d = tmpdir("onlyref");
        write_ztd(&d, "20240101", 8, 8, |_, _| 2.40);
        let mut s = serie(2, 6, 6, &["20240101", "20240113"]);
        let err = correct_gacos_series(&mut s, &d, 0, None).unwrap_err();
        assert!(err.to_string().contains("0 píxeles"), "{err}");
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn directorio_sin_mapas_es_error_explicito() {
        let d = tmpdir("empty");
        let mut s = serie(2, 4, 4, &["20240101", "20240113"]);
        assert!(correct_gacos_series(&mut s, &d, 0, None).is_err());
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn referencia_sin_mapa_es_error_no_noop() {
        // El caso que motivó el hardening: SOLO falta el mapa de la época de
        // referencia. Antes: exit ok, 0 % corregido, salida = entrada.
        let d = tmpdir("refmiss");
        write_ztd(&d, "20240113", 8, 8, |_, _| 2.45); // la referencia 20240101 NO existe
        let mut s = serie(2, 6, 6, &["20240101", "20240113"]);
        let err = correct_gacos_series(&mut s, &d, 0, None).unwrap_err();
        assert!(
            err.to_string().contains("referencia"),
            "el error debe nombrar la época de referencia: {err}"
        );
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn refpixel_fuera_de_cobertura_es_error() {
        // Mapas GACOS que cubren solo la mitad oeste de la serie; el píxel de
        // referencia cae en la mitad este (sin cobertura). Antes: todas las
        // épocas se saltaban en silencio y el comando "terminaba bien".
        let d = tmpdir("refoob");
        // WIDTH 4 con X_STEP 0.0125 cubre lon [-71.5, -71.45); la serie de 8
        // columnas llega hasta -71.4: la columna 6 queda fuera.
        for f in ["20240101", "20240113"] {
            let rsc = "WIDTH  4\nFILE_LENGTH  8\nX_FIRST  -71.5\nY_FIRST  -34.0\n\
                       X_STEP  0.0125\nY_STEP  -0.0125\n";
            fs::write(d.join(format!("{f}.ztd.rsc")), rsc).unwrap();
            let bytes: Vec<u8> = (0..4 * 8).flat_map(|_| 2.4f32.to_le_bytes()).collect();
            fs::write(d.join(format!("{f}.ztd")), bytes).unwrap();
        }
        let mut s = serie(2, 8, 8, &["20240101", "20240113"]);
        let err = correct_gacos_series(&mut s, &d, 0, Some((3, 6))).unwrap_err();
        assert!(
            err.to_string().contains("píxel de referencia"),
            "el error debe nombrar el píxel de referencia: {err}"
        );
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn epoca_con_refpixel_no_finito_se_salta_y_se_reporta() {
        // Referencia sana; una época intermedia tiene su mapa entero en NaN
        // (p. ej. producto corrupto): debe saltarse Y quedar listada, y las
        // demás épocas corregirse igual.
        let d = tmpdir("skiprep");
        // El campo de la época sana VARÍA en columna: con referenciado espacial
        // un retardo uniforme se cancela exactamente (corrección 0 en todas
        // partes, física correcta), así que uniforme no serviría para verificar
        // que la época sana sí se corrige.
        write_ztd(&d, "20240101", 8, 8, |_, _| 2.40);
        write_ztd(&d, "20240113", 8, 8, |_, _| f32::NAN);
        write_ztd(&d, "20240125", 8, 8, |_, c| 2.40 + 0.01 * c as f32);
        let mut s = serie(3, 6, 6, &["20240101", "20240113", "20240125"]);
        let rep = correct_gacos_series(&mut s, &d, 0, Some((2, 2))).unwrap();
        assert_eq!(rep.epochs_skipped, vec!["20240113".to_string()]);
        assert!(rep.epochs_missing.is_empty());
        // La época saltada queda intacta; la sana, corregida con el doble
        // diferencial: (0.01·col) − (0.01·2) = +0.02 m en la columna 4.
        assert!(s.data.index_axis(Axis(0), 1).iter().all(|&v| v == 0.0));
        assert!((s.data[[2, 4, 4]] - 0.02).abs() < 1e-4, "época sana: {}", s.data[[2, 4, 4]]);
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn serie_toda_nan_es_error_cero_aplicados() {
        // Con mapas y referencia válidos pero una serie sin ningún dato finito,
        // el resultado sería la entrada disfrazada de corregida: error.
        let d = tmpdir("allnan");
        write_ztd(&d, "20240101", 8, 8, |_, _| 2.40);
        write_ztd(&d, "20240113", 8, 8, |_, _| 2.45);
        let mut s = serie(2, 6, 6, &["20240101", "20240113"]);
        s.data.fill(f32::NAN);
        let err = correct_gacos_series(&mut s, &d, 0, None).unwrap_err();
        assert!(
            err.to_string().contains("0 píxeles"),
            "el error debe decir que no se corrigió nada: {err}"
        );
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn incidencia_proyecta_a_los_por_secante() {
        let d = tmpdir("los");
        write_ztd(&d, "20240101", 8, 8, |_, _| 2.40);
        write_ztd(&d, "20240113", 8, 8, |_, _| 2.45);
        let mut s = serie(2, 6, 6, &["20240101", "20240113"]);
        s.meta.incidence_deg = 39.0;
        correct_gacos_series(&mut s, &d, 0, None).unwrap();
        let esperado = 0.05 / 39.0_f64.to_radians().cos();
        let v = s.data[[1, 3, 3]] as f64;
        assert!(
            (v - esperado).abs() < 1e-4,
            "LOS {v}, se esperaba {esperado}"
        );
        fs::remove_dir_all(&d).ok();
    }
}

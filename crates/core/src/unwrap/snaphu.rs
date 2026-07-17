//! Backend opcional de desenrollado 2D vía SNAPHU (Chen & Zebker), shell-out
//! a un binario `snaphu` instalado por separado — no vendorizado ni FFI (ver
//! `docs/auditoria-2026-07-02.md`, gap G-3, para la justificación).
//!
//! Requiere el binario `snaphu` en PATH (o [`SnaphuConfig::binary`]
//! apuntando a él): `conda install -c conda-forge snaphu` o
//! `pip install snaphu` lo instalan, pero el ejecutable queda **dentro**
//! del paquete Python (`site-packages/snaphu/snaphu`), no necesariamente en
//! PATH — puede requerir apuntar `binary` explícitamente a esa ruta.
//!
//! Usa `STATCOSTMODE SMOOTH` (mismo modo estadístico que
//! `validation/isce_unwrap.py` ya usa y valida en este proyecto contra
//! datos reales) y formato `FLOAT_DATA` (fase/coherencia como float32 crudo,
//! sin banda de amplitud) — coincide con el `Array2<f32>` que ya maneja
//! [`super::unwrap_2d`].

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use ndarray::{Array2, Array3, Axis};
use rayon::prelude::*;

use super::wrapped_phase_layer;
use crate::error::{InsarError, IoResultExt, Result};
use crate::types::{IfgStack, UnwrappedStack};

/// Configuración del backend SNAPHU.
#[derive(Debug, Clone)]
pub struct SnaphuConfig {
    /// Ruta al binario `snaphu`; default resuelto por `PATH` ("snaphu").
    pub binary: PathBuf,
}

impl Default for SnaphuConfig {
    fn default() -> Self {
        Self { binary: PathBuf::from("snaphu") }
    }
}

/// Nombres de archivo (relativos al directorio temporal de la invocación,
/// ver [`snaphu_config_text`]) para el desenrollado y la coherencia.
const OUTFILE_NAME: &str = "unwrapped.raw";
const CORRFILE_NAME: &str = "corr.raw";

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Directorio temporal único por invocación (PID + contador + `tag`, sin
/// depender de la crate `tempfile` — mismo patrón manual que usan los tests
/// del crate, ver `io/mod.rs`). Necesario porque `unwrap_stack_snaphu`
/// lanza una invocación de snaphu por capa en paralelo (rayon).
fn unique_tmp_dir(tag: &str) -> PathBuf {
    let n = TMP_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
    std::env::temp_dir().join(format!("insar_snaphu_{}_{tag}_{n}", std::process::id()))
}

/// Escribe `data` como raw float32 nativo de la plataforma (formato SNAPHU
/// `FLOAT_DATA`: un canal, sin cabecera).
fn write_float_raw(path: &Path, data: &Array2<f32>) -> Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data.iter() {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    std::fs::write(path, bytes).with_path(path)?;
    Ok(())
}

/// Lee raw float32 nativo (formato SNAPHU `FLOAT_DATA`) con dimensiones
/// conocidas de antemano (snaphu no escribe cabecera).
fn read_float_raw(path: &Path, rows: usize, cols: usize) -> Result<Array2<f32>> {
    let bytes = std::fs::read(path).with_path(path)?;
    let expected = rows * cols * 4;
    if bytes.len() != expected {
        return Err(InsarError::DimensionMismatch(format!(
            "{}: {} bytes, se esperaban {expected} ({rows}x{cols} float32)",
            path.display(),
            bytes.len()
        )));
    }
    let data: Vec<f32> =
        bytes.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect();
    Array2::from_shape_vec((rows, cols), data).map_err(|e| InsarError::DimensionMismatch(e.to_string()))
}

/// Texto del archivo de config de snaphu. Usa nombres de archivo
/// **relativos** ([`OUTFILE_NAME`]/[`CORRFILE_NAME`]) — nunca rutas
/// absolutas — porque el parser de config de snaphu tokeniza por whitespace:
/// una ruta absoluta con espacios (p. ej. `TMPDIR` con espacios en el nombre
/// de usuario, o macOS con `/var/folders/...` con un componente inusual) se
/// trunca en el primer espacio y snaphu escribe en una ruta inesperada o
/// falla con un error críptico. Los nombres relativos funcionan porque
/// [`run_snaphu`] invoca el proceso con `current_dir(dir)`. INFILE/LINELENGTH
/// van como argumentos posicionales del CLI, no en este config — ahí sí es
/// seguro pasar rutas absolutas porque `Command::arg` no tokeniza por
/// whitespace (execve recibe cada argumento intacto, sin shell de por medio).
fn snaphu_config_text(has_corr: bool) -> String {
    let mut conf = format!(
        "INFILEFORMAT FLOAT_DATA\nOUTFILE {OUTFILE_NAME}\nOUTFILEFORMAT FLOAT_DATA\nSTATCOSTMODE SMOOTH\n",
    );
    if has_corr {
        conf.push_str(&format!("CORRFILE {CORRFILE_NAME}\nCORRFILEFORMAT FLOAT_DATA\n"));
    }
    conf
}

/// Desenrolla `wrapped` (radianes) invocando `snaphu` como subproceso.
///
/// Mismo contrato NoData que [`super::unwrap_2d`]: los NaN de entrada se
/// reemplazan por fase 0.0 antes de invocar snaphu (aborta con no-finitos)
/// y, si hay `quality`, por coherencia 0.0 en esas posiciones (señal de "no
/// confiable" a snaphu); los NaN se restauran en la salida en las mismas
/// posiciones. Sin `quality`, snaphu estima su propia coherencia desde los
/// datos (comportamiento documentado del binario) — los NaN de entrada
/// igual se restauran en la salida, pero no hay forma de señalar "no
/// confiable" sin mapa de calidad.
///
/// Errores: [`InsarError::Io`] si el binario no se encuentra o falla el
/// I/O; [`InsarError::Inversion`] si snaphu termina con código de error
/// (mensaje incluye stderr); [`InsarError::DimensionMismatch`] si `quality`
/// no coincide en tamaño con `wrapped`.
pub fn unwrap_2d_snaphu(
    wrapped: &Array2<f32>,
    quality: Option<&Array2<f32>>,
    config: &SnaphuConfig,
) -> Result<Array2<f32>> {
    let (rows, cols) = wrapped.dim();
    if let Some(q) = quality
        && q.dim() != (rows, cols)
    {
        return Err(InsarError::DimensionMismatch(format!(
            "quality {:?} vs wrapped {:?}",
            q.dim(),
            (rows, cols)
        )));
    }

    let dir = unique_tmp_dir("2d");
    std::fs::create_dir_all(&dir).with_path(&dir)?;
    let result = run_snaphu(&dir, wrapped, quality, rows, cols, config);
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// Núcleo de la invocación: escribe input/config, corre snaphu, lee output.
/// `dir` ya debe existir; el caller es responsable de limpiarlo.
fn run_snaphu(
    dir: &Path,
    wrapped: &Array2<f32>,
    quality: Option<&Array2<f32>>,
    rows: usize,
    cols: usize,
    config: &SnaphuConfig,
) -> Result<Array2<f32>> {
    let nan_mask = wrapped.mapv(|v| !v.is_finite());
    let infile_data = wrapped.mapv(|v| if v.is_finite() { v } else { 0.0 });
    let infile = dir.join("wrapped.raw");
    write_float_raw(&infile, &infile_data)?;

    let corr_path = dir.join(CORRFILE_NAME);
    let has_corr = if let Some(q) = quality {
        let masked =
            Array2::from_shape_fn((rows, cols), |(r, c)| if nan_mask[[r, c]] { 0.0 } else { q[[r, c]] });
        write_float_raw(&corr_path, &masked)?;
        true
    } else {
        false
    };

    let outfile = dir.join(OUTFILE_NAME);
    let conf_path = dir.join("snaphu.conf");
    std::fs::write(&conf_path, snaphu_config_text(has_corr)).with_path(&conf_path)?;

    let output = Command::new(&config.binary)
        .arg("-f")
        .arg(&conf_path)
        .arg(&infile)
        .arg(cols.to_string())
        .current_dir(dir)
        .output()
        .with_path(&config.binary)?;
    if !output.status.success() {
        return Err(InsarError::Inversion(format!(
            "snaphu exited con {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    let mut unwrapped = read_float_raw(&outfile, rows, cols)?;
    for ((r, c), &is_nan) in nan_mask.indexed_iter() {
        if is_nan {
            unwrapped[[r, c]] = f32::NAN;
        }
    }
    Ok(unwrapped)
}

/// Como [`unwrap_2d_snaphu`] para cada interferograma del stack — paralelo
/// por capa (mismo patrón que [`super::unwrap_stack`]), cada tarea rayon
/// con su propio directorio temporal.
pub fn unwrap_stack_snaphu(
    stack: &IfgStack,
    coherence: Option<&Array3<f32>>,
    config: &SnaphuConfig,
) -> Result<UnwrappedStack> {
    if let Some(coh) = coherence
        && coh.dim() != stack.data.dim()
    {
        return Err(InsarError::DimensionMismatch(format!(
            "coherence {:?} vs stack {:?}",
            coh.dim(),
            stack.data.dim()
        )));
    }

    let (n_layers, rows, cols) = stack.data.dim();
    let layers: Vec<Array2<f32>> = (0..n_layers)
        .into_par_iter()
        .map(|k| {
            let wrapped = wrapped_phase_layer(stack, k);
            let qual = coherence.map(|coh| coh.index_axis(Axis(0), k).to_owned());
            let dir = unique_tmp_dir(&format!("stack{k}"));
            std::fs::create_dir_all(&dir).with_path(&dir)?;
            let result = run_snaphu(&dir, &wrapped, qual.as_ref(), rows, cols, config);
            let _ = std::fs::remove_dir_all(&dir);
            result
        })
        .collect::<Result<Vec<_>>>()?;

    let mut data = Array3::from_elem((n_layers, rows, cols), f32::NAN);
    for (k, layer) in layers.into_iter().enumerate() {
        data.index_axis_mut(Axis(0), k).assign(&layer);
    }

    let out = UnwrappedStack {
        data,
        epochs: stack.epochs.clone(),
        pairs: stack.pairs.clone(),
        meta: stack.meta.clone(),
    };
    out.validate()?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Epoch, IfgPair, StackMeta, SENTINEL1_WAVELENGTH_M};
    use num_complex::Complex32;
    use surtgis_core::GeoTransform;

    fn ramp(rows: usize, cols: usize, a: f32, b: f32) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |(r, c)| a * r as f32 + b * c as f32)
    }

    fn wrap(phase: &Array2<f32>) -> Array2<f32> {
        phase.map(|&p| p.sin().atan2(p.cos()))
    }

    fn meta() -> StackMeta {
        StackMeta {
            transform: GeoTransform::new(0.0, 0.0, 30.0, -30.0),
            crs: None,
            wavelength_m: SENTINEL1_WAVELENGTH_M,
            incidence_deg: 39.0,
            heading_deg: None,
        }
    }

    fn stack_2_capas(rows: usize, cols: usize) -> (IfgStack, Vec<Array2<f32>>) {
        let truths = vec![ramp(rows, cols, 0.45, 0.35), ramp(rows, cols, -0.3, 0.5)];
        let mut data = Array3::from_elem((2, rows, cols), Complex32::new(0.0, 0.0));
        for (k, t) in truths.iter().enumerate() {
            for ((r, c), &p) in t.indexed_iter() {
                data[[k, r, c]] = Complex32::new(p.cos(), p.sin());
            }
        }
        let epochs = ["2023-01-01", "2023-01-13", "2023-01-25"]
            .iter()
            .map(|s| Epoch(s.parse().unwrap()))
            .collect();
        let stack = IfgStack {
            data,
            epochs,
            pairs: vec![
                IfgPair { reference: 0, secondary: 1, perp_baseline_m: 40.0 },
                IfgPair { reference: 1, secondary: 2, perp_baseline_m: -25.0 },
            ],
            meta: meta(),
        };
        (stack, truths)
    }

    /// Compara `unw` contra `truth` relativo a un píxel de referencia (ambos
    /// algoritmos solo recuperan la fase salvo un offset aditivo 2πk).
    fn assert_matches_ramp(unw: &Array2<f32>, truth: &Array2<f32>, reference: (usize, usize), tol: f32) {
        let (rr, rc) = reference;
        let u0 = unw[[rr, rc]];
        let t0 = truth[[rr, rc]];
        assert!(u0.is_finite(), "referencia {reference:?} quedó NaN");
        for ((r, c), &u) in unw.indexed_iter() {
            let err = ((u - u0) - (truth[[r, c]] - t0)).abs();
            assert!(err < tol, "({r},{c}): unw={u} truth={} err={err}", truth[[r, c]]);
        }
    }

    #[test]
    fn binario_ausente_da_error_io() {
        let wrapped = wrap(&ramp(4, 4, 0.3, 0.2));
        let config = SnaphuConfig { binary: "insar-rs-snaphu-no-existe-nunca".into() };
        let err = unwrap_2d_snaphu(&wrapped, None, &config).unwrap_err();
        assert!(matches!(err, InsarError::Io { .. }), "got: {err:?}");
    }

    /// Regresión A-10: el config de snaphu debe usar nombres de archivo
    /// relativos, nunca rutas absolutas — el parser de snaphu tokeniza por
    /// whitespace, así que una ruta con espacios (p. ej. `TMPDIR` con
    /// espacios en el nombre de usuario) rompía OUTFILE/CORRFILE en silencio.
    /// No requiere el binario snaphu: solo inspecciona el texto generado.
    #[test]
    fn snaphu_config_usa_nombres_relativos_sin_espacios() {
        for has_corr in [false, true] {
            let conf = snaphu_config_text(has_corr);
            // Cada línea debe tener exactamente 2 tokens (clave + valor): si
            // el valor de OUTFILE/CORRFILE fuera una ruta absoluta con
            // espacios, esta línea tendría 3+ tokens.
            for line in conf.lines() {
                let tokens: Vec<&str> = line.split_whitespace().collect();
                assert_eq!(tokens.len(), 2, "línea con más de 2 tokens (¿ruta con espacios?): {line:?}");
            }
            assert!(conf.contains(&format!("OUTFILE {OUTFILE_NAME}\n")), "conf: {conf}");
            assert!(
                !conf.contains('/') && !conf.contains('\\'),
                "OUTFILE/CORRFILE no deben ser rutas (con separador de directorio): {conf}"
            );
            if has_corr {
                assert!(conf.contains(&format!("CORRFILE {CORRFILE_NAME}\n")), "conf: {conf}");
            } else {
                assert!(!conf.contains("CORRFILE"), "conf: {conf}");
            }
        }
    }

    #[test]
    #[ignore = "requiere el binario snaphu en PATH (conda install -c conda-forge snaphu)"]
    fn rampa_lineal_se_recupera_via_snaphu() {
        let truth = ramp(24, 20, 0.45, 0.35);
        let wrapped = wrap(&truth);
        let config = SnaphuConfig::default();
        let unw = unwrap_2d_snaphu(&wrapped, None, &config).unwrap();
        assert_matches_ramp(&unw, &truth, (12, 10), 1e-3);
    }

    #[test]
    #[ignore = "requiere el binario snaphu en PATH (conda install -c conda-forge snaphu)"]
    fn rampa_con_coherencia_se_recupera_via_snaphu() {
        let truth = ramp(24, 20, 0.45, 0.35);
        let wrapped = wrap(&truth);
        let quality = Array2::from_elem((24, 20), 0.9_f32);
        let config = SnaphuConfig::default();
        let unw = unwrap_2d_snaphu(&wrapped, Some(&quality), &config).unwrap();
        assert_matches_ramp(&unw, &truth, (12, 10), 1e-3);
    }

    #[test]
    #[ignore = "requiere el binario snaphu en PATH (conda install -c conda-forge snaphu)"]
    fn unwrap_stack_snaphu_dos_capas() {
        let (stack, truths) = stack_2_capas(16, 14);
        let config = SnaphuConfig::default();
        let out = unwrap_stack_snaphu(&stack, None, &config).unwrap();
        assert_eq!(out.data.dim(), (2, 16, 14));
        assert_eq!(out.pairs.len(), 2);
        for (k, truth) in truths.iter().enumerate() {
            let layer = out.data.index_axis(Axis(0), k).to_owned();
            assert_matches_ramp(&layer, truth, (8, 7), 1e-3);
        }
    }
}

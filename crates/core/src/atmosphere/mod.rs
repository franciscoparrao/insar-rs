//! Corrección atmosférica simple (alcance MVP): el APS se estima como la
//! componente pasa-bajo espacial / pasa-alto temporal de la serie
//! (esquema SBAS clásico) y se resta. APS avanzado (modelos meteorológicos,
//! GACOS) es v0.2.
//!
//! Modelo: el APS está correlado en espacio y descorrelado en tiempo, mientras
//! que la deformación es suave en el tiempo. Por eso:
//!
//! 1. **Pasa-alto temporal**: por píxel, `hp[e] = serie[e] − media_móvil(e)`,
//!    con ventana centrada de [`ApsConfig::temporal_window_epochs`] épocas.
//!    El truncamiento en los bordes es **simétrico**: el semiancho efectivo en
//!    la época `e` es `min(w/2, e, n−1−e)`, de modo que la ventana siempre
//!    queda centrada en `e`. Así una señal lineal en el tiempo (con épocas
//!    equiespaciadas) tiene pasa-alto exactamente cero también en los bordes
//!    y no se confunde con atmósfera. Si la ventana es ≥ n_épocas se usa la
//!    media global de la serie del píxel (la media móvil degenera en eso);
//!    en ese caso una tendencia lineal SÍ deja residuo en el pasa-alto, por
//!    lo que se recomienda ventana < n_épocas.
//! 2. **Pasa-bajo espacial**: por época, filtro gaussiano 2D separable con
//!    sigma [`ApsConfig::spatial_sigma_px`] y radio `ceil(3σ)`, normalizado
//!    por los pesos válidos (convolución normalizada): los NaN no aportan ni
//!    al numerador ni al denominador, y en los bordes del raster el kernel se
//!    renormaliza con los pesos que caen dentro de la grilla.
//! 3. El APS estimado se resta de la serie in place donde es finito.
//!
//! Política NaN (NoData): si la serie temporal de un píxel contiene algún
//! NaN, ese píxel se excluye por completo de la estimación (su pasa-alto es
//! NaN en todas las épocas → APS NaN → no se corrige y no contamina a sus
//! vecinos gracias a la normalización por pesos válidos).

use ndarray::{Array2, Array3, ArrayView2, Axis, Zip};
use rayon::prelude::*;

use crate::error::{InsarError, Result};
use crate::types::DisplacementSeries;

/// Parámetros del filtro espacio-temporal.
#[derive(Debug, Clone)]
pub struct ApsConfig {
    /// Sigma del filtro gaussiano espacial, en píxeles.
    pub spatial_sigma_px: f32,
    /// Ventana de la media móvil temporal, en épocas (impar).
    pub temporal_window_epochs: usize,
}

impl Default for ApsConfig {
    fn default() -> Self {
        Self { spatial_sigma_px: 8.0, temporal_window_epochs: 5 }
    }
}

/// Estima y remueve la señal atmosférica de la serie, in place.
/// NaN se propaga sin contaminar vecinos (filtro con normalización por
/// pesos válidos).
///
/// Errores:
/// - [`InsarError::Metadata`] si `spatial_sigma_px` no es finito y positivo.
/// - [`InsarError::Metadata`] si `temporal_window_epochs` es par.
pub fn correct_aps(series: &mut DisplacementSeries, config: &ApsConfig) -> Result<()> {
    let sigma = config.spatial_sigma_px;
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(InsarError::Metadata(format!(
            "spatial_sigma_px debe ser finito y > 0 (recibido {sigma})"
        )));
    }
    let window = config.temporal_window_epochs;
    if window.is_multiple_of(2) {
        return Err(InsarError::Metadata(format!(
            "temporal_window_epochs debe ser impar (recibido {window})"
        )));
    }

    let n_epochs = series.n_layers();
    let (rows, cols) = series.dims();
    if n_epochs == 0 || rows == 0 || cols == 0 {
        return Ok(());
    }

    // 1) Componente pasa-alto temporal por píxel.
    let hp = temporal_high_pass(&series.data, window);

    // 2) APS por época: pasa-bajo espacial gaussiano con normalización por
    //    pesos válidos. Paralelo por época.
    let kernel = gaussian_kernel(f64::from(sigma));
    let aps: Vec<Array2<f32>> = (0..n_epochs)
        .into_par_iter()
        .map(|e| spatial_low_pass_nan(hp.index_axis(Axis(0), e), &kernel))
        .collect();

    // 3) Restar el APS donde es finito (donde no lo es, el píxel queda
    //    intacto: política de no-corrección para píxeles inválidos).
    for (e, aps_e) in aps.iter().enumerate() {
        let mut layer = series.data.index_axis_mut(Axis(0), e);
        Zip::from(&mut layer).and(aps_e).for_each(|s, &a| {
            if a.is_finite() {
                *s -= a;
            }
        });
    }

    Ok(())
}

/// Pasa-alto temporal: `serie − media móvil centrada` por píxel, en f64.
///
/// - Ventana truncada simétricamente en los bordes (semiancho efectivo
///   `min(half, e, n−1−e)`): en la primera y última época la ventana degenera
///   al propio valor y el pasa-alto es 0 (no se estima APS ahí, decisión
///   conservadora que evita confundir tendencia con atmósfera).
/// - `window >= n` → media global del píxel.
/// - Cualquier NaN en la serie del píxel → NaN en todas sus épocas.
fn temporal_high_pass(data: &Array3<f32>, window: usize) -> Array3<f32> {
    let n = data.shape()[0];
    let (rows, cols) = (data.shape()[1], data.shape()[2]);
    let mut hp = Array3::<f32>::zeros((n, rows, cols));
    let half = window / 2;
    let global = window >= n;

    Zip::from(hp.lanes_mut(Axis(0)))
        .and(data.lanes(Axis(0)))
        .par_for_each(|mut hp_px, px| {
            if px.iter().any(|v| !v.is_finite()) {
                hp_px.fill(f32::NAN);
                return;
            }
            if global {
                let mean = px.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
                for (h, &v) in hp_px.iter_mut().zip(px.iter()) {
                    *h = (f64::from(v) - mean) as f32;
                }
                return;
            }
            // Sumas acumuladas para media móvil O(n).
            let mut prefix = vec![0.0f64; n + 1];
            for (e, &v) in px.iter().enumerate() {
                prefix[e + 1] = prefix[e] + f64::from(v);
            }
            for e in 0..n {
                let k = half.min(e).min(n - 1 - e);
                let (lo, hi) = (e - k, e + k);
                let mean = (prefix[hi + 1] - prefix[lo]) / (hi - lo + 1) as f64;
                hp_px[e] = (f64::from(px[e]) - mean) as f32;
            }
        });

    hp
}

/// Kernel gaussiano 1D sin normalizar (centro = 1), radio `ceil(3σ)`.
/// La normalización se hace por pesos válidos en la convolución.
fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil() as usize;
    let denom = 2.0 * sigma * sigma;
    (-(radius as i64)..=radius as i64)
        .map(|i| {
            let x = i as f64;
            (-x * x / denom).exp()
        })
        .collect()
}

/// Filtro gaussiano 2D separable con normalización por pesos válidos
/// (convolución normalizada): se convoluciona por separado el campo (con
/// NaN→0) y la máscara de validez, y se divide. Como el kernel 2D es el
/// producto de los 1D, dos pasadas separables sobre numerador y denominador
/// equivalen exactamente al filtro 2D normalizado. En los bordes del raster
/// el kernel se trunca y la división renormaliza automáticamente.
///
/// Píxel central NaN → salida NaN (ese píxel no se corrige).
fn spatial_low_pass_nan(input: ArrayView2<'_, f32>, kernel: &[f64]) -> Array2<f32> {
    let (rows, cols) = input.dim();
    let mut val = Array2::<f64>::zeros((rows, cols));
    let mut mask = Array2::<f64>::zeros((rows, cols));
    for ((i, j), &x) in input.indexed_iter() {
        if x.is_finite() {
            val[[i, j]] = f64::from(x);
            mask[[i, j]] = 1.0;
        }
    }

    let val_s = convolve_axis(&convolve_axis(&val, kernel, Axis(1)), kernel, Axis(0));
    let mask_s = convolve_axis(&convolve_axis(&mask, kernel, Axis(1)), kernel, Axis(0));

    let mut out = Array2::<f32>::from_elem((rows, cols), f32::NAN);
    for ((i, j), o) in out.indexed_iter_mut() {
        // Si el centro es válido, mask_s >= peso_central^2 = 1, así que el
        // umbral solo protege contra degeneraciones numéricas.
        if input[[i, j]].is_finite() && mask_s[[i, j]] > 1e-9 {
            *o = (val_s[[i, j]] / mask_s[[i, j]]) as f32;
        }
    }
    out
}

/// Convolución 1D a lo largo de `axis`, truncada en los bordes (los puntos
/// fuera de la grilla no aportan; la renormalización ocurre al dividir por
/// la máscara convolucionada con el mismo kernel).
fn convolve_axis(input: &Array2<f64>, kernel: &[f64], axis: Axis) -> Array2<f64> {
    let radius = kernel.len() / 2;
    let mut out = Array2::<f64>::zeros(input.dim());
    Zip::from(out.lanes_mut(axis))
        .and(input.lanes(axis))
        .for_each(|mut o, lane| {
            let len = lane.len();
            for c in 0..len {
                let lo = c.saturating_sub(radius);
                let hi = (c + radius).min(len - 1);
                let mut acc = 0.0;
                for t in lo..=hi {
                    acc += kernel[t + radius - c] * lane[t];
                }
                o[c] = acc;
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DisplacementSeries, Epoch, SENTINEL1_WAVELENGTH_M, StackMeta};
    use chrono::NaiveDate;
    use ndarray::Array3;
    use surtgis_core::GeoTransform;

    fn meta() -> StackMeta {
        StackMeta {
            transform: GeoTransform::new(0.0, 0.0, 30.0, -30.0),
            crs: None,
            wavelength_m: SENTINEL1_WAVELENGTH_M,
            incidence_deg: 39.0,
            heading_deg: None,
        }
    }

    /// `n` épocas equiespaciadas cada 12 días.
    fn epochs(n: usize) -> Vec<Epoch> {
        let start = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        (0..n)
            .map(|i| Epoch(start + chrono::Duration::days(12 * i as i64)))
            .collect()
    }

    fn series_from(data: Array3<f32>) -> DisplacementSeries {
        let n = data.shape()[0];
        DisplacementSeries { data, epochs: epochs(n), meta: meta() }
    }

    /// Deformación lineal en el tiempo con tasa que varía suavemente en
    /// espacio: `d[e](r,c) = rate(r,c) · e`.
    fn linear_deformation(n: usize, rows: usize, cols: usize) -> Array3<f32> {
        Array3::from_shape_fn((n, rows, cols), |(e, r, c)| {
            let rate = 1e-3 * (r as f32 + c as f32); // m por época
            rate * e as f32
        })
    }

    /// Artefacto espacialmente suave (afín): constante + rampa.
    fn smooth_artifact(rows: usize, cols: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |(r, c)| 0.03 + 5e-4 * (r as f32 + c as f32))
    }

    fn rms(values: impl Iterator<Item = f64>) -> f64 {
        let (mut acc, mut count) = (0.0, 0usize);
        for v in values {
            acc += v * v;
            count += 1;
        }
        (acc / count.max(1) as f64).sqrt()
    }

    /// RMS del residuo (serie − deformación pura) en la época `e`,
    /// excluyendo píxeles no finitos.
    fn rms_residual(series: &Array3<f32>, truth: &Array3<f32>, e: usize) -> f64 {
        rms(
            series
                .index_axis(Axis(0), e)
                .iter()
                .zip(truth.index_axis(Axis(0), e).iter())
                .filter(|(s, _)| s.is_finite())
                .map(|(&s, &t)| f64::from(s) - f64::from(t)),
        )
    }

    /// Ruido pseudo-aleatorio determinista (LCG) en [-0.5, 0.5).
    fn lcg_noise(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    #[test]
    fn ventana_par_es_error() {
        let mut s = series_from(linear_deformation(5, 4, 4));
        let cfg = ApsConfig { spatial_sigma_px: 2.0, temporal_window_epochs: 4 };
        let err = correct_aps(&mut s, &cfg).unwrap_err();
        assert!(matches!(err, InsarError::Metadata(_)), "se esperaba Metadata, hubo {err:?}");
    }

    #[test]
    fn sigma_invalido_es_error() {
        for sigma in [0.0_f32, -1.5, f32::NAN, f32::INFINITY] {
            let mut s = series_from(linear_deformation(5, 4, 4));
            let cfg = ApsConfig { spatial_sigma_px: sigma, temporal_window_epochs: 5 };
            let err = correct_aps(&mut s, &cfg).unwrap_err();
            assert!(
                matches!(err, InsarError::Metadata(_)),
                "sigma={sigma}: se esperaba Metadata, hubo {err:?}"
            );
        }
    }

    #[test]
    fn deformacion_lineal_pasa_intacta() {
        let truth = linear_deformation(9, 16, 16);
        let mut s = series_from(truth.clone());
        let cfg = ApsConfig { spatial_sigma_px: 2.0, temporal_window_epochs: 5 };
        correct_aps(&mut s, &cfg).unwrap();
        let max_diff = s
            .data
            .iter()
            .zip(truth.iter())
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
            .fold(0.0_f64, f64::max);
        // Ventana centrada con truncamiento simétrico → pasa-alto exactamente
        // cero para señal lineal equiespaciada (solo redondeo f32).
        assert!(max_diff < 1e-6, "deformación lineal alterada: max_diff={max_diff}");
    }

    #[test]
    fn artefacto_suave_de_una_epoca_se_atenua_5x() {
        let (n, rows, cols, e_art) = (9, 24, 24, 4);
        let truth = linear_deformation(n, rows, cols);
        let mut data = truth.clone();
        let art = smooth_artifact(rows, cols);
        data.index_axis_mut(Axis(0), e_art)
            .zip_mut_with(&art, |d, &a| *d += a);

        let before = rms_residual(&data, &truth, e_art);
        let mut s = series_from(data);
        let cfg = ApsConfig { spatial_sigma_px: 2.0, temporal_window_epochs: 7 };
        correct_aps(&mut s, &cfg).unwrap();
        let after = rms_residual(&s.data, &truth, e_art);

        assert!(
            after * 5.0 <= before,
            "atenuación insuficiente: antes={before:.6}, después={after:.6} (ratio {:.2}x)",
            before / after
        );
    }

    #[test]
    fn ruido_blanco_no_se_trata_como_aps() {
        let (n, rows, cols) = (9, 16, 16);
        let mut seed = 42_u64;
        let noise =
            Array3::from_shape_fn((n, rows, cols), |_| 0.01 * lcg_noise(&mut seed));
        let mut s = series_from(noise.clone());
        let cfg = ApsConfig { spatial_sigma_px: 2.0, temporal_window_epochs: 5 };
        correct_aps(&mut s, &cfg).unwrap();

        let noise_rms = rms(noise.iter().map(|&v| f64::from(v)));
        let change_rms = rms(
            s.data
                .iter()
                .zip(noise.iter())
                .map(|(&a, &b)| f64::from(a) - f64::from(b)),
        );
        // El filtro espacial aplana el ruido incoherente: la "corrección"
        // aplicada debe ser una fracción menor del ruido, no removerlo.
        assert!(
            change_rms < 0.35 * noise_rms,
            "ruido tratado como APS: cambio={change_rms:.6} vs ruido={noise_rms:.6}"
        );
    }

    #[test]
    fn nan_se_preserva_y_no_contamina() {
        let (n, rows, cols, e_art) = (9, 24, 24, 4);
        let truth = linear_deformation(n, rows, cols);
        let mut data = truth.clone();
        let art = smooth_artifact(rows, cols);
        data.index_axis_mut(Axis(0), e_art)
            .zip_mut_with(&art, |d, &a| *d += a);
        // Un NaN en una sola época invalida todo el píxel (2,3).
        data[[1, 2, 3]] = f32::NAN;

        let original = data.clone();
        let before = rms_residual(&data, &truth, e_art);
        let mut s = series_from(data);
        let cfg = ApsConfig { spatial_sigma_px: 2.0, temporal_window_epochs: 7 };
        correct_aps(&mut s, &cfg).unwrap();

        // El NaN sigue siendo NaN.
        assert!(s.data[[1, 2, 3]].is_nan());
        // El píxel con NaN no se corrige en ninguna época.
        for e in 0..n {
            if e == 1 {
                continue;
            }
            assert_eq!(
                s.data[[e, 2, 3]],
                original[[e, 2, 3]],
                "píxel NaN corregido en época {e}"
            );
        }
        // Los vecinos siguen corrigiéndose bien (el NaN no contamina):
        // la atenuación global del artefacto se mantiene.
        let after = rms_residual(&s.data, &truth, e_art);
        assert!(
            after * 5.0 <= before,
            "el NaN degradó la corrección: antes={before:.6}, después={after:.6}"
        );
        // Y el residuo del vecino inmediato es pequeño frente al artefacto.
        let neighbor_res =
            (f64::from(s.data[[e_art, 2, 4]]) - f64::from(truth[[e_art, 2, 4]])).abs();
        assert!(
            neighbor_res < f64::from(art[[2, 4]]) / 4.0,
            "vecino del NaN mal corregido: residuo={neighbor_res:.6}"
        );
    }

    #[test]
    fn ventana_mayor_que_serie_usa_media_global() {
        // window >= n_épocas → media global. Una serie constante en el tiempo
        // tiene media global = valor → APS = 0 → sin cambios.
        let (n, rows, cols) = (5, 8, 8);
        let data = Array3::from_shape_fn((n, rows, cols), |(_, r, c)| {
            0.01 * (r as f32 - c as f32)
        });
        let mut s = series_from(data.clone());
        let cfg = ApsConfig { spatial_sigma_px: 1.5, temporal_window_epochs: 7 };
        correct_aps(&mut s, &cfg).unwrap();
        let max_diff = s
            .data
            .iter()
            .zip(data.iter())
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 1e-7, "serie constante en el tiempo alterada: {max_diff}");
    }
}

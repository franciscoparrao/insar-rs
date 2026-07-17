//! Corrección troposférica **estratificada por reanálisis** (G-7: ERA5/GACOS).
//!
//! Complementa [`super::correct_topo_correlated`] (empírico, regresión
//! fase-elevación) con un modelo físico del retardo (Smith & Weintraub 1953;
//! Saastamoinen 1972 para el término hidrostático; integración vertical del
//! perfil para el húmedo — el mismo enfoque de PyAPS/RAiDER/GACOS).
//!
//! Este módulo **no fetchea ERA5 ni remuestrea horizontalmente**: consume un
//! perfil atmosférico por columna (temperatura/humedad/altura geopotencial
//! por nivel de presión) ya remuestreado a cada ubicación InSAR — la descarga
//! vía la API de Copernicus CDS (requiere credenciales propias del usuario) y
//! la interpolación horizontal grilla ERA5 (~31 km) → grilla InSAR NO están
//! implementadas todavía (gap abierto de la auditoría 2026-07-05, hallazgo
//! A-9); son el trabajo pendiente para que esta corrección sea operable
//! end-to-end, análogo a `validation/hyp3_*.py` para HyP3. Lo que sí está
//! implementado y testeado es el **kernel físico** por columna: retardo
//! cenital hidrostático + húmedo con interpolación vertical a la altura de
//! superficie real, proyección a LOS, y el ensamblado diferencial sobre la
//! serie ya invertida.
//!
//! ## Modelo físico
//!
//! El retardo cenital total es hidrostático + húmedo:
//!
//! - **Hidrostático** ([`saastamoinen_zhd`], dominante: ~2.3 m a nivel del
//!   mar): depende solo de presión superficial, latitud y altura — no
//!   requiere perfil vertical completo, es la parte "fácil" de modelar.
//! - **Húmedo** ([`integrate_zwd`], ~0.03–0.4 m, mucho más variable en
//!   espacio y tiempo): integración vertical de la refractividad húmeda
//!   `N_wet = k2'·e/T + k3·e/T²` (`k2'` reducido, ver `K2_PRIME`, para no
//!   contar el vapor de agua dos veces junto al ZHD sobre presión total)
//!   sobre el perfil de niveles de presión de ERA5 (temperatura, humedad
//!   específica, altura geopotencial), **interpolado a la altura real de
//!   superficie** ([`saastamoinen_zhd`]/[`integrate_zwd`] no descartan la
//!   rebanada entre la superficie y el primer nivel — la más húmeda).
//!
//! El retardo cenital se proyecta a LOS dividiendo por `cos(incidencia)`
//! ([`project_to_los`]) — aproximación estándar en InSAR, sin mapping
//! function angular tipo Niell (razonable para incidencias S1 IW, 29°–46°).

use ndarray::{Array3, Axis};

use crate::error::{InsarError, Result};
use crate::types::DisplacementSeries;

/// Coeficiente de refractividad seca (Smith & Weintraub 1953, vía Bevis et
/// al. 1994), K/hPa. Solo se usa aquí para derivar `K2_PRIME` — el término
/// hidrostático real se calcula con la fórmula cerrada de Saastamoinen
/// ([`saastamoinen_zhd`]), no con `K1` directamente.
const K1: f64 = 77.6;
/// K2 "crudo" de Bevis et al. 1994, K/hPa. **No usar directamente** en
/// `N_wet`: cuenta el vapor de agua dos veces cuando el ZHD ya se calculó
/// sobre la presión TOTAL (que es lo que hace [`saastamoinen_zhd`] vía
/// Saastamoinen 1972) — el coeficiente correcto para esa combinación es
/// `K2_PRIME` (Bevis et al. 1994, ec. 8: `k2' = k2 - k1·Rd/Rv`).
const K2_BEVIS: f64 = 71.6;
/// K3 de Bevis et al. 1994, K²/hPa — sin el doble conteo de K2, no requiere
/// corrección.
const K3: f64 = 3.75e5;
/// Razón de masas molares agua/aire seco (`Rd/Rv`), para presión de vapor
/// desde humedad específica y para reducir K2.
const EPSILON: f64 = 0.622;
/// K2 reducido (`k2' = k2 - k1·Rd/Rv ≈ 23.3 K/hPa`, Bevis et al. 1994): el
/// coeficiente húmedo correcto para esta formulación (ZHD sobre presión
/// total). Coincide con RAiDER (`k2 = 0.233 K/Pa = 23.3 K/hPa`); usar
/// `K2_BEVIS` sin reducir aquí sobreestimaría el ZWD ~1 cm en atmósfera
/// húmeda (doble conteo del vapor de agua ya presente en el ZHD).
const K2_PRIME: f64 = K2_BEVIS - K1 * EPSILON;

/// Perfil atmosférico vertical en una columna (un píxel ERA5, una fecha):
/// niveles de presión con temperatura, humedad específica y altura
/// geopotencial. No se asume ningún orden particular — las funciones que lo
/// consumen ordenan por altura internamente.
#[derive(Debug, Clone)]
pub struct AtmosphericProfile {
    /// Presión de cada nivel, hPa.
    pub pressure_hpa: Vec<f64>,
    /// Temperatura, K.
    pub temperature_k: Vec<f64>,
    /// Humedad específica, kg/kg.
    pub specific_humidity: Vec<f64>,
    /// Altura geopotencial de cada nivel, m.
    pub height_m: Vec<f64>,
}

impl AtmosphericProfile {
    fn validate(&self) -> Result<()> {
        let n = self.pressure_hpa.len();
        if self.temperature_k.len() != n
            || self.specific_humidity.len() != n
            || self.height_m.len() != n
        {
            return Err(InsarError::DimensionMismatch(format!(
                "AtmosphericProfile: vectores de largo distinto (presión {n}, T {}, q {}, z {})",
                self.temperature_k.len(),
                self.specific_humidity.len(),
                self.height_m.len()
            )));
        }
        Ok(())
    }
}

/// Presión parcial de vapor de agua (hPa) desde humedad específica (kg/kg) y
/// presión total (hPa) — Wallace & Hobbs 2006, aprox. estándar.
fn vapor_pressure_hpa(specific_humidity: f64, pressure_hpa: f64) -> f64 {
    specific_humidity * pressure_hpa / (EPSILON + (1.0 - EPSILON) * specific_humidity)
}

/// Retardo cenital hidrostático (Saastamoinen 1972; Davis et al. 1985), en
/// metros. `pressure_hpa`: presión superficial en el píxel; `lat_deg`:
/// latitud geodésica; `height_m`: altura sobre el elipsoide.
///
/// Domina el retardo total (~2.3 m a nivel del mar, `P₀=1013.25 hPa`) y varía
/// suavemente en espacio — es la parte del retardo que la corrección
/// topo-correlacionada empírica ([`super::correct_topo_correlated`]) ya
/// captura razonablemente bien; el valor real de este módulo está en el
/// término húmedo ([`integrate_zwd`]), que la regresión fase-elevación no
/// separa de la deformación.
pub fn saastamoinen_zhd(pressure_hpa: f64, lat_deg: f64, height_m: f64) -> f64 {
    let lat_rad = lat_deg.to_radians();
    let f = 1.0 - 0.00266 * (2.0 * lat_rad).cos() - 0.00028 * (height_m / 1000.0);
    0.0022768 * pressure_hpa / f
}

/// Interpola `(presión, temperatura, humedad específica)` del perfil a
/// `target_height_m`: T y q **linealmente** en altura, P **log-linealmente**
/// (la presión decae ~exponencialmente con la altura — estándar en
/// PyAPS/RAiDER, evita presiones negativas al extrapolar). Si
/// `target_height_m` cae dentro del rango de alturas del perfil, interpola
/// entre los dos niveles que la bracket-ean; si cae fuera (superficie más
/// baja o más alta que todos los niveles), extrapola con el par de niveles
/// del borde más cercano.
///
/// Antes de esta función, [`integrate_zwd`]/[`zenith_delay`] descartaban la
/// rebanada `[superficie, primer nivel ERA5]` en vez de interpolar — la más
/// húmeda y la que más varía con la topografía, reintroduciendo justo la
/// señal topo-correlacionada que este módulo pretende eliminar.
///
/// Error si el perfil tiene menos de 2 niveles (no hay par para
/// interpolar/extrapolar).
fn interpolate_profile(profile: &AtmosphericProfile, target_height_m: f64) -> Result<(f64, f64, f64)> {
    let n = profile.pressure_hpa.len();
    if n < 2 {
        return Err(InsarError::Inversion(format!(
            "perfil con {n} nivel(es): se requieren ≥2 para interpolar a la superficie"
        )));
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| profile.height_m[a].total_cmp(&profile.height_m[b]));

    let pos = idx.partition_point(|&i| profile.height_m[i] < target_height_m);
    let (lo, hi) = if pos == 0 {
        (idx[0], idx[1])
    } else if pos >= n {
        (idx[n - 2], idx[n - 1])
    } else {
        (idx[pos - 1], idx[pos])
    };

    let (z0, z1) = (profile.height_m[lo], profile.height_m[hi]);
    let dz = z1 - z0;
    if dz.abs() < 1e-9 {
        // Niveles duplicados en altura (o único nivel bracket-eante): sin
        // información para interpolar, usa el primero tal cual.
        return Ok((profile.pressure_hpa[lo], profile.temperature_k[lo], profile.specific_humidity[lo]));
    }
    let frac = (target_height_m - z0) / dz;

    let temperature_k =
        profile.temperature_k[lo] + frac * (profile.temperature_k[hi] - profile.temperature_k[lo]);
    let specific_humidity = profile.specific_humidity[lo]
        + frac * (profile.specific_humidity[hi] - profile.specific_humidity[lo]);
    let ln_p0 = profile.pressure_hpa[lo].ln();
    let ln_p1 = profile.pressure_hpa[hi].ln();
    let pressure_hpa = (ln_p0 + frac * (ln_p1 - ln_p0)).exp();

    Ok((pressure_hpa, temperature_k, specific_humidity))
}

/// Retardo cenital húmedo (m): integra `N_wet = K2'·e/T + K3·e/T²` en altura
/// sobre el perfil (regla del trapecio), desde `surface_height_m` (el punto
/// InSAR real, interpolado entre niveles ERA5 vía `interpolate_profile` —
/// no descartado) hasta el tope del perfil.
///
/// Error si el perfil tiene longitudes de vector inconsistentes o quedan
/// menos de 2 niveles útiles (el punto de superficie interpolado más al
/// menos un nivel real por encima) tras interpolar.
pub fn integrate_zwd(profile: &AtmosphericProfile, surface_height_m: f64) -> Result<f64> {
    profile.validate()?;

    let n_wet_at = |t: f64, q: f64, p: f64| {
        let e = vapor_pressure_hpa(q, p);
        K2_PRIME * e / t + K3 * e / (t * t)
    };

    let (p_surf, t_surf, q_surf) = interpolate_profile(profile, surface_height_m)?;
    let mut levels: Vec<(f64, f64)> = vec![(surface_height_m, n_wet_at(t_surf, q_surf, p_surf))];
    levels.extend((0..profile.pressure_hpa.len()).filter(|&i| profile.height_m[i] > surface_height_m).map(
        |i| {
            let n_wet = n_wet_at(profile.temperature_k[i], profile.specific_humidity[i], profile.pressure_hpa[i]);
            (profile.height_m[i], n_wet)
        },
    ));
    levels.sort_by(|a, b| a.0.total_cmp(&b.0));
    levels.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);

    if levels.len() < 2 {
        return Err(InsarError::Inversion(format!(
            "perfil con {} nivel(es) sobre {surface_height_m} m tras interpolar \
             (se requieren ≥2 para integrar)",
            levels.len()
        )));
    }

    // Trapecio en N_wet(z); N está en unidades de refractividad (ppm) → 1e-6.
    let integral: f64 = levels
        .windows(2)
        .map(|w| {
            let (z0, n0) = w[0];
            let (z1, n1) = w[1];
            0.5 * (n0 + n1) * (z1 - z0)
        })
        .sum();
    Ok(integral * 1e-6)
}

/// Retardo cenital total (hidrostático + húmedo), en metros, para un perfil
/// y una ubicación (latitud, altura del punto InSAR real). La presión
/// superficial para el término hidrostático es la del perfil **interpolado**
/// a `surface_height_m` (`interpolate_profile`) — la misma que usa
/// [`integrate_zwd`] internamente para el término húmedo, evitando la
/// inconsistencia de evaluar ZHD y ZWD en presiones de referencia distintas.
pub fn zenith_delay(
    profile: &AtmosphericProfile,
    lat_deg: f64,
    surface_height_m: f64,
) -> Result<f64> {
    profile.validate()?;
    let (surface_pressure, _, _) = interpolate_profile(profile, surface_height_m)?;
    let zhd = saastamoinen_zhd(surface_pressure, lat_deg, surface_height_m);
    let zwd = integrate_zwd(profile, surface_height_m)?;
    Ok(zhd + zwd)
}

/// Proyecta un retardo cenital a línea de vista (LOS): `delay_los =
/// delay_zenith / cos(incidencia)` — aproximación estándar en InSAR (sin
/// mapping function angular tipo Niell). Error si `incidence_deg ∉ [0, 90)`.
pub fn project_to_los(zenith_delay_m: f64, incidence_deg: f64) -> Result<f64> {
    if !(0.0..90.0).contains(&incidence_deg) {
        return Err(InsarError::Metadata(format!(
            "incidencia {incidence_deg}° fuera de [0, 90)"
        )));
    }
    Ok(zenith_delay_m / incidence_deg.to_radians().cos())
}

/// Suma a la serie la corrección del retardo troposférico por reanálisis, ya
/// proyectado a LOS y resuelto en la grilla del stack. `los_delay_m`: épocas
/// × filas × columnas, mismo layout que [`DisplacementSeries::data`] — cada
/// capa es el retardo LOS (m, positivo = más retardo = más camino óptico)
/// de esa época ([`zenith_delay`] + [`project_to_los`] por píxel, resueltas
/// externamente).
///
/// ## Signo (verificado contra MintPy `tropo_pyaps3.py`/`diff.py`)
///
/// Un retardo positivo alarga el camino óptico ⇒ mayor rango aparente ⇒, con
/// la convención `d = -λ/(4π)·φ` del motor (fase positiva = alejamiento =
/// desplazamiento negativo, ver [`crate::inversion::phase_to_displacement`]),
/// el retardo positivo se propaga a la serie invertida como un desplazamiento
/// NEGATIVO espurio. Recuperar el desplazamiento real exige **sumar** de
/// vuelta `(los_delay[e] − los_delay[reference_epoch])` — no restarlo. MintPy
/// hace la operación equivalente: `tropo_pyaps3.py` invierte el signo del
/// retardo (`pha *= -1`, "reverse the sign for consistency") antes de
/// guardarlo, y `diff.py` resta ese valor ya invertido de la serie — neto,
/// suma el retardo positivo tal como aquí.
///
/// Como la serie ya está referenciada a la primera época, se usa la
/// DIFERENCIA `los_delay[e] − los_delay[reference_epoch]` — el mismo
/// convenio diferencial que la interferometría real (y que
/// [`super::correct_topo_series`]/[`crate::postprocess::deramp_series`]).
/// Píxeles con retardo no finito en la época o la referencia quedan sin
/// corregir (NaN de ERA5 no debe contaminar displacement válido).
pub fn correct_era5_series(
    series: &mut DisplacementSeries,
    los_delay_m: &Array3<f32>,
    reference_epoch: usize,
) -> Result<()> {
    let n_epochs = series.n_layers();
    let (rows, cols) = series.dims();
    if los_delay_m.dim() != (n_epochs, rows, cols) {
        return Err(InsarError::DimensionMismatch(format!(
            "los_delay_m {:?} vs serie ({n_epochs}, {rows}, {cols})",
            los_delay_m.dim()
        )));
    }
    if reference_epoch >= n_epochs {
        return Err(InsarError::Metadata(format!(
            "reference_epoch {reference_epoch} fuera de rango (0..{n_epochs})"
        )));
    }

    let reference = los_delay_m.index_axis(Axis(0), reference_epoch).to_owned();
    for e in 0..n_epochs {
        let delay_e = los_delay_m.index_axis(Axis(0), e);
        let mut layer = series.data.index_axis(Axis(0), e).to_owned();
        ndarray::Zip::from(&mut layer)
            .and(&delay_e)
            .and(&reference)
            .for_each(|d, &de, &dr| {
                if de.is_finite() && dr.is_finite() {
                    *d += de - dr;
                }
            });
        series.data.index_axis_mut(Axis(0), e).assign(&layer);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zhd_nivel_del_mar_reproduce_valor_clasico() {
        // Atmósfera estándar (P₀=1013.25 hPa, lat 45°, nivel del mar):
        // ZHD ≈ 2.3 m es el valor de referencia citado en toda la literatura
        // GNSS/InSAR (Saastamoinen 1972).
        let zhd = saastamoinen_zhd(1013.25, 45.0, 0.0);
        assert!((zhd - 2.3068).abs() < 1e-3, "ZHD {zhd} fuera de tolerancia");
    }

    #[test]
    fn zhd_decrece_con_presion_y_altura_reales() {
        // Sitio de altura (~3000 m, P≈700 hPa, típico Andes) vs nivel del
        // mar: el ZHD real (P y H co-variando) debe ser bastante menor.
        let sea_level = saastamoinen_zhd(1013.25, -33.0, 0.0);
        let altiplano = saastamoinen_zhd(700.0, -33.0, 3000.0);
        assert!(altiplano < sea_level, "{altiplano} vs {sea_level}");
        assert!(altiplano > 1.0 && altiplano < 2.0, "ZHD altiplano {altiplano} implausible");
    }

    #[test]
    fn vapor_pressure_nula_sin_humedad() {
        assert_eq!(vapor_pressure_hpa(0.0, 1000.0), 0.0);
    }

    #[test]
    fn vapor_pressure_rango_plausible() {
        // q=10 g/kg (húmedo, costero) a P=1000 hPa → e de orden 10-20 hPa.
        let e = vapor_pressure_hpa(0.010, 1000.0);
        assert!((10.0..20.0).contains(&e), "e={e} hPa fuera de rango plausible");
    }

    /// Perfil sintético con humedad decayendo exponencialmente con la altura
    /// (razonable: la mayoría del vapor de agua está en los primeros ~2 km).
    fn perfil_humedo(surface_q: f64) -> AtmosphericProfile {
        let heights: Vec<f64> = (0..15).map(|i| i as f64 * 1000.0).collect();
        let pressure: Vec<f64> = heights.iter().map(|h| 1013.25 * (-h / 8000.0).exp()).collect();
        let temperature: Vec<f64> = heights.iter().map(|h| 288.15 - 0.0065 * h).collect();
        let humidity: Vec<f64> = heights.iter().map(|h| surface_q * (-h / 2000.0).exp()).collect();
        AtmosphericProfile {
            pressure_hpa: pressure,
            temperature_k: temperature,
            specific_humidity: humidity,
            height_m: heights,
        }
    }

    #[test]
    fn zwd_atmosfera_seca_es_nula() {
        let perfil = perfil_humedo(0.0);
        let zwd = integrate_zwd(&perfil, 0.0).unwrap();
        assert!(zwd.abs() < 1e-9, "ZWD {zwd} debería ser exactamente 0 sin vapor de agua");
    }

    #[test]
    fn zwd_humeda_en_rango_fisico_plausible() {
        // 12 g/kg en superficie (costa húmeda) → ZWD típico 0.03-0.4 m.
        let perfil = perfil_humedo(0.012);
        let zwd = integrate_zwd(&perfil, 0.0).unwrap();
        assert!((0.03..0.4).contains(&zwd), "ZWD {zwd} m fuera del rango físico esperado");
    }

    #[test]
    fn zwd_mayor_altura_superficie_da_menor_retardo() {
        let perfil = perfil_humedo(0.012);
        let zwd_mar = integrate_zwd(&perfil, 0.0).unwrap();
        let zwd_alto = integrate_zwd(&perfil, 2000.0).unwrap();
        assert!(zwd_alto < zwd_mar, "{zwd_alto} vs {zwd_mar}");
    }

    #[test]
    fn zwd_menos_de_dos_niveles_es_error() {
        let mut perfil = perfil_humedo(0.012);
        // Sube la superficie por encima de todos los niveles salvo el tope.
        let err = integrate_zwd(&perfil, 20000.0).unwrap_err();
        assert!(matches!(err, InsarError::Inversion(_)));
        // Vectores de largo distinto → DimensionMismatch.
        perfil.temperature_k.pop();
        assert!(matches!(
            integrate_zwd(&perfil, 0.0).unwrap_err(),
            InsarError::DimensionMismatch(_)
        ));
    }

    // ---------- interpolate_profile / A-8 (interpolación a superficie) ----------

    #[test]
    fn interpolate_profile_reproduce_nivel_nativo_en_el_borde() {
        let perfil = perfil_humedo(0.012);
        let (p, t, q) = interpolate_profile(&perfil, 0.0).unwrap();
        assert!((p - perfil.pressure_hpa[0]).abs() < 1e-9);
        assert!((t - perfil.temperature_k[0]).abs() < 1e-9);
        assert!((q - perfil.specific_humidity[0]).abs() < 1e-9);
    }

    #[test]
    fn interpolate_profile_es_monotono_entre_niveles() {
        let perfil = perfil_humedo(0.012);
        let (_, t0, q0) = interpolate_profile(&perfil, 0.0).unwrap();
        let (_, t500, q500) = interpolate_profile(&perfil, 500.0).unwrap();
        let (_, t1000, q1000) = interpolate_profile(&perfil, 1000.0).unwrap();
        assert!(t0 > t500 && t500 > t1000, "temperatura debe decrecer con la altura");
        assert!(q0 > q500 && q500 > q1000, "humedad debe decrecer con la altura");
    }

    #[test]
    fn interpolate_profile_extrapola_fuera_de_rango_sin_error() {
        let perfil = perfil_humedo(0.012);
        let (p, _, _) = interpolate_profile(&perfil, 20000.0).unwrap();
        assert!(p > 0.0 && p.is_finite(), "extrapolación log-lineal debe dar presión positiva finita");
    }

    #[test]
    fn interpolate_profile_menos_de_dos_niveles_es_error() {
        let perfil = AtmosphericProfile {
            pressure_hpa: vec![1000.0],
            temperature_k: vec![290.0],
            specific_humidity: vec![0.01],
            height_m: vec![0.0],
        };
        assert!(matches!(interpolate_profile(&perfil, 0.0).unwrap_err(), InsarError::Inversion(_)));
    }

    /// Regresión A-8: antes de interpolar, `integrate_zwd(surface=500)`
    /// descartaba toda la rebanada `[0, 1000]` (el filtro exigía
    /// `height_m[i] >= surface_height_m`, así que con niveles nativos en
    /// 0/1000/2000m una superficie a 500m arrancaba igual que una a 1000m) —
    /// exactamente el mismo resultado que `surface=1000`, perdiendo la mitad
    /// más húmeda de la capa. Tras interpolar, debe ser estrictamente mayor.
    #[test]
    fn zwd_superficie_intermedia_no_colapsa_al_nivel_superior() {
        let perfil = perfil_humedo(0.012);
        let zwd_500 = integrate_zwd(&perfil, 500.0).unwrap();
        let zwd_1000 = integrate_zwd(&perfil, 1000.0).unwrap();
        assert!(
            zwd_500 > zwd_1000,
            "zwd(500)={zwd_500} debería ser mayor que zwd(1000)={zwd_1000} (fix A-8)"
        );
    }

    #[test]
    fn zenith_delay_combina_hidrostatico_y_humedo() {
        let perfil = perfil_humedo(0.010);
        let total = zenith_delay(&perfil, -33.0, 0.0).unwrap();
        let zwd = integrate_zwd(&perfil, 0.0).unwrap();
        let zhd = saastamoinen_zhd(perfil.pressure_hpa[0], -33.0, 0.0);
        assert!((total - (zhd + zwd)).abs() < 1e-9);
        // Rango físico típico de latitudes medias: 2.0-2.8 m total.
        assert!((2.0..2.8).contains(&total), "retardo total {total} implausible");
    }

    #[test]
    fn project_to_los_identidad_en_nadir() {
        let d = project_to_los(2.3, 0.0).unwrap();
        assert!((d - 2.3).abs() < 1e-9);
    }

    #[test]
    fn project_to_los_crece_con_incidencia() {
        let d0 = project_to_los(2.3, 0.0).unwrap();
        let d30 = project_to_los(2.3, 30.0).unwrap();
        let d40 = project_to_los(2.3, 39.0).unwrap();
        assert!(d0 < d30 && d30 < d40);
        // A 30°, 1/cos(30°) ≈ 1.1547.
        assert!((d30 - 2.3 * 1.1547).abs() < 1e-3);
    }

    #[test]
    fn project_to_los_incidencia_invalida_es_error() {
        assert!(project_to_los(2.3, 90.0).is_err());
        assert!(project_to_los(2.3, -1.0).is_err());
        assert!(project_to_los(2.3, 95.0).is_err());
    }

    fn serie_constante(n_epochs: usize, rows: usize, cols: usize, valor: f32) -> DisplacementSeries {
        use crate::types::{Epoch, StackMeta};
        use chrono::NaiveDate;
        DisplacementSeries {
            data: Array3::from_elem((n_epochs, rows, cols), valor),
            epochs: (0..n_epochs)
                .map(|i| Epoch(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap() + chrono::Duration::days(i as i64 * 12)))
                .collect(),
            meta: StackMeta {
                transform: surtgis_core::GeoTransform::default(),
                crs: None,
                wavelength_m: crate::types::SENTINEL1_WAVELENGTH_M,
                incidence_deg: 39.0,
                heading_deg: None,
            },
        }
    }

    /// Regresión C-1: la corrección debe SUMAR la diferencia de retardo, no
    /// restarla — restar duplicaba el error atmosférico en vez de removerlo
    /// (ver doc de [`correct_era5_series`] y la verificación cruzada contra
    /// MintPy `tropo_pyaps3.py`/`diff.py`).
    #[test]
    fn correct_era5_suma_diferencia_respecto_a_referencia() {
        let mut series = serie_constante(3, 2, 2, 0.0);
        // Retardo LOS constante en espacio, creciente en el tiempo: 0, 0.02, 0.05 m.
        let mut delay = Array3::<f32>::zeros((3, 2, 2));
        delay.index_axis_mut(Axis(0), 1).fill(0.02);
        delay.index_axis_mut(Axis(0), 2).fill(0.05);

        correct_era5_series(&mut series, &delay, 0).unwrap();

        for r in 0..2 {
            for c in 0..2 {
                assert!((series.data[[0, r, c]] - 0.0).abs() < 1e-6);
                assert!((series.data[[1, r, c]] - 0.02).abs() < 1e-6);
                assert!((series.data[[2, r, c]] - 0.05).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn correct_era5_respeta_nan_sin_contaminar() {
        let mut series = serie_constante(2, 1, 2, 1.0);
        let mut delay = Array3::<f32>::zeros((2, 1, 2));
        delay[[1, 0, 0]] = 0.03; // píxel 0: corrección normal
        delay[[1, 0, 1]] = f32::NAN; // píxel 1: ERA5 sin dato → no tocar

        correct_era5_series(&mut series, &delay, 0).unwrap();

        assert!((series.data[[1, 0, 0]] - (1.0 + 0.03)).abs() < 1e-6);
        assert!((series.data[[1, 0, 1]] - 1.0).abs() < 1e-6, "píxel NaN no debería cambiar");
    }

    #[test]
    fn correct_era5_dimension_mismatch_es_error() {
        let mut series = serie_constante(2, 2, 2, 0.0);
        let delay = Array3::<f32>::zeros((2, 3, 2));
        assert!(matches!(
            correct_era5_series(&mut series, &delay, 0).unwrap_err(),
            InsarError::DimensionMismatch(_)
        ));
    }

    #[test]
    fn correct_era5_referencia_fuera_de_rango_es_error() {
        let mut series = serie_constante(2, 2, 2, 0.0);
        let delay = Array3::<f32>::zeros((2, 2, 2));
        assert!(matches!(
            correct_era5_series(&mut series, &delay, 5).unwrap_err(),
            InsarError::Metadata(_)
        ));
    }

    /// Regresión C-1 (end-to-end): construye una serie "contaminada" por el
    /// mismo modelo físico que predice cómo el retardo atmosférico sesga la
    /// serie invertida (`d_obs(e) = d_true(e) − (L_atmo(e) − L_atmo(ref))`,
    /// ver doc de [`correct_era5_series`]), usando retardos reales calculados
    /// vía [`zenith_delay`]/[`project_to_los`] sobre perfiles con humedad
    /// creciente en el tiempo. Con el signo del bug (`-=`), la corrección
    /// habría duplicado el sesgo en vez de removerlo; con el signo correcto
    /// (`+=`) recupera la deformación verdadera. Es más fuerte que un test de
    /// retardo constante: usa retardos físicamente derivados y no solo
    /// re-verifica el propio signo bajo prueba de forma aislada.
    #[test]
    fn correct_era5_recupera_deformacion_verdadera_end_to_end() {
        let true_trend = [0.0_f32, -0.01, -0.02]; // subsidencia lineal, m
        let humidities = [0.0, 0.006, 0.012]; // seco → húmedo
        let incidence_deg = 39.0;

        let los_delay: Vec<f32> = humidities
            .iter()
            .map(|&q| {
                let perfil = perfil_humedo(q);
                let zenith = zenith_delay(&perfil, -33.0, 0.0).unwrap();
                project_to_los(zenith, incidence_deg).unwrap() as f32
            })
            .collect();

        let mut series = serie_constante(3, 1, 1, 0.0);
        let ref_delay = los_delay[0];
        for (e, &target) in true_trend.iter().enumerate() {
            series.data[[e, 0, 0]] = target - (los_delay[e] - ref_delay);
        }

        let delay_arr = Array3::from_shape_fn((3, 1, 1), |(e, _, _)| los_delay[e]);
        correct_era5_series(&mut series, &delay_arr, 0).unwrap();

        for (e, &target) in true_trend.iter().enumerate() {
            assert!(
                (series.data[[e, 0, 0]] - target).abs() < 1e-4,
                "época {e}: corregido {} vs verdadero {target}",
                series.data[[e, 0, 0]],
            );
        }
    }
}

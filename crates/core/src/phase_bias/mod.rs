//! Corrección del **sesgo de fase de no-cierre** (*phase bias*, "fading
//! signal") — Maghsoudi et al. (2022), RSE 275:113022.
//!
//! # Qué corrige, y por qué no lo hace [`crate::unwrap_error`]
//!
//! El multilooking rompe la consistencia de fase entre tripletes: la fase del
//! interferograma largo no iguala la suma de los cortos que cubren el mismo
//! lapso. Ese residuo es la **fase de no-cierre**, y su parte sistemática es un
//! **sesgo** que crece cuanto más corto es el par y cuanto más decorrelaciona
//! el terreno (cultivo y bosque mucho más que zona urbana).
//!
//! [`crate::unwrap_error`] también usa cierres, pero resuelve el problema
//! **ortogonal**: saltos de ciclo **enteros** (`φ' = φ − 2π·U`, Yunjun et al.
//! 2019). El phase bias es la parte **fraccional** del cierre — exactamente lo
//! que aquel módulo redondea a cero y descarta por construcción. Los dos deben
//! corregirse, y en puntos distintos del pipeline:
//!
//! | | `phase_bias` | `unwrap_error` |
//! |---|---|---|
//! | Corrige | sesgo fraccional | salto entero de 2π |
//! | Opera sobre | [`IfgStack`] envuelto | [`crate::types::UnwrappedStack`] |
//! | Momento | **antes** de desenrollar | después de desenrollar |
//!
//! Que el sesgo se estime sobre fase **envuelta** es lo que hace la corrección
//! barata: no requiere desenrollar, ni *phase linking*, ni interferogramas
//! largos coherentes (salvo el ancla de §Coeficientes).
//!
//! # El método
//!
//! Con la red que conecta cada época a sus M vecinas temporales, la fase de
//! cierre de la cadena que arranca en la época `i` y abarca `n` intervalos es
//! (Eqs. 2–3 del paper, aquí generalizadas a `n` arbitrario):
//!
//! ```text
//! Δφ_{i,i+n} = φ_{i,i+n} − Σ_{k=0}^{n−1} φ_{i+k,i+k+1}
//! ```
//!
//! Atribuyéndola a sesgo más ruido, con `δ` el sesgo desconocido de cada
//! interferograma (Eqs. 4–5):
//!
//! ```text
//! Δφ_{i,i+n} = δ_{i,i+n} − Σ_{k=0}^{n−1} δ_{i+k,i+k+1} + ε
//! ```
//!
//! Con N épocas eso da 2N−5 observaciones y 3N−6 incógnitas: **indeterminado**.
//! La regularización central del paper (Eqs. 6–7) es que el sesgo del par largo
//! es una **razón constante** `aₙ` de la suma de los sesgos de los cortos que
//! cubren su mismo lapso:
//!
//! ```text
//! δ_{i,i+n} = aₙ · Σ_{k=0}^{n−1} δ_{i+k,i+k+1}
//! ```
//!
//! lo que deja solo N−1 incógnitas (los sesgos de span unitario) y vuelve el
//! sistema **sobredeterminado**, resoluble por mínimos cuadrados (Eq. 10):
//!
//! ```text
//! Δφ_{i,i+n} = (aₙ − 1) · Σ_{k=0}^{n−1} δ_{i+k,i+k+1} + ε
//! ```
//!
//! y cada interferograma se corrige con `φᶜ = φ − aₙ·Σδ̂` (Eqs. 11–13, con
//! `a₁ ≡ 1` para el span unitario).
//!
//! # `max_span` ≥ 3 no es un default, es un requisito
//!
//! Con `max_span = 2` el sistema **nunca** queda determinado: hay N−2
//! observaciones (una por cada cadena de span 2) para N−1 incógnitas, sea cual
//! sea N. Hacen falta al menos **dos longitudes de par largo distintas** — los
//! spans 2 y 3 del paper — para cerrar el sistema: 2N−5 observaciones contra
//! N−1 incógnitas, sobredeterminado desde N ≥ 5 (exactamente determinado en
//! N = 4). Una red que solo tenga pares consecutivos y saltos de 2 no puede
//! estimar el sesgo, y [`correct_phase_bias`] lo dice con un error en vez de
//! devolver un stack sin tocar.
//!
//! # Indexación por época, no por días
//!
//! El paper está escrito para muestreo regular de 6 días y span máximo 3. Aquí
//! el span `n` es la **diferencia de índice de época**, que es la
//! generalización correcta a muestreo irregular (y coincide con lo que el paper
//! hace de facto: "interferograms that connect each epoch i to the three or
//! four nearest acquisitions in time"). Con muestreo regular, span n = n·Δt.
//!
//! # Garantía de consistencia
//!
//! Tras corregir, el cierre de cada observación usada queda **exactamente en el
//! residuo del ajuste** de mínimos cuadrados:
//!
//! ```text
//! Δφᶜ = (φ − aₙΣδ̂) − Σ(φ_unit − δ̂) = Δφ − (aₙ−1)Σδ̂
//! ```
//!
//! o sea que el RMS de cierre no puede subir en los píxeles corregidos. Es la
//! propiedad que verifica [`PhaseBiasReport::closure_rms_before`] /
//! [`PhaseBiasReport::closure_rms_after`], y el test `el_cierre_baja_tras_corregir`.

use std::collections::{HashMap, HashSet};

use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, Array3, Axis};
use num_complex::Complex32;
use rayon::prelude::*;

use crate::error::{InsarError, Result};
use crate::inversion::rcond_pseudo_inverse;
use crate::types::IfgStack;

/// Span máximo por defecto (M = 3): la red LiCSAR conecta cada época a sus 3–4
/// vecinas, y es la configuración para la que el paper publica a₁ y a₂.
pub const DEFAULT_MAX_SPAN: usize = 3;

/// Lapso objetivo del par ancla para estimar los coeficientes, en días. El
/// paper usa 360 (§4, Eqs. 8–9): elegido porque es divisible por 6, 12 y 18, y
/// porque a un año el sesgo ya es despreciable.
pub const DEFAULT_ANCHOR_DAYS: f64 = 360.0;

/// Configuración de [`correct_phase_bias`].
#[derive(Debug, Clone)]
pub struct PhaseBiasConfig {
    /// Span máximo M (en índices de época) que se modela y corrige. Los pares
    /// con span > M se dejan intactos: el modelo no tiene `aₙ` para ellos, y
    /// según el propio paper su sesgo es despreciable.
    pub max_span: usize,
    /// Coeficientes `a₂..a_M`. `None` → se estiman de los datos con
    /// [`estimate_coefficients`]. El paper reporta a₁=0,47 y a₂=0,31 para su
    /// zona de estudio en Turquía, pero son **específicos de escena** (dependen
    /// de la cobertura del suelo): reusarlos a ciegas en otro sitio no está
    /// justificado, por eso el default es estimar.
    pub coefficients: Option<Vec<f64>>,
    /// Lapso objetivo del par ancla en días, si hay que estimar coeficientes.
    pub anchor_days: f64,
    /// Máscara de píxeles admisibles para estimar los coeficientes (típicamente
    /// coherencia ≥ 0,3 promediada sobre los pares largos, como en el paper).
    /// `None` = todos los píxeles con fase válida.
    pub coefficient_mask: Option<Array2<bool>>,
    /// Mínimo de píxeles válidos para aceptar una estimación de coeficientes.
    pub min_coefficient_pixels: usize,
    /// Cierre acumulado mínimo de la cadena unitaria sobre el lapso del ancla,
    /// en radianes por píxel, para considerar que la escena tiene sesgo
    /// medible. Es el **denominador** de las Eqs. 8–9: por debajo de esto los
    /// `aₙ` son un cociente de ruidos. El default (0,05 rad ≈ 0,2 mm LOS
    /// acumulados en un año) está muy por debajo de los sesgos reales
    /// reportados —orden de radianes en cultivo y bosque— así que solo rechaza
    /// el caso degenerado.
    pub min_cumulative_closure_rad: f64,
}

impl Default for PhaseBiasConfig {
    fn default() -> Self {
        Self {
            max_span: DEFAULT_MAX_SPAN,
            coefficients: None,
            anchor_days: DEFAULT_ANCHOR_DAYS,
            coefficient_mask: None,
            min_coefficient_pixels: 100,
            min_cumulative_closure_rad: 0.05,
        }
    }
}

/// Coeficientes `aₙ` estimados de los datos.
#[derive(Debug, Clone)]
pub struct CoefficientEstimate {
    /// `a₂..a_M`, en orden de span creciente (índice `n−2` ↔ span `n`).
    pub coefficients: Vec<f64>,
    /// Span del par ancla usado, en índices de época.
    pub anchor_span: usize,
    /// Lapso real del ancla en días (informativo: puede no ser exactamente
    /// `anchor_days` si el muestreo es irregular).
    pub anchor_days: f64,
    /// Nº de anclas (pares largos) que contribuyeron.
    pub n_anchors: usize,
    /// Nº de píxeles válidos acumulados sobre todas las anclas.
    pub n_pixels: usize,
    /// Cierre acumulado de la cadena unitaria, `Σ Δφ_{L−1}` — el denominador de
    /// las Eqs. 8–9. Es la magnitud total del sesgo que se está midiendo: si es
    /// ~0, la escena no tiene sesgo detectable y los `aₙ` son ruido dividido
    /// por ruido.
    pub cumulative_unit_closure: f64,
}

/// Resultado de [`correct_phase_bias`].
#[derive(Debug, Clone)]
pub struct PhaseBiasReport {
    /// Coeficientes efectivamente aplicados (`a₂..a_M`).
    pub coefficients: Vec<f64>,
    /// Detalle de la estimación, si los coeficientes no venían dados.
    pub estimate: Option<CoefficientEstimate>,
    /// Píxeles donde el sistema de cierres tenía rango completo y se aplicó
    /// corrección.
    pub pixels_corrected: usize,
    /// Píxeles sin corregir: sin observaciones de cierre suficientes (fase no
    /// válida en demasiados pares, o red localmente sin redundancia). Quedan
    /// **intactos**, no a NaN.
    pub pixels_skipped: usize,
    /// Capas del stack que el modelo no cubre (span > `max_span`, o cadena
    /// unitaria incompleta) y quedaron sin corregir.
    pub layers_uncovered: usize,
    /// RMS de la fase de cierre antes de corregir, en radianes.
    pub closure_rms_before: f64,
    /// RMS de la fase de cierre después de corregir, en radianes. Debe bajar:
    /// es el residuo del ajuste (ver doc del módulo).
    pub closure_rms_after: f64,
}

/// Topología de cierres de la red: qué observaciones existen y qué incógnitas
/// tocan. Depende solo de los pares, no de los valores — se arma una vez.
#[derive(Debug, Clone)]
struct ClosureTopology {
    /// Columna de incógnita asignada a cada slot unitario (los slots sin par
    /// observado no tienen columna).
    unit_col: Vec<Option<usize>>,
    /// Nº de incógnitas = slots unitarios observados.
    n_unknowns: usize,
    /// Observaciones de cierre.
    obs: Vec<ClosureObs>,
}

/// Una observación de cierre: la cadena de `span` intervalos que arranca en la
/// época `start`.
#[derive(Debug, Clone)]
struct ClosureObs {
    span: usize,
    /// Capa del par largo (start, start+span).
    long_layer: usize,
    /// Capas de los pares unitarios de la cadena, en orden.
    unit_layers: Vec<usize>,
    /// Columnas de incógnita que toca (una por par unitario).
    cols: Vec<usize>,
}

/// Fase envuelta del cierre de una observación en un píxel, o `None` si alguna
/// de las capas involucradas no es válida.
///
/// Se calcula como el argumento del producto complejo — **no** restando fases
/// desenrolladas — que es lo que hace el sesgo estimable sin desenrollar. Cada
/// factor se normaliza a módulo 1 antes de multiplicar: solo importa el
/// argumento, y con cadenas de 4 factores de amplitud SLC sin normalizar el
/// producto desborda `f32` con facilidad.
fn closure_at(obs: &ClosureObs, read: impl Fn(usize) -> Complex32) -> Option<f64> {
    let unit = |z: Complex32| -> Option<Complex32> {
        let norm = z.norm();
        (norm.is_finite() && norm > 0.0).then(|| z / norm)
    };
    let mut acc = unit(read(obs.long_layer))?;
    for &k in &obs.unit_layers {
        acc *= unit(read(k))?.conj();
    }
    Some(acc.arg() as f64)
}

impl ClosureTopology {
    /// Arma la topología de cierres hasta `max_span`.
    ///
    /// Una observación `(i, n)` existe solo si están **todos** sus ingredientes:
    /// el par largo `(i, i+n)` y los `n` pares unitarios de la cadena. Un par
    /// unitario faltante no solo elimina su incógnita: invalida toda
    /// observación que lo cruce (la Eq. 2 necesita su fase para formar el
    /// cierre).
    fn build(stack: &IfgStack, max_span: usize) -> Self {
        let n_epochs = stack.epochs.len();
        let index: HashMap<(usize, usize), usize> = stack
            .pairs
            .iter()
            .enumerate()
            .map(|(k, p)| ((p.reference, p.secondary), k))
            .collect();

        let n_slots = n_epochs.saturating_sub(1);
        let unit_layer: Vec<Option<usize>> =
            (0..n_slots).map(|i| index.get(&(i, i + 1)).copied()).collect();

        let mut unit_col = vec![None; n_slots];
        let mut n_unknowns = 0;
        for (i, layer) in unit_layer.iter().enumerate() {
            if layer.is_some() {
                unit_col[i] = Some(n_unknowns);
                n_unknowns += 1;
            }
        }

        let mut obs = Vec::new();
        for span in 2..=max_span {
            for start in 0..n_slots.saturating_sub(span - 1) {
                let Some(&long_layer) = index.get(&(start, start + span)) else {
                    continue;
                };
                // La cadena unitaria completa debe existir.
                let chain: Option<Vec<(usize, usize)>> = (start..start + span)
                    .map(|i| unit_layer[i].zip(unit_col[i]))
                    .collect();
                let Some(chain) = chain else { continue };
                let (unit_layers, cols) = chain.into_iter().unzip();
                obs.push(ClosureObs { span, long_layer, unit_layers, cols });
            }
        }

        ClosureTopology { unit_col, n_unknowns, obs }
    }

    /// Matriz de diseño (n_obs × n_unknowns): `(aₙ − 1)` en las columnas de la
    /// cadena. Es la Eq. (10) generalizada.
    fn design(&self, coefficients: &[f64]) -> DMatrix<f64> {
        let mut a = DMatrix::<f64>::zeros(self.obs.len(), self.n_unknowns);
        for (o, ob) in self.obs.iter().enumerate() {
            let coef = coefficients[ob.span - 2] - 1.0;
            for &c in &ob.cols {
                a[(o, c)] = coef;
            }
        }
        a
    }
}

/// Valida `max_span` y la longitud del vector de coeficientes.
fn check_span(max_span: usize) -> Result<()> {
    if max_span < 2 {
        return Err(InsarError::InvalidNetwork(format!(
            "max_span = {max_span}: se necesita al menos 2 para que exista una \
             fase de cierre (un par de span 1 no cierra contra nada)"
        )));
    }
    Ok(())
}

/// RMS de la fase de cierre del stack hasta `max_span`, en radianes, y el nº de
/// observaciones (píxel × cierre) que entraron.
///
/// Es el producto de QC del sesgo: se calcula antes y después de
/// [`correct_phase_bias`] y debe bajar. `Ok((f64::NAN, 0))` si la red no tiene
/// ningún cierre evaluable.
pub fn closure_rms(stack: &IfgStack, max_span: usize) -> Result<(f64, usize)> {
    stack.validate()?;
    check_span(max_span)?;
    let topo = ClosureTopology::build(stack, max_span);
    if topo.obs.is_empty() {
        return Ok((f64::NAN, 0));
    }

    let (n_rows, n_cols) = stack.dims();
    let data = stack.data.view();
    let (sum_sq, count) = (0..n_rows)
        .into_par_iter()
        .map(|r| {
            let mut acc = (0.0_f64, 0usize);
            for c in 0..n_cols {
                for ob in &topo.obs {
                    if let Some(v) = closure_at(ob, |k| data[[k, r, c]]) {
                        acc.0 += v * v;
                        acc.1 += 1;
                    }
                }
            }
            acc
        })
        .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

    if count == 0 {
        return Ok((f64::NAN, 0));
    }
    Ok(((sum_sq / count as f64).sqrt(), count))
}

/// Mapa por píxel del RMS de la fase de cierre hasta `max_span`, en radianes,
/// más el nº de cierres que entró en cada píxel.
///
/// Es [`closure_rms`] desagregado espacialmente: el estadístico global es la
/// media cuadrática de este mapa ponderada por el conteo. Calculado antes y
/// después de [`correct_phase_bias`] muestra **dónde** muerde el sesgo
/// (cultivo/bosque decorrelacionado vs urbano coherente) sin pasar por el
/// desenrollado. NaN donde ningún cierre fue evaluable.
pub fn closure_rms_map(
    stack: &IfgStack,
    max_span: usize,
) -> Result<(Array2<f32>, Array2<u32>)> {
    stack.validate()?;
    check_span(max_span)?;
    let topo = ClosureTopology::build(stack, max_span);
    let (n_rows, n_cols) = stack.dims();
    let data = stack.data.view();

    let mut rms = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut count = Array2::<u32>::zeros((n_rows, n_cols));
    let mut rms_rows: Vec<_> = rms.axis_iter_mut(Axis(0)).collect();
    let mut count_rows: Vec<_> = count.axis_iter_mut(Axis(0)).collect();
    rms_rows
        .par_iter_mut()
        .zip(count_rows.par_iter_mut())
        .enumerate()
        .for_each(|(r, (rms_row, count_row))| {
            for c in 0..n_cols {
                let (mut sum_sq, mut n) = (0.0_f64, 0u32);
                for ob in &topo.obs {
                    if let Some(v) = closure_at(ob, |k| data[[k, r, c]]) {
                        sum_sq += v * v;
                        n += 1;
                    }
                }
                if n > 0 {
                    rms_row[c] = (sum_sq / n as f64).sqrt() as f32;
                    count_row[c] = n;
                }
            }
        });
    Ok((rms, count))
}

/// Estima los coeficientes `a₂..a_M` de los datos (Eqs. 8–9 del paper).
///
/// Elige como **ancla** los pares cuyo lapso se acerca a `anchor_days` y que
/// pueden teselarse exactamente con cadenas de span `1..=max_span` (el paper
/// usa 360 días justamente porque 360 = 60·6 = 30·12 = 20·18). Asumiendo que el
/// ancla tiene sesgo despreciable, su cierre contra la cadena de span `n` mide
/// el sesgo acumulado de esa cadena, y
///
/// ```text
/// aₙ = Σ Δφ_{L−n} / Σ Δφ_{L−1}
/// ```
///
/// **Desviación deliberada del paper:** este es un cociente de sumas, no la
/// media de los cocientes por píxel que calcula el paper. Los mapas de `aₙ` por
/// píxel son ruidosos (el propio paper lo dice, Fig. 8) y su cociente explota
/// donde el denominador cruza cero, de modo que la media muestral no es
/// estimador estable de la razón. El cociente de acumulados es la lectura
/// literal de "cumulative loop closure phases" y coincide con la estimación de
/// mínimos cuadrados de la constante de proporcionalidad. El paper concluye lo
/// mismo que motiva esto: los `aₙ` no tienen patrón espacial, un escalar por
/// escena alcanza.
pub fn estimate_coefficients(
    stack: &IfgStack,
    config: &PhaseBiasConfig,
) -> Result<CoefficientEstimate> {
    stack.validate()?;
    check_span(config.max_span)?;
    let max_span = config.max_span;

    if let Some(mask) = &config.coefficient_mask
        && mask.dim() != stack.dims()
    {
        return Err(InsarError::DimensionMismatch(format!(
            "máscara de coeficientes {:?} vs grilla del stack {:?}",
            mask.dim(),
            stack.dims()
        )));
    }

    let n_epochs = stack.epochs.len();
    let index: HashMap<(usize, usize), usize> = stack
        .pairs
        .iter()
        .enumerate()
        .map(|(k, p)| ((p.reference, p.secondary), k))
        .collect();

    // El span del ancla debe ser divisible por todo n ≤ max_span para que las
    // cadenas teselen exactamente.
    let step = (1..=max_span).fold(1usize, lcm);

    // Candidatos: pares (p, p+K) con K múltiplo de `step`, ordenados por
    // cercanía a `anchor_days`. Se toma el mejor K y se usan TODAS las anclas
    // de ese span (más anclas = más cierre acumulado = estimación más estable).
    let mut anchors_by_span: HashMap<usize, Vec<usize>> = HashMap::new();
    for (k, p) in stack.pairs.iter().enumerate() {
        let span = p.secondary - p.reference;
        if span >= 2 * step && span % step == 0 {
            anchors_by_span.entry(span).or_default().push(k);
        }
    }
    if anchors_by_span.is_empty() {
        return Err(InsarError::InvalidNetwork(format!(
            "no hay par ancla para estimar los coeficientes: se necesita al \
             menos un par cuyo span en épocas sea múltiplo de {step} (mcm de \
             1..={max_span}) y ≥ {}. Densifica la red con un par largo, o pasa \
             coeficientes explícitos en PhaseBiasConfig::coefficients",
            2 * step
        )));
    }

    let days_of = |span: usize| -> f64 {
        // Lapso mediano en días de un salto de `span` épocas, a título
        // informativo para elegir el ancla más cercana al objetivo.
        let mut d: Vec<f64> = (0..n_epochs.saturating_sub(span))
            .map(|i| stack.epochs[i + span].days_since(&stack.epochs[i]) as f64)
            .collect();
        if d.is_empty() {
            return f64::NAN;
        }
        d.sort_by(f64::total_cmp);
        d[d.len() / 2]
    };

    let (anchor_span, anchor_layers) = anchors_by_span
        .into_iter()
        .min_by(|(sa, _), (sb, _)| {
            let da = (days_of(*sa) - config.anchor_days).abs();
            let db = (days_of(*sb) - config.anchor_days).abs();
            da.total_cmp(&db).then(sa.cmp(sb))
        })
        .expect("anchors_by_span no vacío: comprobado arriba");

    // Para cada ancla, la cadena de span n que la tesela (o nada si falta algún
    // eslabón). `chains[a][n-1]` = capas de la cadena de span n del ancla a.
    let mut anchor_obs: Vec<(usize, Vec<Vec<usize>>)> = Vec::new();
    for &k in &anchor_layers {
        let start = stack.pairs[k].reference;
        let chains: Option<Vec<Vec<usize>>> = (1..=max_span)
            .map(|n| {
                (0..anchor_span / n)
                    .map(|j| index.get(&(start + j * n, start + (j + 1) * n)).copied())
                    .collect::<Option<Vec<usize>>>()
            })
            .collect();
        if let Some(chains) = chains {
            anchor_obs.push((k, chains));
        }
    }
    if anchor_obs.is_empty() {
        return Err(InsarError::InvalidNetwork(format!(
            "hay pares ancla de span {anchor_span} épocas pero ninguno se puede \
             teselar con cadenas completas de span 1..={max_span}: falta algún \
             eslabón de la red. Pasa coeficientes explícitos o completa la red"
        )));
    }

    // Cierre acumulado por span: Σ_píxeles Σ_anclas Δφ_{L−n}. Un píxel entra
    // solo si TODAS las cadenas y el ancla son válidas ahí — si no, los
    // numeradores y el denominador se calcularían sobre poblaciones distintas
    // de píxeles y su cociente no sería el aₙ de nadie.
    let (n_rows, n_cols) = stack.dims();
    let data = stack.data.view();
    let mask = config.coefficient_mask.as_ref();

    let (sums, n_pixels) = (0..n_rows)
        .into_par_iter()
        .map(|r| {
            let mut sums = vec![0.0_f64; max_span];
            let mut n = 0usize;
            let mut per_pixel = vec![0.0_f64; max_span];
            for c in 0..n_cols {
                if mask.is_some_and(|m| !m[[r, c]]) {
                    continue;
                }
                let mut ok = true;
                per_pixel.iter_mut().for_each(|v| *v = 0.0);
                'anchors: for (k, chains) in &anchor_obs {
                    for (n_idx, chain) in chains.iter().enumerate() {
                        let ob = ClosureObs {
                            span: n_idx + 1,
                            long_layer: *k,
                            unit_layers: chain.clone(),
                            cols: Vec::new(),
                        };
                        match closure_at(&ob, |l| data[[l, r, c]]) {
                            Some(v) => per_pixel[n_idx] += v,
                            None => {
                                ok = false;
                                break 'anchors;
                            }
                        }
                    }
                }
                if ok {
                    for (s, v) in sums.iter_mut().zip(&per_pixel) {
                        *s += v;
                    }
                    n += 1;
                }
            }
            (sums, n)
        })
        .reduce(
            || (vec![0.0; max_span], 0),
            |mut a, b| {
                for (x, y) in a.0.iter_mut().zip(&b.0) {
                    *x += y;
                }
                (a.0, a.1 + b.1)
            },
        );

    if n_pixels < config.min_coefficient_pixels {
        return Err(InsarError::Inversion(format!(
            "solo {n_pixels} píxeles válidos para estimar los coeficientes \
             (mínimo {}): el par ancla de {anchor_span} épocas no mantiene \
             coherencia en casi ningún lado. Usa un ancla más corta \
             (anchor_days), relaja la máscara, o pasa coeficientes explícitos",
            config.min_coefficient_pixels
        )));
    }

    let denom = sums[0];
    // El denominador es el sesgo acumulado de la cadena unitaria. Si es ~0 no
    // hay sesgo que medir y aₙ = ruido/ruido: devolverlo sería inventar una
    // corrección a partir de nada. El umbral es POR PÍXEL Y ANCLA, en
    // radianes — una escala física fija. Compararlo contra una escala derivada
    // de las propias sumas sería circular (el denominador suele ser la mayor
    // de ellas, así que la prueba pasaría siempre).
    let per_pixel_closure = denom.abs() / (n_pixels * anchor_obs.len()) as f64;
    if !denom.is_finite() || per_pixel_closure < config.min_cumulative_closure_rad {
        return Err(InsarError::Inversion(format!(
            "cierre acumulado de la cadena unitaria ≈ 0 ({per_pixel_closure:.3e} \
             rad por píxel sobre {n_pixels} píxeles y {} anclas, mínimo {:.3}): \
             esta escena no muestra sesgo de no-cierre detectable a un año, así \
             que los coeficientes serían un cociente de ruidos. No hay nada que \
             corregir — o el ancla es demasiado corta para acumular el sesgo",
            anchor_obs.len(),
            config.min_cumulative_closure_rad
        )));
    }

    let coefficients: Vec<f64> = sums[1..].iter().map(|s| s / denom).collect();
    for (i, a) in coefficients.iter().enumerate() {
        if !a.is_finite() || a.abs() > 2.0 {
            return Err(InsarError::Inversion(format!(
                "coeficiente a{} = {a:.3} fuera de rango físico: el modelo dice \
                 que el sesgo DECAE con el largo del par, o sea 0 < aₙ < 1 \
                 (el paper reporta 0,47 y 0,31). Un valor así indica que la \
                 hipótesis de razón constante no se sostiene en esta escena — \
                 revisa la red y la coherencia antes de corregir",
                i + 1
            )));
        }
    }

    Ok(CoefficientEstimate {
        coefficients,
        anchor_span,
        anchor_days: days_of(anchor_span),
        n_anchors: anchor_obs.len(),
        n_pixels,
        cumulative_unit_closure: denom,
    })
}

/// Mínimo común múltiplo (para el span del ancla).
fn lcm(a: usize, b: usize) -> usize {
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    a / gcd(a, b) * b
}

/// Corrige el sesgo de fase de no-cierre del stack **envuelto**, in situ.
///
/// Estima por píxel los sesgos de span unitario `δ̂` resolviendo por mínimos
/// cuadrados el sistema de cierres (Eq. 10) y aplica `φᶜ = φ − aₙ·Σδ̂`
/// (Eqs. 11–13) a cada par de span ≤ `max_span`. Sobre datos complejos eso es
/// una rotación: `c ← c · exp(−i·corrección)`, que preserva la amplitud.
///
/// Píxeles cuyo sistema reducido (por fases no válidas) queda rank-deficiente
/// se dejan **intactos** y se cuentan en [`PhaseBiasReport::pixels_skipped`] —
/// no se les aplica la solución de norma mínima, que repartiría el cierre en
/// fracciones arbitrarias entre incógnitas que las observaciones no restringen.
pub fn correct_phase_bias(
    stack: &mut IfgStack,
    config: &PhaseBiasConfig,
) -> Result<PhaseBiasReport> {
    stack.validate()?;
    check_span(config.max_span)?;
    let max_span = config.max_span;

    // 1. Coeficientes: dados o estimados.
    let (coefficients, estimate) = match &config.coefficients {
        Some(a) => {
            if a.len() != max_span - 1 {
                return Err(InsarError::Inversion(format!(
                    "{} coeficientes dados para max_span = {max_span}: se \
                     esperan {} (a₂..a_{max_span})",
                    a.len(),
                    max_span - 1
                )));
            }
            (a.clone(), None)
        }
        None => {
            let est = estimate_coefficients(stack, config)?;
            (est.coefficients.clone(), Some(est))
        }
    };

    let (closure_rms_before, _) = closure_rms(stack, max_span)?;

    // 2. Topología y matriz de diseño (una vez para toda la escena).
    let topo = ClosureTopology::build(stack, max_span);
    if topo.obs.is_empty() {
        return Err(InsarError::InvalidNetwork(format!(
            "la red no tiene ninguna fase de cierre con span ≤ {max_span}: hacen \
             falta pares largos Y la cadena unitaria completa que cubren. Nada \
             que estimar"
        )));
    }
    let n_obs = topo.obs.len();
    let design = topo.design(&coefficients);

    // 3. Pseudoinversa por patrón de observaciones activas. La matriz de diseño
    //    no depende del píxel; solo depende de QUÉ observaciones son válidas
    //    ahí, así que se cachea por máscara igual que en `invert_sbas`.
    let n_words = n_obs.div_ceil(64);
    let mask_bit = |key: &mut [u64], o: usize| key[o / 64] |= 1u64 << (o % 64);
    let (n_rows, n_cols) = stack.dims();

    let data_ro = stack.data.view();
    let unique_masks: HashSet<Vec<u64>> = (0..n_rows)
        .into_par_iter()
        .map(|r| {
            let mut set = HashSet::new();
            for c in 0..n_cols {
                let mut key = vec![0u64; n_words];
                for (o, ob) in topo.obs.iter().enumerate() {
                    if closure_at(ob, |k| data_ro[[k, r, c]]).is_some() {
                        mask_bit(&mut key, o);
                    }
                }
                set.insert(key);
            }
            set
        })
        .reduce(HashSet::new, |mut a, b| {
            a.extend(b);
            a
        });

    let solvers: HashMap<Vec<u64>, Option<DMatrix<f64>>> = unique_masks
        .into_par_iter()
        .map(|key| {
            let rows: Vec<usize> = (0..n_obs)
                .filter(|&o| key[o / 64] & (1u64 << (o % 64)) != 0)
                .collect();
            if rows.len() < topo.n_unknowns {
                return (key, None);
            }
            let reduced =
                DMatrix::from_fn(rows.len(), topo.n_unknowns, |i, j| design[(rows[i], j)]);
            let pinv = rcond_pseudo_inverse(reduced);
            (key, pinv)
        })
        .collect();

    // 4. Corrección por píxel. Para cada capa cubierta por el modelo, la
    //    corrección es `aₙ·Σδ̂` sobre las columnas de su cadena unitaria.
    //    `layer_plan[k] = Some((aₙ, cols))` para las capas corregibles.
    let layer_plan: Vec<Option<(f64, Vec<usize>)>> = stack
        .pairs
        .iter()
        .map(|p| {
            let span = p.secondary - p.reference;
            if span > max_span {
                return None;
            }
            // a₁ ≡ 1 para el span unitario (Eq. 11).
            let coef = if span == 1 { 1.0 } else { coefficients[span - 2] };
            let cols: Option<Vec<usize>> =
                (p.reference..p.secondary).map(|i| topo.unit_col[i]).collect();
            cols.map(|cols| (coef, cols))
        })
        .collect();
    let layers_uncovered = layer_plan.iter().filter(|p| p.is_none()).count();
    if layer_plan.iter().all(Option::is_none) {
        return Err(InsarError::InvalidNetwork(format!(
            "ninguna capa del stack es corregible con max_span = {max_span}: \
             todos los pares tienen span mayor, o su cadena unitaria está \
             incompleta. Sube max_span o completa la red"
        )));
    }

    let mut row_views: Vec<_> = stack.data.axis_iter_mut(Axis(1)).collect();
    let (pixels_corrected, pixels_skipped) = row_views
        .par_iter_mut()
        .map(|row| {
            let mut local = (0usize, 0usize);
            for c in 0..n_cols {
                // (a) Cierres activos del píxel y su máscara.
                let mut key = vec![0u64; n_words];
                let mut vals: Vec<f64> = Vec::with_capacity(n_obs);
                for (o, ob) in topo.obs.iter().enumerate() {
                    if let Some(v) = closure_at(ob, |k| row[[k, c]]) {
                        mask_bit(&mut key, o);
                        vals.push(v);
                    }
                }

                // (b) Solver del patrón. Rank-deficiente → píxel intacto.
                let Some(Some(pinv)) = solvers.get(&key) else {
                    local.1 += 1;
                    continue;
                };

                // (c) δ̂ = pinv · Δφ_activos.
                let delta = pinv * DVector::from_vec(vals);

                // (d) Aplica φᶜ = φ − aₙ·Σδ̂ como rotación compleja.
                for (k, plan) in layer_plan.iter().enumerate() {
                    let Some((coef, cols)) = plan else { continue };
                    let z = row[[k, c]];
                    if !z.is_finite() {
                        continue;
                    }
                    let corr: f64 = coef * cols.iter().map(|&j| delta[j]).sum::<f64>();
                    row[[k, c]] = z * Complex32::from_polar(1.0, -corr as f32);
                }
                local.0 += 1;
            }
            local
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

    // Cero píxeles corregidos = passthrough silencioso. El caso más común no
    // es "datos malos" sino una red que solo tiene spans 1 y 2, con la que el
    // sistema es indeterminado por construcción (ver doc del módulo): sin este
    // error el pipeline reportaría éxito habiendo dejado el stack intacto.
    if pixels_corrected == 0 {
        return Err(InsarError::Inversion(format!(
            "ningún píxel corregido ({pixels_skipped} saltados): el sistema de \
             cierres quedó rank-deficiente en toda la grilla. Con max_span = \
             {max_span} hay {n_obs} observaciones para {} incógnitas — si \
             max_span es 2, el sistema es indeterminado SIEMPRE y hace falta \
             añadir pares de span 3 a la red",
            topo.n_unknowns
        )));
    }

    let (closure_rms_after, _) = closure_rms(stack, max_span)?;

    Ok(PhaseBiasReport {
        coefficients,
        estimate,
        pixels_corrected,
        pixels_skipped,
        layers_uncovered,
        closure_rms_before,
        closure_rms_after,
    })
}

/// Mapa de sesgo acumulado por época: `Σ_{k<e} δ̂_{k,k+1}`, el producto que
/// permite ver la firma temporal del sesgo (Fig. 4 del paper: acumulación que
/// imita subsidencia en cultivo).
///
/// Devuelve `épocas × filas × cols` con la primera época en 0 (referencia) y
/// NaN en los píxeles cuyo sistema de cierres no tiene rango completo.
pub fn cumulative_bias(stack: &IfgStack, config: &PhaseBiasConfig) -> Result<Array3<f32>> {
    stack.validate()?;
    check_span(config.max_span)?;
    let max_span = config.max_span;

    let coefficients = match &config.coefficients {
        Some(a) => a.clone(),
        None => estimate_coefficients(stack, config)?.coefficients,
    };

    let topo = ClosureTopology::build(stack, max_span);
    if topo.obs.is_empty() {
        return Err(InsarError::InvalidNetwork(
            "la red no tiene fases de cierre evaluables".into(),
        ));
    }
    let design = topo.design(&coefficients);
    let n_obs = topo.obs.len();
    let n_epochs = stack.epochs.len();
    let (n_rows, n_cols) = stack.dims();
    let data = stack.data.view();

    let mut out = Array3::<f32>::from_elem((n_epochs, n_rows, n_cols), f32::NAN);
    let mut planes: Vec<_> = out.axis_iter_mut(Axis(1)).collect();
    planes.par_iter_mut().enumerate().for_each(|(r, plane)| {
        for c in 0..n_cols {
            let mut rows = Vec::with_capacity(n_obs);
            let mut vals = Vec::with_capacity(n_obs);
            for (o, ob) in topo.obs.iter().enumerate() {
                if let Some(v) = closure_at(ob, |k| data[[k, r, c]]) {
                    rows.push(o);
                    vals.push(v);
                }
            }
            if rows.len() < topo.n_unknowns {
                continue;
            }
            let reduced =
                DMatrix::from_fn(rows.len(), topo.n_unknowns, |i, j| design[(rows[i], j)]);
            let Some(pinv) = rcond_pseudo_inverse(reduced) else { continue };
            let delta = pinv * DVector::from_vec(vals);

            plane[[0, c]] = 0.0;
            let mut acc = 0.0_f64;
            for e in 1..n_epochs {
                // Slot e−1 sin par observado → la acumulación se interrumpe:
                // desde ahí el sesgo acumulado es desconocido, no "igual al
                // anterior".
                match topo.unit_col[e - 1] {
                    Some(col) => {
                        acc += delta[col];
                        plane[[e, c]] = acc as f32;
                    }
                    None => break,
                }
            }
        }
    });

    Ok(out)
}

/// Máscara de píxeles con cierre grande — candidatos a estar dominados por el
/// sesgo. `|Δφ| > threshold` en al menos `min_count` observaciones.
pub fn high_closure_mask(
    stack: &IfgStack,
    max_span: usize,
    threshold: f64,
    min_count: usize,
) -> Result<Array2<bool>> {
    stack.validate()?;
    check_span(max_span)?;
    let topo = ClosureTopology::build(stack, max_span);
    let (n_rows, n_cols) = stack.dims();
    let data = stack.data.view();

    let mut out = Array2::<bool>::from_elem((n_rows, n_cols), false);
    let mut rows: Vec<_> = out.axis_iter_mut(Axis(0)).collect();
    rows.par_iter_mut().enumerate().for_each(|(r, out_row)| {
        for c in 0..n_cols {
            let n = topo
                .obs
                .iter()
                .filter_map(|ob| closure_at(ob, |k| data[[k, r, c]]))
                .filter(|v| v.abs() > threshold)
                .count();
            out_row[c] = n >= min_count;
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Epoch, IfgPair, StackMeta, SENTINEL1_WAVELENGTH_M};
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

    fn epochs(n: usize) -> Vec<Epoch> {
        let start: chrono::NaiveDate = "2023-01-01".parse().unwrap();
        (0..n).map(|i| Epoch(start + chrono::Duration::days(6 * i as i64))).collect()
    }

    fn pair(i: usize, j: usize) -> IfgPair {
        IfgPair { reference: i, secondary: j, perp_baseline_m: 0.0 }
    }

    /// Red daisy-chain de spans 1..=max_span sobre `n` épocas.
    fn chain_pairs(n: usize, max_span: usize) -> Vec<IfgPair> {
        let mut p = Vec::new();
        for span in 1..=max_span {
            for i in 0..n.saturating_sub(span) {
                p.push(pair(i, i + span));
            }
        }
        p
    }

    /// Construye un stack sintético a partir de:
    /// - `def[e]`: fase de deformación acumulada (libre de sesgo) por época,
    /// - `bias[i]`: sesgo verdadero del par unitario (i, i+1),
    /// - `a`: coeficientes a₂..a_M,
    ///
    /// de modo que `φ_{i,j} = (def[j] − def[i]) + a_{j−i}·Σ bias`, que es
    /// exactamente el modelo de las Eqs. (6)–(7). Fase envuelta como complejo
    /// unitario.
    fn synth(
        n_epochs: usize,
        max_span: usize,
        def: &[f64],
        bias: &[f64],
        a: &[f64],
        rows: usize,
        cols: usize,
    ) -> IfgStack {
        let pairs = chain_pairs(n_epochs, max_span);
        let mut data = Array3::<Complex32>::zeros((pairs.len(), rows, cols));
        for (k, p) in pairs.iter().enumerate() {
            let span = p.secondary - p.reference;
            let coef = if span == 1 { 1.0 } else { a[span - 2] };
            let b: f64 = bias[p.reference..p.secondary].iter().sum();
            let phi = (def[p.secondary] - def[p.reference]) + coef * b;
            data.index_axis_mut(Axis(0), k)
                .fill(Complex32::from_polar(1.0, phi as f32));
        }
        IfgStack { data, epochs: epochs(n_epochs), pairs, meta: meta() }
    }

    // ---------- topología ----------

    #[test]
    fn topologia_cuenta_observaciones_e_incognitas() {
        // 6 épocas, spans 1..3 → 5 incógnitas unitarias; observaciones:
        // span 2 → 4 (i=0..3), span 3 → 3 (i=0..2) = 7 = 2N−5 con N=6. ✓
        let pairs = chain_pairs(6, 3);
        let stack = IfgStack {
            data: Array3::zeros((pairs.len(), 1, 1)),
            epochs: epochs(6),
            pairs,
            meta: meta(),
        };
        let topo = ClosureTopology::build(&stack, 3);
        assert_eq!(topo.n_unknowns, 5);
        assert_eq!(topo.obs.len(), 7, "2N−5 observaciones para N=6");
    }

    #[test]
    fn par_unitario_faltante_invalida_los_cierres_que_lo_cruzan() {
        // Sin el par (2,3), toda observación cuya cadena lo contenga se cae:
        // la Eq. 2 necesita su fase para formar el cierre.
        let mut pairs = chain_pairs(6, 3);
        pairs.retain(|p| !(p.reference == 2 && p.secondary == 3));
        let stack = IfgStack {
            data: Array3::zeros((pairs.len(), 1, 1)),
            epochs: epochs(6),
            pairs,
            meta: meta(),
        };
        let topo = ClosureTopology::build(&stack, 3);
        assert_eq!(topo.n_unknowns, 4, "el slot (2,3) ya no es incógnita");
        // Sobreviven solo los cierres que no cruzan el slot 2: span 2 en i=0,1
        // y span 3 en i=0. (i=3 span 2 cruza (3,4),(4,5): no toca el slot 2 →
        // también sobrevive.)
        for ob in &topo.obs {
            assert!(
                !ob.unit_layers.is_empty(),
                "toda observación conserva su cadena completa"
            );
        }
        assert!(topo.obs.len() < 7);
    }

    // ---------- estimación de coeficientes ----------

    #[test]
    fn recupera_los_coeficientes_verdaderos() {
        // Ancla de 12 épocas (múltiplo de mcm(1,2,3)=6) con sesgo unitario
        // constante: los cierres acumulados dan aₙ exactos.
        let n = 13;
        let a_true = [0.47, 0.31];
        let bias: Vec<f64> = vec![0.02; n - 1];
        let def: Vec<f64> = (0..n).map(|e| 0.05 * e as f64).collect();

        let mut stack = synth(n, 3, &def, &bias, &a_true, 4, 4);
        // Añade el par ancla (0, 12) con sesgo despreciable, como asume el paper.
        let anchor = pair(0, 12);
        let phi = def[12] - def[0];
        let layer = Array2::from_elem((4, 4), Complex32::from_polar(1.0, phi as f32));
        stack.data.push(Axis(0), layer.view()).unwrap();
        stack.pairs.push(anchor);

        let cfg = PhaseBiasConfig {
            anchor_days: 72.0, // 12 épocas × 6 días
            min_coefficient_pixels: 1,
            ..Default::default()
        };
        let est = estimate_coefficients(&stack, &cfg).unwrap();
        assert_eq!(est.anchor_span, 12);
        assert!(
            (est.coefficients[0] - a_true[0]).abs() < 1e-3,
            "a₁ = {} vs {}",
            est.coefficients[0],
            a_true[0]
        );
        assert!(
            (est.coefficients[1] - a_true[1]).abs() < 1e-3,
            "a₂ = {} vs {}",
            est.coefficients[1],
            a_true[1]
        );
    }

    #[test]
    fn escena_sin_sesgo_es_error_no_correccion_espuria() {
        // Sin sesgo el denominador de las Eqs. 8–9 es 0: devolver aₙ sería
        // dividir ruido por ruido. Debe fallar con mensaje, no correlacionar
        // nada silenciosamente.
        let n = 13;
        let bias = vec![0.0; n - 1];
        let def: Vec<f64> = (0..n).map(|e| 0.05 * e as f64).collect();
        let mut stack = synth(n, 3, &def, &bias, &[0.47, 0.31], 4, 4);
        let phi = def[12] - def[0];
        let layer = Array2::from_elem((4, 4), Complex32::from_polar(1.0, phi as f32));
        stack.data.push(Axis(0), layer.view()).unwrap();
        stack.pairs.push(pair(0, 12));

        let cfg = PhaseBiasConfig { min_coefficient_pixels: 1, ..Default::default() };
        let err = estimate_coefficients(&stack, &cfg).unwrap_err();
        assert!(
            format!("{err}").contains("sesgo de no-cierre detectable"),
            "mensaje inesperado: {err}"
        );
    }

    #[test]
    fn sin_ancla_es_error_accionable() {
        // Red solo de spans 1..3 sobre 6 épocas: no hay par de span múltiplo
        // de 6 ≥ 12 → no se puede estimar.
        let pairs = chain_pairs(6, 3);
        let stack = IfgStack {
            data: Array3::from_elem((pairs.len(), 2, 2), Complex32::new(1.0, 0.0)),
            epochs: epochs(6),
            pairs,
            meta: meta(),
        };
        let err = estimate_coefficients(&stack, &PhaseBiasConfig::default()).unwrap_err();
        assert!(format!("{err}").contains("par ancla"), "mensaje: {err}");
    }

    // ---------- corrección ----------

    #[test]
    fn recupera_el_sesgo_inyectado_y_deja_la_deformacion() {
        // Sesgo variable por época + deformación lineal. Con los coeficientes
        // verdaderos dados, la corrección debe devolver exactamente la fase de
        // deformación en cada par.
        let n = 8;
        let a_true = vec![0.47, 0.31];
        let bias: Vec<f64> = (0..n - 1).map(|i| 0.03 + 0.01 * (i as f64).sin()).collect();
        let def: Vec<f64> = (0..n).map(|e| 0.05 * e as f64).collect();

        let mut stack = synth(n, 3, &def, &bias, &a_true, 3, 3);
        let cfg = PhaseBiasConfig {
            coefficients: Some(a_true.clone()),
            ..Default::default()
        };
        let rep = correct_phase_bias(&mut stack, &cfg).unwrap();
        assert_eq!(rep.pixels_skipped, 0);
        assert_eq!(rep.pixels_corrected, 9);

        for (k, p) in stack.pairs.iter().enumerate() {
            let expected = def[p.secondary] - def[p.reference];
            let got = stack.data[[k, 1, 1]].arg() as f64;
            assert!(
                (got - expected).abs() < 1e-4,
                "par ({},{}) : {got} vs {expected}",
                p.reference,
                p.secondary
            );
        }
    }

    #[test]
    fn el_cierre_baja_tras_corregir() {
        let n = 10;
        let a_true = vec![0.5, 0.3];
        let bias: Vec<f64> = (0..n - 1).map(|i| 0.04 * ((i % 3) as f64 + 1.0)).collect();
        let def: Vec<f64> = (0..n).map(|e| 0.02 * e as f64).collect();

        let mut stack = synth(n, 3, &def, &bias, &a_true, 2, 2);
        let cfg = PhaseBiasConfig { coefficients: Some(a_true), ..Default::default() };
        let before = closure_rms(&stack, 3).unwrap().0;
        assert!(before > 1e-3, "el stack sintético debe tener cierre no nulo");

        let rep = correct_phase_bias(&mut stack, &cfg).unwrap();
        assert!(
            rep.closure_rms_after < rep.closure_rms_before,
            "RMS de cierre {} → {}",
            rep.closure_rms_before,
            rep.closure_rms_after
        );
        assert!(rep.closure_rms_after < 1e-4, "modelo exacto: residuo ≈ 0");
    }

    #[test]
    fn el_mapa_de_cierre_es_consistente_con_el_global() {
        // El RMS global es la media cuadrática del mapa ponderada por conteo;
        // y en un stack sintético espacialmente uniforme cada píxel debe dar
        // el mismo RMS que el global.
        let n = 10;
        let a_true = vec![0.5, 0.3];
        let bias: Vec<f64> = (0..n - 1).map(|i| 0.04 * ((i % 3) as f64 + 1.0)).collect();
        let def: Vec<f64> = (0..n).map(|e| 0.02 * e as f64).collect();
        let stack = synth(n, 3, &def, &bias, &a_true, 3, 4);

        let (global, n_global) = closure_rms(&stack, 3).unwrap();
        let (map, count) = closure_rms_map(&stack, 3).unwrap();

        let mut sum_sq = 0.0_f64;
        let mut total = 0usize;
        for (v, &c) in map.iter().zip(count.iter()) {
            if c > 0 {
                sum_sq += (*v as f64).powi(2) * c as f64;
                total += c as usize;
            }
            assert!((*v as f64 - global).abs() < 1e-5, "stack uniforme: {v} vs {global}");
        }
        assert_eq!(total, n_global, "mismo nº de observaciones que el global");
        // Tolerancia de f32: el mapa redondea cada RMS a f32 antes de agregar.
        assert!(
            ((sum_sq / total as f64).sqrt() - global).abs() < 1e-5,
            "la agregación del mapa reproduce el RMS global: {} vs {global}",
            (sum_sq / total as f64).sqrt()
        );
    }

    #[test]
    fn el_mapa_de_cierre_marca_nan_sin_observaciones() {
        // Un píxel con fase inválida en todas las capas no tiene cierres:
        // RMS NaN y conteo 0.
        let n = 8;
        let bias: Vec<f64> = vec![0.03; n - 1];
        let def: Vec<f64> = (0..n).map(|e| 0.05 * e as f64).collect();
        let mut stack = synth(n, 3, &def, &bias, &[0.47, 0.31], 2, 2);
        for k in 0..stack.n_layers() {
            stack.data[[k, 0, 0]] = Complex32::new(f32::NAN, f32::NAN);
        }
        let (map, count) = closure_rms_map(&stack, 3).unwrap();
        assert!(map[[0, 0]].is_nan());
        assert_eq!(count[[0, 0]], 0);
        assert!(map[[1, 1]].is_finite());
        assert!(count[[1, 1]] > 0);
    }

    #[test]
    fn stack_sin_sesgo_no_se_toca() {
        let n = 8;
        let bias = vec![0.0; n - 1];
        let def: Vec<f64> = (0..n).map(|e| 0.05 * e as f64).collect();
        let mut stack = synth(n, 3, &def, &bias, &[0.47, 0.31], 2, 2);
        let orig = stack.data.clone();

        let cfg = PhaseBiasConfig {
            coefficients: Some(vec![0.47, 0.31]),
            ..Default::default()
        };
        correct_phase_bias(&mut stack, &cfg).unwrap();
        for (a, b) in stack.data.iter().zip(orig.iter()) {
            assert!((a - b).norm() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn pixel_invalido_queda_intacto_y_se_reporta() {
        // Un píxel con casi todas las capas a cero (no válidas) no tiene
        // observaciones suficientes: se cuenta como skipped y NO se toca.
        let n = 8;
        let a_true = vec![0.47, 0.31];
        let bias: Vec<f64> = vec![0.03; n - 1];
        let def: Vec<f64> = (0..n).map(|e| 0.05 * e as f64).collect();
        let mut stack = synth(n, 3, &def, &bias, &a_true, 2, 2);

        for k in 0..stack.pairs.len() {
            stack.data[[k, 0, 0]] = Complex32::new(0.0, 0.0);
        }
        let cfg = PhaseBiasConfig { coefficients: Some(a_true), ..Default::default() };
        let rep = correct_phase_bias(&mut stack, &cfg).unwrap();
        assert_eq!(rep.pixels_skipped, 1);
        assert_eq!(rep.pixels_corrected, 3);
        for k in 0..stack.pairs.len() {
            assert_eq!(stack.data[[k, 0, 0]], Complex32::new(0.0, 0.0));
        }
    }

    #[test]
    fn la_correccion_preserva_la_amplitud() {
        // Sobre datos complejos la corrección es una rotación: el módulo (la
        // coherencia/amplitud del interferograma) no puede cambiar.
        let n = 8;
        let a_true = vec![0.47, 0.31];
        let bias: Vec<f64> = vec![0.05; n - 1];
        let def: Vec<f64> = (0..n).map(|e| 0.05 * e as f64).collect();
        let mut stack = synth(n, 3, &def, &bias, &a_true, 2, 2);
        stack.data.mapv_inplace(|z| z * 7.5); // amplitud arbitraria ≠ 1

        let cfg = PhaseBiasConfig { coefficients: Some(a_true), ..Default::default() };
        correct_phase_bias(&mut stack, &cfg).unwrap();
        for z in stack.data.iter() {
            assert!((z.norm() - 7.5).abs() < 1e-3, "módulo cambió: {}", z.norm());
        }
    }

    #[test]
    fn coeficientes_mal_dimensionados_son_error() {
        let pairs = chain_pairs(8, 3);
        let mut stack = IfgStack {
            data: Array3::from_elem((pairs.len(), 2, 2), Complex32::new(1.0, 0.0)),
            epochs: epochs(8),
            pairs,
            meta: meta(),
        };
        let cfg = PhaseBiasConfig {
            coefficients: Some(vec![0.47]), // falta a₂ para max_span=3
            ..Default::default()
        };
        assert!(correct_phase_bias(&mut stack, &cfg).is_err());
    }

    #[test]
    fn red_de_span_2_es_indeterminada_y_lo_dice() {
        // N−2 observaciones para N−1 incógnitas: indeterminado para todo N.
        // Debe fallar con un mensaje que apunte a la causa (falta span 3), no
        // devolver un stack intacto con éxito.
        let n = 10;
        let pairs = chain_pairs(n, 2);
        let mut stack = IfgStack {
            data: Array3::from_elem((pairs.len(), 3, 3), Complex32::new(1.0, 0.0)),
            epochs: epochs(n),
            pairs,
            meta: meta(),
        };
        let cfg = PhaseBiasConfig {
            max_span: 2,
            coefficients: Some(vec![0.47]),
            ..Default::default()
        };
        let err = correct_phase_bias(&mut stack, &cfg).unwrap_err();
        assert!(
            format!("{err}").contains("indeterminado SIEMPRE"),
            "mensaje inesperado: {err}"
        );
    }

    #[test]
    fn max_span_menor_a_2_es_error() {
        assert!(check_span(1).is_err());
        assert!(check_span(0).is_err());
        assert!(check_span(2).is_ok());
    }

    // ---------- productos derivados ----------

    #[test]
    fn sesgo_acumulado_reproduce_la_firma_temporal() {
        let n = 9;
        let a_true = vec![0.47, 0.31];
        let bias: Vec<f64> = vec![0.03; n - 1];
        let def: Vec<f64> = (0..n).map(|e| 0.05 * e as f64).collect();
        let stack = synth(n, 3, &def, &bias, &a_true, 2, 2);

        let cfg = PhaseBiasConfig { coefficients: Some(a_true), ..Default::default() };
        let cum = cumulative_bias(&stack, &cfg).unwrap();
        assert_eq!(cum.shape(), &[n, 2, 2]);
        assert_eq!(cum[[0, 0, 0]], 0.0);
        for e in 1..n {
            let expected = 0.03 * e as f64;
            assert!(
                (cum[[e, 0, 0]] as f64 - expected).abs() < 1e-3,
                "época {e}: {} vs {expected}",
                cum[[e, 0, 0]]
            );
        }
    }

    #[test]
    fn mascara_de_cierre_alto_separa_pixeles() {
        let n = 8;
        let a_true = vec![0.47, 0.31];
        let def: Vec<f64> = (0..n).map(|e| 0.05 * e as f64).collect();
        // Píxel con sesgo grande vs stack limpio: se arman dos stacks y se
        // compara la máscara.
        let sesgado = synth(n, 3, &def, &vec![0.5; n - 1], &a_true, 1, 1);
        let limpio = synth(n, 3, &def, &vec![0.0; n - 1], &a_true, 1, 1);

        let m1 = high_closure_mask(&sesgado, 3, 0.1, 1).unwrap();
        let m0 = high_closure_mask(&limpio, 3, 0.1, 1).unwrap();
        assert!(m1[[0, 0]], "el píxel sesgado debe marcarse");
        assert!(!m0[[0, 0]], "el píxel limpio no");
    }
}

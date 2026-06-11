//! Red Small-Baseline Subset: generación de pares por doble umbral
//! (temporal, perpendicular), matriz de diseño estilo Berardino et al. 2002
//! y verificación de conectividad.

use ndarray::Array2;

use crate::error::{InsarError, Result};
use crate::types::{Epoch, IfgPair};

/// Umbrales para la construcción de la red SBAS.
#[derive(Debug, Clone)]
pub struct SbasConfig {
    pub max_temporal_baseline_days: i64,
    pub max_perp_baseline_m: f64,
}

impl Default for SbasConfig {
    fn default() -> Self {
        Self { max_temporal_baseline_days: 60, max_perp_baseline_m: 200.0 }
    }
}

/// Genera todos los pares (i < j) que cumplen ambos umbrales.
/// `perp_baselines_m[i]` es la baseline perpendicular de la época `i`
/// respecto a una referencia común; la del par es la diferencia.
/// Error si la red resultante queda vacía o desconectada.
pub fn build_network(
    epochs: &[Epoch],
    perp_baselines_m: &[f64],
    config: &SbasConfig,
) -> Result<Vec<IfgPair>> {
    if epochs.len() != perp_baselines_m.len() {
        return Err(InsarError::InvalidNetwork(format!(
            "{} épocas vs {} baselines perpendiculares",
            epochs.len(),
            perp_baselines_m.len()
        )));
    }

    // Las épocas deben venir estrictamente ordenadas: los índices de IfgPair
    // asumen reference < secondary en el tiempo, y épocas duplicadas no
    // tienen sentido físico en un stack coregistrado.
    if let Some(k) = epochs.windows(2).position(|w| w[0] >= w[1]) {
        return Err(InsarError::InvalidNetwork(format!(
            "épocas no ordenadas estrictamente: posición {} ({:?}) >= posición {} ({:?})",
            k,
            epochs[k].0,
            k + 1,
            epochs[k + 1].0
        )));
    }

    let mut pairs = Vec::new();
    for i in 0..epochs.len() {
        for j in (i + 1)..epochs.len() {
            let dt_days = epochs[j].days_since(&epochs[i]);
            let db_perp = perp_baselines_m[j] - perp_baselines_m[i];
            if dt_days.abs() <= config.max_temporal_baseline_days
                && db_perp.abs() <= config.max_perp_baseline_m
            {
                pairs.push(IfgPair { reference: i, secondary: j, perp_baseline_m: db_perp });
            }
        }
    }

    if pairs.is_empty() {
        return Err(InsarError::InvalidNetwork(format!(
            "ningún par cumple los umbrales (|Δt| <= {} días, |Δb_perp| <= {} m) \
             para {} épocas",
            config.max_temporal_baseline_days,
            config.max_perp_baseline_m,
            epochs.len()
        )));
    }

    if !is_connected(&pairs, epochs.len()) {
        return Err(InsarError::InvalidNetwork(format!(
            "la red de {} pares sobre {} épocas queda desconectada con los \
             umbrales (|Δt| <= {} días, |Δb_perp| <= {} m); relajar umbrales \
             o partir el stack en subsets",
            pairs.len(),
            epochs.len(),
            config.max_temporal_baseline_days,
            config.max_perp_baseline_m
        )));
    }

    Ok(pairs)
}

/// Matriz de diseño A (n_pares × (n_épocas − 1)) que mapea incrementos de
/// desplazamiento entre épocas consecutivas a la fase de cada par.
/// Error si algún par tiene índices fuera de rango.
pub fn design_matrix(pairs: &[IfgPair], n_epochs: usize) -> Result<Array2<f64>> {
    let n_unknowns = n_epochs.saturating_sub(1);
    let mut a = Array2::<f64>::zeros((pairs.len(), n_unknowns));

    for (row, p) in pairs.iter().enumerate() {
        if p.reference >= n_epochs || p.secondary >= n_epochs {
            return Err(InsarError::InvalidNetwork(format!(
                "par {row}: índices ({}, {}) fuera de rango para {n_epochs} épocas",
                p.reference, p.secondary
            )));
        }
        if p.reference == p.secondary {
            return Err(InsarError::InvalidNetwork(format!(
                "par {row}: referencia y secundaria son la misma época ({})",
                p.reference
            )));
        }
        if p.reference > p.secondary {
            // Contrato de IfgPair: `reference` es la época más antigua.
            return Err(InsarError::InvalidNetwork(format!(
                "par {row}: referencia ({}) posterior a secundaria ({}); \
                 el contrato exige reference < secondary",
                p.reference, p.secondary
            )));
        }
        // La fase del par (i, j) es la suma de los incrementos k→k+1
        // para k en i..j (incógnita k = desplazamiento entre épocas k y k+1).
        for k in p.reference..p.secondary {
            a[[row, k]] = 1.0;
        }
    }

    Ok(a)
}

/// `true` si el grafo de épocas inducido por los pares es conexo
/// (requisito para inversión SBAS sin ambigüedad por subsets).
pub fn is_connected(pairs: &[IfgPair], n_epochs: usize) -> bool {
    if n_epochs == 0 {
        return true;
    }

    // Union-find con compresión de caminos (sin rank: n_epochs es pequeño).
    let mut parent: Vec<usize> = (0..n_epochs).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // compresión por mitades
            x = parent[x];
        }
        x
    }

    for p in pairs {
        if p.reference >= n_epochs || p.secondary >= n_epochs {
            return false; // índice fuera de rango: red inválida, no panic
        }
        let (ra, rb) = (find(&mut parent, p.reference), find(&mut parent, p.secondary));
        if ra != rb {
            parent[ra] = rb;
        }
    }

    let root = find(&mut parent, 0);
    (1..n_epochs).all(|i| find(&mut parent, i) == root)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Épocas cada 12 días a partir de 2023-01-01.
    fn epochs_12d(n: usize) -> Vec<Epoch> {
        let start: chrono::NaiveDate = "2023-01-01".parse().unwrap();
        (0..n)
            .map(|i| Epoch(start + chrono::Duration::days(12 * i as i64)))
            .collect()
    }

    fn pair(i: usize, j: usize) -> IfgPair {
        IfgPair { reference: i, secondary: j, perp_baseline_m: 0.0 }
    }

    // ---------- build_network ----------

    #[test]
    fn red_4_epocas_12d_umbral_25d() {
        // Épocas en días: 0, 12, 24, 36. Umbral 25 días.
        // Δt: (0,1)=12 ✓ (0,2)=24 ✓ (0,3)=36 ✗ (1,2)=12 ✓ (1,3)=24 ✓ (2,3)=12 ✓
        let epochs = epochs_12d(4);
        let baselines = [0.0, 50.0, -30.0, 100.0];
        let config = SbasConfig { max_temporal_baseline_days: 25, max_perp_baseline_m: 200.0 };

        let pairs = build_network(&epochs, &baselines, &config).unwrap();

        let idx: Vec<(usize, usize)> =
            pairs.iter().map(|p| (p.reference, p.secondary)).collect();
        assert_eq!(idx, vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]);

        // Baseline del par = b[j] - b[i], verificable a mano.
        let bp: Vec<f64> = pairs.iter().map(|p| p.perp_baseline_m).collect();
        assert_eq!(bp, vec![50.0, -30.0, -80.0, 50.0, 130.0]);
    }

    #[test]
    fn umbral_perpendicular_filtra_pares() {
        // Mismas épocas; baseline de la época 3 muy lejana: caen (1,3) y (2,3)...
        // pero con umbral 90 m: (0,1)=50 ✓ (0,2)=-30 ✓ (1,2)=-80 ✓ (1,3)=450 ✗ (2,3)=530 ✗
        let epochs = epochs_12d(4);
        let baselines = [0.0, 50.0, -30.0, 500.0];
        let config = SbasConfig { max_temporal_baseline_days: 25, max_perp_baseline_m: 90.0 };

        // La época 3 queda sin aristas → red desconectada → error.
        let err = build_network(&epochs, &baselines, &config).unwrap_err();
        assert!(matches!(err, InsarError::InvalidNetwork(_)));
    }

    #[test]
    fn red_vacia_es_error() {
        let epochs = epochs_12d(3);
        let baselines = [0.0, 10.0, 20.0];
        let config = SbasConfig { max_temporal_baseline_days: 5, max_perp_baseline_m: 200.0 };
        let err = build_network(&epochs, &baselines, &config).unwrap_err();
        assert!(matches!(err, InsarError::InvalidNetwork(_)));
    }

    #[test]
    fn longitudes_inconsistentes_es_error() {
        let epochs = epochs_12d(3);
        let baselines = [0.0, 10.0]; // falta una
        let err = build_network(&epochs, &baselines, &SbasConfig::default()).unwrap_err();
        assert!(matches!(err, InsarError::InvalidNetwork(_)));
    }

    #[test]
    fn epocas_desordenadas_es_error() {
        let mut epochs = epochs_12d(3);
        epochs.swap(0, 1);
        let baselines = [0.0, 10.0, 20.0];
        let err = build_network(&epochs, &baselines, &SbasConfig::default()).unwrap_err();
        assert!(matches!(err, InsarError::InvalidNetwork(_)));
    }

    #[test]
    fn epocas_duplicadas_es_error() {
        let mut epochs = epochs_12d(3);
        epochs[1] = epochs[0];
        let baselines = [0.0, 10.0, 20.0];
        let err = build_network(&epochs, &baselines, &SbasConfig::default()).unwrap_err();
        assert!(matches!(err, InsarError::InvalidNetwork(_)));
    }

    #[test]
    fn red_desconectada_por_hueco_temporal_es_error() {
        // Días: 0, 12 | 100, 112 — el hueco de 88 días parte la red en dos.
        let start: chrono::NaiveDate = "2023-01-01".parse().unwrap();
        let epochs: Vec<Epoch> = [0, 12, 100, 112]
            .iter()
            .map(|&d| Epoch(start + chrono::Duration::days(d)))
            .collect();
        let baselines = [0.0, 10.0, 20.0, 30.0];
        let config = SbasConfig { max_temporal_baseline_days: 25, max_perp_baseline_m: 200.0 };
        let err = build_network(&epochs, &baselines, &config).unwrap_err();
        assert!(matches!(err, InsarError::InvalidNetwork(_)));
    }

    // ---------- design_matrix ----------

    #[test]
    fn matriz_diseno_red_chica_elemento_a_elemento() {
        // 4 épocas, incógnitas: d01, d12, d23 (3 columnas).
        // Pares: (0,1) → [1,0,0]; (1,2) → [0,1,0]; (0,2) → [1,1,0];
        //        (2,3) → [0,0,1]; (1,3) → [0,1,1]
        let pairs = [pair(0, 1), pair(1, 2), pair(0, 2), pair(2, 3), pair(1, 3)];
        let a = design_matrix(&pairs, 4).unwrap();

        assert_eq!(a.shape(), &[5, 3]);
        #[rustfmt::skip]
        let expected = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        for (r, row) in expected.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                assert_eq!(a[[r, c]], v, "A[{r},{c}]");
            }
        }
    }

    #[test]
    fn matriz_diseno_sin_pares() {
        let a = design_matrix(&[], 4).unwrap();
        assert_eq!(a.shape(), &[0, 3]);
    }

    #[test]
    fn matriz_diseno_indice_fuera_de_rango_es_error() {
        let err = design_matrix(&[pair(0, 7)], 4).unwrap_err();
        assert!(matches!(err, InsarError::InvalidNetwork(_)));
    }

    #[test]
    fn matriz_diseno_ref_igual_sec_es_error() {
        let err = design_matrix(&[pair(2, 2)], 4).unwrap_err();
        assert!(matches!(err, InsarError::InvalidNetwork(_)));
    }

    #[test]
    fn matriz_diseno_ref_posterior_a_sec_es_error() {
        let err = design_matrix(&[pair(3, 1)], 4).unwrap_err();
        assert!(matches!(err, InsarError::InvalidNetwork(_)));
    }

    // ---------- is_connected ----------

    #[test]
    fn conectividad_red_partida_en_dos_componentes() {
        // {0,1} y {2,3} sin puente.
        let pairs = [pair(0, 1), pair(2, 3)];
        assert!(!is_connected(&pairs, 4));
        // Agregar el puente (1,2) la conecta.
        let pairs = [pair(0, 1), pair(1, 2), pair(2, 3)];
        assert!(is_connected(&pairs, 4));
    }

    #[test]
    fn conectividad_epoca_aislada() {
        // Época 2 sin aristas → desconexo.
        assert!(!is_connected(&[pair(0, 1)], 3));
    }

    #[test]
    fn conectividad_casos_borde() {
        assert!(is_connected(&[], 0)); // grafo vacío: conexo por convención
        assert!(is_connected(&[], 1)); // un solo nodo: conexo
        assert!(!is_connected(&[], 2)); // dos nodos sin arista: desconexo
        assert!(!is_connected(&[pair(0, 9)], 3)); // fuera de rango: false, sin panic
    }
}

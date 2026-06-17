//! Stress-test del camino NaN-por-par y su cache por patrón de máscara: lee el
//! stack ISCE real, invierte con la red completa y con ~20% de los pares
//! descartados por píxel (NaN, patrón determinista que genera varios patrones
//! de máscara distintos), y compara.
//!
//! Qué verifica: (1) el camino por píxel corre a escala en pocos segundos
//! gracias a la pseudoinversa cacheada por patrón (sin él, una SVD por píxel
//! tardaría minutos); (2) ningún píxel se descarta por desconexión si la red
//! reducida sigue conexa.
//!
//! NOTA sobre el RMSE vs full: NO es pequeño y NO debe serlo. Cada píxel
//! descarta un subconjunto de pares *distinto*, así que se comparan dos
//! estimadores LSQ genuinamente diferentes sobre fase con ruido real; el SBAS
//! no ponderado es sensible a qué pares entran. La corrección numérica del
//! camino reducido se verifica aparte con datos sin ruido (tests unitarios de
//! `inversion`, recuperación exacta al quitar un par).
//!
//! Uso: cargo run --release -p insar-core --example robustness_dropout -- <ifg_dir> [baselines]

use std::path::{Path, PathBuf};

use insar_core::inversion::{estimate_velocity, invert_sbas};
use insar_core::io::isce::{IsceLoadConfig, read_isce_unwrapped_stack};

fn main() {
    let mut args = std::env::args().skip(1);
    let ifg_dir = args.next().expect("arg1: dir ISCE");
    let baselines = args.next();

    let config = IsceLoadConfig {
        baselines_dir: baselines.map(PathBuf::from),
        ..Default::default()
    };
    let stack = read_isce_unwrapped_stack(Path::new(&ifg_dir), &config).expect("lectura ISCE");
    let (np, nr, nc) = (stack.pairs.len(), stack.dims().0, stack.dims().1);
    println!("Stack: {} épocas, {np} pares, {nr}×{nc}", stack.epochs.len());

    // Inversión con red completa.
    let vel_full = estimate_velocity(&invert_sbas(&stack, None).unwrap()).unwrap();

    // Copia con ~20% de (par, píxel) puestos a NaN (patrón determinista).
    let mut dropped = stack.clone();
    let mut n_drop = 0u64;
    for k in 0..np {
        for r in 0..nr {
            for c in 0..nc {
                if (k * 131 + r * 17 + c * 7) % 5 == 0 {
                    dropped.data[[k, r, c]] = f32::NAN;
                    n_drop += 1;
                }
            }
        }
    }
    println!("Descartados {:.1}% de las observaciones (par, píxel)", 100.0 * n_drop as f64 / (np * nr * nc) as f64);

    let t = std::time::Instant::now();
    let vel_drop = estimate_velocity(&invert_sbas(&dropped, None).unwrap()).unwrap();
    println!("Inversión con dropout: {:.2}s", t.elapsed().as_secs_f64());

    // Comparación velocidad full vs dropout (mm/año).
    let (mut n, mut sse, mut maxabs, mut nan_drop) = (0u64, 0.0f64, 0.0f64, 0u64);
    for r in 0..nr {
        for c in 0..nc {
            let a = vel_full.data[[r, c]];
            let b = vel_drop.data[[r, c]];
            if b.is_nan() && !a.is_nan() {
                nan_drop += 1;
                continue;
            }
            if a.is_finite() && b.is_finite() {
                let d = (a - b) as f64;
                sse += d * d;
                maxabs = maxabs.max(d.abs());
                n += 1;
            }
        }
    }
    let rmse = (sse / n as f64).sqrt() * 1000.0;
    println!(
        "vs full: RMSE={rmse:.4} mm/año  max|Δ|={:.4} mm/año  ({n} px comparados, {nan_drop} px →NaN por desconexión)",
        maxabs * 1000.0
    );
}

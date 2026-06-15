//! Benchmarks del núcleo SBAS sobre stacks sintéticos de tamaño realista
//! (comparable al caso Fernandina: ~98 épocas, ~288 pares, ~270k píxeles).
//!
//! Correr: `cargo bench -p insar-core`

use chrono::NaiveDate;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::Array3;
use std::hint::black_box;
use surtgis_core::GeoTransform;

use insar_core::inversion::{estimate_velocity, invert_sbas};
use insar_core::ps::amplitude_dispersion;
use insar_core::types::{
    AmplitudeStack, Epoch, IfgPair, SENTINEL1_WAVELENGTH_M, StackMeta, UnwrappedStack,
};

fn meta() -> StackMeta {
    StackMeta {
        transform: GeoTransform::new(0.0, 0.0, 1.0, -1.0),
        crs: None,
        wavelength_m: SENTINEL1_WAVELENGTH_M,
        incidence_deg: 39.0,
        heading_deg: None,
    }
}

fn epochs(n: usize) -> Vec<Epoch> {
    let start = NaiveDate::from_ymd_opt(2018, 1, 1).unwrap();
    (0..n)
        .map(|i| Epoch(start + chrono::Duration::days(12 * i as i64)))
        .collect()
}

/// Red SBAS: pares consecutivos + saltos de 2 y 3 (densa y conexa).
fn pairs(n_epochs: usize) -> Vec<IfgPair> {
    let mut p = Vec::new();
    for i in 0..n_epochs {
        for step in 1..=3 {
            if i + step < n_epochs {
                p.push(IfgPair {
                    reference: i,
                    secondary: i + step,
                    perp_baseline_m: 20.0 * step as f64,
                });
            }
        }
    }
    p
}

/// Stack desenrollado sintético: deformación lineal en el tiempo por píxel.
fn unwrapped(n_epochs: usize, rows: usize, cols: usize) -> UnwrappedStack {
    let eps = epochs(n_epochs);
    let prs = pairs(n_epochs);
    let scale = -4.0 * std::f64::consts::PI / SENTINEL1_WAVELENGTH_M;
    let mut data = Array3::<f32>::zeros((prs.len(), rows, cols));
    for (k, pr) in prs.iter().enumerate() {
        let dt = eps[pr.secondary].years_since(&eps[pr.reference]);
        for r in 0..rows {
            for c in 0..cols {
                // velocidad ~ -0.05 m/año modulada espacialmente
                let v = -0.05 * (1.0 + 0.1 * ((r + c) as f64).sin());
                data[[k, r, c]] = (scale * v * dt) as f32;
            }
        }
    }
    UnwrappedStack { data, epochs: eps, pairs: prs, meta: meta() }
}

fn amp_stack(n_epochs: usize, rows: usize, cols: usize) -> AmplitudeStack {
    let mut data = Array3::<f32>::zeros((n_epochs, rows, cols));
    for e in 0..n_epochs {
        for r in 0..rows {
            for c in 0..cols {
                data[[e, r, c]] = 1.0 + 0.05 * ((e + r + c) as f32).sin();
            }
        }
    }
    AmplitudeStack { data, epochs: epochs(n_epochs), meta: meta() }
}

fn bench_inversion(c: &mut Criterion) {
    let mut g = c.benchmark_group("invert_sbas");
    g.sample_size(10);
    // Tamaño tipo Fernandina y una grilla menor para escala.
    for (ne, rows, cols) in [(98, 150, 200), (98, 450, 600)] {
        let stack = unwrapped(ne, rows, cols);
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("{ne}ep_{rows}x{cols}_{}pares", stack.pairs.len())),
            &stack,
            |b, s| b.iter(|| black_box(invert_sbas(s, None).unwrap())),
        );
    }
    g.finish();
}

fn bench_velocity(c: &mut Criterion) {
    let stack = unwrapped(98, 450, 600);
    let series = invert_sbas(&stack, None).unwrap();
    let mut g = c.benchmark_group("estimate_velocity");
    g.sample_size(20);
    g.bench_function("98ep_450x600", |b| {
        b.iter(|| black_box(estimate_velocity(&series).unwrap()))
    });
    g.finish();
}

fn bench_ps(c: &mut Criterion) {
    let stack = amp_stack(98, 450, 600);
    let mut g = c.benchmark_group("amplitude_dispersion");
    g.sample_size(20);
    g.bench_function("98ep_450x600", |b| {
        b.iter(|| black_box(amplitude_dispersion(&stack).unwrap()))
    });
    g.finish();
}

criterion_group!(benches, bench_inversion, bench_velocity, bench_ps);
criterion_main!(benches);

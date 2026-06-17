use std::path::PathBuf;

use clap::{Parser, Subcommand};
use insar_core::pipeline::{SbasPipelineConfig, run_sbas};

#[derive(Parser)]
#[command(name = "insar", version, about = "Motor InSAR time-series (PS-InSAR + SBAS) en Rust")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Muestra metadata de un stack de interferogramas
    Info {
        /// Directorio del stack (GeoTIFFs + stack.json)
        dir: PathBuf,
    },
    /// Selección de Persistent Scatterers por amplitude dispersion
    Ps {
        /// Directorio del stack de amplitudes
        dir: PathBuf,
        /// Umbral de amplitude dispersion
        #[arg(long, default_value_t = 0.4)]
        threshold: f32,
    },
    /// Reporta la red SBAS del stack y su conectividad
    Network {
        /// Directorio del stack
        dir: PathBuf,
    },
    /// Pipeline SBAS completo: desenrollado + inversión + APS + velocidad
    Run {
        /// Directorio del stack de entrada
        input: PathBuf,
        /// Directorio de salida
        output: PathBuf,
        /// Umbral PS (omitir para invertir toda la grilla)
        #[arg(long)]
        ps_threshold: Option<f32>,
    },
    /// SBAS directo desde interferogramas ISCE (.unw ya desenrollados): inversión + velocidad
    Isce {
        /// Directorio de interferogramas ISCE (subdirs YYYYMMDD_YYYYMMDD)
        input: PathBuf,
        /// Directorio de salida (velocity.tif + series/)
        output: PathBuf,
        /// Directorio de baselines (opcional)
        #[arg(long)]
        baselines: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Info { dir } => {
            let stack = insar_core::io::read_ifg_stack(&dir)?;
            let (rows, cols) = stack.dims();
            println!("interferogramas: {}", stack.n_layers());
            println!("épocas:          {}", stack.epochs.len());
            println!("grilla:          {rows} × {cols}");
            println!("λ (m):           {}", stack.meta.wavelength_m);
        }
        Command::Ps { dir, threshold } => {
            let stack = insar_core::io::read_amplitude_stack(&dir)?;
            let dispersion = insar_core::ps::amplitude_dispersion(&stack)?;
            let candidates = insar_core::ps::select_ps(&dispersion, threshold);
            println!("candidatos PS (D_A ≤ {threshold}): {}", candidates.len());
        }
        Command::Network { dir } => {
            let stack = insar_core::io::read_ifg_stack(&dir)?;
            stack.validate()?;
            let connected = insar_core::network::is_connected(&stack.pairs, stack.epochs.len());
            println!("pares:    {}", stack.pairs.len());
            println!("conexa:   {}", if connected { "sí" } else { "NO" });
        }
        Command::Run { input, output, ps_threshold } => {
            let config = SbasPipelineConfig {
                input_dir: input,
                output_dir: output,
                ps_threshold,
                network: Default::default(),
                aps: Default::default(),
            };
            let products = run_sbas(&config)?;
            let (rows, cols) = products.series.dims();
            println!("serie invertida: {} épocas, {rows} × {cols}", products.series.epochs.len());
        }
        Command::Isce { input, output, baselines } => {
            use insar_core::io::isce::{IsceLoadConfig, read_isce_coherence, read_isce_unwrapped_stack};
            let config = IsceLoadConfig { baselines_dir: baselines, ..Default::default() };
            let mut stack = read_isce_unwrapped_stack(&input, &config)?;
            let (rows, cols) = stack.dims();
            println!("ISCE: {} épocas, {} pares, {rows} × {cols}", stack.epochs.len(), stack.pairs.len());

            // Píxel de referencia: máxima coherencia media (si hay .cor) o el
            // centro. Referenciar la entrada elimina el offset por interferograma
            // del desenrollado (necesario para serie y coherencia temporal válidas).
            let (ref_r, ref_c) = match read_isce_coherence(&input, &config) {
                Ok(coh) => {
                    let n_pairs = coh.shape()[0];
                    let mut best = (rows / 2, cols / 2, f32::MIN);
                    for r in 0..rows {
                        for c in 0..cols {
                            let (mut sum, mut n) = (0.0_f64, 0u32);
                            for k in 0..n_pairs {
                                let v = coh[[k, r, c]];
                                if v.is_finite() {
                                    sum += v as f64;
                                    n += 1;
                                }
                            }
                            if n > 0 {
                                let mean = (sum / n as f64) as f32;
                                if mean > best.2 {
                                    best = (r, c, mean);
                                }
                            }
                        }
                    }
                    println!("referencia (máx coherencia media): ({}, {}) γ̄={:.3}", best.0, best.1, best.2);
                    (best.0, best.1)
                }
                Err(_) => {
                    println!("sin coherencia disponible: referencia al centro ({}, {})", rows / 2, cols / 2);
                    (rows / 2, cols / 2)
                }
            };
            insar_core::inversion::reference_to_pixel(&mut stack, ref_r, ref_c)?;

            let series = insar_core::inversion::invert_sbas(&stack, None)?;
            let velocity = insar_core::inversion::estimate_velocity(&series)?;
            let gamma = insar_core::inversion::temporal_coherence(&stack, &series)?;
            std::fs::create_dir_all(&output)?;
            insar_core::io::write_velocity(&velocity, &output.join("velocity.tif"))?;
            insar_core::io::write_series(&series, &output.join("series"))?;
            // La coherencia temporal comparte el writer Float32 (mismo meta).
            let coh = insar_core::types::VelocityMap { data: gamma, meta: stack.meta.clone() };
            insar_core::io::write_velocity(&coh, &output.join("temporal_coherence.tif"))?;
            println!("escrito: {}/velocity.tif + temporal_coherence.tif + series/", output.display());
        }
    }
    Ok(())
}

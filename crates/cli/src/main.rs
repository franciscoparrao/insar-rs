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
    }
    Ok(())
}

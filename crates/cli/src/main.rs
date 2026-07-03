use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use insar_core::pipeline::{SbasPipelineConfig, run_sbas};
use insar_core::postprocess::RampKind;

#[derive(Parser)]
#[command(name = "insar", version, about = "Motor InSAR time-series (PS-InSAR + SBAS) en Rust")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Superficie de deramp expuesta en la CLI.
#[derive(Clone, Copy, ValueEnum)]
enum DerampArg {
    Linear,
    Quadratic,
}

impl From<DerampArg> for RampKind {
    fn from(d: DerampArg) -> Self {
        match d {
            DerampArg::Linear => RampKind::Linear,
            DerampArg::Quadratic => RampKind::Quadratic,
        }
    }
}

/// Backend de desenrollado 2D expuesto en la CLI.
#[derive(Clone, Copy, ValueEnum)]
enum UnwrapBackendArg {
    FloodFill,
    Snaphu,
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
    /// Pipeline SBAS completo: desenrollado + cierre + inversión + APS + velocidad
    Run {
        /// Directorio del stack de entrada
        input: PathBuf,
        /// Directorio de salida
        output: PathBuf,
        /// Umbral PS (omitir para invertir toda la grilla)
        #[arg(long)]
        ps_threshold: Option<f32>,
        /// Coherencia mínima para desenrollar (requiere 'coherence' en stack.json;
        /// solo aplica con --unwrap-backend flood-fill)
        #[arg(long)]
        min_quality: Option<f32>,
        /// Backend de desenrollado 2D (snaphu requiere el binario en PATH)
        #[arg(long, default_value = "flood-fill")]
        unwrap_backend: UnwrapBackendArg,
        /// Ruta al binario snaphu (solo con --unwrap-backend snaphu; default: PATH)
        #[arg(long)]
        snaphu_bin: Option<PathBuf>,
        /// Deramp de la serie tras las correcciones
        #[arg(long)]
        deramp: Option<DerampArg>,
        /// Desactiva la corrección de errores de desenrollado por cierre de fase
        #[arg(long)]
        no_closure_correction: bool,
    },
    /// Descompone LOS ascendente + descendente en (Up, East) — geometría escalar
    Decompose {
        /// GeoTIFF LOS ascendente (m/año o m; misma unidad que la salida)
        asc_los: PathBuf,
        /// GeoTIFF LOS descendente (misma grilla y unidad que asc_los)
        desc_los: PathBuf,
        /// Directorio de salida (up.tif + east.tif)
        output: PathBuf,
        /// Ángulo de incidencia medio ascendente (grados)
        #[arg(long)]
        asc_incidence_deg: f64,
        /// Heading de la plataforma ascendente (grados)
        #[arg(long)]
        asc_heading_deg: f64,
        /// Ángulo de incidencia medio descendente (grados)
        #[arg(long)]
        desc_incidence_deg: f64,
        /// Heading de la plataforma descendente (grados)
        #[arg(long)]
        desc_heading_deg: f64,
    },
    /// Descriptores por píxel para ML (velocidad, aceleración, estacionalidad...)
    Features {
        /// Directorio con la serie (disp_YYYYMMDD.tif, salida de 'run'/'isce')
        series_dir: PathBuf,
        /// Directorio de salida (un GeoTIFF por feature)
        output: PathBuf,
        /// Longitud de onda radar en metros
        #[arg(long, default_value_t = insar_core::types::SENTINEL1_WAVELENGTH_M)]
        wavelength_m: f64,
        /// Desactiva las features estacionales (amplitud/fase anual)
        #[arg(long)]
        no_seasonal: bool,
        /// Desactiva la feature de aceleración (ajuste cuadrático)
        #[arg(long)]
        no_acceleration: bool,
        /// Mínimo de épocas finitas por píxel para ajustar
        #[arg(long, default_value_t = 5)]
        min_valid_epochs: usize,
        /// Además escribe la tabla (x,y,features) como CSV
        #[arg(long)]
        csv: bool,
    },
    /// Deramp standalone de una serie ya escrita (fuera de 'run'/'isce')
    Deramp {
        /// Directorio con la serie (disp_YYYYMMDD.tif)
        series_dir: PathBuf,
        /// Directorio de salida
        output: PathBuf,
        /// Modelo de rampa
        kind: DerampArg,
        /// Longitud de onda radar en metros
        #[arg(long, default_value_t = insar_core::types::SENTINEL1_WAVELENGTH_M)]
        wavelength_m: f64,
    },
    /// SBAS directo desde interferogramas ISCE (.unw ya desenrollados)
    Isce {
        /// Directorio de interferogramas ISCE (subdirs YYYYMMDD_YYYYMMDD)
        input: PathBuf,
        /// Directorio de salida (velocity.tif + series/ + QC)
        output: PathBuf,
        /// Directorio de baselines (opcional; requerido por --dem-error-range)
        #[arg(long)]
        baselines: Option<PathBuf>,
        /// Pesos WLS por coherencia (inverso de varianza de fase; requiere .cor)
        #[arg(long)]
        wls: bool,
        /// Inversión robusta L1 por IRLS (resiste errores de unwrap residuales)
        #[arg(long)]
        robust: bool,
        /// Estima el error de DEM: rango oblicuo medio en metros (S1 IW ≈ 850000)
        #[arg(long)]
        dem_error_range: Option<f64>,
        /// Deramp de la serie tras las correcciones
        #[arg(long)]
        deramp: Option<DerampArg>,
        /// Desactiva la corrección de errores de desenrollado por cierre de fase
        #[arg(long)]
        no_closure_correction: bool,
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
        Command::Run {
            input,
            output,
            ps_threshold,
            min_quality,
            unwrap_backend,
            snaphu_bin,
            deramp,
            no_closure_correction,
        } => {
            use insar_core::pipeline::UnwrapBackend;
            use insar_core::unwrap::snaphu::SnaphuConfig;

            let unwrap_backend = match unwrap_backend {
                UnwrapBackendArg::FloodFill => UnwrapBackend::FloodFill,
                UnwrapBackendArg::Snaphu => {
                    let mut snaphu_config = SnaphuConfig::default();
                    if let Some(bin) = snaphu_bin {
                        snaphu_config.binary = bin;
                    }
                    UnwrapBackend::Snaphu(snaphu_config)
                }
            };
            let config = SbasPipelineConfig {
                ps_threshold,
                unwrap_min_quality: min_quality,
                unwrap_backend,
                correct_unwrap: !no_closure_correction,
                deramp: deramp.map(RampKind::from),
                ..SbasPipelineConfig::new(input, output.clone())
            };
            let products = run_sbas(&config)?;
            let (rows, cols) = products.series.dims();
            println!(
                "serie invertida: {} épocas, {rows} × {cols}",
                products.series.epochs.len()
            );
            if let Some(rep) = products.unwrap_report {
                println!(
                    "cierre de fase: {} píxeles corregidos, {} detectados sin corregir",
                    rep.corrected, rep.detected_uncorrected
                );
            }
            println!("escrito: {}/velocity.tif + temporal_coherence.tif + series/", output.display());
        }
        Command::Decompose {
            asc_los,
            desc_los,
            output,
            asc_incidence_deg,
            asc_heading_deg,
            desc_incidence_deg,
            desc_heading_deg,
        } => {
            use insar_core::decompose::{LosVector, decompose_asc_desc};
            use insar_core::types::{SENTINEL1_WAVELENGTH_M, StackMeta};

            // El GeoTIFF no guarda wavelength/incidencia/heading (no son
            // geográficos); transform/crs se sobrescriben desde el archivo
            // dentro de read_velocity, así que los valores de placeholder
            // aquí son irrelevantes salvo incidence/heading (documentales).
            let meta_for = |incidence_deg: f64, heading_deg: f64| StackMeta {
                transform: surtgis_core::GeoTransform::default(),
                crs: None,
                wavelength_m: SENTINEL1_WAVELENGTH_M,
                incidence_deg,
                heading_deg: Some(heading_deg),
            };
            let asc = insar_core::io::read_velocity(&asc_los, meta_for(asc_incidence_deg, asc_heading_deg))?;
            let desc =
                insar_core::io::read_velocity(&desc_los, meta_for(desc_incidence_deg, desc_heading_deg))?;

            let geom_asc = LosVector::from_incidence_heading(asc_incidence_deg, asc_heading_deg);
            let geom_desc = LosVector::from_incidence_heading(desc_incidence_deg, desc_heading_deg);
            let decomposed = decompose_asc_desc(&asc.data, geom_asc, &desc.data, geom_desc)?;

            std::fs::create_dir_all(&output)?;
            let wrap = |d| insar_core::types::VelocityMap { data: d, meta: asc.meta.clone() };
            insar_core::io::write_velocity(&wrap(decomposed.up), &output.join("up.tif"))?;
            insar_core::io::write_velocity(&wrap(decomposed.east), &output.join("east.tif"))?;
            println!("escrito: {}/up.tif + east.tif", output.display());
        }
        Command::Features {
            series_dir,
            output,
            wavelength_m,
            no_seasonal,
            no_acceleration,
            min_valid_epochs,
            csv,
        } => {
            use insar_core::features::{FeatureConfig, extract_features};
            use insar_core::types::StackMeta;

            let meta = StackMeta {
                transform: surtgis_core::GeoTransform::default(),
                crs: None,
                wavelength_m,
                incidence_deg: 0.0,
                heading_deg: None,
            };
            let series = insar_core::io::read_series(&series_dir, meta)?;
            let config = FeatureConfig {
                seasonal: !no_seasonal,
                acceleration: !no_acceleration,
                min_valid_epochs,
            };
            let maps = extract_features(&series, None, &config)?;

            std::fs::create_dir_all(&output)?;
            maps.write_geotiffs(&output)?;
            let mut extra = String::new();
            if csv {
                let csv_path = output.join("features.csv");
                maps.write_features_csv(None, &csv_path)?;
                extra.push_str(" + features.csv");
            }
            println!(
                "escrito: {}/{{{}}}.tif{extra}",
                output.display(),
                maps.feature_names().join(",")
            );
        }
        Command::Deramp { series_dir, output, kind, wavelength_m } => {
            use insar_core::postprocess::deramp_series;
            use insar_core::types::StackMeta;

            let meta = StackMeta {
                transform: surtgis_core::GeoTransform::default(),
                crs: None,
                wavelength_m,
                incidence_deg: 0.0,
                heading_deg: None,
            };
            let mut series = insar_core::io::read_series(&series_dir, meta)?;
            deramp_series(&mut series, kind.into(), None)?;
            insar_core::io::write_series(&series, &output)?;
            println!("escrito: {}/ (serie deramp)", output.display());
        }
        Command::Isce {
            input,
            output,
            baselines,
            wls,
            robust,
            dem_error_range,
            deramp,
            no_closure_correction,
        } => {
            use insar_core::inversion::{
                DemErrorConfig, IrlsConfig, SbasSolverConfig, WeightScheme, invert_sbas_ext,
                select_reference_pixel,
            };
            use insar_core::io::isce::{
                IsceLoadConfig, read_isce_coherence, read_isce_unwrapped_stack,
            };

            let config = IsceLoadConfig { baselines_dir: baselines, ..Default::default() };
            let mut stack = read_isce_unwrapped_stack(&input, &config)?;
            let (rows, cols) = stack.dims();
            println!(
                "ISCE: {} épocas, {} pares, {rows} × {cols}",
                stack.epochs.len(),
                stack.pairs.len()
            );

            // Coherencia: calidad para referencia automática y pesos WLS.
            let coherence = match read_isce_coherence(&input, &config) {
                Ok(coh) => Some(coh),
                Err(e) => {
                    if wls {
                        anyhow::bail!("--wls requiere los .cor de coherencia: {e}");
                    }
                    println!("sin coherencia disponible ({e})");
                    None
                }
            };

            // Corrección de errores de desenrollado por cierre de fase.
            let closure_qc = if no_closure_correction {
                None
            } else {
                let rep = insar_core::unwrap_error::correct_unwrap_errors(&mut stack)?;
                println!(
                    "cierre de fase: {} píxeles corregidos, {} detectados sin corregir",
                    rep.corrected, rep.detected_uncorrected
                );
                Some(insar_core::unwrap_error::nonzero_closure_count(&stack)?)
            };

            // Píxel de referencia: máxima coherencia media, o el centro.
            let (ref_r, ref_c) = coherence
                .as_ref()
                .and_then(select_reference_pixel)
                .unwrap_or((rows / 2, cols / 2));
            println!("referencia: ({ref_r}, {ref_c})");
            insar_core::inversion::reference_to_pixel(&mut stack, ref_r, ref_c)?;

            // Inversión: OLS o WLS, con error de DEM opcional.
            let solver = SbasSolverConfig {
                weighting: if wls { WeightScheme::InversePhaseVariance } else { WeightScheme::Unit },
                dem_error: dem_error_range.map(|slant_range_m| DemErrorConfig { slant_range_m }),
                robust: robust.then(IrlsConfig::default),
            };
            let solution = invert_sbas_ext(&stack, None, coherence.as_ref(), &solver)?;
            let mut series = solution.series;

            // Deramp opcional de la serie.
            if let Some(kind) = deramp {
                insar_core::postprocess::deramp_series(&mut series, kind.into(), None)?;
            }

            let velocity = insar_core::inversion::estimate_velocity(&series)?;
            let vel_std = insar_core::inversion::estimate_velocity_uncertainty(&series)?;
            let gamma = insar_core::postprocess::temporal_coherence(&stack, &series)?;

            std::fs::create_dir_all(&output)?;
            insar_core::io::write_velocity(&velocity, &output.join("velocity.tif"))?;
            insar_core::io::write_series(&series, &output.join("series"))?;
            // Los mapas de calidad comparten el writer Float32 (mismo meta).
            let wrap = |d| insar_core::types::VelocityMap { data: d, meta: stack.meta.clone() };
            insar_core::io::write_velocity(&wrap(gamma), &output.join("temporal_coherence.tif"))?;
            insar_core::io::write_velocity(&wrap(vel_std), &output.join("velocity_std.tif"))?;
            let mut extras = String::new();
            if let Some(dem) = solution.dem_error_m {
                insar_core::io::write_velocity(&wrap(dem), &output.join("dem_error.tif"))?;
                extras.push_str(" + dem_error.tif");
            }
            if let Some(qc) = closure_qc {
                insar_core::io::write_velocity(&wrap(qc), &output.join("closure_qc.tif"))?;
                extras.push_str(" + closure_qc.tif");
            }
            println!(
                "escrito: {}/velocity.tif + velocity_std.tif + temporal_coherence.tif{extras} + series/",
                output.display()
            );
        }
    }
    Ok(())
}

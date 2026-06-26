//! Descomposición cosísmica asc + desc → **alzamiento vertical (Up)** + Este.
//!
//! Toma dos directorios de exportación (uno ascendente, uno descendente), cada
//! uno con `meta.json` (grilla + geometría) y `los.f32` (desplazamiento LOS en
//! metros, hacia el satélite positivo). Resuelve por píxel el desplazamiento
//! vertical y horizontal E-O y escribe `uplift.tif` + `east.tif`
//! georreferenciados (writer GeoTIFF nativo de SurtGIS).
//!
//! Pensado para un sismo: un interferograma pre→post por geometría. La señal
//! vertical (alzamiento de la costa) sale de combinar las dos miradas.
//!
//! Uso:
//!   cargo run --release -p insar-core --example decompose_coseismic -- \
//!     --asc validation/venz_asc --desc validation/venz_desc --out validation/venz_updown
//! Opcional (sobreescribe la geometría del meta.json):
//!   --inc-asc 38 --head-asc -12 --inc-desc 41 --head-desc -168

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use surtgis_core::io::write_geotiff;
use surtgis_core::{GeoTransform, Raster};

use insar_core::decompose::{decompose_asc_desc, LosVector};

#[derive(Deserialize)]
struct Geo {
    lon0: f64,
    lat0: f64,
    dlon: f64,
    dlat: f64,
}
#[derive(Deserialize)]
struct Meta {
    rows: usize,
    cols: usize,
    incidence_deg: f64,
    #[serde(default)]
    heading_deg: Option<f64>,
    geo: Geo,
}

fn read_f32(path: &Path, n: usize) -> Vec<f32> {
    let mut b = Vec::new();
    fs::File::open(path)
        .unwrap_or_else(|e| panic!("abrir {}: {e}", path.display()))
        .read_to_end(&mut b)
        .unwrap();
    assert_eq!(b.len(), n * 4, "tamaño inesperado {}", path.display());
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn load(dir: &Path) -> (Meta, Vec<f32>) {
    let meta: Meta = serde_json::from_str(&fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let los = read_f32(&dir.join("los.f32"), meta.rows * meta.cols);
    (meta, los)
}

/// Rumbo por defecto de Sentinel-1 según la geometría, si el meta no lo trae.
fn default_heading(label: &str) -> f64 {
    match label {
        "asc" => -12.0,
        _ => -168.0,
    }
}

/// Lee `--clave valor` simple de los argumentos.
fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).cloned()
}
fn argf(args: &[String], key: &str) -> Option<f64> {
    arg(args, key).map(|v| v.parse().unwrap())
}

fn geom(meta: &Meta, args: &[String], label: &str) -> LosVector {
    let inc = argf(args, &format!("--inc-{label}")).unwrap_or(meta.incidence_deg);
    let head = argf(args, &format!("--head-{label}"))
        .or(meta.heading_deg)
        .unwrap_or_else(|| {
            eprintln!("aviso: {label} sin heading_deg en meta ni CLI → uso default Sentinel-1");
            default_heading(label)
        });
    let g = LosVector::from_incidence_heading(inc, head);
    println!("  {label}: inc={inc:.1}° head={head:.1}° → ê=(E {:.3}, N {:.3}, U {:.3})", g.east, g.north, g.up);
    g
}

fn write_tif(path: &Path, data: &[f32], m: &Meta) {
    let mut raster = Raster::from_vec(data.to_vec(), m.rows, m.cols).unwrap();
    raster.set_transform(GeoTransform::new(m.geo.lon0, m.geo.lat0, m.geo.dlon, m.geo.dlat));
    raster.set_nodata(Some(f32::NAN));
    write_geotiff(&raster, path, None).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let asc_dir = PathBuf::from(arg(&args, "--asc").expect("--asc <dir>"));
    let desc_dir = PathBuf::from(arg(&args, "--desc").expect("--desc <dir>"));
    let out_dir = PathBuf::from(arg(&args, "--out").expect("--out <dir>"));
    fs::create_dir_all(&out_dir).unwrap();

    let (ma, la) = load(&asc_dir);
    let (md, ld) = load(&desc_dir);
    assert_eq!((ma.rows, ma.cols), (md.rows, md.cols), "asc y desc deben compartir grilla recortada");
    println!("grilla {}×{}", ma.rows, ma.cols);

    println!("geometrías de vista (suelo→satélite, ENU):");
    let ga = geom(&ma, &args, "asc");
    let gd = geom(&md, &args, "desc");

    let la2 = ndarray::Array2::from_shape_vec((ma.rows, ma.cols), la).unwrap();
    let ld2 = ndarray::Array2::from_shape_vec((md.rows, md.cols), ld).unwrap();
    let out = decompose_asc_desc(&la2, ga, &ld2, gd).unwrap();

    // Estadística rápida del alzamiento (cm).
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut k = 0usize;
    for &v in out.up.iter() {
        if v.is_finite() {
            mn = mn.min(v);
            mx = mx.max(v);
            k += 1;
        }
    }
    println!(
        "descompuesto: {k} píxeles válidos; vertical {:.1}..{:.1} cm (Up>0 = alzamiento)",
        mn * 100.0,
        mx * 100.0
    );

    write_tif(&out_dir.join("uplift.tif"), out.up.as_slice().unwrap(), &ma);
    write_tif(&out_dir.join("east.tif"), out.east.as_slice().unwrap(), &ma);
    println!("OK → {}/uplift.tif + east.tif", out_dir.display());
}

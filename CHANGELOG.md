# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/es/1.1.0/);
versionado: [SemVer](https://semver.org/lang/es/).

## [Unreleased]

## [0.2.0] — 2026-07-17

Primer release publicado a crates.io/PyPI. Reúne el trabajo del backlog v0.2
acumulado desde el tag v0.1.0 (nunca publicado): pipeline ISCE end-to-end
unificado, backend SNAPHU, corrección troposférica ERA5, mmap del camino ISCE
y un batch de hardening numérico/validación.

### Backend SNAPHU opcional para el desenrollado 2D (G-3 del backlog v0.2)

Shell-out a un binario `snaphu` instalado por separado (patrón estándar del
ecosistema InSAR — MintPy/ISCE2 hacen lo mismo, vía Python; sin FFI ni
vendorizar SNAPHU dentro de insar-rs, cero dependencias Rust nuevas).
Verificado de punta a punta con SNAPHU real (conda-forge): resultados
idénticos al flood-fill propio sobre el stack sintético de referencia. Ver
`docs/auditoria-2026-07-02.md` (G-3) para el detalle de la decisión.

- `unwrap::snaphu`: `unwrap_2d_snaphu`/`unwrap_stack_snaphu` (mismo
  contrato NoData que `unwrap::unwrap_2d`), `SnaphuConfig` (ruta al
  binario). Formato `FLOAT_DATA`/`STATCOSTMODE SMOOTH` (mismo modo ya
  usado por `validation/isce_unwrap.py` en este proyecto).
- `pipeline`: `SbasPipelineConfig.unwrap_backend: UnwrapBackend`
  (`FloodFill` default, sin cambios de comportamiento; `Snaphu(SnaphuConfig)`
  opcional).
- CLI: `insar run --unwrap-backend {flood-fill,snaphu} [--snaphu-bin PATH]`.
  El subcomando `isce` no cambia (no invoca `unwrap::*` — los `.unw` llegan
  ya desenrollados).

### Escalabilidad de memoria: mmap del camino ISCE (G-9 del backlog v0.2)

Investigación previa (3 agentes en paralelo) acotó el alcance: el unwrap 2D
(flood-fill) bloquea el tiling espacial completo sin rediseño; `surtgis-core`
no soporta lectura por ventana (cross-repo, fuera de alcance). Se implementó
la parte autocontenida y de bajo riesgo — ver `docs/auditoria-2026-07-02.md`
(G-9) para el detalle de la decisión y lo diferido.

- `io::isce`: `read_raw_band`/`read_raw_band_complex`/`read_raw_band_byte`/
  `read_unw_phase_masked` mapean el archivo con `memmap2` en vez de
  `fs::read` a un `Vec<u8>` — evita duplicar en RAM el contenido de rasters
  `.unw`/`.int`. Primer `unsafe` del crate (inevitable para mmap, acotado a
  una función de 3 líneas con la invariante documentada).

### Exponer los diferenciadores (G-17 del backlog v0.2)

`features`, `decompose` y `deramp` (`postprocess::remove_ramp`/
`deramp_series`) ya existían y estaban testeados, pero solo eran alcanzables
escribiendo Rust contra `examples/`. `unwrap_error::correct_unwrap_errors` ya
estaba integrado en `run`/`isce` de la CLI; le faltaba el binding Python.

- `io`: lectores `read_velocity`/`read_series`, simétricos de
  `write_velocity`/`write_series` (permiten releer un `VelocityMap`/
  `DisplacementSeries` ya escritos por un pipeline previo).
- `features`: `FeatureMaps::write_features_csv` (tabla `x,y,<features>`,
  además de los GeoTIFF individuales de `write_geotiffs`).
- CLI: subcomandos nuevos `decompose` (LOS asc+desc → up.tif/east.tif,
  geometría escalar), `features` (`--csv` opcional) y `deramp` (standalone).
- Python (`insar_rs`): `decompose_asc_desc`, `decompose_per_pixel`
  (geometría por píxel — sin superficie externa hasta ahora),
  `extract_features` (dict de arrays), `remove_ramp`, `correct_unwrap_errors`.

## [0.1.0] — 2026-07-02

Primera versión funcional del motor: SBAS end-to-end validado contra MintPy
(paridad numérica r = 1.000000 en el camino OLS) con tres casos chilenos
reales y el path SLC-stack coregistrado (burst2safe → topsStack → insar-rs)
probado con datos Sentinel-1.

### Núcleo (insar-core)

- Inversión SBAS por incrementos (Berardino et al. 2002): OLS con
  pseudoinversa cacheada por patrón de validez; **WLS por coherencia**
  (esquemas `coh` y `var`/Cramér-Rao, Yunjun et al. 2019); **inversión
  robusta L1 por IRLS** (Lauknes et al. 2011) con análisis de
  identificabilidad documentado; **corrección de error de DEM** (columna
  ∝ B⊥, Fattahi & Amelung 2013) con Δz por píxel.
- Velocidad: ajuste lineal + SE formal, **bootstrap de épocas**
  (determinista, SplitMix64) y **modelo temporal** con polinomio,
  componentes periódicas y saltos cosísmicos (estilo
  `timeseries2velocity`).
- Desenrollado 2D quality-guided (flood-fill por coherencia, islas
  re-sembradas, umbral `min_quality`); **corrección de errores de
  desenrollado por cierre de fase** con verificación de efectividad y
  reporte (`corrected` / `detected_uncorrected`); QC
  `nonzero_closure_count` (≙ `numTriNonzeroIntAmbiguity` de MintPy).
- Corrección APS espacio-temporal (ajuste lineal local en tiempo real —
  exacto con gaps de adquisición); corrección troposférica estratificada
  fase-elevación (Doin et al. 2009); deramp polinomial; referenciado
  espacial y coherencia temporal (Pepe & Lanari 2006).
- Descomposición LOS → (Up, East) con geometría escalar o **por píxel**
  (incidencia/heading de `los.rdr`), y conversión azimut-ISCE → heading.
- I/O: formato de stack propio (`stack.json` + GeoTIFF, coherencia
  opcional) vía writer nativo de SurtGIS (sin GDAL); **lector ISCE nativo**
  (VRT + raw binario): `.unw` (con enmascarado NoData por amplitud),
  `.int` (CFloat32), `.cor`, `.unw.conncomp` (Byte), `los.rdr`, baselines
  de topsStack; dtypes Float32/Float64/CFloat32/Byte con verificación de
  cotas checked.
- Features por píxel para ML (velocidad, aceleración, estacionalidad, R²,
  saltos) con ajuste sobre épocas finitas y esquema de tabla determinista;
  tabla lista para smelt-ml con coordenadas para CV espacial.
- Pipeline `run_sbas` integrado con el orden físico documentado: unwrap
  (calidad) → cierre → referencia → inversión → APS → deramp → productos
  (`velocity`, `series/`, `temporal_coherence`, `dem_error`, `closure_qc`).

### CLI (insar)

- `info`, `ps`, `network`, `run` (pipeline con `--min-quality`, `--deramp`,
  `--no-closure-correction`), `isce` (SBAS directo con `--wls`, `--robust`,
  `--dem-error-range`, `--deramp` y productos de QC).

### Python (insar-rs en PyPI, módulo `insar_rs`)

- `invert_sbas`, `estimate_velocity`, `estimate_velocity_uncertainty`,
  `amplitude_dispersion`, `temporal_coherence`, `sbas_from_isce`
  (end-to-end). Cómputo sin GIL (`allow_threads`); errores mapeados a
  excepciones idiomáticas (`OSError` / `ValueError` / `RuntimeError`).

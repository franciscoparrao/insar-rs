# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/es/1.1.0/);
versionado: [SemVer](https://semver.org/lang/es/).

## [Unreleased]

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

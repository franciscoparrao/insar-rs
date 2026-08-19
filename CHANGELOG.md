# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/es/1.1.0/);
versionado: [SemVer](https://semver.org/lang/es/).

## [Unreleased]

### Corrección de phase bias (sesgo de fase de no-cierre)

Nuevo módulo `phase_bias` e integración como paso 3 de `run_sbas`
(`SbasPipelineConfig::phase_bias`, `SbasProducts::phase_bias_report`).
Implementa la corrección empírica de Maghsoudi et al. (2022), RSE 275:113022
(Eqs. 2–13) **dentro del pipeline nativo**, no como post-proceso externo.

Corrige el sesgo sistemático que el multilooking introduce en los
interferogramas cortos (la *fading signal*): mayor cuanto más corto el par y
más decorrelaciona el terreno, y que en una cadena SBAS se acumula hasta
producir velocidades sesgadas — imita subsidencia en cultivo y bosque.

**No es lo que hacía `unwrap_error`**, y conviene que quede escrito porque los
dos módulos usan la misma palabra "cierre": `unwrap_error` resuelve saltos
**enteros** de 2π (Yunjun et al. 2019) sobre fase desenrollada; `phase_bias`
resuelve la parte **fraccional** del cierre —justo lo que el otro redondea a
cero y descarta— sobre fase **envuelta**. Son ortogonales y van en puntos
distintos del pipeline: phase bias antes de desenrollar (es una perturbación
de la fase envuelta; corregirlo después obligaría a re-desenrollar), errores de
desenrollado después. El doc del módulo lleva la tabla comparativa.

Que la estimación trabaje sobre fase envuelta —cierres calculados como
argumento del producto complejo, no restando fases desenrolladas— es lo que
hace la corrección barata: no necesita desenrollar, ni *phase linking*, ni
interferogramas largos coherentes salvo un ancla para los coeficientes.

Generalizaciones sobre el paper, que está escrito para muestreo regular de
6 días y span máximo 3:

- **Span arbitrario `M`** indexado por **orden de época**, no por días de
  calendario. Es la generalización correcta a muestreo irregular y coincide con
  lo que el paper hace de facto ("interferograms that connect each epoch i to
  the three or four nearest acquisitions in time"). La fila del diseño lleva
  `(aₙ−1)` en las n columnas de la cadena, que recupera la Eq. (10) para M=3.
- **Coeficientes estimados de los datos** (`estimate_coefficients`) en vez de
  reusar los 0,47 / 0,31 publicados: el paper muestra que dependen de la
  cobertura del suelo, así que son específicos de escena. Se estiman como
  **cociente de acumulados** y no como media de los cocientes por píxel que
  calcula el paper — los mapas por píxel son ruidosos (el propio paper lo dice,
  Fig. 8) y su cociente explota donde el denominador cruza cero, así que la
  media muestral no es estimador estable de la razón.
- **Red incompleta**: un par unitario faltante no solo elimina su incógnita,
  invalida toda observación que lo cruce (la Eq. 2 necesita su fase para formar
  el cierre). La topología lo maneja explícitamente.

Productos adicionales: `closure_rms` (QC antes/después), `cumulative_bias`
(firma temporal del sesgo, Fig. 4 del paper) y `high_closure_mask`.

**Hardening anti no-op silencioso**, siguiendo la doctrina de GACOS. Son
errores duros, no reportes que nadie mira:

- `max_span = 2` **nunca** determina el sistema: N−2 observaciones para N−1
  incógnitas, sea cual sea N. Hacen falta dos longitudes de par largo distintas
  (spans 2 y 3) para cerrarlo. Sin este chequeo, una red de solo consecutivos y
  saltos de 2 devolvía éxito con el stack intacto.
- Cero píxeles corregidos al terminar.
- Escena sin sesgo medible: el denominador de las Eqs. 8–9 es el sesgo
  acumulado, y si es ~0 los coeficientes son un cociente de ruidos. El umbral
  es por píxel y en radianes — compararlo contra una escala derivada de las
  propias sumas sería circular, porque el denominador suele ser la mayor de
  ellas y la prueba pasaría siempre.
- Coeficiente fuera de rango físico (el modelo dice que el sesgo decae con el
  largo del par: 0 < aₙ < 1).
- Sin par ancla teselable para estimar coeficientes.

Píxeles cuyo sistema reducido queda rank-deficiente se dejan **intactos** y se
cuentan en `pixels_skipped`: no se les aplica la solución de norma mínima, que
repartiría el cierre en fracciones arbitrarias entre incógnitas que las
observaciones no restringen.

15 tests unitarios (topología y observaciones 2N−5, par unitario faltante,
recuperación exacta de los coeficientes verdaderos, escena sin sesgo, sin
ancla, recuperación del sesgo inyectado dejando la deformación intacta, caída
del RMS de cierre, stack sin sesgo no se toca, píxel inválido intacto,
preservación de la amplitud —la corrección es una rotación compleja—, red de
span 2 indeterminada, sesgo acumulado, máscara de cierre alto) más un test
**end-to-end del pipeline** (`tests/phase_bias_e2e.rs`) que es la prueba del
argumento: sobre un stack sintético de 10 épocas con sesgo de 0,15 rad por par
de 12 días, la velocidad sale sesgada en **8,6 mm/año** (43 % de la señal
verdadera de −20 mm/año) y la corrección la devuelve a 0,00 mm/año de error,
con el RMS de cierre cayendo de 0,242 a 1,1e−8 rad.

Limitación conocida: el camino `run_sbas_isce` lee `.unw` ya desenrollados, así
que la corrección **no aplica** ahí — requiere fase envuelta.

### Corrección troposférica GACOS (P0.2 del roadmap de datos abiertos)

Nuevo módulo `troposphere::gacos` y subcomando `insar tropo-gacos`. Es la
**primera vía troposférica del motor operable end-to-end**: lee los pares
`YYYYMMDD.ztd` + `.ztd.rsc` tal como los entrega el servicio GACOS, los
remuestrea bilinealmente a la grilla de la serie, proyecta a LOS por la secante
de la incidencia y aplica el diferencial. Sin credenciales, sin dependencias
nuevas y sin resolver perfiles atmosféricos fuera del motor.

Contraste con `troposphere::era5` (G-7), que mantiene abierto el gap A-9 (no
descarga ni remuestrea horizontalmente): además de operabilidad, GACOS gana en
resolución para AOIs chicos — ERA5 tiene celdas de ~31 km, de modo que un AOI de
~10 km cae dentro de una sola celda y el reanálisis no puede resolver ningún
gradiente topo-correlacionado *dentro* del AOI; GACOS entrega ZTD a **~90 m**
(0,000833°, interpolado con SRTM; el paso real siempre se lee del `.rsc`).

La corrección de serie hace **streaming**: carga, remuestrea y libera un mapa a
la vez — nunca materializa los N mapas simultáneamente (a 90 m, un frame
continental son cientos de MB por época; el camino ingenuo escala a GBs).
`load_for_epochs` queda como building block explícitamente marcado con esa
advertencia. La época de referencia se salta (su corrección es idénticamente 0)
y las métricas del reporte quedan definidas sobre las épocas no-referencia.

Incluye **referenciado espacial**, que `era5` no hace: una serie InSAR está
referenciada en tiempo *y* en espacio, así que la corrección aplica el doble
diferencial `(D[e,p] − D[e₀,p]) − (D[e,p_ref] − D[e₀,p_ref])`. Omitirlo
introduce un offset uniforme espurio cuando la serie sí fue referenciada a un
píxel.

Queda documentada además una **trampa metodológica**: si el objetivo científico
es testear una relación entre la señal y la elevación, `correct_topo_correlated`
no sirve como corrección previa — al ajustar y remover una pendiente
fase-elevación global deja residuos de signo arbitrario por subregión y vuelve
circular ese test. La tabla de "cuál usar" está en el doc del módulo.

**Hardening anti no-op silencioso** (auditoría 2026-08-01): los estados que
convertían la corrección en un passthrough con exit 0 — época de referencia sin
mapa GACOS, píxel de referencia fuera de la cobertura, o cero píxeles
corregidos al terminar — ahora son **errores duros** con mensaje accionable, y
las épocas con mapa pero sin retardo finito en el píxel de referencia se
reportan en `GacosReport::epochs_skipped` en vez de saltarse en silencio.
Verificado end-to-end: los dos gatillos reproducidos contra el binario pasan de
"escrito: serie corregida (cobertura 0.0 %)" a exit 1.

15 tests unitarios (parseo `.rsc`, binario de tamaño inconsistente, remuestreo
identidad incluyendo bordes, remuestreo grueso→fino exacto en campo lineal —el
camino de producción—, no-extrapolación fuera de cobertura, diferencial
temporal, referenciado espacial, época faltante, única época corregible sin
mapa, directorio vacío, proyección a LOS, y los 4 del hardening: referencia sin
mapa, refpixel fuera de cobertura, época saltada reportada, serie toda-NaN) más
verificación end-to-end por CLI
sobre una serie sintética con retardo espurio en columna: recupera la
deformación verdadera exacta (−10,00 y −20,00 mm) y anula el gradiente espurio
(49,33 mm → 0,000 mm).

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

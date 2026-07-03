# Auditoría del motor insar-rs

**Fecha**: 2026-07-02
**Scope**: los 3 crates del workspace (`core` 14 módulos, `cli`, `python`), tests, benches, examples, empaquetado y docs. ~6.600 líneas de Rust.
**Método**: revisión en 2 planos — (1) calidad de código (7 dimensiones: complejidad, acoplamiento, cohesión, abstracciones, error handling, naming, deuda) y (2) rigor científico InSAR contra el estado del arte (MintPy, StaMPS, SNAPHU, ISCE2). 4 revisores paralelos por clúster de módulos + clippy + lectura de los contratos centrales.

---

## Resumen ejecutivo

| Métrica | Resultado |
|---|---|
| Clippy (workspace, all-targets) | **0 warnings** en insar-rs |
| Tests unitarios | **113** `#[test]` en 14/14 módulos, con ground truth analítico (no smoke tests) |
| TODOs/FIXMEs/código muerto | esencialmente **cero** (1 falso positivo) |
| Convención Result-sin-panic | respetada, con **1 excepción real** (`temporal_coherence`) |
| Paridad MintPy | r=1.000000 declarada — **condicionada** (ver hallazgo C-2 y gap G-2) |
| Calidad general | **Excelente para un MVP** — la deuda no está en el código sino en 3 bugs silenciosos, la integración del pipeline y el empaquetado |

**Veredicto**: la matemática central es correcta y en algunos puntos *más rigurosa* que las implementaciones de referencia (manejo per-píxel de dropout con re-verificación de conectividad de la red reducida; convolución normalizada con máscara; geometría LOS verificada idéntica a la convención MintPy right-looking). La distancia a "mejor motor InSAR existente" está en cuatro frentes, en este orden: **(A)** tres bugs/limitaciones que fallan *en silencio* sobre datos reales, **(B)** el pipeline desintegrado (los módulos diferenciadores existen pero `run_sbas` no los invoca), **(C)** los gaps algorítmicos frente a MintPy/StaMPS (WLS, DEM error, closure bias, unwrapping global), y **(D)** distribución (hoy es un excelente motor *privado*: nadie externo puede compilarlo).

---

## 1. Hallazgos — CRÍTICOS y ALTOS

### [C-1] Path-dep obligatoria a surtgis-core bloquea crates.io y PyPI
- **Archivo**: `Cargo.toml:37`, `crates/core/Cargo.toml`
- **Dimensión**: distribución
- `surtgis-core = { path = "../surtgis/crates/core" }` sin `version` es dependencia normal (no dev): `cargo publish` la rechaza, y `maturin sdist`/`pip install` desde fuente es imposible (la ruta apunta fuera del repo). Las path-deps dev (geostat, smelt, swarm) son benignas — cargo las strippea al publicar.
- **Fix**: publicar `surtgis-core` en crates.io y usar `{ version = "0.x", path = "..." }` (path para dev, version para publish). Plan B: feature-gate del I/O GeoTIFF para que el núcleo numérico compile sin surtgis. Plan C interino: dependencia `git` con rev pinneado.

### [C-2] Ceros de ISCE entran como fase válida (viola NoData=NaN)
- **Archivo**: `crates/core/src/io/isce/mod.rs:357-434`
- **Dimensión**: InSAR-IO / correctness científica
- ISCE escribe `0.0` en la banda de fase de `filt_fine.unw` donde el unwrapping falló o hay agua/máscara. El reader lee solo la banda 2 y conserva esos ceros como fase legítima → `invert_sbas` los trata como observaciones válidas de fase 0, **sesgando velocidades cerca de zonas enmascaradas**. Además condiciona la validación contra MintPy, que sí enmascara.
- **Fix**: leer también la banda 1 (amplitud) y poner NaN donde `amp == 0`, detrás de un flag `mask_zero_amplitude: bool = true` en `IsceLoadConfig`; idealmente aceptar umbral de coherencia con el `.cor`.

### [A-1] `temporal_coherence` puede hacer panic con inputs no validados
- **Archivo**: `crates/core/src/inversion/mod.rs:392-441`
- **Dimensión**: error handling
- A diferencia de `invert_sbas` (que llama `stack.validate()`), no valida stack ni serie: si `pairs.len() != data.shape()[0]` o `series.data.shape()[0] != epochs.len()`, el indexado ndarray hace panic. Única violación real de la convención "sin panic en rutas públicas". Fix de 3 líneas: `stack.validate()?` + verificar `series.n_layers() == series.epochs.len()`.

### [A-2] Corrección por cierre de fase silenciosamente nula en redes ralas
- **Archivo**: `crates/core/src/unwrap_error/mod.rs:119-130, 181-193`
- **Dimensión**: InSAR-proc
- La solución L2 de norma mínima redondeada a enteros falla en el caso mínimo: con 1 solo lazo (3 pares), un salto de 2π da U=±1/3 en cada par → `round()` → 0 en todos: **se detecta el cierre pero no se corrige nada, sin señal al caller**. Las redes SBAS reales son ralas; el propio test usa una red completa "para que la redundancia localice el salto".
- **Fix corto**: recomputar cierres tras aplicar U; si no bajaron, revertir y devolver `{corrected, detected_uncorrected}` en vez de `usize`. **Fix real**: solución entera L1/ILP (recomendación explícita de Yunjun et al. 2019).

### [A-3] Ventana temporal del APS en índices de época, no en días
- **Archivo**: `crates/core/src/atmosphere/mod.rs:112-147`
- **Dimensión**: InSAR-proc
- La media móvil asume muestreo equiespaciado; Sentinel-1 real tiene gaps (12→24→48 días, S1B muerto, inviernos sin adquisiciones): la separación deformación/atmósfera se degrada exactamente en los huecos. `series.epochs` está disponible y no se usa. MintPy/StaMPS filtran con gaussiana **en días**.
- **Fix**: media/gaussiana temporal ponderada por Δt real. Es el gap más barato de cerrar con impacto directo en la validación vs MintPy.

### [A-4] Los bindings PyO3 nunca liberan el GIL
- **Archivo**: `crates/python/src/lib.rs` (0 ocurrencias de `allow_threads`)
- `sbas_from_isce` (lectura de 288 pares + búsqueda de referencia + inversión: segundos a minutos) corre con el GIL tomado — bloquea cualquier otro thread Python (dask, GUI, servidores). Las entradas ya son owned: envolver la llamada al core en `py.allow_threads(|| ...)` es directo.

### [A-5] Higiene de release ausente (licencia, numpy, CI)
- `license = "MIT OR Apache-2.0"` declarada **sin archivos** LICENSE-MIT/LICENSE-APACHE → sin licencia efectiva; es lo primero que mira un revisor JOSS/ESIN.
- `pyproject.toml` sin `dependencies = ["numpy>=1.23"]` → `import insar_rs` en venv limpio falla con ImportError críptico.
- Sin `.github/` (CI), sin CHANGELOG, sin tags: nada verifica que los 113 tests sigan verdes; sin wheels multi-plataforma (el abi3-py39 ya elegido hace los wheels casi gratis con maturin-action).

---

## 2. Hallazgos — MEDIOS (selección con impacto real)

| # | Hallazgo | Archivo | Fix |
|---|---|---|---|
| M-1 | **Lazos con NaN = "cierre perfecto"**: un par NaN pone RHS=0 pero la fila sigue en la pinv → asevera cierre falso, sesga correcciones a 0 | `unwrap_error/mod.rs:163-169` | cachear pinv por máscara de pares válidos (patrón que ya existe en `inversion:141-178`) |
| M-2 | **Cache de pseudoinversas sin cota**: NaN post-unwrap pueden generar O(n_píxeles) patrones únicos → GBs en el HashMap | `inversion/mod.rs:144-180` | cota N_max + resolver patrones raros on-the-fly (o LRU) |
| M-3 | **APS todo-o-nada por píxel**: 1 NaN en una época excluye el píxel de la corrección en TODAS las épocas → APS en parches | `atmosphere/mod.rs:121-124` | media móvil sobre épocas finitas (contador de válidos por ventana) |
| M-4 | **Decompose con θ constante**: incidencia S1 IW varía 30°→46° a lo ancho del swath → error sistemático ~10-15% en Up/East en los bordes | `decompose/mod.rs:45,84` | sobrecarga con `Array2` de incidencia/heading por píxel (sistema 2×2 por píxel, costo trivial) |
| M-5 | **Decompose sin contrato de grillas/fechas asc-desc**: asc y desc nunca comparten grilla nativa ni fechas; lo estándar es descomponer velocidades sobre grilla geocodificada común | `decompose/mod.rs:98-107` | documentar contrato (resample vía SurtGIS) + helper `decompose_velocity` |
| M-6 | **Pipeline desintegrado**: `run_sbas` no llama a `correct_unwrap_errors`, `reference_to_pixel`, `correct_topo_*` ni `remove_ramp`; la coherencia no llega al unwrapper. Además el orden físico correcto es: unwrap → closure-correction → invert → reference → **tropo estratificada → APS turbulento** → deramp (la estratificada es estacional: el filtro HP-temporal NO la remueve) | `pipeline/mod.rs` | integrar los módulos con ese orden como contrato documentado |
| M-7 | **`read_baseline` degrada silenciosamente a 0.0** con typo en `baselines_dir` o formato distinto → Bperp=0 invalida la futura corrección de DEM error | `io/isce/mod.rs:317-349` | `Err` si el usuario configuró el dir y `count==0` |
| M-8 | **Épocas duplicadas no se detectan** en `stack.json`: `write_series` sobrescribe `disp_YYYYMMDD.tif` silenciosamente; par entre duplicadas → matriz singular con error críptico | `io/mod.rs:109-119, 305-322` | exigir épocas estrictamente crecientes en `load_manifest` |
| M-9 | **`min_valid_epochs` en features no hace lo que dice**: doc promete semántica por píxel; implementación exige TODAS las épocas finitas → con decorrelación parcial mata la gracia del módulo (features ML sobre píxeles imperfectos) | `features/mod.rs:58-59, 218-229` | subsetear filas de la matriz de diseño a épocas finitas si `n ≥ max(min_valid_epochs, n_coef)`, pinv cacheada por máscara |
| M-10 | **Esquema de columnas de `to_table` depende de los DATOS**: `feature_names()` decide si "acceleration"/"seasonal_*" existen escaneando finitud → dos corridas con la misma config pueden dar distinto número de columnas y romper silenciosamente un modelo Smelt entrenado | `features/mod.rs:321-371` | guardar `FeatureConfig` dentro de `FeatureMaps`; esquema determinista |
| M-11 | **Errores I/O sin ruta**: `InsarError::Io` propaga "No such file (os error 2)" sin decir cuál de los 288 pares faltó | `io/mod.rs:98`, `io/isce/mod.rs:82,217,271`, `cli/main.rs` | helper `read_with_ctx(path)` o variante `Io { path, source }`; en CLI, `.with_context()` |
| M-12 | **Selección de píxel de referencia duplicada carácter por carácter** entre CLI y PyO3 (~35 líneas, loop serial O(rows·cols·pares)) | `python/lib.rs:223-250` vs `cli/main.rs:98-130` | mover a `inversion::select_reference_pixel()` con rayon |
| M-13 | **Mapeo de errores plano en Python**: todo `InsarError` → `PyValueError`; un directorio inexistente debería ser `FileNotFoundError` | `python/lib.rs:28-30` | `impl From<InsarError> for PyErr` con match por variante |
| M-14 | **Cuádruple duplicación** del patrón "acumular capas + check dims + from_shape_vec" | `io/mod.rs:190-231, 252-277`; `io/isce/mod.rs:372-424, 443-474` | extraer `StackBuilder` |
| M-15 | **Alocación antes de validar** contra tamaño de archivo: VRT malformado con dims gigantes → OOM abort en vez de `Err`; multiplicaciones sin `checked_mul` en la verificación de cota | `io/isce/mod.rs:220-245` | mover chequeo de cota antes de `Array2::zeros` + `checked_mul` |
| M-16 | **`postprocess` fragmentado**: `reference_to_pixel` y `temporal_coherence` viven en `inversion`; falta `coherence_mask(gamma, thr) -> Array2<bool>` que alimente `remove_ramp`/`correct_topo_*` | `postprocess/mod.rs` vs `inversion/mod.rs:40,392` | re-exportar/mover + helper de 5 líneas |
| M-17 | **Unwrapping sin umbral de calidad mínima**: píxeles de coherencia ~0 igual se integran al árbol y propagan error aguas abajo | `unwrap/mod.rs:79` | parámetro `min_quality: Option<f32>` |
| M-18 | **Cero tests del lado Python** (ninguna regresión automática de orden de ejes/dtype/errores) | `crates/python/` | `tests/test_bindings.py` con el caso sintético del e2e, vía `maturin develop && pytest` |
| M-19 | **PLAN.md/README no reflejan la mitad del crate**: `features`, `decompose`, `troposphere`, `postprocess` invisibles — justo los módulos diferenciadores | `PLAN.md:41-52`, `README.md` | actualizar tabla de arquitectura + sección de capacidades |

## 3. Hallazgos — BAJOS (lista corta)

- `pixel_dispersion`: umbral absoluto `f32::EPSILON` para μ≈0 (escala-dependiente; amplitudes negativas no rechazadas) → umbral relativo + `v<0 → NaN` (`ps/mod.rs:67`). Política todo-o-nada de NaN reduce cobertura (StaMPS/MintPy usan `n_valid ≥ min`).
- `estimate_velocity` / `estimate_velocity_uncertainty`: duplicación estructural (doble pasada sobre el cubo) → `fit_linear()` compartido; asimetría de API (`VelocityMap` vs `Array2` pelado, se pierde el meta para escribir GeoTIFF) (`inversion/mod.rs:257-380`).
- `invert_sbas` ~150 líneas / 3 pasadas → extraer `collect_partial_masks`, `build_reduced_solvers`, `invert_pixel_rows`.
- `reduced_pinv` reimplementa la matriz de diseño inline (segunda fuente de verdad vs `network::design_matrix`) (`inversion/mod.rs:244-247`).
- `find_seed` re-escanea toda la imagen por isla → O(N·islas) con máscaras reales; pre-ordenar por calidad + cursor (`unwrap/mod.rs:187-215`).
- Acumulación de fase desenrollada en f32 por caminos largos → acumular en f64 (`unwrap/mod.rs:141`).
- Triple duplicación del boilerplate LSQ-SVD (`troposphere:118-125`, `postprocess:94-101`, `inversion`) → helper `lstsq_svd()`.
- `build_closure_loops` asume `reference < secondary` en silencio; lazos O(C(n,3)) sin cap por Δépoca (`unwrap_error/mod.rs:48-77`).
- Cita a Bekaert 2015 engañosa en troposphere (lo implementado es el modelo global tipo Doin 2009); sin advertencia de que en volcanes el edificio ES la topografía → `correct_topo_correlated` sin mask borra la señal de inflación — caso de uso declarado del proyecto.
- Referenciado solo a píxel único (MintPy promedia ventana) → `reference_to_region(row, col, radius)`.
- Parser VRT: atributo `band` ignorado (asume orden documental), `relativeToVRT="0"` no manejado, `pixel_offset==0` no rechazado, loop de decodificación píxel a píxel sin fast-path contiguo.
- Georreferencia entre archivos del stack no verificada (se toma la del primero); `incidence_deg` sin validar.
- e2e: usar `tempfile::TempDir` (RAII) + assert apretado (1e-6) sobre el caso sin APS; benchmark sin guardia anti-regresión en CI.
- CLI: `run` no informa rutas de salida (`isce` sí); sin subcomandos deramp/features/unwrap_error; sin smoke test `assert_cmd`.
- `dummy_meta` en bindings fija `incidence_deg=39.0` silenciosa — inocuo hoy, incorrecto sutil cuando el core la consuma.
- Helpers `read_f32()`/`Meta` copiados en ≥4 examples → `examples/common/mod.rs`.
- Pirámide de 9 `.zip()` de rayon en features → `ndarray::Zip` o struct de vistas por fila.

---

## 4. Gaps vs estado del arte — el camino a "mejor motor InSAR"

Priorizados por (impacto científico × costo). Los primeros 5 definen la credibilidad del paper; los siguientes definen el diferenciador.

### Nivel 1 — condicionan la validez de resultados sobre datos reales

- **G-1 · WLS por coherencia**. Toda la inversión es OLS sin pesos. MintPy ofrece pesos por varianza de fase/coherencia/Fisher info (Yunjun et al. 2019). Mejora de exactitud de mayor retorno: `W^{1/2}A x = W^{1/2}b` reutilizando la misma maquinaria SVD. Requiere que la coherencia entre a `invert_sbas` (hoy `read_isce_coherence` devuelve un Array3 desacoplado).
- **G-2 · Corrección de error de DEM (residuo topográfico ∝ B⊥)**. `IfgPair.perp_baseline_m` ya existe y **no se usa en la inversión**. Agregar la columna `(4π/λ)·B⊥/(r·sinθ)` estima Δz por píxel junto con la serie (Fattahi & Amelung 2013). Sin esto, la paridad r=1.0 vs MintPy solo se sostiene si MintPy corrió sin `dem_error` — conviene explicitarlo en `docs/validation.md`.
- **G-3 · Unwrapping global**. El flood-fill quality-guided integra por camino único: un residuo atravesado propaga error a todo el subárbol, sin optimización global. Mitigación corta: umbral de coherencia + closure correction arreglada (A-2). Para paridad real sobre datos ruidosos: MCF/network-flow propio, o binding opcional a SNAPHU como backend (como hace todo el ecosistema).
  - **Cerrado (2026-07-03)**: backend SNAPHU opcional vía shell-out (`unwrap::snaphu`, `pipeline::UnwrapBackend::Snaphu`, CLI `--unwrap-backend snaphu`). Investigado antes de implementar: SNAPHU no estaba disponible como binario standalone en la máquina (solo embebido dentro de ISCE2, inaccesible desde Rust); se instaló vía `conda install -c conda-forge snaphu` (paquete Python moderno `isce-framework/snaphu`, vendoriza el binario C real dentro de `site-packages/snaphu/snaphu`) para verificar la integración de punta a punta. Se eligió shell-out sobre FFI/vendorizar (evita temas de licencia de redistribución de C ajeno, cero dependencias Rust nuevas, mismo patrón que MintPy/ISCE2). Verificado con el binario real: recupera una rampa sintética exacta (con y sin coherencia, y en el paralelismo por capa de `unwrap_stack_snaphu`), y el pipeline CLI completo (`insar run --unwrap-backend snaphu`) da resultados **idénticos** (correlación 1.0) al flood-fill propio sobre el stack sintético de referencia. Fuera de alcance de este cierre: variante `_min_quality`, modo `--tile` de snaphu (relevante para G-9), y agregar snaphu a la matriz de CI (decisión de costo/beneficio aparte).
- **G-4 · Términos temporales en velocidad**. MintPy ajusta polinomio + anual/semianual + steps cosísmicos. El fit lineal puro sesga v con estacionalidad fuerte (valles agrícolas chilenos: crítico). Extensión natural del mismo LSQ — y `features/` ya ajusta seasonal: reutilizar.
- **G-5 · Phase closure / bias mitigation como QC**. Nada de triplets cerrados (`φ_ij+φ_jk−φ_ik≠0`) ni fading-signal bias (Ansari 2021; Zheng 2022, en MintPy como `closure_phase_bias`). El conteo de non-zero closure como máscara de calidad es barato, y MintPy lo reporta como producto estándar (`numTriNonzeroIntAmbiguity`).

### Nivel 2 — completitud del pipeline físico

- **G-6 · Geometría ISCE por píxel**: leer `merged/geom_reference/{lat,lon,hgt,los}.rdr` → incidencia por píxel (29°–46° en IW, hoy escalar 39.0 hardcodeado), heading real, geocodificación de salidas. Depende de G-8.
- **G-7 · Troposfera por reanálisis (ERA5/GACOS)**: la empírica fase-elevación no separa deformación topo-correlacionada del retardo. Ya declarado v0.2 — mantener como prioridad 1 de esa versión.
- **G-8 · dtypes del reader ISCE**: solo Float32. Sin `CFloat32` no se leen `.int`/`.slc` (¡el `IfgStack` central no es alimentable desde ISCE, solo `UnwrappedStack`!); sin `Byte` no se lee `.conncomp` (EL producto para descartar errores de unwrap por componente conexa). Fix pequeño (~50 líneas genéricas sobre dtype), gap individual más limitante del I/O.
- **G-9 · Escalabilidad de memoria**: todo en RAM (`fs::read` completo incluyendo la banda de amplitud que se descarta: 2× I/O y RAM), stack denso `Array3`. Para full-res (problema ya vivido: disco/RAM explotan) se necesita mmap (`memmap2` mapea directo el layout por offsets) + lectura por tiles/ventanas + multilooking on-the-fly. **Decisión arquitectónica: tomarla antes de que más código asuma `Array3` denso.**
  - **Investigado y parcialmente cerrado (2026-07-03)**: se exploró con 3 agentes en paralelo (I/O actual, tileabilidad por algoritmo, capacidades de `surtgis-core`/ecosistema) antes de decidir el alcance. Hallazgos que acotan la decisión: (a) el unwrap 2D (`unwrap::unwrap_2d`, flood-fill quality-guided) **no es tileable sin rediseño real** — propaga fase por vecindad 4-conexa a través de toda la componente conexa; un halo fijo no basta, hace falta *stitching* entre tiles — es el único paso genuinamente bloqueante del pipeline; todo lo demás (inversión, cierre de fase, deramp, velocidad, features, decompose) es per-píxel en el eje temporal o "fit global barato (3-6 coeficientes) + apply local", trivialmente tileable; (b) `surtgis-core` (el reader GeoTIFF que usa `io/mod.rs`) no tiene lectura por ventana ni mmap — agregarla es trabajo cross-repo, fuera de alcance de una sesión; (c) la corrección "2× I/O" del párrafo original era imprecisa: el caso estándar ISCE (banda amplitud+fase en el mismo archivo) ya se leía una sola vez del disco (comentario del propio módulo), el costo real era 2× **RAM transitoria** (buffer `raw: Vec<u8>` + `Array2` de amplitud descartada tras el masking), no I/O duplicada.
  - **Hecho**: mmap (`memmap2`) del camino ISCE (`io::isce::{read_raw_band, read_raw_band_complex, read_raw_band_byte, read_unw_phase_masked}`), que es autocontenido en insar-rs y ya conoce los offsets exactos por banda vía el `.vrt` — sin tocar `surtgis-core` ni el contrato de `types.rs`. Primer `unsafe` del crate (inevitable para mmap, acotado a una función de 3 líneas con la invariante documentada). Medido contra Fernandina (98 épocas/288 pares/450×600, release): el pico de RSS del proceso **no cambia de forma medible** (~600 MiB antes y después) porque está dominado por el `Array3` acumulado del stack completo (~296 MiB), no por el buffer transitorio por-archivo que elimina el mmap (~2 MiB por par) — la ganancia real es de robustez (sin exigir una asignación contigua del tamaño del archivo completo) y es la base para datasets de mayor resolución por archivo (bursts sin recortar), no una mejora medible a esta escala.
  - **Diferido explícitamente**: tiling espacial 2D de cualquier algoritmo (bloqueado por unwrap — candidato natural para cuando se aborde G-3/SNAPHU-MCF), lectura por ventana del camino GeoTIFF (requiere extender `surtgis-core`), multilooking on-the-fly (depende de que exista lectura por ventana).
- **G-10 · Red small-baseline**: doble umbral genera redes O(n²) densas y falla en huecos largos. Delaunay en (t, B⊥) + poda MST garantizando conectividad (relajar umbral solo en aristas puente) es lo estándar. Red desconectada = error duro está bien (mejor que el min-norm silencioso), pero ofrecer "quedarse con la componente mayor" como opción.
- **G-11 · Incertidumbre honesta**: SE(v) actual asume residuos i.i.d. (= paridad con MintPy, no es bug). Diferenciador: propagación de covarianza completa `A⁺Σ_bA⁺ᵀ` o bootstrap por remuestreo de épocas.
- **G-12 · Inversión robusta L1** (IRLS, Lauknes et al. 2011): supera claramente a L2 con errores de unwrap residuales. **MintPy no la trae de serie → diferenciador fuerte y muy citable.**

### Nivel 3 — PS de verdad + formatos + producto

- **G-13 · PS por estabilidad de fase**: D_A es solo el filtro inicial de Ferretti; StaMPS refina con análisis iterativo de coherencia de fase. Para "PS-InSAR" real (no SBAS-sobre-candidatos-D_A) este es el siguiente bloque grande. Previo barato: normalización radiométrica inter-época antes de D_A (StaMPS lo hace; sin eso las diferencias de calibración inflan σ_A).
- **G-14 · HDF5 (ifgramStack.h5/timeseries.h5)**: el cross-check MintPy hoy va por exports manuales — frágil. Crate `hdf5-metno`, o pragmático: script de export en `validation/` (patrón ya existente).
- **G-15 · HyP3 casi gratis**: los productos ASF on-demand son GeoTIFF float32 1-banda — un generador de `stack.json` desde un directorio HyP3 es el tercer formato más barato y de mayor impacto (usuarios chilenos sin ISCE instalado).
- **G-16 · Referencia configurable**: ref_date arbitrario + referencia por ventana k×k (hoy: época 0 + píxel único).
- **G-17 · Exponer los diferenciadores**: features→ML, deramp, decompose, unwrap_error no tienen binding Python ni subcomando CLI — hoy solo accesibles escribiendo Rust contra examples. El motor "subvende" sus capacidades más originales.

---

## 5. Distribución y producto (checklist para el paper ESIN/C&G)

1. ✅ `surtgis-core` **publicado en crates.io (0.16.4, 2026-07-02)**; dep dual `{ version = "0.16", path = ... }`; examples excluidos del paquete; **`cargo publish --dry-run -p insar-core` PASA** (compila contra el crates.io real). Publicar cuando se quiera: `cargo publish -p insar-core`.
2. ✅ LICENSE-MIT + LICENSE-APACHE en la raíz; `license` en pyproject.
3. ✅ pyproject completo: `dependencies = ["numpy>=1.23"]`, authors, readme, urls, keywords, classifiers (incl. licencias).
4. ✅ CI GitHub Actions (`.github/workflows/ci.yml`): test + clippy -D warnings en Linux con checkout de los 4 repos hermanos; wheels abi3 linux/mac/windows vía maturin-action. **Pendiente del usuario**: secret `SIBLINGS_TOKEN` si los repos hermanos son privados.
5. ✅ CHANGELOG.md (keep-a-changelog, v0.1.0 documentada). **Pendiente del usuario**: `git tag v0.1.0` al commitear.
6. ✅ `py.allow_threads` en TODO el cómputo (incl. sbas_from_isce completo) + `InsarError → {OSError, ValueError, RuntimeError}` idiomáticos.
7. ✅ 6 tests pytest contra el wheel real (maturin develop → verde) + smoke test CLI (`CARGO_BIN_EXE`, sin deps nuevas).
8. ✅ README con tabla de los 14 módulos y capacidades; PLAN.md con la tabla de arquitectura real.
9. ✅ `insar-rs` **disponible en PyPI** (verificado 2026-07-02); `insar-core` disponible en crates.io.
10. ☐ DOI Zenodo del release (usar `/zenodo package` cuando se publique v0.1.0).

---

## 6. Plan de acción priorizado

**Sprint 1 — correctness silenciosa (antes de cualquier validación nueva vs MintPy)** — ✅ **COMPLETADO 2026-07-02**
C-2 (ceros ISCE→NaN vía `mask_zero_amplitude`, default on) · A-1 (validate en temporal_coherence) · A-2 (closure verificada: aplica solo si reduce Σ|n_l|; reporta `UnwrapCorrectionReport{corrected, detected_uncorrected}`) · A-3 (APS: ajuste lineal local en tiempo real — idéntico a la media móvil con épocas equiespaciadas, exacto con gaps) · M-1 (lazos NaN excluidos del solver, pinv cacheada por máscara como en invert_sbas) · M-7 (baselines: dir inexistente o cero matches = error; parciales documentadas) · M-8 (épocas estrictamente crecientes en load_manifest).
*Resultado: 128 tests verdes (127 unit + 1 e2e; 15 nuevos), clippy limpio. API cambiada: `correct_unwrap_errors` ahora devuelve `UnwrapCorrectionReport` (caller `validate_maule.rs` actualizado).*
*Razón: son exactamente el tipo de discrepancia que la validación numérica va a atribuir erróneamente a la inversión.*

**Sprint 2 — pipeline integrado + credibilidad científica** — ✅ **COMPLETADO 2026-07-02**
- G-1: `invert_sbas_ext` con `WeightScheme::{Unit, Coherence, InversePhaseVariance}` — WLS por ecuaciones normales + Cholesky por píxel; par sin coherencia finita se excluye; `invert_sbas` queda como wrapper OLS.
- G-2: `DemErrorConfig { slant_range_m }` — columna `g_k = −B⊥/(R·sinθ)` en la matriz de diseño (vía `build_design_ext`, que ahora reutiliza `network::design_matrix` — de paso eliminó la doble fuente de verdad de `reduced_pinv`); Δz por píxel en `SbasSolution::dem_error_m`; error claro si todas las B⊥ son 0.
- G-5: `unwrap_error::nonzero_closure_count` (≙ `numTriNonzeroIntAmbiguity` de MintPy); producto `closure_qc.tif`.
- M-17: `unwrap_2d_min_quality` / `unwrap_stack_min_quality` (umbral de coherencia; islas re-sembradas).
- M-16: `postprocess` re-exporta `reference_to_pixel`/`temporal_coherence` + `coherence_mask(gamma, thr)`.
- M-6: `run_sbas` integrado — coherencia del manifiesto (campo opcional `coherence` en stack.json) → unwrap con calidad → closure correction (default on) → referencia (configurada o automática) → `invert_sbas_ext` → APS → deramp opcional → productos (`velocity`, `series/`, `temporal_coherence.tif`, `dem_error.tif`, `closure_qc.tif`). Tropo estratificada documentada como paso vía API (requiere DEM, fuera del formato v0.1).
- Bonus M-12: `inversion::select_reference_pixel` compartida por pipeline, CLI e `insar_rs.sbas_from_isce` (elimina las ~35 líneas duplicadas).
- CLI: `run --min-quality --deramp --no-closure-correction`; `isce --wls --dem-error-range --deramp --no-closure-correction` + productos QC.
*Resultado: 142 tests verdes (141 unit + 1 e2e; 14 nuevos), clippy limpio, test de datos reales Fernandina (288 pares) verde. Examples reparados tras cambio de API de geostat-core (KrigingConfig `..Default::default()`).*

**Sprint 3 — publicable** — ✅ **COMPLETADO 2026-07-02**
Sección 5 completa (ver checklist arriba: 9/10 ✅, falta solo Zenodo post-release). Además:
- M-9: `features` con semántica por píxel real — ajuste sobre épocas finitas con `EpochSolver` cacheado por patrón (umbral efectivo `max(min_valid_epochs, n_coef)`); acumulado y max_step sobre épocas finitas consecutivas; SE con dof del subset.
- M-10: esquema de tabla determinista — `FeatureMaps::{has_seasonal, has_acceleration}` desde la config, no de los datos; test: serie 100% NaN conserva las columnas.
- M-4 (bindings): validación temprana de largos en `temporal_coherence` (antes zip truncaba en silencio).
*Resultado: 159 unit + 1 e2e + 1 smoke CLI + 6 pytest, clippy limpio en los crates propios. geostat-core quedó a medio editar en otra sesión durante la verificación (mismo patrón que smelt en la mañana): los tests se corrieron con `--lib --tests` esquivando examples; re-verificar `cargo test --workspace` completo cuando geostat compile.*

**Sprint 4 — diferenciadores del paper** — ✅ **COMPLETADO 2026-07-02**
- G-12: inversión robusta L1 por IRLS (`IrlsConfig` en `SbasSolverConfig::robust`; `w ← w_base/max(|r|,ε)`, compone con WLS y error de DEM; CLI `--robust`). **Hallazgo científico documentado**: la robustez L1 exige margen ≥2:1 — un par que es suma exacta de otros dos (p. ej. (0,2)=(0,1)+(1,2) sin más cobertura) tiene trade-off 1:1 y el outlier es solo débilmente identificable (verificado con probe numérico; nota en el doc de `IrlsConfig`). Test con red completa de 4 épocas: aísla +3 rad sin conocer la coherencia.
- G-11: `estimate_velocity_bootstrap(series, n_resamples, seed)` — remuestreo de épocas con reemplazo, SplitMix64 determinista sembrado por píxel (reproducible bajo rayon), honesto frente a residuos correlacionados.
- G-4: `fit_temporal_model` / `TemporalModel { polynomial_order, periods_yr, steps }` — estilo `timeseries2velocity` de MintPy: pinv de G compartida, `TemporalFit { velocity, velocity_std (SE formal), coefficients, names }`; valida saltos dentro del rango y rank de G. Test: estacionalidad de 1.28 años sesga el fit lineal >1e-3 y el modelo lo reduce a <1e-5; salto cosísmico de 5 cm recuperado.
- G-8: reader ISCE con decodificación genérica por dtype (`decode_raw_with`, cota checked_mul ANTES de alocar — cierra también M-15): `CFloat32` → `read_isce_wrapped_stack` (¡el `IfgStack` ya es alimentable desde ISCE!), `Byte` → `read_isce_conncomp`, `Float64` → casteado (lat/lon/hgt), `read_isce_los` (incidencia + azimut por píxel, NoData (0,0)). De paso `stack_pair_layers` + `epochs_pairs_baselines` deduplican el apilado (cierra M-14 en isce).
- M-4/M-5: `decompose_per_pixel` con `PerPixelGeometry` (incidencia/heading por píxel; colinealidad local → NaN, no error) + `isce_azimuth_to_heading` (heading = 90°−azimut, derivado y verificado contra la convención right-looking) + contrato de grillas/tiempos asc-desc documentado en el módulo (resamplear a grilla común; descomponer velocidades, no épocas).
*Resultado: 158 tests verdes (16 nuevos), clippy limpio.*

**Backlog v0.2+**: G-3 (MCF/SNAPHU backend) · G-7 (ERA5) · G-9 (mmap/tiles — decidir arquitectura ya) · G-13 (PS fase-estabilidad) · G-10 (Delaunay/MST) · G-14/G-15 (HDF5/HyP3).

---

## 7. Lo que ya está bien (no tocar)

- Contrato central (`types.rs`) chico y estable; macro `impl_stack_dims` proporcionada; validación de pares con mensajes útiles.
- Manejo per-píxel de dropout con re-verificación de conectividad de la red reducida: **más riguroso que el enfoque de máscara global de implementaciones de referencia**.
- γ_temporal implementa Pepe & Lanari 2006 fielmente (2π-periódico vía exponencial compleja).
- Geometría LOS de `decompose` verificada idéntica a la convención MintPy right-looking, con test de signos.
- `with_ramp` en troposphere (plano en el ajuste, no en la resta): práctica correcta anti-sesgo.
- Convolución normalizada separable con máscara de validez en atmosphere: correcta y elegante.
- Separación atmosphere (turbulenta estocástica) vs troposphere (estratificada determinista): físicamente correcta, misma partición que MintPy. Solo mejorar naming/docs cruzados (`aps::{turbulent, stratified}`).
- 113 tests con ground truth analítico y propiedades físicas; tests negativos de I/O (JSON malformado, truncados, dims) por encima de la media del software InSAR.
- Parser VRT minimalista pero honesto: rechaza explícitamente lo que no soporta.
- `expect()` post-loop protegidos por invariantes comentados; deuda TODO ≈ 0; clippy limpio.

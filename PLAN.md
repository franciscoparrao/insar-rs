# Plan de Proyecto: insar-rs (MVP v0.1)

> Generado 2026-06-11 con patrón orquestador. Fuente de verdad del estado del proyecto.
> Doc de contexto: `CLAUDE.md` (este directorio) y `~/proyectos/ideas-motores-rust.md` (idea N2).

## Objetivo

Motor Rust para análisis de series temporales InSAR (Sentinel-1): selección de
Persistent Scatterers por amplitude dispersion, red SBAS e inversión de la serie
temporal de desplazamiento LOS, con validación numérica contra MintPy.

## Stack

- Lenguaje: Rust (edition 2024, toolchain ≥ 1.94)
- Numérica: `ndarray` 0.16 (+rayon), `num-complex` 0.4, `nalgebra` 0.33 (LSQ/SVD)
- I/O raster: **`surtgis-core`** (path dep `../surtgis/crates/core`,
  `default-features = false, features = ["parallel", "complex"]` → GeoTIFF nativo
  puro Rust, sin GDAL). API confirmada:
  - `surtgis_core::io::{read_geotiff, write_geotiff}` (genéricas sobre `Raster<T>`)
  - `surtgis_core::{Raster, GeoTransform, CRS}` (GeoTransform/CRS: Clone+Debug)
  - feature `complex`: `complex_from_parts`, `complex_to_parts`, `magnitude`, `phase`
    — complejos persisten como pares de bandas (re, im)
- CLI: `clap` 4 (derive)
- Python (PyO3): **diferido a fase tardía v0.1** — crate fuera de members hasta que el core esté estable

## Decisiones de diseño

| Decisión | Razón |
|----------|-------|
| Input = stacks ya coregistrados (GeoTIFF por interferograma + metadata; lector ISCE binario plano en segunda iteración) | CLAUDE.md difiere generación de interferogramas a v0.2; GeoTIFF aprovecha surtgis-core sin GDAL |
| Stacks en memoria como `ndarray::Array3` + `StackMeta` propio (no `Vec<Raster>`) | Inversión y filtros operan por píxel a través del eje temporal; conversión a `Raster` solo en frontera I/O |
| ndarray 0.16 / num-complex 0.4 alineados con surtgis | Evita dos instancias incompatibles de los tipos |
| Errores propios (`InsarError` con thiserror); errores de surtgis se mapean con `.map_err(... to_string)` | Evita acoplar el enum público al tipo de error de surtgis |
| Unwrapping mínimo: flood-fill guiado por coherencia/calidad 2D | MVP explícitamente "mínimo"; SNAPHU-equivalente queda fuera del alcance |
| APS simple: filtro pasa-alto temporal + pasa-bajo espacial (estilo SBAS clásico) | Suficiente para paridad MintPy en stack de prueba; APS avanzado es v0.2 |

## Arquitectura

### Módulos (todos dentro de `crates/core`, salvo cli)

| Módulo | Responsabilidad | Nivel | Dependencias | Tamaño est. |
|--------|----------------|-------|--------------|-------------|
| `types` + `error` | Contratos: stacks, pares, metadata, resultados | scaffold (✔ completo) | — | ~250 líneas |
| `io` | Leer stacks GeoTIFF (+ISCE plano), escribir velocidad/series | 0 | types, surtgis-core | ~400 |
| `ps` | Amplitude dispersion + selección de PS | 0 | types | ~200 |
| `network` | Red SBAS (umbrales temporal/perp), matriz de diseño, conectividad | 0 | types | ~250 |
| `unwrap` | Desenrollado 2D mínimo (flood-fill por calidad) | 0 | types | ~300 |
| `inversion` | LSQ SBAS (nalgebra SVD), fase→desplazamiento, velocidad | 0 | types, network (solo `design_matrix`) | ~350 |
| `atmosphere` | Corrección APS simple (filtro espacio-temporal) | 0 | types | ~200 |
| `pipeline` | Orquesta el flujo completo SBAS end-to-end | 1 | todos los anteriores | ~200 |
| `cli` (crate) | Subcomandos: info, ps, network, invert, run | 2 | insar-core | ~250 |
| validación | Cross-check vs MintPy sobre stack público | 3 | pipeline + datos | notebook/script |

### Grafo de dependencias

```
[types/error] (scaffold, completo)
   ├──→ [io] ──────────┐
   ├──→ [ps] ──────────┤
   ├──→ [network] ─────┤   Nivel 0: paralelizables (contratos en types)
   ├──→ [unwrap] ──────┤
   ├──→ [inversion] ───┤   (usa design_matrix de network vía firma del contrato)
   └──→ [atmosphere] ──┘
                        ↓
                   [pipeline]  Nivel 1
                        ↓
                      [cli]    Nivel 2
                        ↓
              [validación MintPy]  Nivel 3 (requiere stack de prueba)
```

Sin ciclos. `inversion` depende de `network` solo por la firma
`design_matrix(pairs, n_epochs) -> Array2<f64>`, ya fijada en el scaffold.

### Contratos entre módulos

Los contratos completos viven en `crates/core/src/types.rs` y en las firmas stub
(`todo!()`) de cada `mod.rs`. Resumen:

```rust
// types.rs (núcleo del contrato)
Epoch(NaiveDate)
IfgPair { reference: usize, secondary: usize, perp_baseline_m: f64 }
StackMeta { transform: GeoTransform, crs: Option<CRS>, wavelength_m: f64, incidence_deg: f64, heading_deg: Option<f64> }
IfgStack        { data: Array3<Complex32>, epochs, pairs, meta }   // pares × filas × cols
AmplitudeStack  { data: Array3<f32>, epochs, meta }                // épocas × filas × cols
UnwrappedStack  { data: Array3<f32>, epochs, pairs, meta }         // radianes
PsCandidate     { row, col, amp_dispersion }
DisplacementSeries { data: Array3<f32>, epochs, meta }             // metros LOS
VelocityMap     { data: Array2<f32>, meta }                        // m/año LOS
```

Convenciones transversales:
- Layout `Array3`: eje 0 = tiempo/par, eje 1 = filas, eje 2 = columnas.
- NoData = `f32::NAN` (complejos: ambas partes NaN, convención surtgis).
- Fase→desplazamiento: `d = -λ/(4π)·φ` (signo: alejamiento del sensor = negativo).
- Errores: `insar_core::Result<T>`; no `panic!` en API pública (los `todo!()` del scaffold se eliminan al implementar).

## Fases de Implementación

### Fase 1: Scaffold (orquestador)
- [x] PLAN.md
- [x] Workspace Cargo.toml + crates `core`, `cli`
- [x] `types.rs` + `error.rs` completos (contrato)
- [x] Firmas stub en io/ps/network/unwrap/inversion/atmosphere/pipeline
- [x] CLI esqueleto con clap
- [x] `cargo check` verde + `git init`

### Fase 2: Implementación paralela Nivel 0 (subagentes, máx 3-4 simultáneos)
- [x] Lote A: `ps` + `network` + `inversion` (2026-06-11; 36 tests nuevos, 39 totales verdes, clippy limpio)
  - `ps`: D_A muestral (n−1), two-pass en f64, NaN estricto, select_ps inclusivo y ordenado
  - `network`: doble umbral + chequeo de orden estricto de épocas; design_matrix rechaza ref>sec; union-find
  - `inversion`: SVD cacheada (rcond estilo LAPACK), solve en f64, serie relativa a época 0; NaN por píxel (NaN por par anotado para después)
- [x] Lote B: `io` + `unwrap` + `atmosphere` (2026-06-11; 29 tests nuevos, 68 totales verdes, clippy limpio)
  - `io`: formato stack.json definido y documentado en el módulo; complejos como pares `*_re.tif`/`*_im.tif` (el reader nativo de surtgis ignora el parámetro band en multibanda); round-trip testeado
  - `unwrap`: flood-fill quality-guided, vecindad 4, islas por NaN con re-siembra (offset 2πk por isla documentado)
  - `atmosphere`: pasa-alto temporal (prefix sums, ventana truncada simétrica) + gaussiano espacial separable normalizado por máscara; artefacto suave atenuado ~7×, deformación lineal intacta
- [x] Verificador entre lotes: `cargo test` + cumplimiento de contratos (Nivel 0 completo ✔)

### Fase 3: Integración Nivel 1-2 (subagente + orquestador) — COMPLETA (2026-06-13)
- [x] `pipeline` end-to-end (run_sbas: io→unwrap→inversion→atmosphere→velocity)
- [x] `cli` funcional (verificado real: info/network/run sobre stack generado por el ejemplo synthetic_stack)
- [x] Test e2e: recupera velocidad central con error ~2.4e-9 m/año (70 tests verdes, clippy limpio)
- [x] Review de código: sin bugs críticos/altos; núcleo numérico correcto y auto-consistente

#### Caveats del review para la paridad con MintPy (Fase 4) — metodológicos, no defectos
1. **NoData all-or-nothing** (inversion, estimate_velocity): un solo par/época no-finito anula TODA la serie del píxel. MintPy invierte sobre el subset de pares disponible → diferirá en *cuántos píxeles sobreviven* en datos reales con decorrelación parcial. Fix futuro: subsetear filas de A por píxel y re-resolver.
2. **APS=0 en épocas extremas** (atmosphere): el pasa-alto temporal da 0 en primera/última época; esos extremos conservan ruido atmosférico y tienen el mayor leverage en el ajuste lineal de velocidad. Considerar estimación one-sided en bordes.
3. **Velocidad = OLS no ponderado** sobre serie completa; MintPy puede ponderar por coherencia. Consistente con (1).
4. **Orden APS post-inversión**: variante legítima de SBAS; al comparar con MintPy, configurar MintPy con el mismo orden (filtrado temporal-espacial post-inversión) o las velocidades no calzarán aunque ambos sean correctos.
5. Salida es **LOS** (m/año), no vertical (incidence_deg/heading_deg se almacenan pero no se usan en v0.1). El cross-check debe comparar LOS.

### Fase 4: Validación (Nivel 3) — COMPLETA (2026-06-14) ✔ PARIDAD CONFIRMADA
- [x] Stack: FernandinaSenDT128 (tutorial MintPy, Sentinel-1 ISCE, 98 épocas / 288 ifgs / 450×600)
- [x] MintPy 1.6.2 instalado en .venv-mintpy; ifgramStack.h5 desde load_data; inversión -w no + velocidad
- [x] Ingesta en insar-rs (examples/validate_fernandina.rs) de las fases exactas exportadas
- [x] Comparación (validation/compare.py): **serie RMSE 0.0029 mm, r=1.000000; velocidad RMSE 0.0070 mm/año, r=1.000000, pendiente 0.9995** — paridad al nivel del redondeo f32
- [x] Documentado en docs/validation.md + figura validation/validation_velocity.png
- Metodología: misma fase desenrollada a ambos lados, inversión no ponderada sin correcciones, referenciados al mismo píxel (147/579). Aísla la inversión SBAS.
- Scripts en validation/ (export_ifgstack.py, compare.py); data/ y .venv-mintpy/ gitignored.

### Fase 5: Tardía v0.1 — EN CURSO (2026-06-14)
- [x] **Lector ISCE nativo** (`io::isce`) — COMPLETO (2026-06-15). Cierra el loop "lee ISCE e invierte sin Python".
  - `parse_vrt` (roxmltree) + `read_raw_band` (offsets GDAL VRTRawRasterBand, Float32 LSB) + `read_isce_unwrapped_stack` (descubre pares YYYYMMDD_YYYYMMDD, banda 2 = fase, baselines promediando "Bperp (average):") + `read_isce_coherence` (banda 1 de .cor).
  - 4 tests unitarios (vrt sintético) + 1 test #[ignore] con datos reales (98 épocas/288 pares/450×600, red conexa) — pasa en 4.8s.
  - CLI: subcomando `insar isce <ifg_dir> <out> [--baselines DIR]` → velocity.tif + series/. Probado real.
  - **Validación nativa end-to-end**: leer .unw → invert_sbas → estimate_velocity vs velocity.h5 de MintPy → RMSE 0.0070 mm/año, r=1.000000, pendiente 0.9995 (idéntico al camino vía h5). Lectura 3.3s + inversión 1.8s para 270k píxeles.
- [x] Benchmarks (criterion) vs MintPy — COMPLETO (2026-06-15). `benches/sbas.rs`; docs/benchmarks.md. invert_sbas 270k px ~2.9s; inversión real 1.8s vs MintPy 55.7s.
- [x] Crate `python` (PyO3) — COMPLETO (2026-06-15). `crates/python` (insar_rs): invert_sbas, estimate_velocity, amplitude_dispersion, sbas_from_isce (numpy float32). maturin develop; validado desde Python vs MintPy (RMSE 0.0070 mm/año, r=1.000000). README + pyproject.

Dependencia añadida: `roxmltree` (parser XML read-only) para el `.vrt`; pyo3 0.22 + numpy 0.22 (abi3-py39) en el crate python.

## Estado: MVP v0.1 COMPLETO (Fases 1-5). Listo para paper.

### Polish post-v0.1 (2026-06-17)
- [x] **Inversión NaN-por-par** (resuelve caveat #1 del review): cada píxel se
  invierte con su subconjunto de pares válidos; NaN solo si la red reducida
  queda desconectada. Pseudoinversa cacheada por patrón de máscara (3 pasadas:
  máscaras únicas → pinv por patrón → inversión). Stress-test real (20% dropout,
  270k px): 545s → **1.34s** con el cache (400×), resultado idéntico. 2 tests
  nuevos (recupera con par faltante; NaN si desconexión). examples/robustness_dropout.rs.
- [x] **Coherencia temporal** (Pepe & Lanari 2006) como calidad de inversión:
  `inversion::temporal_coherence` (γ = |Σ exp(j(φ_obs−φ_model))|/M). En Fernandina
  mediana 0.996, 62% γ>0.9. 4 tests. Expuesta en CLI (temporal_coherence.tif) y PyO3.
- [x] **Referencia espacial de la entrada** (`inversion::reference_to_pixel`):
  resta el offset constante por interferograma del desenrollado. Sin ella la
  coherencia temporal salía ~0.1 (offsets de ISCE como residuo). El CLI/PyO3
  eligen el píxel de máxima coherencia media. No altera la paridad de velocidad
  vs MintPy (RMSE 0.0070, r=1.000000 confirmado tras el cambio). 2 tests.
- [x] **Incertidumbre de velocidad** (`inversion::estimate_velocity_uncertainty`):
  error estándar del ajuste OLS (estilo velocityStd de MintPy). Fernandina:
  mediana 0.83 mm/año. CLI (velocity_std.tif) + PyO3. 3 tests.
- [x] **Corrección de errores de desenrollado** (`unwrap_error`, Yunjun et al. 2019):
  cierre de fase sobre la red SBAS (lazos de tripletes), estima el entero de
  corrección por par y píxel (pinv L2 redondeada) y aplica φ−=2π·U. Maneja los
  saltos 2π entre componentes conexas de productos ISCE/GUNW sin heurística
  espacial. 6 tests. Integrado en el flujo ARIA.
- [ ] Pendientes de los caveats: APS en épocas extremas (#2), velocidad OLS no ponderada (#3, opcional — MintPy también usa OLS por defecto).

### Caso chileno (en curso): Laguna del Maule vía ARIA S1-GUNW (2026-06-19)
- Pipeline ARIA completo: descarga autenticada (ASF/Earthdata), `aria_to_stack.py`
  (recorte + máscara connectedComponents + export), `examples/validate_maule.rs`
  (corrección de unwrap errors → referencia → inversión → velocidad + coherencia).
- 43 épocas / 80 pares (track 83 desc, 2017-05..2018-10), red conexa, inversión 1.2s.
- `unwrap_error` corrigió 305k píxeles; coherencia temporal 0.44 → 0.75.
- **Hallazgo (físico, no del motor)**: el centro deformante de Maule está
  DECORRELACIONADO en este subset (inflación rápida + nieve andina estacional →
  componente 0 en la mayoría de pares). Solo sobrevive coherencia en el terreno
  bajo circundante. Maule es de los casos InSAR más difíciles justo por ser el
  más rápido. Recomendación reforzada: caso chileno de ALTA coherencia (norte
  árido / Atacama: subsidencia minera o de salar) recuperaría limpio.
- LiCSAR (formato ideal) sigue inalcanzable desde este entorno (JASMIN bloqueado).
- **Salar de Atacama** (track 156 desc, 67 ifgs 2019-2020): coherencia 0.999,
  cobertura 100%, pipeline limpio. Velocidad cruda −6 cm/año dominada por rampa;
  con deramp queda ~5 cm/año de señal localizada. Confirma la física de coherencia.
- [x] **Deramp nativo** (`postprocess::remove_ramp` / `deramp_series`): ajuste
  planar/cuadrático LSQ con máscara, resta in situ. 4 tests. Resuelve el caveat #2-rampa.

## Prompts para Subagentes (Fase 2)

Template — reemplazar `{módulo}`:

```
Implementa el módulo `{módulo}` del proyecto insar-rs (motor InSAR time-series en Rust).

Contexto:
- Workspace: /home/franciscoparrao/proyectos/insar-rs
- Lee PRIMERO: PLAN.md (sección Decisiones + Contratos), crates/core/src/types.rs,
  crates/core/src/error.rs, y crates/core/src/{módulo}/mod.rs (firmas stub con todo!()).
- Tu directorio: crates/core/src/{módulo}/ — NO modifiques nada fuera de él
  (en particular NO cambies types.rs ni las firmas públicas; si un contrato te
  parece insuficiente, repórtalo en vez de cambiarlo).

Requisitos:
1. Implementa todas las funciones públicas del mod.rs eliminando los todo!().
2. Tests unitarios en el mismo módulo (#[cfg(test)]) con casos sintéticos
   verificables a mano (ej. para inversion: red de 4 épocas con desplazamiento
   lineal conocido debe recuperarse exacto).
3. Paraleliza con rayon donde el costo lo amerite (por píxel/por fila).
4. NoData = NaN; propagar, no abortar.
5. cargo test -p insar-core debe quedar verde.

Restricciones:
- No agregues dependencias nuevas sin reportarlo como pregunta.
- Sin unsafe. Sin panic! en rutas públicas.

Al terminar reporta: archivos tocados, resumen (3-5 líneas), número de tests y
si pasan, decisiones numéricas tomadas (tolerancias, algoritmos elegidos).
```

Notas específicas por módulo:
- `ps`: amplitude dispersion D_A = σ_A/μ_A por píxel sobre el eje temporal; umbral típico 0.25-0.4.
- `network`: pares por umbral doble (días, metros); `design_matrix` estilo SBAS clásico (Berardino 2002) sobre incrementos de velocidad entre épocas; `is_connected` con union-find o BFS.
- `inversion`: LSQ vía SVD de nalgebra; manejar redes con subsets desconectados devolviendo error claro; velocidad = ajuste lineal sobre la serie.
- `unwrap`: flood-fill desde semilla de mayor calidad, ±2π en cruces; coherencia opcional como mapa de calidad.
- `atmosphere`: pasa-bajo espacial (gaussiano separable) + pasa-alto temporal (media móvil) sobre la serie; restar el estimado APS.
- `io`: GeoTIFF stack = directorio con un .tif por interferograma (2 bandas re/im o fase+coherencia) + `stack.json` con épocas/pares/baselines/metadata; usar surtgis-core para leer/escribir; definir y documentar `stack.json` en el módulo.

## Estado de sesión

Contexto persistente: `~/.claude/session_state/insar-rs.json` (crear al cerrar Fase 1).

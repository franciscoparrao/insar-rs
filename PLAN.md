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

### Fase 3: Integración Nivel 1-2 (subagente + orquestador)
- [ ] `pipeline` end-to-end
- [ ] `cli` funcional
- [ ] Review de código (subagente /rust o /review)

### Fase 4: Validación (Nivel 3)
- [ ] Conseguir stack público coregistrado (ej. tutorial MintPy: Fernandina/Galápagos ARIA, o ISCE San Francisco)
- [ ] Script Python de comparación: serie insar-rs vs MintPy (RMSE, correlación)
- [ ] Documentar paridad en `docs/validation.md`

### Fase 5: Tardía v0.1
- [ ] Crate `python` (PyO3) sobre el core estable
- [ ] Lector ISCE binario plano (.int/.unw + XML)
- [ ] Benchmarks (criterion) vs MintPy en tiempo de ejecución

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

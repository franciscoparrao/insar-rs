# insar-rs — Hoja de ruta: maximizar el camino de datos abiertos

> **Estado:** propuesta de integración · creado 2026-06-28
> **Documento madre:** `PLAN.md` (arquitectura del motor) · `CLAUDE.md` (contexto)
> **Premisa:** geofísica sin instrumentos caros. `insar-rs` ya consume datos
> gratis del espacio (ARIA-GUNW + HyP3). Esta ruta enchufa las fuentes abiertas
> que aún NO se explotan y que el motor ya está en condiciones de consumir.

## Estado actual (línea base)

- **Motor:** MVP validado. Paridad numérica con MintPy en **Fernandina**
  (Sentinel-1): serie RMSE 0.0029 mm, velocidad RMSE 0.0070 mm/año, r = 1.000000.
- **Fuentes en uso:** ARIA-GUNW (interferogramas desenrollados listos) + HyP3
  (procesamiento on-demand de ASF). Ambas gratis.
- **Casos Chile cargados:** Algarrobo, Atacama, Maule (con descomposición
  asc+desc → up/down). Validación dura aún solo en Fernandina.
- **Sinergias internas disponibles:** SurtGIS (I/O GeoTIFF sin GDAL), FABDEM
  corregido (línea super-resolution-dem), Planetary Computer (vía MCP gateway).

**Lo que falta:** cobertura continental sin ordenar AOI por AOI, eliminar la
coregistración externa, corrección atmosférica publicable, referencia absoluta
GNSS, y un benchmark independiente además de MintPy.

## Estrella polar

`insar-rs` + (LiCSAR/OPERA + GACOS + GNSS + FABDEM propio) =
**pipeline InSAR completamente abierto, Rust-nativo y reproducible para los
Andes**, sin un solo instrumento propio. Framing del paper (ESIN / C&G):
*instrument-scarce pero data-rich-from-space*, en una región con subsidencia
minera/acuífera, inflación volcánica y deslizamientos lentos.

---

## Fase P0 — Quick wins de alto impacto (semanas)

### P0.1 — Reader de COMET-LiCSAR  ⭐ máxima palanca
- **Qué:** interferogramas Sentinel-1 **automáticos y globales**, ya
  desenrollados, con coherencia, descargables por *frame*. Cubre todos los Andes.
- **Por qué:** elimina la dependencia del on-demand AOI-por-AOI (ARIA/HyP3).
  Cobertura continental sentada esperando.
- **Acceso:** portal COMET-LiCSAR (descarga por frame); LiCSBAS define el formato.
- **Plug a insar-rs:** nuevo reader hermano del lector ARIA en `insar-core`
  (stack desenrollado + coherencia → red SBAS existente). Bajo esfuerzo, gran
  multiplicador de cobertura.
- **Esfuerzo:** S (un reader + parseo de metadata de frame).

### P0.2 — Enganche GACOS (corrección troposférica)
- **Qué:** mapas de retardo troposférico gratis por fecha/lugar. Reemplaza el
  TODO "corrección atmosférica simple" del PLAN.
- **Por qué:** convierte la APS de "simple" a publicable sin escribir un modelo
  atmosférico propio.
- **Acceso:** servicio GACOS (registro + request por AOI/fecha). Alternativa
  local/reproducible: ERA5 vía PyAPS.
- **Plug a insar-rs:** etapa de corrección de fase antes de la inversión de
  series; aplicar sobre stacks Atacama/Maule ya cargados.
- **Esfuerzo:** S–M (descarga + resampleo al grid del stack + resta de fase).

### P0.3 — GNSS gratis como referencia absoluta y validación
- **Qué:** series diarias de miles de estaciones GNSS (incluida Chile).
- **Por qué:** amarra el LOS a un marco de referencia absoluto y valida con dato
  terrestre independiente. InSAR-GNSS es el clásico que piden los reviewers.
- **Acceso:** Nevada Geodetic Lab (Blewitt, global) + red del CSN (Chile).
- **Plug a insar-rs:** módulo de referencing (proyección GNSS→LOS, tie-point) +
  script de validación cruzada en `validation/`.
- **Esfuerzo:** M (proyección al LOS por geometría de órbita + co-registro temporal).

### P0.4 — Housekeeping de contexto
- Actualizar `CLAUDE.md`: dice "Estado: IDEA (sin código)" pero hay MVP validado.
  Corregir para no perder contexto al retomar.
- **Esfuerzo:** XS.

---

## Fase P1 — Estratégico / future-proofing (1–2 meses)

### P1.1 — OPERA CSLC-S1 (SLCs coregistrados globales)
- **Qué:** SLCs **coregistrados, gratis, globales** (NASA/JPL via ASF).
- **Por qué:** elimina la dependencia de ISCE para coregistrar — el input
  externo que hoy el motor asume dado.
- **Plug a insar-rs:** habilita la rama "generación de interferogramas desde
  SLC" (v0.2 del PLAN) sin montar ISCE.
- **Esfuerzo:** L (generación de interferogramas + red de pares).

### P1.2 — OPERA DISP-S1 como benchmark independiente
- **Qué:** producto oficial de desplazamiento (sucesor de ARIA).
- **Por qué:** segundo punto de validación además de MintPy; robustez ante reviewer.
- **Plug a insar-rs:** comparación en `validation/` (cuando la cobertura
  Andes/Chile esté disponible; inicialmente Norteamérica, en expansión).
- **Esfuerzo:** S (otro comparador de series).

### P1.3 — FABDEM corregido para fase topográfica
- **Qué:** usar el DEM corregido propio (línea super-resolution-dem) en vez de
  GLO-30 crudo.
- **Por qué:** mejor corrección de fase topográfica; sinergia interna que da
  novedad metodológica al paper.
- **Plug a insar-rs:** swap del DEM en la etapa de corrección topográfica.
- **Esfuerzo:** S.

### P1.4 — EGMS como dataset de validación QC (Europa)
- **Qué:** velocidades PS/SBAS totalmente procesadas y controladas (Copernicus,
  solo Europa).
- **Por qué:** aunque el caso sea Chile, sirve de benchmark ultra-validado para
  certificar la inversión del motor sobre un frame europeo.
- **Esfuerzo:** S.

---

## Fase P2 — Fusión multi-sensor (línea de papers, abierta a futuro)

- **GRACE/GRACE-FO + InSAR:** subsidencia ↔ almacenamiento de acuíferos
  (Atacama: minería + acuífero = caso natural).
- **Catálogos sísmicos (CSN, USGS):** contexto co-sísmico para Maule/Algarrobo.
- **Inventarios de deformación volcánica** (COMET volcano portal, Smithsonian
  GVP) y **NASA Global Landslide Catalog:** labels para la rama ML
  (geofisica-ml / nowcast).
- **Sentinel-1 RTC vía Planetary Computer** (ya accesible por MCP): backscatter
  para `paper2_unet_sar`.

---

## Resumen de prioridades

| ID | Fuente | Aporte | Esfuerzo | Plug |
|----|--------|--------|----------|------|
| P0.1 | COMET-LiCSAR | Cobertura continental sin AOI on-demand | S | reader stack |
| P0.2 | GACOS | APS publicable | S–M | corrección de fase |
| P0.3 | GNSS (NGL + CSN) | Referencia absoluta + validación | M | referencing |
| P0.4 | — | Actualizar CLAUDE.md al estado real | XS | docs |
| P1.1 | OPERA CSLC-S1 | Sin coregistración ISCE | L | gen. interferog. |
| P1.2 | OPERA DISP-S1 | Benchmark independiente | S | validación |
| P1.3 | FABDEM propio | Fase topográfica mejor | S | corrección topo |
| P1.4 | EGMS | Benchmark QC (Europa) | S | validación |
| P2.* | GRACE / sismos / inventarios / RTC | Fusión multi-sensor + labels ML | var | ramas futuras |

## Secuencia sugerida

1. **P0.4** (XS, corrige contexto) → **P0.1** (desbloquea cobertura) →
   **P0.2** (sube calidad) → **P0.3** (da validación dura para el paper).
2. Con P0 completo: el motor consume Andes entero, corregido y validado contra
   GNSS. Ese es el estado mínimo para el paper ESIN/C&G.
3. **P1** vuelve el pipeline independiente de ISCE (CSLC) y multi-benchmark.
4. **P2** abre la línea de fusión multi-sensor (papers siguientes).

## Notas de reproducibilidad (para el paper)

- Cada fuente es gratuita y citable → empaquetar AOIs + scripts de descarga vía
  `/zenodo` (convención de la familia de motores Rust: SurtGIS, Hydroflux, etc.).
- Mantener los `download_*.py` de `validation/` como receta reproducible por caso.

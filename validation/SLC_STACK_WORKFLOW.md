# Path SLC-stack: el salto de calidad (span largo + baseline corto + PS)

> Estado al 2026-06-29: **gate cleared**. ISCE2 instalado, scaffolding listo,
> motor preparado. Falta la corrida de coregistración (sesión dedicada).

## Por qué este path

Demostrado empíricamente en El Canelo (ver comparación 9m/2y/short):
- HyP3 par-por-par NO puede dar barato **span largo + baseline corto** a la vez
  (coherencia pide baseline corto; precisión de velocidad pide span largo).
- El stack SLC **coregistrado** los une: una sola coregistración → cualquier par
  (baselines cortos) sobre años + **stack de amplitud** que habilita **PS-InSAR**.

## Arquitectura

```
bursts/SLC Sentinel-1  →  ISCE topsStack (coregistra + interferogramas + amplitud)
                            │
                            ▼
                       insar-rs:  io::isce::read_isce_unwrapped_stack  (pares)
                                  ps::amplitude_dispersion + select_ps  (PS)
                                  inversion::invert_sbas                (SBAS)
                                  decompose (asc+desc) + tropo/deramp/ref
```

Lo único que el motor NO hace —la coregistración SLC— lo hace ISCE. El resto ya
está y validado (lector ISCE: paridad con Fernandina; PS y SBAS: tests verdes).

## Infraestructura instalada

- ISCE 2.6.3 en `~/miniforge3/envs/isce2` (sin tocar el shell).
  - `stackSentinel.py`: `~/miniforge3/envs/isce2/share/isce2/topsStack/`
  - `topsApp.py`: `.../site-packages/isce/applications/`
  - Ejecutar con: `~/miniforge3/envs/isce2/bin/python <script>` o `mamba run -n isce2 ...`

## Decisión de tooling (el punto a resolver en la corrida)

`stackSentinel.py` (topsStack) está diseñado para **SLC SAFE completos** (~5 GB
c/u). Con 81 GB libres, una pila de 15 SLC (~75 GB) es ajustada → dos rutas:

1. **Full-SLC topsStack con AOI bbox** (estándar, mejor documentada):
   - Descargar SAFE SLC (asf_search, `dataset='SENTINEL-1'`, processingLevel SLC).
   - `stackSentinel.py -s <slc_dir> -o <orbits> -a <aux> -d <dem> -b '<S N W E>' -c <n>`
     el `-b` recorta al AOI → salida pequeña; gestionar disco borrando SAFE tras
     el unpack. Recomendado para la **primera corrida** (pila corta, 8-10 SLC).
2. **Burst stack** (disco liviano, ~133 MB/burst → cabe holgado):
   - `download_bursts_algarrobo.py` ya baja la pila de bursts El Canelo.
   - Requiere coregistración burst-específica (ISCE2 burst tools / COMPASS), no el
     `stackSentinel.py` estándar. Más liviano pero menos trillado.

Recomendación: **primera corrida con full-SLC topsStack + bbox El Canelo, pila
corta (~8-10 fechas, asc y desc), gestionando disco**; validar la cadena
SLC→ISCE→insar-rs end-to-end. Luego escalar / pasar a bursts si el disco aprieta.

## Pasos de la corrida (sesión dedicada)

1. Órbitas precisas (POEORB) + DEM (ya sabemos bajarlo: `surtgis stac fetch-mosaic
   cop-dem-glo-30`, o el de ISCE).
2. `stackSentinel.py` → genera `run_files/` (coregistro + interferogramas).
3. Ejecutar los `run_files` (horas de cómputo).
4. `insar-rs`: apuntar `io::isce::read_isce_unwrapped_stack` al stack → SBAS + PS
   → decompose asc+desc → velocidad vertical con piso de ruido mucho menor.

## Unwrap (RESUELTO 2026-07-01)

El wrapper `topsApp.py` muere con **exit 1 al terminar `runFilter`**, sin llegar a
`runUnwrap`: deja `merged/` con `filt_topophase.flat` + `phsig.cor` pero **sin
`.unw`**, y sin `PICKLE/` para reanudar con `--start=unwrap`. snaphu va embebido
en ISCE2 (no hay CLI).

Solución: **`validation/isce_unwrap.py`** — llama la clase `Snaphu` de ISCE
directamente sobre los productos de `merged/`, replicando `runUnwrapMcf`
(costMode=SMOOTH, initMethod=MCF, initOnly=True) sin necesitar el pickle.
Parámetros geométricos (earthRadius/altitude) nominales S1 — irrelevantes con
MCF+initOnly. **Sanea NaN/inf → 0** antes de correr (snaphu aborta con NaN; el
filtro deja ~0.4% de NaN en bordes).

```
~/miniforge3/envs/isce2/bin/python validation/isce_unwrap.py <merged_dir> \
    [--range-looks 19] [--azimuth-looks 7]
# genera filt_topophase.unw (+ .conncomp) legible por io::isce
```

Probado en el par El Canelo 20260605_20260617: 99.6% desenrollado, 7.6 ciclos,
20 componentes conexos (usar `.conncomp==0` como máscara de baja fiabilidad).

## Receta bursts → topsStack (PROBADA 2026-07-01)

El path bursts SÍ usa `stackSentinel.py` estándar: **`burst2safe`** empaqueta los
bursts ASF en mini-SAFE que topsStack lee (evita COMPASS). Env: `.venv-mintpy`
tiene `asf_search`+`burst2safe`; env `isce2` corre topsStack.

```bash
# 1. mini-SAFE por geometría (re-descarga bursts + anotaciones desde ASF)
.venv-mintpy/bin/burst2stack --rel-orbit 18  --start-date 2026-01-01 --end-date 2026-06-26 \
    --extent -71.708 -33.397 -71.648 -33.337 --pols VV --swaths IW1 \
    --output-dir data/algarrobo_safe/asc          # asc: 28 SAFE
.venv-mintpy/bin/burst2stack --rel-orbit 156 ... --swaths IW2 --output-dir .../desc  # desc: 27 SAFE

# 2. órbitas precisas (POEORB + RESORB para fechas <20d)
~/miniforge3/envs/isce2/bin/python -m eof --search-path data/algarrobo_safe/asc \
    --save-dir data/algarrobo_safe/orbits --orbit-type precise

# 3. config topsStack (genera run_files/; segundos) + ejecución (horas)
bash validation/run_topsstack.sh asc IW1        # → 28 épocas, 102 pares, 11 run steps
bash validation/run_topsstack.sh exec asc       # corre run_01..run_11
```

Deps que faltaban en isce2/venv (instaladas): `shapely` (isce2), `aiohttp`
+`gdal==3.11.4 pinned` (venv, para burst2safe). DEM reutilizado:
`data/algarrobo_isce/dem/dem.wgs84` (cubre el AOI). aux dir vacío basta.

`run_11_unwrap` usa el snaphu de topsStack; si su wrapper falla, cada par se
rescata con `isce_unwrap.py` sobre su `merged/`.

## Estado corridas (2026-07-02)

- **asc: COMPLETO** ✅ — 26 épocas, 49 pares desenrollados
  (`data/algarrobo_stack/asc/merged/interferograms/*/filt_fine.unw`). insar-rs
  leyó el stack (946×2408) e invirtió SBAS → velocidad LOS en ~2.7s
  (`validation/slcstack/algarrobo_asc_los_velocity.f32`).
- **desc: FALLA en run_07 (merge)** ❌ — `min() arg is an empty sequence` /
  "Skipping processing of swath 2". Causa: el AOI cae sobre **1 solo burst** en
  IW2 desc, y `mergeBursts.py` SALTA swaths con `minBurst==maxBurst` (necesita
  ≥2 bursts). asc no cae en esto (3 bursts en IW1). **Fix pendiente**: re-armar
  desc con `burst2stack ... --min-bursts 2` (fuerza 2 bursts/SAFE), re-fetch
  órbitas, y re-correr `run_topsstack.sh desc IW2 2`.

## Lo que falta (honesto)

- Re-hacer desc con `--min-bursts 2` (arriba). Luego decompose asc+desc.
- Apuntar `io::isce::read_isce_unwrapped_stack` al stack → SBAS + decompose
  asc/desc → velocidad vertical. Config: `unw_filename="filt_fine.unw"`.
- Integrar PS en el pipeline de alto nivel (el módulo existe; falta el ejemplo que
  encadene amplitude_dispersion → select_ps → serie PS).

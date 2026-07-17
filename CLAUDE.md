# insar-rs — Motor de InSAR time-series (deformación del terreno) en Rust

> **Estado:** MADURO v0.1.0 (actualizado 2026-07-11 tras auditoría). ~13.800 LOC, 206 tests. insar-core con 14 módulos: I/O stack propio + lector ISCE nativo, PS, red SBAS, unwrap quality-guided (+SNAPHU opcional), corrección de fase, inversión OLS/WLS/L1-IRLS + error DEM + bootstrap, APS/troposfera, deramp, descomposición LOS→up/east, features ML, pipeline `run_sbas`. Sin stubs. Bindings PyO3. Validado: paridad con MintPy sobre Fernandina (serie RMSE 0.0029 mm, velocidad 0.0070 mm/año, r=1.000000). Requiere sibling `../surtgis`. Sin paper aún (venue: ESIN/C&G) — sería el próximo hito. *(Antes decía "IDEA sin código" — era incorrecto.)*
> Familia de motores Rust del autor: SurtGIS, Hydroflux, Smelt, Anvil, Cantus, Criterium.
> Doc madre: `~/proyectos/ideas-motores-rust.md` (idea N2, Parte 5).

## Qué es
Motor para análisis de series temporales InSAR (Sentinel-1): deformación del
terreno por Persistent Scatterers (PS-InSAR) y Small-Baseline Subset (SBAS).

## El gap que llena
El campo es **ISCE, StaMPS, MintPy, EZ-InSAR** — todos Python/MATLAB, pesados de
montar. No hay motor Rust nativo. Chile: subsidencia (minería, acuíferos),
inflación volcánica, deslizamientos lentos = casos urgentes con datos
Sentinel-1 gratuitos.

## Alcance MVP (v0.1)
- [ ] Lectura de interferogramas / stacks (coregistrados, ej. salida ISCE).
- [ ] Selección de PS (amplitude dispersion) y red SBAS.
- [ ] Inversión de la serie temporal de desplazamiento (LOS).
- [ ] Corrección atmosférica simple; desenrollado de fase (mínimo).
- [ ] (v0.2) Generación de interferogramas desde SLC; APS avanzado.

## Arquitectura tentativa
- `insar-core`: fase compleja, redes de pares, inversión de series, FFT.
- Targets: native (Rayon, FFT) + Python (PyO3) + CLI.
- Salida raster (velocidad LOS, series) vía writer GeoTIFF de **SurtGIS**.

## Validación / paridad numérica
Cross-check series de desplazamiento contra **MintPy** sobre un stack público.

## Venue objetivo
**Earth Science Informatics** (donde se publicó EZ-InSAR) o **Computers &
Geosciences**.

## Conexiones con tu ecosistema
- **fisica-upskill** (`bridges/metodos-fourier-sar-processing.md`): grounding
  físico de SAR (Fourier, backscatter). Internalizar antes de codear el core.
- **postdoc/papers/paper2_unet_sar**: SAR backscatter forward physics; sinergia
  directa de datos/método.
- **nowcast**: deformación pre-falla como variable dinámica de geohazard.
- **SurtGIS**: DEM para corrección topográfica de fase; salidas raster.

## Próximos pasos al retomar
1. Leer un stack coregistrado de prueba y seleccionar PS.
2. Implementar inversión SBAS de la serie LOS; validar vs MintPy.
3. Definir si se incluye generación de interferogramas o se asume input ISCE.

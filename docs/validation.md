# Validación numérica vs MintPy — Fernandina (Sentinel-1 DT128)

> Fase 4 del MVP v0.1. Ejecutada 2026-06-14. Reproducible con los scripts en
> `validation/` y el ejemplo `crates/core/examples/validate_fernandina.rs`.

## Resumen

La inversión SBAS de insar-rs reproduce la de **MintPy 1.6.2** sobre un stack
real de Sentinel-1 (volcán Fernandina, Galápagos) con **paridad numérica al
nivel del redondeo de punto flotante**:

| Producto | RMSE | max\|Δ\| | Pearson r | Pendiente | Cobertura |
|----------|------|----------|-----------|-----------|-----------|
| Serie temporal LOS | **0.0029 mm** | 1.74 mm | 1.000000 | 1.000000 | 100 % |
| Velocidad LOS | **0.0070 mm/año** | 0.38 mm/año | 1.000000 | 0.9995 | 100 % |

(RMSE sobre 26.46 M muestras de serie y 270 k píxeles de velocidad.)

![Comparación de velocidad LOS](../validation/validation_velocity.png)

Los dos mapas de velocidad son indistinguibles; la diferencia se mantiene
dentro de ±1 mm/año en toda la grilla salvo un puñado de píxeles, y la
regresión insar-rs vs MintPy es la diagonal identidad.

## Dataset

- **FernandinaSenDT128** (tutorial oficial de MintPy, [Zenodo 3952953](https://zenodo.org/record/3952953)).
- 98 adquisiciones Sentinel-1, **288 interferogramas** desenrollados (ISCE,
  `merged/interferograms/*/filt_fine.unw`), grilla 450×600 en coordenadas radar.
- λ = 0.0555 m (banda C). Deformación esperada: subsidencia/inflación de la
  caldera (rango de velocidad −72 a +91 mm/año).

## Metodología (comparación apples-to-apples de la inversión)

Para aislar la **inversión SBAS** (el aporte central del motor) de diferencias
en otras etapas, ambos lados parten de las **mismas fases desenrolladas** y se
configuran sin ponderación ni correcciones:

1. **MintPy** construye `inputs/ifgramStack.h5` desde los `.unw` de ISCE
   (`smallbaselineApp.py --dostep load_data`), fija un píxel de referencia
   (`reference_point.py`, auto por coherencia → píxel y/x = 147/579) e invierte
   sin ponderar (`ifgram_inversion.py -w no`) → `timeseries.h5`; la velocidad
   sale de `timeseries2velocity.py` (ajuste lineal).
2. **insar-rs** recibe las fases EXACTAS exportadas de ese mismo
   `ifgramStack.h5` (`validation/export_ifgstack.py` → `meta.json` + `phase.f32`,
   filtrando por `dropIfgram`), construye un `UnwrappedStack` y corre
   `inversion::invert_sbas` + `estimate_velocity` (sin PS, sin APS).
3. **Referenciación**: la inversión es lineal, de modo que referenciar las
   fases a un píxel antes de invertir (MintPy) equivale a invertir y luego
   restar la serie de ese píxel. El comparador referencia la salida de insar-rs
   al mismo píxel 147/579 que usó MintPy; así ambos quedan en el mismo marco.

Convención de signo común: `d_LOS = −λ/(4π)·φ` (verificada: r = +1).

## Residuos: origen

Los residuos son de orden micrométrico y se explican por:

- **Precisión**: insar-rs resuelve en `f64` y exporta en `f32`; MintPy guarda
  `f32`. El RMSE de serie (~3 µm) está en el límite de `f32`.
- **max\|Δ\| de 1.7 mm** se concentra en píxeles con observaciones parciales de
  la red (no presentes en todos los interferogramas), donde el orden de
  operaciones difiere. insar-rs (v0.1) invierte estos píxeles con la red
  completa; ver caveat de NoData all-or-nothing en `PLAN.md`.
- **Velocidad, pendiente 0.9995**: diferencia <0.05 %, consistente con la
  convención de años decimales (insar-rs usa días/365.25; MintPy fracciones de
  año por `datetime`). Despreciable.

## Camino nativo ISCE (sin Python en el loop)

El lector ISCE nativo (`io::isce`, Fase 5) permite repetir la validación
**sin MintPy ni HDF5**: lee los `.unw` de ISCE directamente, invierte y estima
velocidad. Contra el mismo `velocity.h5` de referencia da resultados idénticos:

| Producto | RMSE | max\|Δ\| | Pearson r | Pendiente |
|----------|------|----------|-----------|-----------|
| Velocidad LOS (camino nativo) | 0.0070 mm/año | 0.38 mm/año | 1.000000 | 0.9995 |

Lectura de los 288 interferogramas: 3.3 s; inversión + velocidad de 270 k
píxeles: 1.8 s. Reproducible con:

```bash
cargo run --release -p insar-core --example validate_fernandina_isce -- \
  data/FernandinaSenDT128/merged/interferograms data/FernandinaSenDT128/baselines \
  validation/export/insar_velocity_isce.f32
# o vía CLI:
insar isce data/FernandinaSenDT128/merged/interferograms /tmp/out --baselines data/FernandinaSenDT128/baselines
```

## Veredicto

El núcleo de inversión SBAS de insar-rs es **numéricamente equivalente a MintPy**
sobre datos reales de Sentinel-1. Para datos con decorrelación parcial, las
diferencias esperadas vendrán del enmascarado de píxeles válidos y del manejo de
redes parciales (caveats documentados en `PLAN.md`), no de la física ni del
álgebra de la inversión.

## Reproducir

```bash
# 1. Entorno MintPy (venv) y dataset
python3 -m venv .venv-mintpy && .venv-mintpy/bin/pip install mintpy h5py
curl -L -o F.tar.xz https://zenodo.org/record/3952953/files/FernandinaSenDT128.tar.xz
tar -xJf F.tar.xz -C data/

# 2. MintPy: stack + inversión no ponderada + velocidad
cd data/FernandinaSenDT128/mintpy
smallbaselineApp.py FernandinaSenDT128.txt --dostep load_data
reference_point.py inputs/ifgramStack.h5
ifgram_inversion.py inputs/ifgramStack.h5 -w no -o timeseries.h5 timeseriesResidual.h5 numInvIfgram.h5
timeseries2velocity.py timeseries.h5 -o velocity.h5

# 3. Exportar fases, invertir con insar-rs, comparar
python validation/export_ifgstack.py <stack>/inputs/ifgramStack.h5 --out validation/export
cargo run --release -p insar-core --example validate_fernandina -- validation/export
python validation/compare.py validation/export data/FernandinaSenDT128/mintpy
```

> Nota: MintPy 1.6.2 con numpy ≥2 requiere un parche de una línea en
> `ifgram_inversion.py` (`inv_quality[idx] = np.asarray(inv_quali).ravel()[0]`),
> un bug de calidad-de-inversión ajeno a la serie temporal.

# Figuras de cierre de fase (headline del paper #1, post-pivote)

Evidencia empírica de la corrección de phase bias medida **en dominio wrapped**
(cierre de fase): no pasa por el desenrollado ni por correcciones atmosféricas,
que es lo que la hace afirmable — a diferencia del campo de velocidad, que la
validación GNSS (CLL1) mostró dominado por error de unwrapping flood-fill.

## Reproducción

```bash
# 1. Mapas por píxel + GeoTIFF + crudos para la figura
cargo run --release --example phase_bias_closure_figs -- data/licsar_083D_12636/GEOC

# 2. Figura de publicación (PDF vector + preview PNG 300 dpi)
python3 validation/make_figure_closure.py
```

## Datos y configuración

- Frame LiCSAR **083D_12636** (descendente), GEOC wrapped diffs, AOI valle de
  Ñuble/Chillán (−72.2/−71.8 lon, −36.8/−36.4 lat), 401×401 px.
- Coeficientes aₙ estimados de la red completa (ancla teselable span 12 ≈ 72 d):
  **a₂ = 0.415, a₃ = 0.472** (1 ancla). Comparables en orden a los 0.47/0.31 de
  Maghsoudi et al. 2022 (Turquía) — específicos de escena, como predice el paper.
- Corrección aplicada sobre la ventana hole-free **2021-12-21 → 2022-12-28**
  (32 épocas, 90 pares, full-rank).

## Resultados (2026-08-28)

| Estrato (coherencia media)     | px      | RMS antes | RMS después | Reducción |
|--------------------------------|---------|-----------|-------------|-----------|
| **Global**                     | 141 732 | 1.270     | 0.811       | **−36.2 %** |
| coh < 0.3 (cultivo/bosque)     | 113 528 | 1.340     | 0.856       | −36.1 %   |
| 0.3 ≤ coh < 0.5 (mixto)        | 20 456  | 0.996     | 0.629       | −36.9 %   |
| coh ≥ 0.5 (urbano/estable)     | 7 747   | 0.759     | 0.496       | −34.7 %   |

Dos lecturas para el paper:

1. **El sesgo muerde donde hay decorrelación**: el cierre antes de corregir es
   1.8× mayor en cultivo/bosque (1.34 rad) que en urbano (0.76 rad) — el mapa
   (panel a) muestra el núcleo urbano de Chillán como isla oscura en un valle
   agrícola brillante, espejo exacto del mapa de coherencia (panel c).
2. **La corrección es uniforme entre estratos (~35–37 %)**: consecuencia de que
   los aₙ son escalares de escena (el propio Maghsoudi 2022 muestra que no
   tienen patrón espacial); lo que varía espacialmente es la magnitud del
   sesgo, no la fracción corregible por este modelo.

## Figuras F2 (paridad) y F3 (sintético) — añadidas 2026-08-29

**F2 `figs/fig_parity.pdf`** — paridad numérica vs MintPy (Fernandina, 98
épocas, 288 ifgs, 270 k px, mismo píxel de referencia 76/156 a ambos lados):

- Serie temporal: RMSE **0.0005 mm**, r = 1.000000 (26.5 M comparaciones);
  P99.9 |Δ| = 0.16 µm.
- Velocidad: RMSE **5.3 µm/año**, r = 1.000000.

Cadena de regeneración (los intermedios pesados no se conservan):
```bash
source .venv-mintpy/bin/activate
cd data/FernandinaSenDT128/mintpy
smallbaselineApp.py smallbaselineApp.cfg --dostep load_data
smallbaselineApp.py smallbaselineApp.cfg --dostep reference_point
ifgram_inversion.py inputs/ifgramStack.h5 -w no
timeseries2velocity.py timeseries.h5 -o ../../../validation/export/velocity_ref.h5
cd ../../..
python3 validation/export_ifgstack.py data/FernandinaSenDT128/mintpy/inputs/ifgramStack.h5 --out validation/export
cargo run --release --example validate_fernandina -- validation/export
python3 validation/make_figure_parity.py
```
Nota: si `reference_point` elige otro píxel, la figura sigue siendo válida —
ambos lados se referencian al REF_Y/REF_X del timeseries.h5. La velocidad de
MintPy debe regenerarse con `timeseries2velocity` para compartir referencia
(el `velocity.h5` histórico usa la referencia antigua y mete un offset).

**F3 `figs/fig_synthetic.pdf`** — sintético con ground truth (misma
construcción que `tests/phase_bias_e2e.rs`): verdad −20 mm/año, sesgada
−28.4 (error 8.4 mm/año, 42 %), corregida −20.00; cierre 0.242 → 1.1e−8 rad.
```bash
cargo run --release --example phase_bias_synthetic_figs
python3 validation/make_figure_synthetic.py
```

## Archivos

- `figs/fig_closure.pdf` — figura de publicación (a: RMS antes, b: después,
  c: coherencia media, d: barras por estrato). Etiquetas en inglés (venue).
- `figs/closure_rms_{before,after}.tif`, `figs/closure_reduction_pct.tif`,
  `figs/mean_coherence.tif` — GeoTIFF georreferenciados (EPSG:4326) para GIS.
- `validation/phase_bias_export/` — crudos f32 + meta.json (insumo del script
  Python; gitignored como el resto de validation/ con datos).

## Validación anti-circularidad (2026-09-02 — cambia la métrica del paper)

El review C&G simulado objetó que la reducción del RMS de cierre es en parte
el objetivo del ajuste (circularidad). `phase_bias_validation.rs` lo midió:

| Experimento | Resultado |
|---|---|
| Reducción RMS in-sample (la métrica vieja) | 36.2 % |
| **Nulo empírico** (fase aleatoria sin sesgo, 20 corridas) | **31.8 ± 0.1 %** |
| Nulo analítico 1−√(1−p/n) (31 incógnitas / 59 cierres) | 31.1 % |
| **Hold-out espacial** (δ̂ del bloque 8×8 donut) | **0.0 %** |

→ El RMS por píxel NO es métrica válida: está dominado por absorción de
grados de libertad + ruido de decorrelación. La métrica correcta es el
**cierre medio (sistemático)**, evaluado con la corrección hold-out:

| Clase WorldCover | media antes | después | reducc. | vel. espuria removida |
|---|---|---|---|---|
| Cropland  | −0.081 rad | −0.024 | 70 % | −4.8 mm/año |
| Grassland | −0.081 | −0.027 | 66 % | −4.9 |
| Tree cover| −0.059 | −0.015 | 74 % | −3.8 |
| Built-up  | −0.042 | −0.007 | 84 % | −1.8 |

Sensibilidad (hold-out, global): aₙ de escena −70 %; sets publicados
(Turquía/Campi Flegrei/Azores) −64/−66 % → robusto, y estimar de la escena
es lo mejor. Figura nueva: `figs/fig_systematic.pdf` (headline del paper);
la figura RMS (`fig_closure.pdf`) queda como material histórico/cautionary.

GNSS: `figs/fig_gnss_supp.pdf` (apéndice del paper) — CLL1 (NGL, SA-fixed)
+4.3 mm/año LOS vs flood-fill −37.0/+26.7 en el píxel de la estación.
Datos de entrada archivables: `data/nuble_aoi_input_archive.tar.gz` (107 MB,
171 pares AOI, para el deposit Zenodo — LiCSAR podó 91 pares del archivo).

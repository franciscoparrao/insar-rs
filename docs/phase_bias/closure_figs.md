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

## Archivos

- `figs/fig_closure.pdf` — figura de publicación (a: RMS antes, b: después,
  c: coherencia media, d: barras por estrato). Etiquetas en inglés (venue).
- `figs/closure_rms_{before,after}.tif`, `figs/closure_reduction_pct.tif`,
  `figs/mean_coherence.tif` — GeoTIFF georreferenciados (EPSG:4326) para GIS.
- `validation/phase_bias_export/` — crudos f32 + meta.json (insumo del script
  Python; gitignored como el resto de validation/ con datos).

# Casos chilenos — ARIA Sentinel-1 GUNW

> Procesados 2026-06-18/19 con el pipeline ARIA nativo de insar-rs
> (descarga ASF/Earthdata → `aria_to_stack.py` → `examples/validate_maule.rs`,
> que aplica `unwrap_error` → referencia → inversión SBAS → velocidad + coherencia).
> Dos casos que ilustran el rol decisivo de la coherencia InSAR.

## Resumen

| Caso | Dato | Coherencia temporal | Resultado |
|------|------|---------------------|-----------|
| **Laguna del Maule** (volcán) | track 83 desc, 80 ifgs, 2017–2018 | 0.75 (centro decorrelacionado) | Centro deformante sin señal recuperable |
| **Salar de Atacama** (subsidencia salmuera) | track 156 desc, 67 ifgs, 2019–2020 | **0.999**, cobertura 100% | Pipeline limpio end-to-end; señal de deformación de varios cm/año |

## Capacidad nueva del motor: `unwrap_error`

Ambos casos usan productos ARIA GUNW, donde cada interferograma se desenrolla en
**componentes conexas independientes** con offset 2π propio. Esto motivó el
módulo `inversion`-adyacente `unwrap_error` (corrección por cierre de fase,
Yunjun et al. 2019): usa los lazos de tripletes de la red SBAS para estimar el
entero de corrección por par y píxel. En Maule corrigió 305 k píxeles (coherencia
0.44 → 0.75); en Atacama, 706 k (toda la escena), con coherencia final 0.999.

## Laguna del Maule — el límite físico

Inflación récord (18–27 cm/año publicado). insar-rs ingirió 43 épocas / 80 pares
e invirtió en 1.2 s, pero **el centro deformante quedó decorrelacionado**: la
inflación rápida (alta tasa de franjas) más la nieve estacional andina destruyen
la coherencia sobre los pares de 12–36 días, y el desenrollado marca el centro
como componente 0 (no fiable). Solo sobrevive coherencia en el terreno bajo
circundante. **No es una falla del motor** — es una limitación del dato de
entrada; Maule es de los casos InSAR más difíciles justo por ser el más rápido.

## Salar de Atacama — coherencia de clase mundial

Superficie de sal en el desierto más árido del planeta → coherencia temporal
**mediana 0.999, cobertura 100 %**. El pipeline corre limpio de extremo a extremo
e invierte 705 600 píxeles en 0.65 s. Se observa deformación de varios cm/año,
con subsidencia concentrada hacia el sur del salar (zona de campos de pozos de
salmuera de litio, SQM/Albemarle).

**Caveat honesto**: la velocidad cruda contiene una rampa de larga longitud de
onda (−3 cm/año N-S, típica de atmósfera/órbita). Tras un deramp planar queda una
señal localizada de ~3 cm/año, pero separarla limpiamente de la atmósfera
residual sobre una ventana de 7 meses, una sola órbita y **sin corrección
troposférica (GACOS/ERA5)** requiere refinamientos estándar aún no en el motor
(deramp nativo, corrección troposférica, serie temporal más larga, asc+desc).

## Veredicto

El pipeline ARIA + `unwrap_error` de insar-rs procesa datos chilenos reales de
extremo a extremo. Sobre terreno coherente (Atacama) el resultado es impecable
en cobertura y consistencia; la interpretación geofísica fina pide los
refinamientos habituales. El contraste Maule (decorrelacionado) vs Atacama
(prístino) es, en sí, una buena ilustración de la física de coherencia InSAR.

## Próximos refinamientos sugeridos (orden de valor)

1. **Deramp nativo** (`remove_ramp`) — ajuste planar/cuadrático, estándar y barato.
2. **Corrección troposférica** (ERA5/GACOS) — clave para señales lentas (mm-cm/año).
3. Serie temporal más larga (≥2 años) y combinación ascendente+descendente.

## Reproducir (Atacama)

```bash
python validation/download_atacama.py            # ~7 GB, 67 GUNW (necesita ~/.netrc Earthdata)
python validation/aria_to_stack.py data/atacama_gunw --out validation/atacama_export \
    --lon -68.25 --lat -23.55 --half 0.35
cargo run --release -p insar-core --example validate_maule -- validation/atacama_export
```

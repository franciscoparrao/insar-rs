# Casos chilenos — ARIA Sentinel-1 GUNW

> Procesados 2026-06-18/19 con el pipeline ARIA nativo de insar-rs
> (descarga ASF/Earthdata → `aria_to_stack.py` → `examples/validate_maule.rs`,
> que aplica `unwrap_error` → referencia → inversión SBAS → velocidad + coherencia).
> Dos casos que ilustran el rol decisivo de la coherencia InSAR.

## Resumen

| Caso | Dato | Coherencia temporal | Resultado |
|------|------|---------------------|-----------|
| **Laguna del Maule** (volcán) — red **solo-verano** | track 83 desc, 100 ifgs, 2016–2020 | **0.92**, cobertura 52% | **Inflación −21.3 cm/año LOS recuperada** (lit. 18–27 cm/año) |
| **Salar de Atacama** (subsidencia salmuera) | track 156 desc, 67 ifgs, 2019–2020 | **0.999**, cobertura 100% | Pipeline limpio; tras corrección, residuo 0.30 cm/año |

> Nota histórica: un primer intento con red de estaciones MEZCLADAS (2017–2018,
> pares de 12–36 días) falló — la nieve invernal andina decorrelacionaba el
> centro de Maule (componente 0 → enmascarado → red local desconectada). El
> diagnóstico llevó a la red estacional; ver más abajo.

## Capacidad nueva del motor: `unwrap_error`

Ambos casos usan productos ARIA GUNW, donde cada interferograma se desenrolla en
**componentes conexas independientes** con offset 2π propio. Esto motivó el
módulo `inversion`-adyacente `unwrap_error` (corrección por cierre de fase,
Yunjun et al. 2019): usa los lazos de tripletes de la red SBAS para estimar el
entero de corrección por par y píxel. En Maule corrigió 305 k píxeles (coherencia
0.44 → 0.75); en Atacama, 706 k (toda la escena), con coherencia final 0.999.

## Laguna del Maule — recuperada con red estacional

Inflación récord (18–27 cm/año publicado), uno de los volcanes de deformación
más rápida del planeta. **Primer intento (fallido):** red de 80 pares de 12–36
días sobre 2017–2018 mezclando estaciones → el centro quedó decorrelacionado
(coherencia 0.44, centro en componente 0 por la nieve invernal) y sin señal.

**Diagnóstico:** una prueba barata mostró que un par de 12 días de *pleno verano*
(enero 2018) tiene el centro de Maule a **coherencia 0.96, 100 % fiable**. El
problema no era el motor ni la velocidad de deformación, sino **mezclar
estaciones**: las parejas de invierno (nieve) decorrelaban el centro y, al
enmascarar el componente 0, desconectaban su red SBAS local.

**Solución (red estacional):** 100 interferogramas **solo de verano austral**
(dic–mar) — pares cortos de 12–24 días *dentro* de cada verano + pares anuales
(~360–384 días) que puentean veranos consecutivos (misma estación → coherentes),
abarcando 2016–2020. Con la cadena completa (`unwrap_error` → `troposphere` →
`deramp`):

- coherencia temporal **0.92**, cobertura coherente **52 %** (vs 4 %);
- **inflación de −21.3 cm/año LOS** en el pico, **centrado exactamente en
  Laguna del Maule** (−70.51, −36.06), con serie temporal lineal limpísima sobre
  3.3 años — consistente con los 18–27 cm/año (vertical) publicados.

Lección reutilizable: en terreno nival/vegetado, **redes estacionales +
puentes anuales** mantienen la coherencia donde las redes de baseline corto
pero multi-estación fallan.

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

## Deramp nativo (`postprocess::remove_ramp`)

Añadido al motor: ajuste planar/cuadrático por mínimos cuadrados sobre píxeles
coherentes (con máscara opcional) y resta; `deramp_series` lo aplica por época.
Sobre Atacama, la velocidad cruda (−6 cm/año, dominada por la rampa de −3 cm/año
N-S) tras deramp deja **~5 cm/año de señal localizada residual** — deformación
local separada del gradiente orbital/atmosférico, consistente con subsidencia.
La interpretación geofísica fina aún pide los pasos de abajo.

## Corrección troposférica topo-correlacionada (`troposphere`)

Añadida al motor (`troposphere::correct_topo_correlated` / `correct_topo_series`,
Doin 2009 / Bekaert 2015): ajusta la relación fase-elevación y remueve la
componente dependiente de la altura. Sobre Atacama, con un DEM Copernicus GLO-30
remuestreado a la grilla (`aria_add_dem.py`), reduce drásticamente el scatter de
velocidad sobre píxeles coherentes:

| Corrección | σ velocidad (γ>0.7) | Reducción |
|------------|---------------------|-----------|
| Ninguna | 1.27 cm/año | — |
| Troposférica (topo) | **0.56 cm/año** | −56 % |
| Troposférica + deramp | 0.30 cm/año | −76 % |

La corrección topo-correlacionada sola elimina más de la mitad del scatter:
confirma que la "deformación" cruda de Atacama era mayormente atmósfera
correlacionada con la elevación (Andes altos) + rampa orbital. El residuo final
(0.30 cm/año) está al nivel de ruido para 7 meses — la subsidencia genuina de
litio (~1–2 cm/año) requiere serie más larga para emerger limpiamente.

## Relleno de huecos por kriging (enganche con geostat-rs)

Donde la coherencia cae, el campo de velocidad queda con huecos (NaN). El
motor hermano **geostat-rs** (mismo `ndarray 0.16`) los rellena por kriging
ordinario: los píxeles coherentes condicionan un variograma ajustado
automáticamente, y el kriging estima la velocidad en los huecos con su
**varianza** como incertidumbre. Ejemplo `gapfill_kriging.rs` sobre Maule
(verano): de **52 % a 100 % de cobertura**, rellenando 247 k huecos en **5.1 s**,
con el bullseye de inflación intacto y un mapa de σ que marca alta incertidumbre
donde se interpoló lejos de datos. La salida (campo continuo + capa de
incertidumbre) es la entrada natural del módulo `features` / Smelt.

## Regression kriging (insar-rs + Smelt + geostat-rs)

El método estándar-oro de predicción espacial, uniendo los tres motores
(`regression_kriging.rs`): la tendencia la modela un RandomForest de **Smelt**
sobre covariables de terreno (elevación + pendiente del DEM, disponibles en
todos los píxeles), y los residuos se krigean con **geostat-rs**; la predicción
es `tendencia(terreno) + residuo kriged` con incertidumbre. Sobre Maule rellena
247 k huecos en ~4 s.

**Hallazgo honesto:** la skill *out-of-sample* del terreno es baja (R² held-out
≈ 0.12 — el in-sample 0.86 del RF es sobreajuste), porque la inflación de Maule
es volcánica, no la manda el terreno. Ahí RK ≈ kriging ordinario, que es el
comportamiento correcto. RK aporta cuando la covariable **sí** explica la
deformación (deslizamientos ↔ pendiente, subsidencia ↔ litología): el pipeline
está listo para esos casos.

## Impacto: modelo basado en agentes (insar-rs + swarm-abm)

El último eslabón cierra la cadena **deformación → impacto** (`exposure_abm.rs`):
el campo de velocidad InSAR es el *entorno* de un modelo basado en agentes de
**swarm-abm**. El peligro por celda = |velocidad| (cm/año); agentes-población
hacen random walk sobre el terreno coherente y acumulan exposición al pisar
celdas de alta deformación. Sobre Maule (720×720, 4001 agentes, 300 pasos): la
fracción de población en peligro converge a ~3.4 % ≈ el 3.3 % de área peligrosa
(equilibrio correcto del random walk, sanity-check), y la exposición acumulada
se concentra **exactamente en el bullseye de inflación**.

Es una demo de mecánica (random walk uniforme); un estudio DRR real usaría
distribución de población real + modelo de comportamiento (evitación/evacuación).
Pero el enganche está probado: insar-rs dice *dónde y cuánto* se mueve el suelo,
swarm-abm simula *a quién afecta*.

### El stack nativo Rust, demostrado end-to-end

| Motor | Rol | Ejemplo |
|-------|-----|---------|
| insar-rs | mide deformación (SBAS, correcciones) | (núcleo) |
| geostat-rs | rellena huecos + incertidumbre | `gapfill_kriging.rs` |
| Smelt | ML (clasif/regr, CV espacial, conformal) | `landslide_smelt.rs` |
| Smelt + geostat-rs | regression kriging | `regression_kriging.rs` |
| swarm-abm | ABM de exposición/impacto | `exposure_abm.rs` |

## Próximos refinamientos sugeridos

1. ~~Deramp nativo~~ ✓ (`postprocess::remove_ramp`).
2. ~~Corrección troposférica~~ ✓ (`troposphere::correct_topo_correlated`, topo-correlacionada).
3. Serie temporal más larga (≥2 años) y combinación ascendente+descendente.
4. (Opcional) Corrección troposférica por modelo meteorológico (ERA5/GACOS) cuando las capas GUNW no sean placeholders.

## Reproducir (Atacama)

```bash
python validation/download_atacama.py            # ~7 GB, 67 GUNW (necesita ~/.netrc Earthdata)
python validation/aria_to_stack.py data/atacama_gunw --out validation/atacama_export \
    --lon -68.25 --lat -23.55 --half 0.35
cargo run --release -p insar-core --example validate_maule -- validation/atacama_export
```

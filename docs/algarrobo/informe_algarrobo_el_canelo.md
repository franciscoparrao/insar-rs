---
title: "Medición satelital de cambios de altura del terreno en Algarrobo"
subtitle: "Resultado preliminar en Playa El Canelo y propuesta de colaboración"
author: "Preparado por F. Parra"
date: "27 de junio de 2026"
lang: es-CL
geometry: "margin=2.4cm"
fontsize: 11pt
mainfont: "DejaVu Sans"
colorlinks: true
linkcolor: "RoyalBlue"
---

## 1. De qué se trata

A raíz de tu pregunta sobre si las imágenes Sentinel podrían captar cambios de
altura en las playas, montamos el flujo completo y lo probamos sobre **Playa El
Canelo, Algarrobo**. Este documento resume qué se puede medir, un resultado
preliminar concreto, y cómo se conectaría con los perfiles de playa que levantan
los memoristas.

La técnica es **InSAR**: se comparan imágenes de radar del satélite Sentinel-1
(europeo, datos gratuitos) tomadas en fechas distintas. La diferencia de fase
del radar mide cuánto se acercó o alejó el suelo del satélite, con precisión de
**milímetros a centímetros**. Combinando una pasada *ascendente* y una
*descendente* se separa el movimiento **vertical** (alzamiento o subsidencia) del
horizontal. El procesamiento lo hace un motor propio (*insar-rs*), validado
contra el estándar internacional del campo con coincidencia numérica.

## 2. Qué se puede y qué no

- **Sí se mide** el movimiento vertical del **terreno firme**: roca, terrazas,
  estructuras (casas, muelle). Sobre esos blancos la señal de radar se mantiene
  estable y coherente.
- **No se mide bien sobre la arena** suelta de la playa: las olas, la humedad y
  el movimiento del sedimento "ensucian" la señal del radar (decorrelación). Para
  la arena, los **perfiles de terreno siguen siendo insustituibles**.
- Solo se mide la costa que quedó **en tierra**; el radar no ve bajo el agua.

Esta es justamente la complementariedad útil: el radar entrega el **movimiento
del terreno** (¿la tierra subió o bajó?), y los perfiles entregan el **cambio de
la arena**. Un perfil que "subió" puede ser arena acumulada **o** tierra alzada;
el InSAR ayuda a separar esas dos causas.

## 3. Resultado preliminar en El Canelo

![Velocidad vertical del terreno sobre El Canelo y el casco de Algarrobo
(referencia: punto estable inland). Tonos azules = subsidencia; rojos =
alzamiento. El círculo marca el rasgo costero de interés.](figs/fig1_vertical.png){width=72%}

Se observa un **lóbulo de subsidencia de ~0,5–0,7 cm/año concentrado justo en la
costa de El Canelo / El Canelillo** (círculo), respecto al casco urbano de
Algarrobo tomado como referencia estable. El resto del área se mantiene
prácticamente sin movimiento.

## 4. ¿Es señal real o un artefacto? La evidencia

Antes de interpretar, sometimos el rasgo a las pruebas estándar:

![Las dos geometrías satelitales —ascendente y descendente—, que son
adquisiciones **independientes** (otras fechas, otra atmósfera), muestran ambas
el mismo descenso en la costa de El Canelo (círculo).](figs/fig2_consistencia.png){width=92%}

- **A favor de señal real:** las dos pasadas independientes coinciden
  cuantitativamente en el lóbulo (ascendente −0,67 cm/año; descendente −0,64
  cm/año). Que dos geometrías con atmósferas distintas coincidan **descarta que
  sea ruido atmosférico**, y que ambas tengan el mismo signo indica movimiento
  **vertical** (si fuera horizontal, tendrían signos opuestos). Además la
  coherencia del radar en ese punto es alta (no es un vacío de datos).

- **La cautela necesaria:** la serie temporal es **corta y ruidosa** (6 fechas
  en ~9 meses), y la tendencia la carga en parte la última fecha. No alcanza para
  **afirmar** una subsidencia sostenida.

![Serie temporal en el punto costero (respecto a la referencia). La tendencia
existe pero es ruidosa: con 9 meses no se puede confirmar.](figs/fig3_serie.png){width=78%}

**En síntesis:** es un **candidato de señal creíble, no un artefacto evidente,
pero todavía sin confirmar.** El dato más sólido —la coincidencia entre
geometrías independientes— justifica investigarlo en serio.

## 5. Qué falta para confirmarlo

1. **Más fechas** (12–20 imágenes sobre 2–3 años, con intervalos cortos) para
   bajar el piso de ruido y ver si la tendencia se sostiene.
2. **Un punto GPS/GNSS** cercano para amarrar el movimiento absoluto.
3. **Los perfiles de los memoristas** en el mismo sector: si el terreno realmente
   baja ~0,5 cm/año, debería verse en el registro de terreno, y permitiría
   separar lo que es subsidencia del suelo de lo que es dinámica de la arena.

## 6. Propuesta

Hay un **candidato concreto de subsidencia costera en El Canelo**, justo donde
ustedes ya están trabajando con perfiles. Propongo:

- Que me indiques la **zona exacta** y las **fechas de las campañas** de los
  memoristas, para alinear la serie satelital con el registro de terreno.
- Procesar una **serie más larga** (2–3 años) sobre ese sector y, si hay un GNSS
  disponible, amarrarla.
- Cruzar ambos registros: el radar dice *dónde y cuánto se mueve el terreno*; los
  perfiles dicen *qué hace la arena*. Juntos cierran la interpretación.

El flujo ya está montado y probado de extremo a extremo; replicarlo sobre el
sector y período que ustedes definan es directo.

#!/usr/bin/env python3
"""Ensambla el showcase HTML autocontenido embebiendo las figuras reales."""
import base64
import json

FIG = "/tmp/insar_isce_out/figs"
ROOT = "/home/franciscoparrao/proyectos/insar-rs"
OUT = f"{ROOT}/showcase.html"
TEMPLATE = open(f"{ROOT}/validation/showcase.tmpl.html").read()
stats = json.load(open(f"{FIG}/stats.json"))


def png_uri(name):
    b = base64.b64encode(open(f"{FIG}/{name}", "rb").read()).decode()
    return f"data:image/png;base64,{b}"


def svg(name):
    s = open(f"{FIG}/{name}").read()
    return s[s.index("<svg"):]


html = TEMPLATE
for k, v in {
    "__VELOCITY__": png_uri("velocity.png"),
    "__COHERENCE__": png_uri("coherence.png"),
    "__TIMESERIES__": svg("timeseries.svg"),
    "__PARITY__": svg("parity.svg"),
    "__PEAK_V__": f"{stats['peak']['vel_mm_yr']:.0f}",
    "__PEAK_DISP__": f"{abs(stats['peak']['total_disp_cm']):.0f}",
    "__COH_MED__": f"{stats['coh_median']:.3f}",
}.items():
    html = html.replace(k, str(v))

open(OUT, "w").write(html)
print("escrito", OUT, f"({len(html)//1024} KB)")

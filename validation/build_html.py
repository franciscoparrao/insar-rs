#!/usr/bin/env python3
"""Ensambla el showcase HTML autocontenido embebiendo las figuras reales."""
import base64
import json

FIG = "/tmp/insar_isce_out/figs"     # Fernandina
CL = "/tmp/insar_chile_figs"         # casos chilenos
ROOT = "/home/franciscoparrao/proyectos/insar-rs"
OUT = f"{ROOT}/showcase.html"
TEMPLATE = open(f"{ROOT}/validation/showcase.tmpl.html").read()
stats = json.load(open(f"{FIG}/stats.json"))
cl = json.load(open(f"{CL}/stats.json"))


def png_uri(name, base=FIG):
    b = base64.b64encode(open(f"{base}/{name}", "rb").read()).decode()
    return f"data:image/png;base64,{b}"


def svg(name, base=FIG):
    s = open(f"{base}/{name}").read()
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
    "__MAULE_VEL__": png_uri("maule_vel.png", CL),
    "__MAULE_TS__": svg("maule_ts.svg", CL),
    "__ATACAMA_VEL__": png_uri("atacama_vel.png", CL),
    "__CHAIN__": png_uri("chain.png", CL),
    "__MAULE_V__": f"{abs(cl['maule_v']):.1f}",
    "__MAULE_DISP__": f"{cl['maule_disp']:.0f}",
}.items():
    html = html.replace(k, str(v))

open(OUT, "w").write(html)
print("escrito", OUT, f"({len(html)//1024} KB)")
